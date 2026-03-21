"""Unit tests for query routing over pattern extraction outputs."""

from __future__ import annotations

import unittest

from rag.narrative.models import NarrativeGap, NarrativePhase, NarrativeReconstruction, NarrativeTransition
from rag.patterns.models import (
    EntityClusterLink,
    EntityOccurrence,
    PatternReport,
    RecurringEntity,
    TemporalBurst,
    TopicCluster,
)
from rag.patterns.router import Intent, classify_intent, route_answer, _detect_entity


def _empty_report() -> PatternReport:
    return PatternReport(
        query="test",
        entities=(),
        clusters=(),
        entity_cluster_links=(),
        temporal_bursts=(),
        evidence_count=0,
    )


def _populated_report() -> PatternReport:
    marc = RecurringEntity(
        name="Marc",
        occurrences=(
            EntityOccurrence("e1", "2025-02-04", "Marc said hello."),
            EntityOccurrence("e2", "2025-07-22", "Marc returned."),
        ),
        occurrence_count=2,
    )
    craig = RecurringEntity(
        name="Craig",
        occurrences=(
            EntityOccurrence("e1", "2025-02-04", "Craig was there."),
            EntityOccurrence("e3", "2025-10-22", "Craig at villa."),
        ),
        occurrence_count=2,
    )
    cluster_1 = TopicCluster(
        label="Marc, Craig: villa, drama",
        phase_labels=("P1", "P2"),
        evidence_ids=("e1", "e2"),
        date_range="2025-02-04 to 2025-02-15",
        key_entities=("Marc", "Craig"),
        key_terms=("villa", "drama"),
        phase_count=2,
    )
    cluster_2 = TopicCluster(
        label="Marc: home, routine",
        phase_labels=("P3", "P4"),
        evidence_ids=("e3", "e4"),
        date_range="2025-06-01 to 2025-06-05",
        key_entities=("Marc",),
        key_terms=("home", "routine"),
        phase_count=2,
    )
    link = EntityClusterLink(
        entity="Marc",
        cluster_labels=("Marc, Craig: villa, drama", "Marc: home, routine"),
        cluster_count=2,
        total_phase_count=4,
    )
    burst = TemporalBurst(
        date_range="2025-02-13 to 2025-02-15",
        phase_labels=("P1", "P2", "P3"),
        entities=("Craig", "Marc"),
        burst_size=3,
    )
    return PatternReport(
        query="Marc; Benz",
        entities=(marc, craig),
        clusters=(cluster_1, cluster_2),
        entity_cluster_links=(link,),
        temporal_bursts=(burst,),
        evidence_count=10,
    )


def _make_narrative() -> NarrativeReconstruction:
    phase = NarrativePhase(
        label="2025-02-15: Marc, Craig",
        description="Marc and Craig discussed the villa.",
        evidence_ids=("e1",),
        date_range="2025-02-15",
        support="supported",
    )
    transition = NarrativeTransition(
        from_phase="2025-02-15: Marc, Craig",
        to_phase="2025-06-01: Marc",
        description="Shift from villa drama to reconciliation",
        evidence_ids=("e1", "e2"),
        support="partially_supported",
    )
    gap = NarrativeGap(
        description="90-day gap between Feb and Jun activity",
        reason="Retrieval did not surface messages from this period",
    )
    return NarrativeReconstruction(
        query="test",
        summary="Test summary.",
        timeline=(phase,),
        transitions=(transition,),
        gaps=(gap,),
        limitations=(),
        evidence_count=5,
    )


class IntentClassificationTests(unittest.TestCase):
    """Tests for intent classification."""

    def test_entity_intent(self) -> None:
        self.assertEqual(classify_intent("who are the main people"), Intent.ENTITY)

    def test_entity_intent_variant(self) -> None:
        self.assertEqual(classify_intent("who shows up the most"), Intent.ENTITY)

    def test_theme_intent(self) -> None:
        self.assertEqual(classify_intent("what are the main themes"), Intent.THEME)

    def test_theme_intent_variant(self) -> None:
        self.assertEqual(classify_intent("what topics keep coming up"), Intent.THEME)

    def test_cross_topic_intent(self) -> None:
        self.assertEqual(classify_intent("who connects multiple topics"), Intent.CROSS_TOPIC)

    def test_cross_topic_variant(self) -> None:
        self.assertEqual(classify_intent("who appears across themes"), Intent.CROSS_TOPIC)

    def test_temporal_intent(self) -> None:
        self.assertEqual(classify_intent("when were things most active"), Intent.TEMPORAL)

    def test_temporal_variant(self) -> None:
        self.assertEqual(classify_intent("what periods were intense"), Intent.TEMPORAL)

    def test_timeline_intent(self) -> None:
        self.assertEqual(classify_intent("what changed over time"), Intent.TIMELINE)

    def test_timeline_variant(self) -> None:
        self.assertEqual(classify_intent("how did things evolve"), Intent.TIMELINE)

    def test_unknown_intent(self) -> None:
        self.assertEqual(classify_intent("tell me a joke"), Intent.UNKNOWN)

    def test_empty_query(self) -> None:
        self.assertEqual(classify_intent(""), Intent.UNKNOWN)

    def test_case_insensitive(self) -> None:
        self.assertEqual(classify_intent("WHO ARE THE MAIN PEOPLE"), Intent.ENTITY)

    def test_deterministic(self) -> None:
        """Same input always produces same intent."""
        q = "who are the recurring entities in the themes"
        results = [classify_intent(q) for _ in range(10)]
        self.assertTrue(all(r == results[0] for r in results))


class AnswerRoutingTests(unittest.TestCase):
    """Tests for answer generation via routing."""

    def test_entity_answer_lists_entities(self) -> None:
        answer = route_answer("who are the main people", _populated_report(), [])
        self.assertIn("Marc", answer)
        self.assertIn("Craig", answer)
        self.assertIn("2 occurrences", answer)

    def test_theme_answer_lists_clusters(self) -> None:
        answer = route_answer("what are the main themes", _populated_report(), [])
        self.assertIn("villa, drama", answer)
        self.assertIn("home, routine", answer)

    def test_cross_topic_answer(self) -> None:
        answer = route_answer("who connects multiple topics", _populated_report(), [])
        self.assertIn("Marc", answer)
        self.assertIn("2 clusters", answer)

    def test_temporal_answer(self) -> None:
        answer = route_answer("when were things most active", _populated_report(), [])
        self.assertIn("2025-02-13", answer)
        self.assertIn("3 phases", answer)

    def test_timeline_answer_uses_transitions(self) -> None:
        answer = route_answer("what changed over time", _populated_report(), [_make_narrative()])
        self.assertIn("reconciliation", answer)

    def test_timeline_answer_uses_gaps(self) -> None:
        answer = route_answer("what changed over time", _populated_report(), [_make_narrative()])
        self.assertIn("90-day gap", answer)

    def test_unknown_query_lists_capabilities(self) -> None:
        answer = route_answer("tell me a joke", _populated_report(), [])
        self.assertIn("I can answer", answer)
        self.assertIn("main people", answer)

    def test_entity_answer_empty_report(self) -> None:
        answer = route_answer("who are the main people", _empty_report(), [])
        self.assertIn("No recurring entities", answer)

    def test_theme_answer_empty_report(self) -> None:
        answer = route_answer("what are the main themes", _empty_report(), [])
        self.assertIn("No topic clusters", answer)

    def test_cross_topic_empty(self) -> None:
        answer = route_answer("who connects multiple topics", _empty_report(), [])
        self.assertIn("No entities bridge", answer)

    def test_temporal_empty(self) -> None:
        answer = route_answer("when were things most active", _empty_report(), [])
        self.assertIn("No periods", answer)

    def test_answers_contain_no_speculation(self) -> None:
        """Answers should not contain speculative language."""
        report = _populated_report()
        for q in ["who are the main people", "what are the main themes",
                   "who connects multiple topics", "when were things most active"]:
            answer = route_answer(q, report, [])
            for word in ("probably", "might", "maybe", "possibly", "I think"):
                self.assertNotIn(word, answer, f"Speculative word '{word}' in answer for '{q}'")

    def test_deterministic_answers(self) -> None:
        """Same input always produces same answer."""
        report = _populated_report()
        answers = [route_answer("who are the main people", report, []) for _ in range(5)]
        self.assertTrue(all(a == answers[0] for a in answers))


class EntityDetectionTests(unittest.TestCase):
    """Tests for entity detection from queries."""

    def test_entity_detected_from_query(self) -> None:
        report = _populated_report()
        self.assertEqual(_detect_entity("what happened with Marc", report), "Marc")

    def test_entity_detected_case_insensitive(self) -> None:
        report = _populated_report()
        self.assertEqual(_detect_entity("what happened with marc", report), "Marc")

    def test_highest_frequency_entity_selected(self) -> None:
        """When multiple entities match, highest occurrence count wins."""
        marc = RecurringEntity(
            name="Marc",
            occurrences=(
                EntityOccurrence("e1", "2025-02-04", "Marc said hello."),
                EntityOccurrence("e2", "2025-07-22", "Marc returned."),
                EntityOccurrence("e3", "2025-08-01", "Marc again."),
            ),
            occurrence_count=3,
        )
        craig = RecurringEntity(
            name="Craig",
            occurrences=(EntityOccurrence("e4", "2025-02-04", "Craig was there."),),
            occurrence_count=1,
        )
        report = PatternReport(
            query="test", entities=(marc, craig), clusters=(),
            entity_cluster_links=(), temporal_bursts=(), evidence_count=0,
        )
        self.assertEqual(_detect_entity("Marc and Craig were there", report), "Marc")

    def test_no_entity_match_returns_none(self) -> None:
        report = _populated_report()
        self.assertIsNone(_detect_entity("what happened with Zephyr", report))

    def test_whole_word_match_only(self) -> None:
        """'Marc' should not match inside 'Marcelino'."""
        entity = RecurringEntity(
            name="Marc",
            occurrences=(EntityOccurrence("e1", "2025-01-01", "Marc."),),
            occurrence_count=1,
        )
        report = PatternReport(
            query="test", entities=(entity,), clusters=(),
            entity_cluster_links=(), temporal_bursts=(), evidence_count=0,
        )
        self.assertIsNone(_detect_entity("Marcelino went home", report))

    def test_entity_with_punctuation_in_query(self) -> None:
        report = _populated_report()
        self.assertEqual(_detect_entity("what happened with Marc?", report), "Marc")


class EntityScopedIntentTests(unittest.TestCase):
    """Tests for ENTITY_SCOPED intent classification."""

    def test_entity_scoped_intent_basic(self) -> None:
        report = _populated_report()
        self.assertEqual(classify_intent("what happened with Marc", report), Intent.ENTITY_SCOPED)

    def test_entity_scoped_with_how(self) -> None:
        report = _populated_report()
        self.assertEqual(classify_intent("how did things go with Craig", report), Intent.ENTITY_SCOPED)

    def test_entity_name_only_not_scoped(self) -> None:
        """Entity name alone without question pattern should not trigger ENTITY_SCOPED."""
        report = _populated_report()
        result = classify_intent("Marc", report)
        self.assertNotEqual(result, Intent.ENTITY_SCOPED)

    def test_question_without_entity_not_scoped(self) -> None:
        """Question patterns without a known entity should fall through to keyword scoring."""
        report = _populated_report()
        result = classify_intent("what happened with Zephyr", report)
        self.assertNotEqual(result, Intent.ENTITY_SCOPED)

    def test_pure_entity_query_not_overridden(self) -> None:
        """'who are the main people' should stay ENTITY, not become ENTITY_SCOPED."""
        report = _populated_report()
        self.assertEqual(classify_intent("who are the main people", report), Intent.ENTITY)

    def test_no_report_falls_through(self) -> None:
        """Without a report, classify_intent should not produce ENTITY_SCOPED."""
        result = classify_intent("what happened with Marc")
        self.assertNotEqual(result, Intent.ENTITY_SCOPED)

    def test_deterministic(self) -> None:
        report = _populated_report()
        results = [classify_intent("what happened with Marc", report) for _ in range(10)]
        self.assertTrue(all(r == Intent.ENTITY_SCOPED for r in results))


class EntityScopedAnswerTests(unittest.TestCase):
    """Tests for entity-scoped answer formatting."""

    def test_scoped_answer_contains_entity_name(self) -> None:
        report = _populated_report()
        answer = route_answer("what happened with Marc", report, [])
        self.assertIn("Marc", answer)

    def test_scoped_answer_contains_occurrence_count(self) -> None:
        report = _populated_report()
        answer = route_answer("what happened with Marc", report, [])
        self.assertIn("2 occurrences", answer)

    def test_scoped_answer_shows_relevant_clusters_only(self) -> None:
        report = _populated_report()
        answer = route_answer("what happened with Marc", report, [])
        # Marc is in both clusters.
        self.assertIn("villa, drama", answer)
        self.assertIn("home, routine", answer)

    def test_scoped_answer_filters_clusters(self) -> None:
        """Craig should only see clusters containing Craig."""
        report = _populated_report()
        answer = route_answer("what happened with Craig", report, [])
        self.assertIn("villa, drama", answer)
        # Craig is NOT in the "Marc: home, routine" cluster.
        self.assertNotIn("home, routine", answer)

    def test_scoped_answer_shows_bursts(self) -> None:
        report = _populated_report()
        answer = route_answer("what happened with Marc", report, [])
        self.assertIn("2025-02-13", answer)

    def test_scoped_answer_filters_bursts(self) -> None:
        """Entity not in any burst should not show burst section."""
        entity = RecurringEntity(
            name="Benz",
            occurrences=(EntityOccurrence("e9", "2025-05-01", "Benz arrived."),),
            occurrence_count=1,
        )
        report = PatternReport(
            query="test",
            entities=(entity,),
            clusters=(),
            entity_cluster_links=(),
            temporal_bursts=_populated_report().temporal_bursts,
            evidence_count=5,
        )
        answer = route_answer("what happened with Benz", report, [])
        self.assertNotIn("burst", answer.lower())

    def test_scoped_answer_no_cross_entity_leakage(self) -> None:
        """Scoped answer for Craig should not mention Marc's solo clusters or links."""
        report = _populated_report()
        answer = route_answer("what happened with Craig", report, [])
        # Craig has no entity_cluster_links.
        self.assertNotIn("Bridges", answer)

    def test_scoped_answer_shows_key_mentions(self) -> None:
        report = _populated_report()
        answer = route_answer("what happened with Marc", report, [])
        self.assertIn("Marc said hello", answer)

    def test_scoped_answer_unknown_entity_fallback(self) -> None:
        report = _populated_report()
        answer = route_answer("what happened with Zephyr", report, [])
        # Should fall through to keyword scoring, not ENTITY_SCOPED.
        self.assertNotIn("occurrences", answer)

    def test_scoped_answer_deterministic(self) -> None:
        report = _populated_report()
        answers = [route_answer("what happened with Marc", report, []) for _ in range(5)]
        self.assertTrue(all(a == answers[0] for a in answers))

    def test_scoped_answer_no_speculation(self) -> None:
        report = _populated_report()
        answer = route_answer("what happened with Marc", report, [])
        for word in ("probably", "might", "maybe", "possibly", "I think"):
            self.assertNotIn(word, answer)


if __name__ == "__main__":
    unittest.main()
