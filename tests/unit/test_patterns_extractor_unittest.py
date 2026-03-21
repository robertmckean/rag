"""Unit tests for recurring-entity extraction (Phase 7)."""

from __future__ import annotations

import unittest

from rag.narrative.models import NarrativePhase, NarrativeReconstruction
from rag.patterns.extractor import extract_recurring_entities
from rag.patterns.models import PatternReport


def _make_phase(
    label: str,
    description: str,
    evidence_ids: tuple[str, ...],
    date_range: str | None = None,
) -> NarrativePhase:
    return NarrativePhase(
        label=label,
        description=description,
        evidence_ids=evidence_ids,
        date_range=date_range,
        support="supported" if len(evidence_ids) == 1 else "partially_supported",
    )


def _make_narrative(
    query: str,
    phases: tuple[NarrativePhase, ...],
    evidence_count: int = 0,
) -> NarrativeReconstruction:
    return NarrativeReconstruction(
        query=query,
        summary="test",
        timeline=phases,
        transitions=(),
        gaps=(),
        limitations=(),
        evidence_count=evidence_count,
    )


class RecurringEntityExtractionTests(unittest.TestCase):
    """Core extraction logic tests."""

    def test_entity_in_two_phases_included(self) -> None:
        """Same entity in 2+ distinct phases -> included."""
        phases = (
            _make_phase("Phase 1", '(2025-01-01) [user] "Marc said hello."', ("e1",), "2025-01-01"),
            _make_phase("Phase 2", '(2025-02-01) [user] "Marc returned from trip."', ("e2",), "2025-02-01"),
        )
        narrative = _make_narrative("What about Marc?", phases, evidence_count=2)
        report = extract_recurring_entities([narrative])

        entity_names = [e.name for e in report.entities]
        self.assertIn("Marc", entity_names)

    def test_entity_in_one_phase_excluded(self) -> None:
        """Entity appearing in only 1 phase -> excluded."""
        phases = (
            _make_phase("Phase 1", '(2025-01-01) [user] "Marc said hello."', ("e1",), "2025-01-01"),
            _make_phase("Phase 2", '(2025-02-01) [user] "Benz went home."', ("e2",), "2025-02-01"),
        )
        narrative = _make_narrative("test", phases, evidence_count=2)
        report = extract_recurring_entities([narrative])

        entity_names = [e.name for e in report.entities]
        self.assertNotIn("Marc", entity_names)
        self.assertNotIn("Benz", entity_names)

    def test_entity_across_multiple_narratives(self) -> None:
        """Same entity across different narratives -> included."""
        narr_1 = _make_narrative(
            "query 1",
            (_make_phase("P1", '[user] "Craig was there."', ("e1",)),),
            evidence_count=1,
        )
        narr_2 = _make_narrative(
            "query 2",
            (_make_phase("P2", '[user] "Craig called back."', ("e3",)),),
            evidence_count=1,
        )
        report = extract_recurring_entities([narr_1, narr_2])

        entity_names = [e.name for e in report.entities]
        self.assertIn("Craig", entity_names)

    def test_noise_terms_excluded(self) -> None:
        """Common English words and noise terms should not appear as entities."""
        phases = (
            _make_phase("P1", 'The Maybe Really good day was fine.', ("e1",)),
            _make_phase("P2", 'The Maybe Actually Really important thing.', ("e2",)),
        )
        narrative = _make_narrative("test", phases, evidence_count=2)
        report = extract_recurring_entities([narrative])

        entity_names = {e.name for e in report.entities}
        # These are all in _NON_ENTITY_WORDS or _LABEL_NOISE_WORDS.
        self.assertNotIn("The", entity_names)
        self.assertNotIn("Maybe", entity_names)
        self.assertNotIn("Really", entity_names)
        self.assertNotIn("Actually", entity_names)

    def test_deterministic_entity_ordering(self) -> None:
        """Entities ordered by occurrence_count desc, then name asc."""
        phases = (
            _make_phase("P1", 'Marc and Benz and Zeno talked.', ("e1",), "2025-01-01"),
            _make_phase("P2", 'Marc and Benz discussed plans.', ("e2",), "2025-02-01"),
            _make_phase("P3", 'Marc went alone.', ("e3",), "2025-03-01"),
        )
        narrative = _make_narrative("test", phases, evidence_count=3)
        report = extract_recurring_entities([narrative])

        names = [e.name for e in report.entities]
        # Marc: 3 phases, Benz: 2 phases, Zeno: 1 phase (excluded).
        self.assertEqual(names[0], "Marc")
        self.assertEqual(names[1], "Benz")
        self.assertNotIn("Zeno", names)

    def test_deterministic_occurrence_ordering(self) -> None:
        """Occurrences ordered by created_at asc, nulls last."""
        phases = (
            _make_phase("P1", '[user] "Marc early"', ("e1",), "2025-03-01"),
            _make_phase("P2", '[user] "Marc middle"', ("e2",), "2025-01-15"),
            _make_phase("P3", '[user] "Marc late"', ("e3",), None),
        )
        narrative = _make_narrative("test", phases, evidence_count=3)
        report = extract_recurring_entities([narrative])

        marc = [e for e in report.entities if e.name == "Marc"][0]
        dates = [o.created_at for o in marc.occurrences]
        # Sorted: 2025-01-15, 2025-03-01, None.
        self.assertEqual(dates, ["2025-01-15", "2025-03-01", None])

    def test_occurrence_count_matches_occurrences(self) -> None:
        """occurrence_count must equal len(occurrences)."""
        phases = (
            _make_phase("P1", 'Marc was here.', ("e1",), "2025-01-01"),
            _make_phase("P2", 'Marc was there.', ("e2",), "2025-02-01"),
            _make_phase("P3", 'Marc went home.', ("e3",), "2025-03-01"),
        )
        narrative = _make_narrative("test", phases, evidence_count=3)
        report = extract_recurring_entities([narrative])

        for entity in report.entities:
            self.assertEqual(entity.occurrence_count, len(entity.occurrences))

    def test_empty_narratives_returns_empty_report(self) -> None:
        """No narratives -> empty PatternReport."""
        report = extract_recurring_entities([])
        self.assertEqual(report.query, "")
        self.assertEqual(report.entities, ())
        self.assertEqual(report.evidence_count, 0)

    def test_narrative_with_no_phases(self) -> None:
        """Narrative with empty timeline -> empty entities."""
        narrative = _make_narrative("test", (), evidence_count=0)
        report = extract_recurring_entities([narrative])
        self.assertEqual(report.entities, ())

    def test_same_phase_not_double_counted(self) -> None:
        """Same phase object referenced should not create duplicate occurrences."""
        phase = _make_phase("P1", 'Marc talked about Craig.', ("e1",), "2025-01-01")
        # Two narratives sharing the exact same phase (same evidence_ids).
        narr_1 = _make_narrative("q1", (phase,), evidence_count=1)
        narr_2 = _make_narrative("q2", (phase,), evidence_count=1)
        report = extract_recurring_entities([narr_1, narr_2])

        # Even though Marc appears in 2 narratives, it's the same phase
        # (same evidence_ids), so only 1 unique occurrence — not recurring.
        entity_names = [e.name for e in report.entities]
        self.assertNotIn("Marc", entity_names)

    def test_evidence_id_grounded_to_phase(self) -> None:
        """Each occurrence's evidence_id is the first evidence_id from the phase."""
        phases = (
            _make_phase("P1", 'Marc and Craig talked.', ("e1", "e2"), "2025-01-01"),
            _make_phase("P2", 'Marc returned.', ("e3",), "2025-02-01"),
        )
        narrative = _make_narrative("test", phases, evidence_count=3)
        report = extract_recurring_entities([narrative])

        marc = [e for e in report.entities if e.name == "Marc"][0]
        evidence_ids = [o.evidence_id for o in marc.occurrences]
        self.assertEqual(evidence_ids, ["e1", "e3"])

    def test_excerpt_is_phase_label(self) -> None:
        """Each occurrence's excerpt is the phase label for compact context."""
        phases = (
            _make_phase("2025-01-01: Marc", 'Marc did X.', ("e1",), "2025-01-01"),
            _make_phase("2025-02-01: Marc", 'Marc did Y.', ("e2",), "2025-02-01"),
        )
        narrative = _make_narrative("test", phases, evidence_count=2)
        report = extract_recurring_entities([narrative])

        marc = [e for e in report.entities if e.name == "Marc"][0]
        self.assertEqual(marc.occurrences[0].excerpt, "2025-01-01: Marc")
        self.assertEqual(marc.occurrences[1].excerpt, "2025-02-01: Marc")

    def test_query_joined_from_multiple_narratives(self) -> None:
        """PatternReport.query joins unique queries from all narratives."""
        narr_1 = _make_narrative(
            "What about Marc?",
            (_make_phase("P1", 'Marc said hi.', ("e1",)),),
            evidence_count=1,
        )
        narr_2 = _make_narrative(
            "Tell me about Craig.",
            (_make_phase("P2", 'Marc called Craig.', ("e3",)),),
            evidence_count=1,
        )
        report = extract_recurring_entities([narr_1, narr_2])
        self.assertEqual(report.query, "What about Marc?; Tell me about Craig.")

    def test_evidence_count_summed(self) -> None:
        """PatternReport.evidence_count sums across all narratives."""
        narr_1 = _make_narrative("q1", (), evidence_count=5)
        narr_2 = _make_narrative("q2", (), evidence_count=3)
        report = extract_recurring_entities([narr_1, narr_2])
        self.assertEqual(report.evidence_count, 8)

    def test_report_serialization(self) -> None:
        """Full report serializes without error."""
        phases = (
            _make_phase("P1", 'Marc and Craig talked.', ("e1",), "2025-01-01"),
            _make_phase("P2", 'Marc returned.', ("e2",), "2025-02-01"),
        )
        narrative = _make_narrative("test query", phases, evidence_count=2)
        report = extract_recurring_entities([narrative])
        d = report.to_dict()
        self.assertEqual(d["query"], "test query")
        self.assertIsInstance(d["entities"], list)


class AliasNormalizationTests(unittest.TestCase):
    """Tests for alias handling during extraction."""

    def test_alias_merge_across_phases(self) -> None:
        """Mark in one phase + Marc in another -> merged under 'Marc'."""
        phases = (
            _make_phase("P1", 'Mark said hello.', ("e1",), "2025-01-01"),
            _make_phase("P2", 'Marc returned.', ("e2",), "2025-02-01"),
        )
        narrative = _make_narrative("test", phases, evidence_count=2)
        report = extract_recurring_entities([narrative])

        entity_names = [e.name for e in report.entities]
        self.assertIn("Marc", entity_names)
        self.assertNotIn("Mark", entity_names)

    def test_no_merge_for_unmapped_names(self) -> None:
        """Names not in alias map stay separate."""
        phases = (
            _make_phase("P1", 'Craig was here.', ("e1",), "2025-01-01"),
            _make_phase("P2", 'Benz was there.', ("e2",), "2025-02-01"),
        )
        narrative = _make_narrative("test", phases, evidence_count=2)
        report = extract_recurring_entities([narrative])

        entity_names = [e.name for e in report.entities]
        self.assertNotIn("Craig", entity_names)
        self.assertNotIn("Benz", entity_names)

    def test_cooccurrence_does_not_block_explicit_alias(self) -> None:
        """Both Marc and Mark in the same phase -> still merged (alias is authoritative)."""
        phases = (
            _make_phase("P1", 'Marc and Mark both appeared.', ("e1",), "2025-01-01"),
            _make_phase("P2", 'Marc returned alone.', ("e2",), "2025-02-01"),
        )
        narrative = _make_narrative("test", phases, evidence_count=2)
        report = extract_recurring_entities([narrative])

        entity_names = [e.name for e in report.entities]
        # Explicit alias merges unconditionally: Marc appears in both phases.
        self.assertIn("Marc", entity_names)
        self.assertNotIn("Mark", entity_names)

    def test_cooccurrence_single_phase_occurrence_per_canonical(self) -> None:
        """Both variants in one phase produce only one occurrence for that phase."""
        phases = (
            _make_phase("P1", 'Marc and Mark both appeared.', ("e1",), "2025-01-01"),
            _make_phase("P2", 'Marc returned.', ("e2",), "2025-02-01"),
        )
        narrative = _make_narrative("test", phases, evidence_count=2)
        report = extract_recurring_entities([narrative])

        marc = [e for e in report.entities if e.name == "Marc"][0]
        # P1 should produce exactly one occurrence despite both variants present.
        self.assertEqual(marc.occurrence_count, 2)
        evidence_ids = [o.evidence_id for o in marc.occurrences]
        self.assertEqual(evidence_ids, ["e1", "e2"])

    def test_alias_merge_occurrence_count_correct(self) -> None:
        """occurrence_count reflects merged occurrences."""
        phases = (
            _make_phase("P1", 'Mark said hello.', ("e1",), "2025-01-01"),
            _make_phase("P2", 'Marc returned.', ("e2",), "2025-02-01"),
            _make_phase("P3", 'Marc again.', ("e3",), "2025-03-01"),
        )
        narrative = _make_narrative("test", phases, evidence_count=3)
        report = extract_recurring_entities([narrative])

        marc = [e for e in report.entities if e.name == "Marc"][0]
        self.assertEqual(marc.occurrence_count, 3)
        self.assertEqual(marc.occurrence_count, len(marc.occurrences))

    def test_alias_merge_deterministic_ordering(self) -> None:
        """Merged entity preserves deterministic occurrence ordering."""
        phases = (
            _make_phase("P1", 'Mark started.', ("e1",), "2025-03-01"),
            _make_phase("P2", 'Marc continued.', ("e2",), "2025-01-15"),
        )
        narrative = _make_narrative("test", phases, evidence_count=2)
        report = extract_recurring_entities([narrative])

        marc = [e for e in report.entities if e.name == "Marc"][0]
        dates = [o.created_at for o in marc.occurrences]
        self.assertEqual(dates, ["2025-01-15", "2025-03-01"])

    def test_existing_nonalias_behavior_unchanged(self) -> None:
        """Entities not in alias map behave exactly as before."""
        phases = (
            _make_phase("P1", 'Craig and Benz talked.', ("e1",), "2025-01-01"),
            _make_phase("P2", 'Craig returned.', ("e2",), "2025-02-01"),
            _make_phase("P3", 'Benz went home.', ("e3",), "2025-03-01"),
        )
        narrative = _make_narrative("test", phases, evidence_count=3)
        report = extract_recurring_entities([narrative])

        entity_names = [e.name for e in report.entities]
        # Craig: P1 + P2 = 2 phases -> included.
        self.assertIn("Craig", entity_names)
        # Benz: P1 + P3 = 2 phases -> included.
        self.assertIn("Benz", entity_names)


if __name__ == "__main__":
    unittest.main()
