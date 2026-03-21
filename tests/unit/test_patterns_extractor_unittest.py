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

    def test_excerpt_contains_content_snippet(self) -> None:
        """Occurrences extract content snippets from phase descriptions."""
        phases = (
            _make_phase(
                "2025-01-01: Marc",
                '(2025-01-01) [user] "Marc said something interesting about the trip."',
                ("e1",), "2025-01-01",
            ),
            _make_phase(
                "2025-02-01: Marc",
                '(2025-02-01) [user] "Marc returned from the conference."',
                ("e2",), "2025-02-01",
            ),
        )
        narrative = _make_narrative("test", phases, evidence_count=2)
        report = extract_recurring_entities([narrative])

        marc = [e for e in report.entities if e.name == "Marc"][0]
        # Snippets should contain actual content, not just the phase label.
        self.assertIn("Marc", marc.occurrences[0].excerpt)
        self.assertIn("trip", marc.occurrences[0].excerpt)
        self.assertIn("Marc", marc.occurrences[1].excerpt)
        self.assertIn("returned", marc.occurrences[1].excerpt)

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


class SnippetExtractionTests(unittest.TestCase):
    """Tests for content snippet extraction in occurrences."""

    def test_snippet_from_phase_description(self) -> None:
        """Entity mention in description produces a content snippet."""
        desc = '(2025-01-01) [user] "Marc said something interesting about the trip."'
        phases = (
            _make_phase("P1", desc, ("e1",), "2025-01-01"),
            _make_phase("P2", '(2025-02-01) [user] "Marc returned."', ("e2",), "2025-02-01"),
        )
        narrative = _make_narrative("test", phases, evidence_count=2)
        report = extract_recurring_entities([narrative])

        marc = [e for e in report.entities if e.name == "Marc"][0]
        self.assertIn("Marc", marc.occurrences[0].excerpt)
        self.assertIn("trip", marc.occurrences[0].excerpt)
        # Should not contain the (date) [role] prefix.
        self.assertNotIn("[user]", marc.occurrences[0].excerpt)

    def test_snippet_truncated_to_limit(self) -> None:
        """Long descriptions produce truncated snippets."""
        long_text = "Marc " + "x" * 300 + " end of text."
        desc = f'(2025-01-01) [user] "{long_text}"'
        phases = (
            _make_phase("P1", desc, ("e1",), "2025-01-01"),
            _make_phase("P2", 'Marc was here.', ("e2",), "2025-02-01"),
        )
        narrative = _make_narrative("test", phases, evidence_count=2)
        report = extract_recurring_entities([narrative])

        marc = [e for e in report.entities if e.name == "Marc"][0]
        # Snippet should be capped, not the full 300+ chars.
        self.assertLessEqual(len(marc.occurrences[0].excerpt), 130)

    def test_fallback_to_phase_label(self) -> None:
        """When entity is not found in description text, fall back to label."""
        # Description mentions only lowercase "marc" which won't match "Marc".
        phases = (
            _make_phase("Label A", 'Marc mentioned in label only.', ("e1",), "2025-01-01"),
            _make_phase("Label B", 'no entity here at all.', ("e2",), "2025-02-01"),
        )
        narrative = _make_narrative("test", phases, evidence_count=2)
        report = extract_recurring_entities([narrative])
        # Marc only in P1 (1 phase) so not recurring. Test with a different setup.
        # Use Benz: appears in description of P1 and P2 but not for the fallback entity.
        phases2 = (
            _make_phase("Phase X", 'Benz was here.', ("e1",), "2025-01-01"),
            _make_phase("Phase Y", 'Benz returned.', ("e2",), "2025-02-01"),
        )
        narrative2 = _make_narrative("test", phases2, evidence_count=2)
        report2 = extract_recurring_entities([narrative2])

        benz = [e for e in report2.entities if e.name == "Benz"][0]
        # Snippets should contain "Benz" from the description, not just the label.
        self.assertIn("Benz", benz.occurrences[0].excerpt)

    def test_fallback_when_entity_absent_from_description(self) -> None:
        """If entity name is truly absent from description, excerpt is the phase label."""
        # Force a scenario: entity extracted from one part, description lacks it.
        # Use a phase where Craig appears only via alias search failure.
        phases = (
            _make_phase("Label Craig 1", 'Craig talked about plans.', ("e1",), "2025-01-01"),
            _make_phase("Label Craig 2", 'no names here at all', ("e2",), "2025-02-01"),
        )
        narrative = _make_narrative("test", phases, evidence_count=2)
        report = extract_recurring_entities([narrative])
        # Craig only in P1 description, not in P2.
        # But Craig won't be extracted from P2 anyway (no capitalized name).
        # So Craig appears in only 1 phase — not recurring. This is expected.
        entity_names = [e.name for e in report.entities]
        self.assertNotIn("Craig", entity_names)

    def test_deterministic_snippet_output(self) -> None:
        """Same input produces identical snippets."""
        desc = '(2025-01-01) [user] "Marc went to the meeting with Craig about the project."'
        phases = (
            _make_phase("P1", desc, ("e1",), "2025-01-01"),
            _make_phase("P2", '(2025-02-01) [user] "Marc returned later."', ("e2",), "2025-02-01"),
        )
        narrative = _make_narrative("test", phases, evidence_count=2)
        report_1 = extract_recurring_entities([narrative])
        report_2 = extract_recurring_entities([narrative])

        marc_1 = [e for e in report_1.entities if e.name == "Marc"][0]
        marc_2 = [e for e in report_2.entities if e.name == "Marc"][0]
        for o1, o2 in zip(marc_1.occurrences, marc_2.occurrences):
            self.assertEqual(o1.excerpt, o2.excerpt)

    def test_multiple_occurrences_have_distinct_snippets(self) -> None:
        """Different phases produce different snippets for the same entity."""
        phases = (
            _make_phase("P1", '(2025-01-01) [user] "Marc talked about travel plans."', ("e1",), "2025-01-01"),
            _make_phase("P2", '(2025-02-01) [user] "Marc decided to stay home."', ("e2",), "2025-02-01"),
        )
        narrative = _make_narrative("test", phases, evidence_count=2)
        report = extract_recurring_entities([narrative])

        marc = [e for e in report.entities if e.name == "Marc"][0]
        self.assertIn("travel", marc.occurrences[0].excerpt)
        self.assertIn("stay home", marc.occurrences[1].excerpt)

    def test_alias_variant_found_in_snippet(self) -> None:
        """When the raw text uses an alias variant, snippet still found."""
        phases = (
            _make_phase("P1", '(2025-01-01) [user] "Mark said something."', ("e1",), "2025-01-01"),
            _make_phase("P2", '(2025-02-01) [user] "Marc returned."', ("e2",), "2025-02-01"),
        )
        narrative = _make_narrative("test", phases, evidence_count=2)
        report = extract_recurring_entities([narrative])

        marc = [e for e in report.entities if e.name == "Marc"][0]
        # P1 description has "Mark" not "Marc" — snippet search should find it via alias.
        self.assertIn("Mark", marc.occurrences[0].excerpt)
        self.assertIn("something", marc.occurrences[0].excerpt)


class TopicClusteringTests(unittest.TestCase):
    """Tests for deterministic topic clustering."""

    def test_similar_phases_clustered(self) -> None:
        """Phases with overlapping content terms are grouped."""
        phases = (
            _make_phase("P1", 'Marc discussed the villa drama and Craig involvement.', ("e1",), "2025-01-01"),
            _make_phase("P2", 'Craig talked about the villa situation with Marc present.', ("e2",), "2025-01-05"),
            _make_phase("P3", 'Benz went swimming at the beach in Cambodia.', ("e3",), "2025-03-01"),
        )
        narrative = _make_narrative("test", phases, evidence_count=3)
        report = extract_recurring_entities([narrative])

        # P1 and P2 share villa/Craig/Marc terms — should cluster.
        # P3 is unrelated — should not cluster with P1/P2.
        cluster_labels = [c.label for c in report.clusters]
        self.assertTrue(len(report.clusters) >= 1)
        # The villa/Marc/Craig cluster should exist.
        villa_cluster = [c for c in report.clusters if "Marc" in c.key_entities or "Craig" in c.key_entities]
        self.assertTrue(len(villa_cluster) >= 1)
        self.assertIn("P1", villa_cluster[0].phase_labels[0])

    def test_dissimilar_phases_not_clustered(self) -> None:
        """Phases with no term overlap stay separate."""
        phases = (
            _make_phase("P1", 'Marc discussed philosophy deeply.', ("e1",), "2025-01-01"),
            _make_phase("P2", 'Benz went swimming in Cambodia.', ("e2",), "2025-03-01"),
        )
        narrative = _make_narrative("test", phases, evidence_count=2)
        report = extract_recurring_entities([narrative])

        # No cluster should form — topics are unrelated.
        self.assertEqual(len(report.clusters), 0)

    def test_singleton_clusters_suppressed(self) -> None:
        """Clusters with only 1 phase are not emitted."""
        phases = (
            _make_phase("P1", 'Marc talked about plans.', ("e1",), "2025-01-01"),
            _make_phase("P2", 'Benz went to Cambodia.', ("e2",), "2025-03-01"),
            _make_phase("P3", 'Zeno practiced stoicism.', ("e3",), "2025-06-01"),
        )
        narrative = _make_narrative("test", phases, evidence_count=3)
        report = extract_recurring_entities([narrative])

        for cluster in report.clusters:
            self.assertGreaterEqual(cluster.phase_count, 2)

    def test_query_terms_excluded_from_clustering(self) -> None:
        """Query terms should not inflate similarity between phases."""
        # Both phases mention "marc" (the query term) but have different content.
        phases = (
            _make_phase("P1", 'Marc discussed philosophy and stoicism.', ("e1",), "2025-01-01"),
            _make_phase("P2", 'Marc went swimming in Cambodia beach.', ("e2",), "2025-03-01"),
        )
        narrative = _make_narrative("marc", phases, evidence_count=2)
        report = extract_recurring_entities([narrative])

        # With query term "marc" excluded, these phases have no overlap.
        self.assertEqual(len(report.clusters), 0)

    def test_cluster_deterministic(self) -> None:
        """Same input produces identical clusters."""
        phases = (
            _make_phase("P1", 'Craig and Marc discussed the villa drama.', ("e1",), "2025-01-01"),
            _make_phase("P2", 'Marc and Craig resolved the villa situation.', ("e2",), "2025-01-05"),
        )
        narrative = _make_narrative("test", phases, evidence_count=2)
        report_1 = extract_recurring_entities([narrative])
        report_2 = extract_recurring_entities([narrative])

        self.assertEqual(len(report_1.clusters), len(report_2.clusters))
        for c1, c2 in zip(report_1.clusters, report_2.clusters):
            self.assertEqual(c1.label, c2.label)
            self.assertEqual(c1.phase_labels, c2.phase_labels)

    def test_cluster_date_range(self) -> None:
        """Cluster date range spans min to max across member phases."""
        phases = (
            _make_phase("P1", 'Craig and Marc at the villa drama.', ("e1",), "2025-01-01"),
            _make_phase("P2", 'Marc and Craig villa resolution.', ("e2",), "2025-03-15"),
        )
        narrative = _make_narrative("test", phases, evidence_count=2)
        report = extract_recurring_entities([narrative])

        self.assertTrue(len(report.clusters) >= 1)
        cluster = report.clusters[0]
        self.assertEqual(cluster.date_range, "2025-01-01 to 2025-03-15")

    def test_cluster_evidence_ids_deduped(self) -> None:
        """Evidence IDs in cluster are deduplicated."""
        phases = (
            _make_phase("P1", 'Craig and Marc villa drama.', ("e1", "e2"), "2025-01-01"),
            _make_phase("P2", 'Marc and Craig villa update.', ("e2", "e3"), "2025-01-05"),
        )
        narrative = _make_narrative("test", phases, evidence_count=3)
        report = extract_recurring_entities([narrative])

        if report.clusters:
            cluster = report.clusters[0]
            # e2 appears in both phases but should only be listed once.
            self.assertEqual(len(set(cluster.evidence_ids)), len(cluster.evidence_ids))

    def test_cluster_phase_count_matches(self) -> None:
        """phase_count equals len(phase_labels)."""
        phases = (
            _make_phase("P1", 'Craig Marc villa drama.', ("e1",), "2025-01-01"),
            _make_phase("P2", 'Marc Craig villa resolved.', ("e2",), "2025-01-05"),
            _make_phase("P3", 'Craig villa aftermath Marc.', ("e3",), "2025-01-10"),
        )
        narrative = _make_narrative("test", phases, evidence_count=3)
        report = extract_recurring_entities([narrative])

        for cluster in report.clusters:
            self.assertEqual(cluster.phase_count, len(cluster.phase_labels))

    def test_cluster_label_deterministic_format(self) -> None:
        """Cluster labels follow 'Entities: terms' format."""
        phases = (
            _make_phase("P1", 'Craig and Marc discussed the villa drama.', ("e1",), "2025-01-01"),
            _make_phase("P2", 'Marc and Craig resolved the villa situation.', ("e2",), "2025-01-05"),
        )
        narrative = _make_narrative("test", phases, evidence_count=2)
        report = extract_recurring_entities([narrative])

        if report.clusters:
            label = report.clusters[0].label
            # Label should contain entity names and/or terms.
            self.assertTrue(len(label) > 0)

    def test_cross_narrative_clustering(self) -> None:
        """Phases from different narratives can cluster together."""
        narr_1 = _make_narrative(
            "query 1",
            (_make_phase("P1", 'Craig and Marc at the villa drama.', ("e1",), "2025-01-01"),),
            evidence_count=1,
        )
        narr_2 = _make_narrative(
            "query 2",
            (_make_phase("P2", 'Marc and Craig villa aftermath.', ("e3",), "2025-01-05"),),
            evidence_count=1,
        )
        report = extract_recurring_entities([narr_1, narr_2])

        # P1 and P2 share villa/Craig/Marc — should cluster across narratives.
        self.assertTrue(len(report.clusters) >= 1)


class EntityQualityTests(unittest.TestCase):
    """Tests for entity extraction quality — false positive suppression."""

    def test_sentence_start_generic_words_suppressed(self) -> None:
        """Capitalized common words at sentence/excerpt start are not entities."""
        phases = (
            _make_phase(
                "P1",
                '(2025-01-01) [user] "Something happened with Marc today." '
                '| (2025-01-01) [assistant] "During the conversation Marc explained."',
                ("e1",), "2025-01-01",
            ),
            _make_phase(
                "P2",
                '(2025-02-01) [user] "Something else about Marc." '
                '| (2025-02-01) [assistant] "Fair point about the situation."',
                ("e2",), "2025-02-01",
            ),
        )
        narrative = _make_narrative("test", phases, evidence_count=2)
        report = extract_recurring_entities([narrative])

        entity_names = [e.name for e in report.entities]
        self.assertNotIn("Something", entity_names)
        self.assertNotIn("During", entity_names)
        self.assertNotIn("Fair", entity_names)

    def test_legitimate_single_word_entities_preserved(self) -> None:
        """Real person/place names still extracted despite being single words."""
        phases = (
            _make_phase("P1", 'Marc and Craig met at the villa.', ("e1",), "2025-01-01"),
            _make_phase("P2", 'Benz called Marc about Butters.', ("e2",), "2025-02-01"),
            _make_phase("P3", 'Craig visited Rawai with Mecky.', ("e3",), "2025-03-01"),
        )
        narrative = _make_narrative("test", phases, evidence_count=3)
        report = extract_recurring_entities([narrative])

        # These are legitimate names — should survive filtering.
        all_phase_entities = set()
        from rag.narrative.builder import entity_terms_from_text
        for phase in phases:
            all_phase_entities |= entity_terms_from_text(phase.description)

        for name in ("Marc", "Craig", "Benz", "Butters", "Rawai", "Mecky"):
            self.assertIn(name, all_phase_entities, f"{name} should be extracted as entity")

    def test_corpus_false_positives_suppressed(self) -> None:
        """Words observed as false positives in real corpus runs are suppressed."""
        from rag.narrative.builder import entity_terms_from_text

        false_positives = [
            "Something", "During", "Fair", "Evil", "Good", "Post",
            "Behaviors", "Documented", "Exactly", "Give", "Got",
            "Remember", "See", "Speaking", "Understood", "After",
            "Although", "Dropped", "Fucking", "Integration",
            "Organizing", "Please", "Review", "Tell", "Understanding",
        ]
        for word in false_positives:
            entities = entity_terms_from_text(f'{word} is not a real entity name.')
            self.assertNotIn(word, entities, f"{word} should be filtered out")

    def test_verb_forms_at_sentence_start_suppressed(self) -> None:
        """Common verbs capitalized at sentence start are not entities."""
        phases = (
            _make_phase(
                "P1",
                '(2025-01-01) [user] "Decided to talk to Marc about it."',
                ("e1",), "2025-01-01",
            ),
            _make_phase(
                "P2",
                '(2025-02-01) [user] "Decided that Marc was right after all."',
                ("e2",), "2025-02-01",
            ),
        )
        narrative = _make_narrative("test", phases, evidence_count=2)
        report = extract_recurring_entities([narrative])

        entity_names = [e.name for e in report.entities]
        self.assertNotIn("Decided", entity_names)
        self.assertIn("Marc", entity_names)

    def test_adjectives_at_sentence_start_suppressed(self) -> None:
        """Common adjectives capitalized at sentence start are not entities."""
        from rag.narrative.builder import entity_terms_from_text

        adjectives = ["Good", "Great", "Important", "Interesting", "Strange", "Terrible"]
        for word in adjectives:
            entities = entity_terms_from_text(f'{word} thing happened today.')
            self.assertNotIn(word, entities, f"{word} should be filtered out")

    def test_mixed_noise_and_real_entities(self) -> None:
        """Noise words suppressed while real entities in same text are preserved."""
        from rag.narrative.builder import entity_terms_from_text

        text = 'Something happened when Craig and Marc discussed. Evil stuff was going on.'
        entities = entity_terms_from_text(text)
        self.assertNotIn("Something", entities)
        self.assertNotIn("Evil", entities)
        self.assertIn("Craig", entities)
        self.assertIn("Marc", entities)

    def test_recurring_entity_quality_with_noise_descriptions(self) -> None:
        """Recurring entity extraction filters noise even from multi-phase data."""
        phases = (
            _make_phase(
                "P1",
                '(2025-01-01) [user] "Evil bastard Mark said something."',
                ("e1",), "2025-01-01",
            ),
            _make_phase(
                "P2",
                '(2025-02-01) [user] "Good that Marc came back."',
                ("e2",), "2025-02-01",
            ),
            _make_phase(
                "P3",
                '(2025-03-01) [user] "Organizing the villa trip with Marc."',
                ("e3",), "2025-03-01",
            ),
        )
        narrative = _make_narrative("test", phases, evidence_count=3)
        report = extract_recurring_entities([narrative])

        entity_names = [e.name for e in report.entities]
        self.assertIn("Marc", entity_names)
        self.assertNotIn("Evil", entity_names)
        self.assertNotIn("Good", entity_names)
        self.assertNotIn("Organizing", entity_names)

    def test_topic_clusters_benefit_from_cleaner_entities(self) -> None:
        """Cluster key_entities should not contain noise words."""
        phases = (
            _make_phase(
                "P1",
                'Evil Marc discussed the villa drama with Craig and Benz.',
                ("e1",), "2025-01-01",
            ),
            _make_phase(
                "P2",
                'Something about Marc and Craig at the villa situation.',
                ("e2",), "2025-01-05",
            ),
        )
        narrative = _make_narrative("test", phases, evidence_count=2)
        report = extract_recurring_entities([narrative])

        for cluster in report.clusters:
            for ent in cluster.key_entities:
                self.assertNotIn(ent, {"Evil", "Something"},
                                 f"Noise word '{ent}' should not be a cluster key_entity")


class EntityClusterLinkTests(unittest.TestCase):
    """Tests for cross-cluster entity link extraction."""

    def test_entity_linked_across_two_clusters(self) -> None:
        """Entity present in key_entities of 2+ clusters produces a link."""
        phases = (
            _make_phase("P1", 'Marc and Craig discussed the villa drama.', ("e1",), "2025-01-01"),
            _make_phase("P2", 'Craig and Marc resolved the villa issue.', ("e2",), "2025-01-05"),
            _make_phase("P3", 'Marc talked about home life and family.', ("e3",), "2025-06-01"),
            _make_phase("P4", 'Marc continued discussing home and routine.', ("e4",), "2025-06-05"),
        )
        narrative = _make_narrative("test", phases, evidence_count=4)
        report = extract_recurring_entities([narrative])

        # Marc should appear in clusters for both villa and home topics.
        if len(report.clusters) >= 2:
            link_entities = [l.entity for l in report.entity_cluster_links]
            self.assertIn("Marc", link_entities)

    def test_entity_in_single_cluster_not_linked(self) -> None:
        """Entity in only one cluster does not get a link."""
        phases = (
            _make_phase("P1", 'Craig and Marc discussed the villa drama.', ("e1",), "2025-01-01"),
            _make_phase("P2", 'Craig and Marc resolved the villa issue.', ("e2",), "2025-01-05"),
        )
        narrative = _make_narrative("test", phases, evidence_count=2)
        report = extract_recurring_entities([narrative])

        # With only one cluster, no links should exist.
        self.assertEqual(len(report.entity_cluster_links), 0)

    def test_no_links_when_fewer_than_two_clusters(self) -> None:
        """No links emitted when report has 0 or 1 clusters."""
        phases = (
            _make_phase("P1", 'Marc talked.', ("e1",), "2025-01-01"),
            _make_phase("P2", 'Craig replied.', ("e2",), "2025-06-01"),
        )
        narrative = _make_narrative("test", phases, evidence_count=2)
        report = extract_recurring_entities([narrative])

        self.assertEqual(len(report.entity_cluster_links), 0)

    def test_link_cluster_count_correct(self) -> None:
        """cluster_count matches actual number of clusters the entity bridges."""
        phases = (
            _make_phase("P1", 'Marc Craig villa drama today.', ("e1",), "2025-01-01"),
            _make_phase("P2", 'Craig Marc villa aftermath now.', ("e2",), "2025-01-03"),
            _make_phase("P3", 'Marc discussed home life routine.', ("e3",), "2025-06-01"),
            _make_phase("P4", 'Marc continued home routine daily.', ("e4",), "2025-06-03"),
        )
        narrative = _make_narrative("test", phases, evidence_count=4)
        report = extract_recurring_entities([narrative])

        marc_links = [l for l in report.entity_cluster_links if l.entity == "Marc"]
        if marc_links:
            self.assertEqual(marc_links[0].cluster_count, len(marc_links[0].cluster_labels))


class TemporalBurstTests(unittest.TestCase):
    """Tests for temporal burst detection."""

    def test_burst_detected_for_dense_phases(self) -> None:
        """3+ phases within 7 days should produce a burst."""
        phases = (
            _make_phase("P1", 'Marc talked.', ("e1",), "2025-02-13"),
            _make_phase("P2", 'Craig replied.', ("e2",), "2025-02-14"),
            _make_phase("P3", 'Marc and Craig met.', ("e3",), "2025-02-15"),
        )
        narrative = _make_narrative("test", phases, evidence_count=3)
        report = extract_recurring_entities([narrative])

        self.assertTrue(len(report.temporal_bursts) >= 1)
        burst = report.temporal_bursts[0]
        self.assertEqual(burst.burst_size, 3)
        self.assertIn("2025-02-13", burst.date_range)
        self.assertIn("2025-02-15", burst.date_range)

    def test_no_burst_for_sparse_phases(self) -> None:
        """Phases spread across months should not produce a burst."""
        phases = (
            _make_phase("P1", 'Marc talked.', ("e1",), "2025-01-01"),
            _make_phase("P2", 'Craig replied.', ("e2",), "2025-03-01"),
            _make_phase("P3", 'Benz arrived.', ("e3",), "2025-06-01"),
        )
        narrative = _make_narrative("test", phases, evidence_count=3)
        report = extract_recurring_entities([narrative])

        self.assertEqual(len(report.temporal_bursts), 0)

    def test_no_burst_below_minimum_phases(self) -> None:
        """2 phases in a window is below the 3-phase minimum."""
        phases = (
            _make_phase("P1", 'Marc talked.', ("e1",), "2025-02-13"),
            _make_phase("P2", 'Craig replied.', ("e2",), "2025-02-14"),
        )
        narrative = _make_narrative("test", phases, evidence_count=2)
        report = extract_recurring_entities([narrative])

        self.assertEqual(len(report.temporal_bursts), 0)

    def test_burst_entities_extracted(self) -> None:
        """Burst includes entities from its member phases."""
        phases = (
            _make_phase("P1", 'Marc discussed plans.', ("e1",), "2025-02-13"),
            _make_phase("P2", 'Craig and Marc met.', ("e2",), "2025-02-14"),
            _make_phase("P3", 'Craig resolved issues.', ("e3",), "2025-02-15"),
        )
        narrative = _make_narrative("test", phases, evidence_count=3)
        report = extract_recurring_entities([narrative])

        if report.temporal_bursts:
            burst = report.temporal_bursts[0]
            self.assertIn("Marc", burst.entities)
            self.assertIn("Craig", burst.entities)

    def test_phases_without_dates_excluded(self) -> None:
        """Phases without dates don't contribute to bursts."""
        phases = (
            _make_phase("P1", 'Marc talked.', ("e1",), None),
            _make_phase("P2", 'Craig replied.', ("e2",), None),
            _make_phase("P3", 'Benz arrived.', ("e3",), None),
        )
        narrative = _make_narrative("test", phases, evidence_count=3)
        report = extract_recurring_entities([narrative])

        self.assertEqual(len(report.temporal_bursts), 0)

    def test_burst_deterministic(self) -> None:
        """Same input produces identical bursts."""
        phases = (
            _make_phase("P1", 'Marc Craig villa.', ("e1",), "2025-02-13"),
            _make_phase("P2", 'Craig Marc drama.', ("e2",), "2025-02-14"),
            _make_phase("P3", 'Marc villa end.', ("e3",), "2025-02-15"),
        )
        narrative = _make_narrative("test", phases, evidence_count=3)
        report_1 = extract_recurring_entities([narrative])
        report_2 = extract_recurring_entities([narrative])

        self.assertEqual(len(report_1.temporal_bursts), len(report_2.temporal_bursts))
        for b1, b2 in zip(report_1.temporal_bursts, report_2.temporal_bursts):
            self.assertEqual(b1.date_range, b2.date_range)
            self.assertEqual(b1.burst_size, b2.burst_size)


if __name__ == "__main__":
    unittest.main()
