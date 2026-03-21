"""Unit tests for pattern report rendering (Phase 7)."""

from __future__ import annotations

import json
import unittest

from rag.patterns.models import EntityClusterLink, EntityOccurrence, PatternReport, RecurringEntity, TemporalBurst, TopicCluster
from rag.patterns.renderer import render_json, render_text


def _make_report() -> PatternReport:
    """Populated report with two recurring entities."""
    return PatternReport(
        query="What about Marc?",
        entities=(
            RecurringEntity(
                name="Marc",
                occurrences=(
                    EntityOccurrence("e1", "2025-01-15", "2025-01-15: Marc"),
                    EntityOccurrence("e3", "2025-03-01", "2025-03-01: Marc"),
                    EntityOccurrence("e5", None, "Phase (evidence e5)"),
                ),
                occurrence_count=3,
            ),
            RecurringEntity(
                name="Craig",
                occurrences=(
                    EntityOccurrence("e1", "2025-01-15", "2025-01-15: Marc"),
                    EntityOccurrence("e4", "2025-02-20", "2025-02-20: Craig"),
                ),
                occurrence_count=2,
            ),
        ),
        clusters=(),
        entity_cluster_links=(),
        temporal_bursts=(),
        evidence_count=5,
    )


def _make_empty_report() -> PatternReport:
    return PatternReport(query="nothing", entities=(), clusters=(), entity_cluster_links=(), temporal_bursts=(), evidence_count=0)


class JsonRenderTests(unittest.TestCase):
    """Tests for JSON rendering."""

    def test_populated_report_structure(self) -> None:
        output = render_json(_make_report())
        d = json.loads(output)
        self.assertEqual(d["query"], "What about Marc?")
        self.assertEqual(d["evidence_count"], 5)
        self.assertEqual(len(d["entities"]), 2)
        self.assertEqual(d["entities"][0]["name"], "Marc")
        self.assertEqual(d["entities"][0]["occurrence_count"], 3)
        self.assertEqual(len(d["entities"][0]["occurrences"]), 3)

    def test_empty_report_structure(self) -> None:
        output = render_json(_make_empty_report())
        d = json.loads(output)
        self.assertEqual(d["query"], "nothing")
        self.assertEqual(d["entities"], [])
        self.assertEqual(d["evidence_count"], 0)

    def test_json_is_valid(self) -> None:
        output = render_json(_make_report())
        # Should not raise.
        json.loads(output)

    def test_occurrence_fields_present(self) -> None:
        output = render_json(_make_report())
        d = json.loads(output)
        occ = d["entities"][0]["occurrences"][0]
        self.assertIn("evidence_id", occ)
        self.assertIn("created_at", occ)
        self.assertIn("excerpt", occ)

    def test_null_date_in_json(self) -> None:
        output = render_json(_make_report())
        d = json.loads(output)
        # Marc's third occurrence has no date.
        occ = d["entities"][0]["occurrences"][2]
        self.assertIsNone(occ["created_at"])


class TextRenderTests(unittest.TestCase):
    """Tests for human-readable text rendering."""

    def test_populated_report_content(self) -> None:
        output = render_text(_make_report())
        self.assertIn("Pattern Report: What about Marc?", output)
        self.assertIn("Recurring entities: 2", output)
        self.assertIn("[1] Marc (3 occurrences)", output)
        self.assertIn("[2] Craig (2 occurrences)", output)
        self.assertIn("Evidence count: 5", output)

    def test_empty_report_content(self) -> None:
        output = render_text(_make_empty_report())
        self.assertIn("Pattern Report: nothing", output)
        self.assertIn("No recurring entities found.", output)
        self.assertIn("Evidence count: 0", output)

    def test_deterministic_ordering(self) -> None:
        """Entity numbering matches input order (already sorted by extractor)."""
        output = render_text(_make_report())
        marc_pos = output.index("[1] Marc")
        craig_pos = output.index("[2] Craig")
        self.assertLess(marc_pos, craig_pos)

    def test_date_shown_when_present(self) -> None:
        output = render_text(_make_report())
        self.assertIn("(2025-01-15)", output)
        self.assertIn("(2025-03-01)", output)

    def test_date_omitted_when_absent(self) -> None:
        output = render_text(_make_report())
        # The null-date occurrence for Marc (e5) should not have empty parens.
        lines = output.split("\n")
        e5_lines = [l for l in lines if "e5" in l]
        self.assertTrue(len(e5_lines) > 0)
        for line in e5_lines:
            self.assertNotIn("()", line)
            self.assertNotIn("(None)", line)

    def test_multiple_occurrences_all_rendered(self) -> None:
        output = render_text(_make_report())
        # Marc has 3 occurrences — all should appear.
        lines = output.split("\n")
        marc_occ_lines = [l for l in lines if l.strip().startswith("- e") and "Marc" in output]
        # More precise: count lines under Marc's section.
        in_marc = False
        marc_items = 0
        for line in lines:
            if "[1] Marc" in line:
                in_marc = True
                continue
            if in_marc and line.strip().startswith("- "):
                marc_items += 1
            elif in_marc and not line.strip().startswith("- "):
                break
        self.assertEqual(marc_items, 3)

    def test_excerpt_shown_for_each_occurrence(self) -> None:
        output = render_text(_make_report())
        self.assertIn("2025-01-15: Marc", output)
        self.assertIn("Phase (evidence e5)", output)


def _make_cluster_report() -> PatternReport:
    """Report with entities and clusters."""
    cluster = TopicCluster(
        label="Marc, Craig: villa, drama",
        phase_labels=("2025-01-01: Marc, Craig", "2025-01-05: Craig, Marc"),
        evidence_ids=("e1", "e2", "e3"),
        date_range="2025-01-01 to 2025-01-05",
        key_entities=("Marc", "Craig"),
        key_terms=("villa", "drama"),
        phase_count=2,
    )
    return PatternReport(
        query="What about Marc?",
        entities=(),
        clusters=(cluster,),
        entity_cluster_links=(),
        temporal_bursts=(),
        evidence_count=3,
    )


class ClusterRenderTests(unittest.TestCase):
    """Tests for cluster rendering."""

    def test_json_clusters_present(self) -> None:
        output = render_json(_make_cluster_report())
        d = json.loads(output)
        self.assertEqual(len(d["clusters"]), 1)
        c = d["clusters"][0]
        self.assertEqual(c["label"], "Marc, Craig: villa, drama")
        self.assertEqual(c["phase_count"], 2)

    def test_text_clusters_rendered(self) -> None:
        output = render_text(_make_cluster_report())
        self.assertIn("Topic clusters: 1", output)
        self.assertIn("[1] Marc, Craig: villa, drama", output)
        self.assertIn("2 phases", output)
        self.assertIn("Entities: Marc, Craig", output)
        self.assertIn("Terms: villa, drama", output)

    def test_text_cluster_date_range(self) -> None:
        output = render_text(_make_cluster_report())
        self.assertIn("(2025-01-01 to 2025-01-05)", output)

    def test_text_cluster_evidence_ids(self) -> None:
        output = render_text(_make_cluster_report())
        self.assertIn("Evidence: e1, e2, e3", output)

    def test_text_no_clusters_section_when_empty(self) -> None:
        output = render_text(_make_report())
        self.assertNotIn("Topic clusters:", output)

    def test_json_empty_clusters(self) -> None:
        output = render_json(_make_report())
        d = json.loads(output)
        self.assertEqual(d["clusters"], [])


def _make_full_report() -> PatternReport:
    """Report with entities, clusters, links, and bursts."""
    cluster_1 = TopicCluster(
        label="Marc, Craig: villa, drama",
        phase_labels=("2025-01-01: Marc, Craig", "2025-01-05: Craig, Marc"),
        evidence_ids=("e1", "e2"),
        date_range="2025-01-01 to 2025-01-05",
        key_entities=("Marc", "Craig"),
        key_terms=("villa", "drama"),
        phase_count=2,
    )
    cluster_2 = TopicCluster(
        label="Marc: home, really",
        phase_labels=("2025-02-01: Marc",),
        evidence_ids=("e3",),
        date_range="2025-02-01",
        key_entities=("Marc",),
        key_terms=("home", "really"),
        phase_count=1,
    )
    link = EntityClusterLink(
        entity="Marc",
        cluster_labels=("Marc, Craig: villa, drama", "Marc: home, really"),
        cluster_count=2,
        total_phase_count=3,
    )
    burst = TemporalBurst(
        date_range="2025-01-01 to 2025-01-05",
        phase_labels=("P1: Marc", "P2: Craig", "P3: Marc"),
        entities=("Craig", "Marc"),
        burst_size=3,
    )
    return PatternReport(
        query="What about Marc?",
        entities=(
            RecurringEntity(
                name="Marc",
                occurrences=(EntityOccurrence("e1", "2025-01-01", "Marc snippet"),),
                occurrence_count=1,
            ),
        ),
        clusters=(cluster_1, cluster_2),
        entity_cluster_links=(link,),
        temporal_bursts=(burst,),
        evidence_count=5,
    )


class CrossClusterRenderTests(unittest.TestCase):
    """Tests for cross-cluster entity link rendering."""

    def test_text_links_section_present(self) -> None:
        output = render_text(_make_full_report())
        self.assertIn("Cross-cluster entities: 1", output)
        self.assertIn("Marc (2 clusters, 3 phases)", output)

    def test_text_link_cluster_labels_listed(self) -> None:
        output = render_text(_make_full_report())
        self.assertIn("- Marc, Craig: villa, drama", output)
        self.assertIn("- Marc: home, really", output)

    def test_json_links_present(self) -> None:
        output = render_json(_make_full_report())
        d = json.loads(output)
        self.assertEqual(len(d["entity_cluster_links"]), 1)
        self.assertEqual(d["entity_cluster_links"][0]["entity"], "Marc")
        self.assertEqual(d["entity_cluster_links"][0]["cluster_count"], 2)

    def test_text_no_links_section_when_empty(self) -> None:
        output = render_text(_make_report())
        self.assertNotIn("Cross-cluster entities:", output)


class TemporalBurstRenderTests(unittest.TestCase):
    """Tests for temporal burst rendering."""

    def test_text_bursts_section_present(self) -> None:
        output = render_text(_make_full_report())
        self.assertIn("Temporal bursts: 1", output)
        self.assertIn("2025-01-01 to 2025-01-05 -- 3 phases", output)

    def test_text_burst_entities_listed(self) -> None:
        output = render_text(_make_full_report())
        self.assertIn("Entities: Craig, Marc", output)

    def test_text_burst_phases_listed(self) -> None:
        output = render_text(_make_full_report())
        self.assertIn("- P1: Marc", output)
        self.assertIn("- P2: Craig", output)

    def test_json_bursts_present(self) -> None:
        output = render_json(_make_full_report())
        d = json.loads(output)
        self.assertEqual(len(d["temporal_bursts"]), 1)
        self.assertEqual(d["temporal_bursts"][0]["burst_size"], 3)

    def test_text_no_bursts_section_when_empty(self) -> None:
        output = render_text(_make_report())
        self.assertNotIn("Temporal bursts:", output)


class SummaryRenderTests(unittest.TestCase):
    """Tests for the summary emphasis section."""

    def test_summary_section_present(self) -> None:
        output = render_text(_make_full_report())
        self.assertIn("Summary", output)
        self.assertIn("Top entities:", output)
        self.assertIn("Top themes:", output)

    def test_summary_cross_topic_present(self) -> None:
        output = render_text(_make_full_report())
        self.assertIn("Cross-topic: Marc", output)

    def test_summary_bursts_present(self) -> None:
        output = render_text(_make_full_report())
        self.assertIn("Activity bursts:", output)

    def test_no_summary_when_empty(self) -> None:
        output = render_text(_make_empty_report())
        self.assertNotIn("Summary", output)


if __name__ == "__main__":
    unittest.main()
