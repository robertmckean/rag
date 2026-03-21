"""Unit tests for Phase 7 pattern extraction schema models."""

from __future__ import annotations

import unittest

from rag.patterns.models import (
    EntityClusterLink,
    EntityOccurrence,
    PatternReport,
    RecurringEntity,
    TemporalBurst,
    TopicCluster,
)


class EntityOccurrenceTests(unittest.TestCase):
    """Tests for EntityOccurrence construction and serialization."""

    def test_construction_with_date(self) -> None:
        occ = EntityOccurrence(
            evidence_id="msg-001",
            created_at="2025-06-15T10:00:00Z",
            excerpt="Marc mentioned the trip.",
        )
        self.assertEqual(occ.evidence_id, "msg-001")
        self.assertEqual(occ.created_at, "2025-06-15T10:00:00Z")
        self.assertEqual(occ.excerpt, "Marc mentioned the trip.")

    def test_construction_with_null_date(self) -> None:
        occ = EntityOccurrence(evidence_id="msg-002", created_at=None, excerpt="No date.")
        self.assertIsNone(occ.created_at)

    def test_frozen(self) -> None:
        occ = EntityOccurrence(evidence_id="msg-001", created_at=None, excerpt="x")
        with self.assertRaises(AttributeError):
            occ.evidence_id = "changed"  # type: ignore[misc]

    def test_to_dict(self) -> None:
        occ = EntityOccurrence(
            evidence_id="msg-001",
            created_at="2025-06-15T10:00:00Z",
            excerpt="Marc mentioned the trip.",
        )
        d = occ.to_dict()
        self.assertEqual(d, {
            "evidence_id": "msg-001",
            "created_at": "2025-06-15T10:00:00Z",
            "excerpt": "Marc mentioned the trip.",
        })

    def test_to_dict_null_date(self) -> None:
        occ = EntityOccurrence(evidence_id="msg-002", created_at=None, excerpt="No date.")
        d = occ.to_dict()
        self.assertIsNone(d["created_at"])


class RecurringEntityTests(unittest.TestCase):
    """Tests for RecurringEntity construction and serialization."""

    def _make_entity(self) -> RecurringEntity:
        return RecurringEntity(
            name="Marc",
            occurrences=(
                EntityOccurrence("msg-001", "2025-06-15T10:00:00Z", "Marc said hello."),
                EntityOccurrence("msg-005", "2025-07-01T12:00:00Z", "Marc returned."),
            ),
            occurrence_count=2,
        )

    def test_construction(self) -> None:
        entity = self._make_entity()
        self.assertEqual(entity.name, "Marc")
        self.assertEqual(len(entity.occurrences), 2)
        self.assertEqual(entity.occurrence_count, 2)

    def test_frozen(self) -> None:
        entity = self._make_entity()
        with self.assertRaises(AttributeError):
            entity.name = "changed"  # type: ignore[misc]

    def test_to_dict(self) -> None:
        entity = self._make_entity()
        d = entity.to_dict()
        self.assertEqual(d["name"], "Marc")
        self.assertEqual(len(d["occurrences"]), 2)
        self.assertEqual(d["occurrence_count"], 2)
        self.assertIsInstance(d["occurrences"][0], dict)
        self.assertEqual(d["occurrences"][0]["evidence_id"], "msg-001")

    def test_to_dict_deterministic_order(self) -> None:
        """Occurrences serialize in the order they were provided."""
        entity = self._make_entity()
        d = entity.to_dict()
        ids = [occ["evidence_id"] for occ in d["occurrences"]]
        self.assertEqual(ids, ["msg-001", "msg-005"])

    def test_single_occurrence(self) -> None:
        entity = RecurringEntity(
            name="Benz",
            occurrences=(EntityOccurrence("msg-010", None, "Benz appeared."),),
            occurrence_count=1,
        )
        d = entity.to_dict()
        self.assertEqual(d["occurrence_count"], 1)
        self.assertEqual(len(d["occurrences"]), 1)


class PatternReportTests(unittest.TestCase):
    """Tests for PatternReport construction and serialization."""

    def _make_report(self) -> PatternReport:
        return PatternReport(
            query="What happened with Marc?",
            entities=(
                RecurringEntity(
                    name="Marc",
                    occurrences=(
                        EntityOccurrence("msg-001", "2025-06-15T10:00:00Z", "Marc said hello."),
                        EntityOccurrence("msg-005", "2025-07-01T12:00:00Z", "Marc returned."),
                    ),
                    occurrence_count=2,
                ),
            ),
            clusters=(),
            entity_cluster_links=(),
            temporal_bursts=(),
            evidence_count=5,
        )

    def test_construction(self) -> None:
        report = self._make_report()
        self.assertEqual(report.query, "What happened with Marc?")
        self.assertEqual(len(report.entities), 1)
        self.assertEqual(report.clusters, ())
        self.assertEqual(report.evidence_count, 5)

    def test_frozen(self) -> None:
        report = self._make_report()
        with self.assertRaises(AttributeError):
            report.query = "changed"  # type: ignore[misc]

    def test_to_dict(self) -> None:
        report = self._make_report()
        d = report.to_dict()
        self.assertEqual(d["query"], "What happened with Marc?")
        self.assertEqual(d["evidence_count"], 5)
        self.assertEqual(len(d["entities"]), 1)
        self.assertIsInstance(d["entities"][0], dict)

    def test_to_dict_roundtrip_values(self) -> None:
        """Values survive serialization unchanged."""
        report = self._make_report()
        d = report.to_dict()
        entity_d = d["entities"][0]
        self.assertEqual(entity_d["name"], "Marc")
        occ_d = entity_d["occurrences"][0]
        self.assertEqual(occ_d["evidence_id"], "msg-001")
        self.assertEqual(occ_d["created_at"], "2025-06-15T10:00:00Z")
        self.assertEqual(occ_d["excerpt"], "Marc said hello.")

    def test_empty_report(self) -> None:
        report = PatternReport(query="nothing", entities=(), clusters=(), entity_cluster_links=(), temporal_bursts=(), evidence_count=0)
        d = report.to_dict()
        self.assertEqual(d["query"], "nothing")
        self.assertEqual(d["entities"], [])
        self.assertEqual(d["evidence_count"], 0)

    def test_empty_report_construction(self) -> None:
        report = PatternReport(query="nothing", entities=(), clusters=(), entity_cluster_links=(), temporal_bursts=(), evidence_count=0)
        self.assertEqual(len(report.entities), 0)
        self.assertEqual(report.evidence_count, 0)

    def test_multiple_entities_ordering(self) -> None:
        """Entities serialize in the order they were provided."""
        report = PatternReport(
            query="test",
            entities=(
                RecurringEntity("Zeno", (), 0),
                RecurringEntity("Marc", (), 0),
                RecurringEntity("Benz", (), 0),
            ),
            clusters=(),
            entity_cluster_links=(),
            temporal_bursts=(),
            evidence_count=0,
        )
        d = report.to_dict()
        names = [e["name"] for e in d["entities"]]
        self.assertEqual(names, ["Zeno", "Marc", "Benz"])


class TopicClusterTests(unittest.TestCase):
    """Tests for TopicCluster construction and serialization."""

    def _make_cluster(self) -> TopicCluster:
        return TopicCluster(
            label="Marc, Craig: meeting, villa",
            phase_labels=("2025-02-04: Marc, Craig", "2025-02-13: Craig, Marc"),
            evidence_ids=("e1", "e3", "e7", "e11"),
            date_range="2025-02-04 to 2025-02-13",
            key_entities=("Marc", "Craig"),
            key_terms=("meeting", "villa"),
            phase_count=2,
        )

    def test_construction(self) -> None:
        cluster = self._make_cluster()
        self.assertEqual(cluster.label, "Marc, Craig: meeting, villa")
        self.assertEqual(cluster.phase_count, 2)
        self.assertEqual(len(cluster.phase_labels), 2)
        self.assertEqual(len(cluster.evidence_ids), 4)

    def test_frozen(self) -> None:
        cluster = self._make_cluster()
        with self.assertRaises(AttributeError):
            cluster.label = "changed"  # type: ignore[misc]

    def test_to_dict(self) -> None:
        cluster = self._make_cluster()
        d = cluster.to_dict()
        self.assertEqual(d["label"], "Marc, Craig: meeting, villa")
        self.assertEqual(d["phase_count"], 2)
        self.assertIsInstance(d["phase_labels"], list)
        self.assertIsInstance(d["evidence_ids"], list)
        self.assertIsInstance(d["key_entities"], list)
        self.assertIsInstance(d["key_terms"], list)

    def test_to_dict_roundtrip_values(self) -> None:
        cluster = self._make_cluster()
        d = cluster.to_dict()
        self.assertEqual(d["phase_labels"], ["2025-02-04: Marc, Craig", "2025-02-13: Craig, Marc"])
        self.assertEqual(d["evidence_ids"], ["e1", "e3", "e7", "e11"])
        self.assertEqual(d["date_range"], "2025-02-04 to 2025-02-13")
        self.assertEqual(d["key_entities"], ["Marc", "Craig"])
        self.assertEqual(d["key_terms"], ["meeting", "villa"])

    def test_null_date_range(self) -> None:
        cluster = TopicCluster(
            label="test", phase_labels=(), evidence_ids=(),
            date_range=None, key_entities=(), key_terms=(), phase_count=0,
        )
        d = cluster.to_dict()
        self.assertIsNone(d["date_range"])

    def test_pattern_report_with_clusters(self) -> None:
        """PatternReport serializes clusters alongside entities."""
        cluster = self._make_cluster()
        report = PatternReport(
            query="test",
            entities=(),
            clusters=(cluster,),
            entity_cluster_links=(),
            temporal_bursts=(),
            evidence_count=5,
        )
        d = report.to_dict()
        self.assertEqual(len(d["clusters"]), 1)
        self.assertEqual(d["clusters"][0]["label"], "Marc, Craig: meeting, villa")

    def test_empty_clusters_in_report(self) -> None:
        report = PatternReport(query="test", entities=(), clusters=(), entity_cluster_links=(), temporal_bursts=(), evidence_count=0)
        d = report.to_dict()
        self.assertEqual(d["clusters"], [])


class EntityClusterLinkTests(unittest.TestCase):
    """Tests for EntityClusterLink construction and serialization."""

    def _make_link(self) -> EntityClusterLink:
        return EntityClusterLink(
            entity="Marc",
            cluster_labels=("Marc, Craig: villa, drama", "Marc: home, really"),
            cluster_count=2,
            total_phase_count=4,
        )

    def test_construction(self) -> None:
        link = self._make_link()
        self.assertEqual(link.entity, "Marc")
        self.assertEqual(link.cluster_count, 2)
        self.assertEqual(link.total_phase_count, 4)
        self.assertEqual(len(link.cluster_labels), 2)

    def test_frozen(self) -> None:
        link = self._make_link()
        with self.assertRaises(AttributeError):
            link.entity = "changed"  # type: ignore[misc]

    def test_to_dict(self) -> None:
        link = self._make_link()
        d = link.to_dict()
        self.assertEqual(d["entity"], "Marc")
        self.assertEqual(d["cluster_count"], 2)
        self.assertEqual(d["total_phase_count"], 4)
        self.assertIsInstance(d["cluster_labels"], list)
        self.assertEqual(d["cluster_labels"], ["Marc, Craig: villa, drama", "Marc: home, really"])


class TemporalBurstTests(unittest.TestCase):
    """Tests for TemporalBurst construction and serialization."""

    def _make_burst(self) -> TemporalBurst:
        return TemporalBurst(
            date_range="2025-02-13 to 2025-02-15",
            phase_labels=("P1: Marc, Craig", "P2: Craig", "P3: Marc"),
            entities=("Craig", "Marc"),
            burst_size=3,
        )

    def test_construction(self) -> None:
        burst = self._make_burst()
        self.assertEqual(burst.date_range, "2025-02-13 to 2025-02-15")
        self.assertEqual(burst.burst_size, 3)
        self.assertEqual(len(burst.phase_labels), 3)
        self.assertEqual(len(burst.entities), 2)

    def test_frozen(self) -> None:
        burst = self._make_burst()
        with self.assertRaises(AttributeError):
            burst.date_range = "changed"  # type: ignore[misc]

    def test_to_dict(self) -> None:
        burst = self._make_burst()
        d = burst.to_dict()
        self.assertEqual(d["date_range"], "2025-02-13 to 2025-02-15")
        self.assertEqual(d["burst_size"], 3)
        self.assertIsInstance(d["phase_labels"], list)
        self.assertIsInstance(d["entities"], list)
        self.assertEqual(d["entities"], ["Craig", "Marc"])

    def test_single_date_burst(self) -> None:
        burst = TemporalBurst(
            date_range="2025-02-15",
            phase_labels=("P1", "P2", "P3"),
            entities=(),
            burst_size=3,
        )
        self.assertEqual(burst.date_range, "2025-02-15")

    def test_pattern_report_with_links_and_bursts(self) -> None:
        """PatternReport serializes entity_cluster_links and temporal_bursts."""
        link = EntityClusterLink(entity="Marc", cluster_labels=("C1", "C2"), cluster_count=2, total_phase_count=4)
        burst = TemporalBurst(date_range="2025-02-15", phase_labels=("P1", "P2", "P3"), entities=("Marc",), burst_size=3)
        report = PatternReport(
            query="test",
            entities=(),
            clusters=(),
            entity_cluster_links=(link,),
            temporal_bursts=(burst,),
            evidence_count=5,
        )
        d = report.to_dict()
        self.assertEqual(len(d["entity_cluster_links"]), 1)
        self.assertEqual(d["entity_cluster_links"][0]["entity"], "Marc")
        self.assertEqual(len(d["temporal_bursts"]), 1)
        self.assertEqual(d["temporal_bursts"][0]["burst_size"], 3)

    def test_empty_links_and_bursts_in_report(self) -> None:
        report = PatternReport(query="test", entities=(), clusters=(), entity_cluster_links=(), temporal_bursts=(), evidence_count=0)
        d = report.to_dict()
        self.assertEqual(d["entity_cluster_links"], [])
        self.assertEqual(d["temporal_bursts"], [])


if __name__ == "__main__":
    unittest.main()
