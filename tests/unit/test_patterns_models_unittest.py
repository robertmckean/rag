"""Unit tests for Phase 7 pattern extraction schema models."""

from __future__ import annotations

import unittest

from rag.patterns.models import EntityOccurrence, PatternReport, RecurringEntity


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
            evidence_count=5,
        )

    def test_construction(self) -> None:
        report = self._make_report()
        self.assertEqual(report.query, "What happened with Marc?")
        self.assertEqual(len(report.entities), 1)
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
        report = PatternReport(query="nothing", entities=(), evidence_count=0)
        d = report.to_dict()
        self.assertEqual(d["query"], "nothing")
        self.assertEqual(d["entities"], [])
        self.assertEqual(d["evidence_count"], 0)

    def test_empty_report_construction(self) -> None:
        report = PatternReport(query="nothing", entities=(), evidence_count=0)
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
            evidence_count=0,
        )
        d = report.to_dict()
        names = [e["name"] for e in d["entities"]]
        self.assertEqual(names, ["Zeno", "Marc", "Benz"])


if __name__ == "__main__":
    unittest.main()
