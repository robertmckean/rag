"""Unit tests for pattern report rendering (Phase 7)."""

from __future__ import annotations

import json
import unittest

from rag.patterns.models import EntityOccurrence, PatternReport, RecurringEntity
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
        evidence_count=5,
    )


def _make_empty_report() -> PatternReport:
    return PatternReport(query="nothing", entities=(), evidence_count=0)


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


if __name__ == "__main__":
    unittest.main()
