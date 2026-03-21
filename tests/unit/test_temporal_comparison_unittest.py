"""Unit tests for Phase 13B — deterministic temporal position comparison."""

import unittest

from rag.narrative.positions import (
    Position,
    ThinkingEvolution,
    build_thinking_evolution,
    collect_positions_for_entity,
    _detect_shift,
    _sort_key,
)


def _pos(
    text: str,
    *,
    date: str | None = "2025-06-15",
    entity: str | None = None,
    evidence_id: str = "e1",
    stance_marker: str = "i think",
) -> Position:
    return Position(
        text=text,
        date=date,
        entity=entity,
        evidence_id=evidence_id,
        stance_marker=stance_marker,
    )


class ChronologicalSortTests(unittest.TestCase):
    """Verify positions are sorted by date ascending."""

    def test_ascending_date_order(self) -> None:
        positions = (
            _pos("Later thought", date="2025-09-01"),
            _pos("Earlier thought", date="2025-03-01"),
            _pos("Middle thought", date="2025-06-01"),
        )
        evo = build_thinking_evolution("test", positions)
        dates = [p.date for p in evo.positions]
        self.assertEqual(dates, ["2025-03-01", "2025-06-01", "2025-09-01"])

    def test_undated_positions_sorted_last(self) -> None:
        positions = (
            _pos("Undated", date=None),
            _pos("Dated", date="2025-03-01"),
        )
        evo = build_thinking_evolution("test", positions)
        self.assertEqual(evo.positions[0].date, "2025-03-01")
        self.assertIsNone(evo.positions[1].date)

    def test_same_date_deterministic_by_text(self) -> None:
        positions = (
            _pos("Zebra thought", date="2025-06-01"),
            _pos("Alpha thought", date="2025-06-01"),
        )
        evo = build_thinking_evolution("test", positions)
        self.assertEqual(evo.positions[0].text, "Alpha thought")
        self.assertEqual(evo.positions[1].text, "Zebra thought")


class StablePositionTests(unittest.TestCase):
    """Verify no shifts for consistent positions."""

    def test_no_shifts_for_stable_positions(self) -> None:
        positions = (
            _pos("I think Marc is trustworthy", date="2025-03-01", entity="Marc"),
            _pos("I believe Marc is a good friend", date="2025-06-01", entity="Marc"),
        )
        evo = build_thinking_evolution("Marc", positions)
        self.assertEqual(len(evo.shifts), 0)

    def test_single_position_no_shifts(self) -> None:
        positions = (_pos("I think this is right", date="2025-03-01"),)
        evo = build_thinking_evolution("test", positions)
        self.assertEqual(len(evo.shifts), 0)

    def test_empty_positions_no_shifts(self) -> None:
        evo = build_thinking_evolution("test", ())
        self.assertEqual(len(evo.shifts), 0)
        self.assertEqual(len(evo.positions), 0)


class NegationShiftTests(unittest.TestCase):
    """Verify negation-based shift detection."""

    def test_negation_introduced(self) -> None:
        positions = (
            _pos("I think this works", date="2025-03-01"),
            _pos("I don't think this works", date="2025-06-01",
                 stance_marker="i don't think"),
        )
        evo = build_thinking_evolution("test", positions)
        self.assertEqual(len(evo.shifts), 1)
        self.assertIn("negation introduced", evo.shifts[0])

    def test_negation_removed(self) -> None:
        positions = (
            _pos("I don't trust him anymore", date="2025-03-01",
                 stance_marker="i don't think"),
            _pos("I think I can trust him now", date="2025-06-01"),
        )
        evo = build_thinking_evolution("test", positions)
        self.assertEqual(len(evo.shifts), 1)
        self.assertIn("negation removed", evo.shifts[0])


class ExplicitRevisionTests(unittest.TestCase):
    """Verify explicit self-revision marker detection."""

    def test_changed_my_mind_marker(self) -> None:
        positions = (
            _pos("I think X is fine", date="2025-03-01"),
            _pos("I changed my mind about X", date="2025-06-01",
                 stance_marker="i changed my mind"),
        )
        evo = build_thinking_evolution("test", positions)
        self.assertEqual(len(evo.shifts), 1)
        self.assertIn("explicit self-revision", evo.shifts[0])

    def test_was_wrong_marker(self) -> None:
        positions = (
            _pos("I think this is correct", date="2025-03-01"),
            _pos("I was wrong about that", date="2025-06-01",
                 stance_marker="i was wrong about"),
        )
        evo = build_thinking_evolution("test", positions)
        self.assertEqual(len(evo.shifts), 1)
        self.assertIn("explicit self-revision", evo.shifts[0])

    def test_no_longer_marker(self) -> None:
        positions = (
            _pos("I believe in this method", date="2025-03-01"),
            _pos("I no longer follow that method", date="2025-06-01",
                 stance_marker="i no longer"),
        )
        evo = build_thinking_evolution("test", positions)
        self.assertEqual(len(evo.shifts), 1)
        self.assertIn("explicit self-revision", evo.shifts[0])

    def test_used_to_think_marker(self) -> None:
        positions = (
            _pos("I think this is ideal", date="2025-03-01"),
            _pos("I used to think it was ideal", date="2025-06-01",
                 stance_marker="i used to think"),
        )
        evo = build_thinking_evolution("test", positions)
        self.assertEqual(len(evo.shifts), 1)
        self.assertIn("explicit self-revision", evo.shifts[0])


class SentimentShiftTests(unittest.TestCase):
    """Verify sentiment-bearing term shift detection."""

    def test_positive_to_negative(self) -> None:
        positions = (
            _pos("I think this is great and I trust the process", date="2025-03-01"),
            _pos("I think this is bad and I'm frustrated", date="2025-06-01"),
        )
        evo = build_thinking_evolution("test", positions)
        self.assertEqual(len(evo.shifts), 1)
        self.assertIn("positive to negative", evo.shifts[0])

    def test_negative_to_positive(self) -> None:
        positions = (
            _pos("I'm worried and frustrated about this", date="2025-03-01"),
            _pos("I feel confident and comfortable now", date="2025-06-01",
                 stance_marker="i feel like"),
        )
        evo = build_thinking_evolution("test", positions)
        self.assertEqual(len(evo.shifts), 1)
        self.assertIn("negative to positive", evo.shifts[0])


class EntityScopingTests(unittest.TestCase):
    """Verify entity-level position filtering."""

    def test_entity_specific_filtering(self) -> None:
        positions = (
            _pos("I think Marc is great", entity="Marc"),
            _pos("I think Craig is fine", entity="Craig"),
            _pos("I believe Marc is honest", entity="Marc"),
        )
        marc_positions = collect_positions_for_entity(positions, "Marc")
        self.assertEqual(len(marc_positions), 2)
        self.assertTrue(all(p.entity == "Marc" for p in marc_positions))

    def test_different_entities_not_conflated(self) -> None:
        positions = (
            _pos("I trust Marc completely", date="2025-03-01", entity="Marc"),
            _pos("I'm frustrated with Craig", date="2025-06-01", entity="Craig"),
        )
        evo = build_thinking_evolution("Marc", collect_positions_for_entity(positions, "Marc"))
        # Only one Marc position — no shifts possible.
        self.assertEqual(len(evo.positions), 1)
        self.assertEqual(len(evo.shifts), 0)

    def test_case_insensitive_entity_match(self) -> None:
        positions = (
            _pos("I think marc is great", entity="Marc"),
            _pos("I believe MARC is honest", entity="Marc"),
        )
        result = collect_positions_for_entity(positions, "marc")
        self.assertEqual(len(result), 2)

    def test_text_fallback_when_no_entity_field(self) -> None:
        positions = (
            _pos("I think Marc is great", entity=None),
            _pos("I believe something else", entity=None),
        )
        result = collect_positions_for_entity(positions, "Marc")
        self.assertEqual(len(result), 1)
        self.assertIn("Marc", result[0].text)


class DeterminismTests(unittest.TestCase):
    """Verify deterministic output."""

    def test_deterministic_across_runs(self) -> None:
        positions = (
            _pos("I trust Marc", date="2025-03-01", entity="Marc"),
            _pos("I don't trust Marc anymore", date="2025-06-01", entity="Marc",
                 stance_marker="i don't think"),
        )
        results = [build_thinking_evolution("Marc", positions) for _ in range(5)]
        self.assertTrue(all(r == results[0] for r in results))


class SerializationTests(unittest.TestCase):
    """Verify to_dict serialization."""

    def test_to_dict(self) -> None:
        positions = (
            _pos("I think X", date="2025-03-01"),
            _pos("I don't think X", date="2025-06-01", stance_marker="i don't think"),
        )
        evo = build_thinking_evolution("test", positions)
        d = evo.to_dict()
        self.assertEqual(d["entity"], "test")
        self.assertIsInstance(d["positions"], list)
        self.assertIsInstance(d["shifts"], list)
        self.assertEqual(len(d["positions"]), 2)

    def test_shift_strings_in_dict(self) -> None:
        positions = (
            _pos("I think this is great", date="2025-03-01"),
            _pos("I changed my mind about this", date="2025-06-01",
                 stance_marker="i changed my mind"),
        )
        evo = build_thinking_evolution("test", positions)
        d = evo.to_dict()
        self.assertGreater(len(d["shifts"]), 0)
        self.assertIsInstance(d["shifts"][0], str)


if __name__ == "__main__":
    unittest.main()
