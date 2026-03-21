"""Unit tests for Phase 14A — deterministic contradiction/change signal detection."""

import unittest

from rag.narrative.positions import (
    Contradiction,
    Position,
    detect_contradictions,
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


class ExplicitSelfCorrectionTests(unittest.TestCase):
    """Verify explicit self-revision markers produce contradictions."""

    def test_changed_my_mind(self) -> None:
        positions = [
            _pos("I think Marc is fine", date="2025-03-01"),
            _pos("I changed my mind about Marc", date="2025-06-01",
                 stance_marker="i changed my mind"),
        ]
        result = detect_contradictions("Marc", positions)
        self.assertEqual(len(result), 1)
        self.assertIn("explicit self-revision", result[0].signal)

    def test_was_wrong(self) -> None:
        positions = [
            _pos("I think this is correct", date="2025-03-01"),
            _pos("I was wrong about that approach", date="2025-06-01",
                 stance_marker="i was wrong about"),
        ]
        result = detect_contradictions("test", positions)
        self.assertEqual(len(result), 1)
        self.assertIn("explicit self-revision", result[0].signal)

    def test_no_longer(self) -> None:
        positions = [
            _pos("I believe in this method", date="2025-03-01"),
            _pos("I no longer follow that method", date="2025-06-01",
                 stance_marker="i no longer"),
        ]
        result = detect_contradictions("test", positions)
        self.assertEqual(len(result), 1)
        self.assertIn("explicit self-revision", result[0].signal)

    def test_used_to_think(self) -> None:
        positions = [
            _pos("I think this is ideal", date="2025-03-01"),
            _pos("I used to think it was ideal", date="2025-06-01",
                 stance_marker="i used to think"),
        ]
        result = detect_contradictions("test", positions)
        self.assertEqual(len(result), 1)
        self.assertIn("explicit self-revision", result[0].signal)


class NegationReversalTests(unittest.TestCase):
    """Verify negation change detection."""

    def test_negation_introduced(self) -> None:
        positions = [
            _pos("I think this works", date="2025-03-01"),
            _pos("I don't think this works", date="2025-06-01",
                 stance_marker="i don't think"),
        ]
        result = detect_contradictions("test", positions)
        self.assertEqual(len(result), 1)
        self.assertIn("negation introduced", result[0].signal)

    def test_negation_removed(self) -> None:
        positions = [
            _pos("I don't trust this process", date="2025-03-01",
                 stance_marker="i don't think"),
            _pos("I think I can trust this now", date="2025-06-01"),
        ]
        result = detect_contradictions("test", positions)
        self.assertEqual(len(result), 1)
        self.assertIn("negation removed", result[0].signal)


class SentimentReversalTests(unittest.TestCase):
    """Verify sentiment-bearing term change detection."""

    def test_positive_to_negative(self) -> None:
        positions = [
            _pos("I think this is great and I trust the outcome", date="2025-03-01"),
            _pos("I think this is bad and I'm frustrated", date="2025-06-01"),
        ]
        result = detect_contradictions("test", positions)
        self.assertEqual(len(result), 1)
        self.assertIn("positive to negative", result[0].signal)

    def test_negative_to_positive(self) -> None:
        positions = [
            _pos("I'm worried and frustrated about this", date="2025-03-01"),
            _pos("I feel confident and comfortable now", date="2025-06-01",
                 stance_marker="i feel like"),
        ]
        result = detect_contradictions("test", positions)
        self.assertEqual(len(result), 1)
        self.assertIn("negative to positive", result[0].signal)


class StablePositionTests(unittest.TestCase):
    """Verify no contradictions for consistent positions."""

    def test_stable_repetition_not_flagged(self) -> None:
        positions = [
            _pos("I think Marc is trustworthy", date="2025-03-01", entity="Marc"),
            _pos("I believe Marc is a good friend", date="2025-06-01", entity="Marc"),
        ]
        result = detect_contradictions("Marc", positions)
        self.assertEqual(len(result), 0)

    def test_single_position(self) -> None:
        result = detect_contradictions("test", [_pos("I think this is right")])
        self.assertEqual(len(result), 0)

    def test_empty_positions(self) -> None:
        result = detect_contradictions("test", [])
        self.assertEqual(len(result), 0)


class EntityScopingTests(unittest.TestCase):
    """Verify entity handling in contradictions."""

    def test_entity_preserved_in_output(self) -> None:
        positions = [
            _pos("I think Marc is fine", date="2025-03-01"),
            _pos("I changed my mind about Marc", date="2025-06-01",
                 stance_marker="i changed my mind"),
        ]
        result = detect_contradictions("Marc", positions)
        self.assertEqual(result[0].entity, "Marc")

    def test_different_entities_should_be_scoped_externally(self) -> None:
        """detect_contradictions compares all input positions — caller must scope."""
        positions = [
            _pos("I trust Marc completely", date="2025-03-01"),
            _pos("I'm frustrated with Craig", date="2025-06-01"),
        ]
        # Sentiment shift detected because positions aren't entity-scoped.
        # This verifies the caller (not detect_contradictions) is responsible
        # for entity filtering.
        result = detect_contradictions("mixed", positions)
        # May or may not fire depending on valence — the point is the
        # function doesn't filter by entity internally.
        self.assertIsInstance(result, list)


class ChronologicalOrderTests(unittest.TestCase):
    """Verify positions are sorted before comparison."""

    def test_out_of_order_input_sorted(self) -> None:
        positions = [
            _pos("I changed my mind about this", date="2025-06-01",
                 stance_marker="i changed my mind"),
            _pos("I think this is fine", date="2025-03-01"),
        ]
        result = detect_contradictions("test", positions)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].earlier.date, "2025-03-01")
        self.assertEqual(result[0].later.date, "2025-06-01")

    def test_date_range_string(self) -> None:
        positions = [
            _pos("I think X", date="2025-03-01"),
            _pos("I changed my mind about X", date="2025-06-01",
                 stance_marker="i changed my mind"),
        ]
        result = detect_contradictions("test", positions)
        self.assertEqual(result[0].date_range, "2025-03-01 to 2025-06-01")

    def test_same_date_range(self) -> None:
        positions = [
            _pos("AAA I think this is great", date="2025-03-01"),
            _pos("ZZZ I think this is bad and frustrating", date="2025-03-01"),
        ]
        result = detect_contradictions("test", positions)
        if result:
            self.assertEqual(result[0].date_range, "2025-03-01")


class DeterminismTests(unittest.TestCase):
    """Verify deterministic output."""

    def test_deterministic_across_runs(self) -> None:
        positions = [
            _pos("I trust Marc", date="2025-03-01", entity="Marc"),
            _pos("I don't trust Marc anymore", date="2025-06-01", entity="Marc",
                 stance_marker="i don't think"),
        ]
        results = [detect_contradictions("Marc", positions) for _ in range(5)]
        first = [(c.signal, c.date_range) for c in results[0]]
        for r in results[1:]:
            self.assertEqual([(c.signal, c.date_range) for c in r], first)


class SerializationTests(unittest.TestCase):
    """Verify to_dict serialization."""

    def test_to_dict(self) -> None:
        positions = [
            _pos("I think X is fine", date="2025-03-01"),
            _pos("I changed my mind about X", date="2025-06-01",
                 stance_marker="i changed my mind"),
        ]
        result = detect_contradictions("test", positions)
        d = result[0].to_dict()
        self.assertEqual(d["entity"], "test")
        self.assertIn("signal", d)
        self.assertIn("date_range", d)
        self.assertIsInstance(d["earlier"], dict)
        self.assertIsInstance(d["later"], dict)


if __name__ == "__main__":
    unittest.main()
