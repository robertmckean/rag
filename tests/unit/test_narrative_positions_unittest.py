"""Unit tests for Phase 13A — deterministic user-position extraction."""

import unittest

from rag.narrative.models import NarrativePhase
from rag.narrative.positions import (
    Position,
    _parse_segments,
    extract_positions,
)


def _make_phase(
    description: str,
    *,
    evidence_ids: tuple[str, ...] = ("e1",),
    date_range: str | None = "2024-06-15",
    label: str = "test-phase",
) -> NarrativePhase:
    return NarrativePhase(
        label=label,
        description=description,
        evidence_ids=evidence_ids,
        date_range=date_range,
        support="supported",
    )


class SegmentParsingTests(unittest.TestCase):
    """Verify description segment parsing."""

    def test_single_user_segment(self) -> None:
        desc = '(2024-06-15) [user] "I think Marc is great"'
        segments = _parse_segments(desc)
        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0], ("2024-06-15", "user", "I think Marc is great"))

    def test_mixed_segments(self) -> None:
        desc = '(2024-06-15) [user] "I think this is right" | (2024-06-15) [assistant] "That makes sense"'
        segments = _parse_segments(desc)
        self.assertEqual(len(segments), 2)
        self.assertEqual(segments[0][1], "user")
        self.assertEqual(segments[1][1], "assistant")

    def test_no_date_segment(self) -> None:
        desc = '[user] "I believe something"'
        segments = _parse_segments(desc)
        self.assertEqual(len(segments), 1)
        self.assertIsNone(segments[0][0])
        self.assertEqual(segments[0][1], "user")


class StanceMarkerDetectionTests(unittest.TestCase):
    """Verify stance markers are detected in user text only."""

    def test_basic_stance_detected(self) -> None:
        phase = _make_phase('(2024-06-15) [user] "I think this approach works best"')
        positions = extract_positions(phase)
        self.assertEqual(len(positions), 1)
        self.assertIn("I think", positions[0].text)
        self.assertEqual(positions[0].stance_marker, "i think")

    def test_assistant_text_excluded(self) -> None:
        phase = _make_phase('(2024-06-15) [assistant] "I think you should try this"')
        positions = extract_positions(phase)
        self.assertEqual(len(positions), 0)

    def test_no_false_positives_on_non_stance(self) -> None:
        phase = _make_phase('(2024-06-15) [user] "The weather was nice today"')
        positions = extract_positions(phase)
        self.assertEqual(len(positions), 0)

    def test_multiple_positions_from_one_phase(self) -> None:
        desc = (
            '(2024-06-15) [user] "I think Marc is trustworthy. '
            'I realized that boundaries matter"'
        )
        phase = _make_phase(desc)
        positions = extract_positions(phase)
        self.assertEqual(len(positions), 2)
        markers = {p.stance_marker for p in positions}
        self.assertEqual(markers, {"i think", "i realized"})

    def test_case_insensitive_matching(self) -> None:
        phase = _make_phase('(2024-06-15) [user] "I BELIEVE this is correct"')
        positions = extract_positions(phase)
        self.assertEqual(len(positions), 1)
        self.assertEqual(positions[0].stance_marker, "i believe")

    def test_extracted_text_preserves_wording(self) -> None:
        phase = _make_phase(
            '(2024-06-15) [user] "After talking to him, I decided to set firm boundaries. It felt right."'
        )
        positions = extract_positions(phase)
        self.assertEqual(len(positions), 1)
        self.assertIn("I decided to set firm boundaries", positions[0].text)

    def test_smart_quote_apostrophe_handled(self) -> None:
        """Unicode right single quotation mark (U+2019) should match."""
        phase = _make_phase('(2024-06-15) [user] "I don\u2019t think it went well"')
        positions = extract_positions(phase)
        self.assertEqual(len(positions), 1)
        self.assertEqual(positions[0].stance_marker, "i don't think")

    def test_all_stance_markers_recognized(self) -> None:
        """Each defined marker should be detectable."""
        markers_to_test = [
            "I think", "I believe", "I decided", "I realized",
            "I concluded", "my view is", "I was wrong about",
            "I changed my mind", "I no longer", "I used to think",
            "I don't think", "I don't believe", "I feel like",
            "I noticed", "I learned", "I understand",
            "I'm sure", "I figured out",
        ]
        for marker in markers_to_test:
            desc = f'(2024-01-01) [user] "{marker} something about this"'
            phase = _make_phase(desc)
            positions = extract_positions(phase)
            self.assertGreater(
                len(positions), 0,
                f"Marker '{marker}' was not detected",
            )


class EntityAssociationTests(unittest.TestCase):
    """Verify entity detection within extracted positions."""

    def test_entity_captured_when_present(self) -> None:
        phase = _make_phase('(2024-06-15) [user] "I think Marc is a good friend"')
        positions = extract_positions(phase)
        self.assertEqual(len(positions), 1)
        self.assertEqual(positions[0].entity, "Marc")

    def test_entity_none_when_absent(self) -> None:
        phase = _make_phase('(2024-06-15) [user] "I think this approach is correct"')
        positions = extract_positions(phase)
        self.assertEqual(len(positions), 1)
        self.assertIsNone(positions[0].entity)

    def test_entity_from_sentence_not_other_text(self) -> None:
        desc = (
            '(2024-06-15) [user] "Marc told me something. '
            'I believe the plan is solid"'
        )
        phase = _make_phase(desc)
        positions = extract_positions(phase)
        self.assertEqual(len(positions), 1)
        # "I believe the plan is solid" — no entity in that sentence.
        self.assertIsNone(positions[0].entity)


class PositionMetadataTests(unittest.TestCase):
    """Verify date, evidence_id, and other metadata."""

    def test_date_from_phase(self) -> None:
        phase = _make_phase(
            '(2024-06-15) [user] "I think this is right"',
            date_range="2024-06-15",
        )
        positions = extract_positions(phase)
        self.assertEqual(positions[0].date, "2024-06-15")

    def test_date_none_when_phase_has_none(self) -> None:
        phase = _make_phase(
            '[user] "I think this is right"',
            date_range=None,
        )
        positions = extract_positions(phase)
        self.assertIsNone(positions[0].date)

    def test_evidence_id_assigned(self) -> None:
        phase = _make_phase(
            '(2024-06-15) [user] "I realized the pattern"',
            evidence_ids=("e3",),
        )
        positions = extract_positions(phase)
        self.assertEqual(positions[0].evidence_id, "e3")

    def test_multiple_evidence_ids_cycled(self) -> None:
        desc = (
            '(2024-06-15) [user] "I think this is key" | '
            '(2024-06-16) [user] "I believe that too"'
        )
        phase = _make_phase(desc, evidence_ids=("e1", "e2"))
        positions = extract_positions(phase)
        self.assertEqual(len(positions), 2)
        self.assertEqual(positions[0].evidence_id, "e1")
        self.assertEqual(positions[1].evidence_id, "e2")

    def test_empty_evidence_ids_returns_nothing(self) -> None:
        phase = _make_phase(
            '(2024-06-15) [user] "I think this is right"',
            evidence_ids=(),
        )
        positions = extract_positions(phase)
        self.assertEqual(len(positions), 0)


class DeterminismTests(unittest.TestCase):
    """Verify deterministic output."""

    def test_deterministic_across_runs(self) -> None:
        desc = (
            '(2024-06-15) [user] "I think Marc is great. '
            'I also realized something about Craig"'
        )
        phase = _make_phase(desc)
        run1 = extract_positions(phase)
        run2 = extract_positions(phase)
        self.assertEqual(run1, run2)


class ToDictTests(unittest.TestCase):
    """Verify serialization."""

    def test_to_dict(self) -> None:
        pos = Position(
            text="I think this is right",
            date="2024-06-15",
            entity="Marc",
            evidence_id="e1",
            stance_marker="i think",
        )
        d = pos.to_dict()
        self.assertEqual(d["text"], "I think this is right")
        self.assertEqual(d["date"], "2024-06-15")
        self.assertEqual(d["entity"], "Marc")
        self.assertEqual(d["evidence_id"], "e1")
        self.assertEqual(d["stance_marker"], "i think")


if __name__ == "__main__":
    unittest.main()
