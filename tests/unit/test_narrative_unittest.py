"""Unit tests for the narrative reconstruction layer (Phase 6)."""

import json
import unittest

from rag.answering.models import Citation, EvidenceItem
from rag.narrative.builder import (
    DEFAULT_GAP_THRESHOLD_DAYS,
    DEFAULT_PHASE_WINDOW_DAYS,
    build_narrative,
    _days_between,
    _detect_gaps,
    _detect_limitations,
    _detect_transitions,
    _group_into_phases,
    _parse_date,
    _sort_chronologically,
)
from rag.narrative.models import (
    NarrativeGap,
    NarrativeLimitation,
    NarrativePhase,
    NarrativeReconstruction,
    NarrativeTransition,
)
from rag.narrative.renderer import render_debug, render_json, render_text


def _make_evidence(
    rank: int,
    excerpt: str,
    created_at: str | None = None,
    author_role: str = "user",
) -> EvidenceItem:
    return EvidenceItem(
        rank=rank,
        source_result_rank=rank,
        window_id=f"w{rank}",
        score=1.0,
        retrieval_mode="relevance",
        author_role=author_role,
        matched_terms=(),
        citation=Citation(
            provider="chatgpt",
            conversation_id=f"conv{rank}",
            title=f"Conversation {rank}",
            message_id=f"msg{rank}",
            created_at=created_at,
            excerpt=excerpt,
        ),
    )


class NarrativeModelsTests(unittest.TestCase):
    """Test schema definitions and serialization."""

    def test_narrative_phase_to_dict(self) -> None:
        phase = NarrativePhase(
            label="2025-03-06: Benz",
            description="test description",
            evidence_ids=("e1", "e2"),
            date_range="2025-03-06",
            support="partially_supported",
        )
        d = phase.to_dict()
        self.assertEqual(d["label"], "2025-03-06: Benz")
        self.assertEqual(d["support"], "partially_supported")
        self.assertIsInstance(d["evidence_ids"], tuple)

    def test_narrative_reconstruction_to_dict_has_all_keys(self) -> None:
        recon = NarrativeReconstruction(
            query="test",
            summary="summary",
            timeline=(),
            transitions=(),
            gaps=(),
            limitations=(),
            evidence_count=0,
        )
        d = recon.to_dict()
        for key in ("query", "summary", "timeline", "transitions", "gaps", "limitations", "evidence_count"):
            self.assertIn(key, d)

    def test_narrative_reconstruction_json_roundtrip(self) -> None:
        recon = NarrativeReconstruction(
            query="test",
            summary="summary",
            timeline=(NarrativePhase("l", "d", ("e1",), None, "supported"),),
            transitions=(),
            gaps=(NarrativeGap("gap desc", "reason"),),
            limitations=(NarrativeLimitation("lim desc", "truncated_excerpt"),),
            evidence_count=1,
        )
        serialized = json.dumps(recon.to_dict())
        parsed = json.loads(serialized)
        self.assertEqual(parsed["query"], "test")
        self.assertEqual(len(parsed["timeline"]), 1)
        self.assertEqual(len(parsed["gaps"]), 1)
        self.assertEqual(len(parsed["limitations"]), 1)


class ChronologicalSortTests(unittest.TestCase):
    """Test evidence sorting."""

    def test_sorts_by_timestamp(self) -> None:
        e1 = _make_evidence(1, "later event", "2025-07-13T15:00:00Z")
        e2 = _make_evidence(2, "earlier event", "2025-03-06T12:00:00Z")
        sorted_ev = _sort_chronologically((e1, e2))
        self.assertEqual(sorted_ev[0].rank, 2)
        self.assertEqual(sorted_ev[1].rank, 1)

    def test_null_timestamps_sort_last(self) -> None:
        e1 = _make_evidence(1, "no timestamp", None)
        e2 = _make_evidence(2, "has timestamp", "2025-01-01T00:00:00Z")
        sorted_ev = _sort_chronologically((e1, e2))
        self.assertEqual(sorted_ev[0].rank, 2)
        self.assertEqual(sorted_ev[1].rank, 1)


class PhaseGroupingTests(unittest.TestCase):
    """Test evidence grouping into phases."""

    def test_single_evidence_creates_single_supported_phase(self) -> None:
        evidence = (_make_evidence(1, "I met Marc at the bar", "2025-02-07T08:00:00Z"),)
        phases = _group_into_phases(list(evidence), ("marc",), DEFAULT_PHASE_WINDOW_DAYS)
        self.assertEqual(len(phases), 1)
        self.assertEqual(phases[0].support, "supported")
        self.assertEqual(phases[0].evidence_ids, ("e1",))

    def test_same_date_different_topics_creates_separate_phases(self) -> None:
        e1 = _make_evidence(1, "I talked about Marc and the meeting with Craig", "2025-02-07T08:00:00Z")
        e2 = _make_evidence(2, "I went swimming at the beach and saw dolphins playing", "2025-02-07T10:00:00Z")
        phases = _group_into_phases([e1, e2], ("marc",), DEFAULT_PHASE_WINDOW_DAYS)
        self.assertEqual(len(phases), 2, "Same-date evidence with different topics should not auto-group")

    def test_same_date_same_topic_groups_together(self) -> None:
        e1 = _make_evidence(1, "I met Marc at the bar to discuss the meeting", "2025-02-07T08:00:00Z")
        e2 = _make_evidence(2, "Marc said he wanted to discuss the project at the meeting", "2025-02-07T10:00:00Z")
        phases = _group_into_phases([e1, e2], ("marc",), DEFAULT_PHASE_WINDOW_DAYS)
        self.assertEqual(len(phases), 1, "Same-date evidence with shared topic should group")
        self.assertEqual(phases[0].support, "partially_supported")

    def test_distant_dates_create_separate_phases(self) -> None:
        e1 = _make_evidence(1, "I talked about Marc at the meeting", "2025-02-07T08:00:00Z")
        e2 = _make_evidence(2, "I saw Marc again at the villa", "2025-07-13T15:00:00Z")
        phases = _group_into_phases([e1, e2], ("marc",), DEFAULT_PHASE_WINDOW_DAYS)
        self.assertEqual(len(phases), 2)

    def test_multi_evidence_phase_is_partially_supported(self) -> None:
        e1 = _make_evidence(1, "Shadow work journey begins with reflection", "2025-11-22T06:00:00Z")
        e2 = _make_evidence(2, "Shadow work continues with deeper reflection", "2025-11-24T04:00:00Z")
        phases = _group_into_phases([e1, e2], ("shadow", "work"), DEFAULT_PHASE_WINDOW_DAYS)
        self.assertEqual(len(phases), 1)
        self.assertEqual(phases[0].support, "partially_supported")

    def test_phase_label_includes_date_and_entities(self) -> None:
        e1 = _make_evidence(1, "I met Marc at the villa", "2025-02-07T08:00:00Z")
        phases = _group_into_phases([e1], ("marc",), DEFAULT_PHASE_WINDOW_DAYS)
        self.assertIn("2025-02-07", phases[0].label)
        self.assertIn("Marc", phases[0].label)

    def test_phase_date_range_spans_multiple_dates(self) -> None:
        e1 = _make_evidence(1, "Marc discussion at the bar", "2025-02-07T08:00:00Z")
        e2 = _make_evidence(2, "Marc follow-up at the bar", "2025-02-10T10:00:00Z")
        phases = _group_into_phases([e1, e2], ("marc",), DEFAULT_PHASE_WINDOW_DAYS)
        self.assertEqual(len(phases), 1)
        self.assertEqual(phases[0].date_range, "2025-02-07 to 2025-02-10")

    def test_configurable_phase_window(self) -> None:
        e1 = _make_evidence(1, "Marc discussion at the meeting", "2025-02-07T08:00:00Z")
        e2 = _make_evidence(2, "Marc follow-up at the meeting", "2025-02-10T10:00:00Z")
        # With 1-day window, these should be separate.
        phases = _group_into_phases([e1, e2], ("marc",), phase_window_days=1)
        self.assertEqual(len(phases), 2)
        # With 7-day window, they should group.
        phases = _group_into_phases([e1, e2], ("marc",), phase_window_days=7)
        self.assertEqual(len(phases), 1)


class TransitionDetectionTests(unittest.TestCase):
    """Test transition detection between phases."""

    def test_topic_change_detected(self) -> None:
        e1 = _make_evidence(1, "I discussed Marc and the meeting with Craig", "2025-02-07T08:00:00Z")
        e2 = _make_evidence(2, "I went to see Benz at the pool hall tonight", "2025-07-13T15:00:00Z")
        sorted_ev = [e1, e2]
        phases = _group_into_phases(sorted_ev, ("marc", "benz"), phase_window_days=7)
        self.assertEqual(len(phases), 2)
        transitions = _detect_transitions(phases, sorted_ev)
        self.assertEqual(len(transitions), 1)
        self.assertIn("Topic shift", transitions[0].description)
        self.assertEqual(transitions[0].support, "partially_supported")

    def test_transitions_have_evidence_ids(self) -> None:
        e1 = _make_evidence(1, "Marc and Craig meeting discussion", "2025-02-07T08:00:00Z")
        e2 = _make_evidence(2, "Benz at the pool hall was fun", "2025-07-13T15:00:00Z")
        sorted_ev = [e1, e2]
        phases = _group_into_phases(sorted_ev, ("marc", "benz"), phase_window_days=7)
        transitions = _detect_transitions(phases, sorted_ev)
        self.assertTrue(len(transitions[0].evidence_ids) > 0)
        for eid in transitions[0].evidence_ids:
            self.assertTrue(eid.startswith("e"))

    def test_all_transitions_are_partially_supported(self) -> None:
        e1 = _make_evidence(1, "Marc meeting at the villa", "2025-02-07T08:00:00Z")
        e2 = _make_evidence(2, "Benz at pool hall was great", "2025-07-13T15:00:00Z")
        e3 = _make_evidence(3, "Shadow work and stoicism practice", "2025-12-10T06:00:00Z")
        sorted_ev = [e1, e2, e3]
        phases = _group_into_phases(sorted_ev, ("test",), phase_window_days=7)
        transitions = _detect_transitions(phases, sorted_ev)
        for trans in transitions:
            self.assertEqual(trans.support, "partially_supported")

    def test_stated_change_detected(self) -> None:
        e1 = _make_evidence(1, "I was living at the villa with the group", "2025-06-01T00:00:00Z")
        e2 = _make_evidence(2, "I moved out and started traveling to Cambodia", "2025-11-01T00:00:00Z")
        sorted_ev = [e1, e2]
        phases = _group_into_phases(sorted_ev, ("living",), phase_window_days=7)
        transitions = _detect_transitions(phases, sorted_ev)
        self.assertEqual(len(transitions), 1)
        self.assertIn("Stated change", transitions[0].description)


class GapDetectionTests(unittest.TestCase):
    """Test temporal gap detection."""

    def test_large_gap_detected(self) -> None:
        phases = [
            NarrativePhase("a", "desc", ("e1",), "2025-02-07", "supported"),
            NarrativePhase("b", "desc", ("e2",), "2025-07-13", "supported"),
        ]
        gaps = _detect_gaps(phases, DEFAULT_GAP_THRESHOLD_DAYS)
        self.assertEqual(len(gaps), 1)
        self.assertIn("No evidence between", gaps[0].description)

    def test_small_gap_not_flagged(self) -> None:
        phases = [
            NarrativePhase("a", "desc", ("e1",), "2025-02-07", "supported"),
            NarrativePhase("b", "desc", ("e2",), "2025-02-20", "supported"),
        ]
        gaps = _detect_gaps(phases, DEFAULT_GAP_THRESHOLD_DAYS)
        self.assertEqual(len(gaps), 0)

    def test_configurable_gap_threshold(self) -> None:
        phases = [
            NarrativePhase("a", "desc", ("e1",), "2025-02-07", "supported"),
            NarrativePhase("b", "desc", ("e2",), "2025-02-20", "supported"),
        ]
        # 10-day threshold should flag a 13-day gap.
        gaps = _detect_gaps(phases, gap_threshold_days=10)
        self.assertEqual(len(gaps), 1)

    def test_no_gap_when_dates_missing(self) -> None:
        phases = [
            NarrativePhase("a", "desc", ("e1",), None, "supported"),
            NarrativePhase("b", "desc", ("e2",), "2025-07-13", "supported"),
        ]
        gaps = _detect_gaps(phases, DEFAULT_GAP_THRESHOLD_DAYS)
        self.assertEqual(len(gaps), 0)


class LimitationDetectionTests(unittest.TestCase):
    """Test data-quality limitation detection."""

    def test_truncated_excerpt_detected(self) -> None:
        long_text = "x" * 200
        evidence = [_make_evidence(1, long_text, "2025-01-01T00:00:00Z")]
        limitations = _detect_limitations(evidence)
        kinds = [lim.kind for lim in limitations]
        self.assertIn("truncated_excerpt", kinds)

    def test_null_timestamp_detected(self) -> None:
        evidence = [_make_evidence(1, "some text", None)]
        limitations = _detect_limitations(evidence)
        kinds = [lim.kind for lim in limitations]
        self.assertIn("null_timestamp", kinds)

    def test_ambiguous_ordering_detected(self) -> None:
        e1 = _make_evidence(1, "first message today", "2025-02-07T08:00:00Z")
        e2 = _make_evidence(2, "second message today", "2025-02-07T10:00:00Z")
        limitations = _detect_limitations([e1, e2])
        kinds = [lim.kind for lim in limitations]
        self.assertIn("ambiguous_ordering", kinds)

    def test_clean_evidence_has_no_limitations(self) -> None:
        e1 = _make_evidence(1, "Short clean text", "2025-02-07T08:00:00Z")
        e2 = _make_evidence(2, "Another clean text", "2025-02-08T10:00:00Z")
        limitations = _detect_limitations([e1, e2])
        self.assertEqual(len(limitations), 0)


class FullNarrativeTests(unittest.TestCase):
    """Integration tests for build_narrative."""

    def test_empty_evidence_produces_empty_narrative(self) -> None:
        result = build_narrative("test query", ())
        self.assertEqual(result.evidence_count, 0)
        self.assertEqual(len(result.timeline), 0)
        self.assertIn("No evidence", result.summary)

    def test_multi_conversation_narrative(self) -> None:
        evidence = (
            _make_evidence(1, "Shadow work journey begins with my mother's story", "2025-11-22T06:00:00Z"),
            _make_evidence(2, "We're having an argument about shadow work topics", "2025-11-24T04:00:00Z"),
            _make_evidence(3, "I embraced stoicism to understand shadow work", "2025-12-10T06:00:00Z"),
        )
        result = build_narrative("what was my path to shadow work", evidence)
        self.assertGreater(len(result.timeline), 0)
        self.assertEqual(result.evidence_count, 3)
        # All evidence should be represented.
        all_eids = set()
        for phase in result.timeline:
            all_eids.update(phase.evidence_ids)
        self.assertEqual(all_eids, {"e1", "e2", "e3"})

    def test_fragmented_evidence_across_time(self) -> None:
        evidence = (
            _make_evidence(1, "I met Benz at the bar in March", "2025-03-06T12:00:00Z"),
            _make_evidence(2, "Thinking of sending Benz a message about pool", "2025-07-13T15:00:00Z"),
            _make_evidence(3, "I stopped at the bar to play pool with Benz", "2026-01-23T14:00:00Z"),
        )
        result = build_narrative("what happened with Benz", evidence)
        # Should produce multiple phases due to time gaps.
        self.assertGreaterEqual(len(result.timeline), 2)
        # Should detect at least one gap.
        self.assertGreater(len(result.gaps), 0)

    def test_ambiguous_timeline_surfaces_limitations(self) -> None:
        evidence = (
            _make_evidence(1, "Something happened first", None),
            _make_evidence(2, "Something happened second", None),
        )
        result = build_narrative("what happened", evidence)
        self.assertGreater(len(result.limitations), 0)
        kinds = [lim.kind for lim in result.limitations]
        self.assertIn("null_timestamp", kinds)

    def test_gaps_surfaced_not_filled(self) -> None:
        evidence = (
            _make_evidence(1, "Early event about Marc and Craig", "2025-02-07T08:00:00Z"),
            _make_evidence(2, "Much later event about Marc alone", "2025-08-15T10:00:00Z"),
        )
        result = build_narrative("what did I learn about Marc", evidence)
        self.assertGreater(len(result.gaps), 0)
        for gap in result.gaps:
            self.assertIn("No evidence between", gap.description)

    def test_no_hallucinated_evidence_ids(self) -> None:
        evidence = (
            _make_evidence(1, "Marc meeting discussion", "2025-02-07T08:00:00Z"),
            _make_evidence(2, "Marc follow-up conversation", "2025-02-12T16:00:00Z"),
        )
        result = build_narrative("what did I learn about Marc", evidence)
        valid_ids = {"e1", "e2"}
        for phase in result.timeline:
            for eid in phase.evidence_ids:
                self.assertIn(eid, valid_ids, f"Phase references non-existent evidence: {eid}")
        for trans in result.transitions:
            for eid in trans.evidence_ids:
                self.assertIn(eid, valid_ids, f"Transition references non-existent evidence: {eid}")


class RendererTests(unittest.TestCase):
    """Test output formatters."""

    def _sample_narrative(self) -> NarrativeReconstruction:
        return NarrativeReconstruction(
            query="test query",
            summary="Test summary across 2 phases.",
            timeline=(
                NarrativePhase("Phase A", "desc A", ("e1",), "2025-02-07", "supported"),
                NarrativePhase("Phase B", "desc B", ("e2",), "2025-07-13", "supported"),
            ),
            transitions=(
                NarrativeTransition("Phase A", "Phase B", "Topic shift", ("e1", "e2"), "partially_supported"),
            ),
            gaps=(NarrativeGap("150-day gap", "retrieval miss"),),
            limitations=(NarrativeLimitation("e1 truncated", "truncated_excerpt"),),
            evidence_count=2,
        )

    def test_json_output_is_valid_json(self) -> None:
        output = render_json(self._sample_narrative())
        parsed = json.loads(output)
        self.assertEqual(parsed["query"], "test query")
        self.assertEqual(len(parsed["timeline"]), 2)

    def test_text_output_contains_key_sections(self) -> None:
        output = render_text(self._sample_narrative())
        self.assertIn("Narrative:", output)
        self.assertIn("Summary:", output)
        self.assertIn("Timeline:", output)
        self.assertIn("Transitions:", output)
        self.assertIn("Gaps:", output)
        self.assertIn("Limitations:", output)

    def test_debug_output_includes_debug_section(self) -> None:
        output = render_debug(self._sample_narrative())
        self.assertIn("=== DEBUG ===", output)
        self.assertIn("evidence_ids:", output)


class HelperTests(unittest.TestCase):
    """Test internal helper functions."""

    def test_parse_date_from_iso(self) -> None:
        self.assertEqual(_parse_date("2025-02-07T08:00:00Z"), "2025-02-07")

    def test_parse_date_from_none(self) -> None:
        self.assertIsNone(_parse_date(None))

    def test_parse_date_from_short_string(self) -> None:
        self.assertIsNone(_parse_date("2025"))

    def test_days_between_same_date(self) -> None:
        self.assertEqual(_days_between("2025-02-07", "2025-02-07"), 0)

    def test_days_between_different_dates(self) -> None:
        days = _days_between("2025-02-07", "2025-02-14")
        self.assertGreater(days, 0)
        self.assertLess(days, 15)


if __name__ == "__main__":
    unittest.main()
