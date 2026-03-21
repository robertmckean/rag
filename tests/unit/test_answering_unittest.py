import json
import unittest
from pathlib import Path
from unittest.mock import patch

from rag.answering.answer import answer_query, answer_result_json
from rag.answering.generator_llm import LLMSynthesisRequest, synthesize_answer_with_llm
from rag.answering.models import AnswerStatus


# These tests lock the first grounded-answer slice against the existing retrieval contract.
# The fixture run is intentionally small so status classification is deterministic and reviewable.
# Assertions focus on observable grounded-output behavior instead of internal helper implementation details.
class AnsweringTests(unittest.TestCase):
    # Point answer tests at the dedicated grounded-answer fixture run.
    def setUp(self) -> None:
        self.run_dir = Path("tests/fixtures/answering/sample_run")

    # Verify that a clearly supported topic returns a supported answer and bounded evidence.
    def test_supported_answer_from_strong_evidence(self) -> None:
        result = answer_query(self.run_dir, "What have I said about burnout late 2025?", limit=8, max_evidence=5)

        self.assertEqual(result.answer_status, AnswerStatus.SUPPORTED)
        self.assertIn("Burnout was getting worse in late 2025.", result.answer)
        self.assertGreaterEqual(len(result.evidence_used), 1)
        self.assertEqual(result.citations[0].message_id, "chatgpt:message:burnout-user-1")

    # Verify that no meaningful lexical support yields insufficient evidence.
    def test_insufficient_evidence_when_no_meaningful_support_exists(self) -> None:
        result = answer_query(self.run_dir, "What have I said about marathon training?", limit=8, max_evidence=5)

        self.assertEqual(result.answer_status, AnswerStatus.INSUFFICIENT_EVIDENCE)
        self.assertEqual(result.evidence_used, ())
        self.assertIn("not find enough relevant evidence", result.answer)

    # Verify that tokenization noise does not let weak adjacent excerpts count as entity-specific support.
    def test_entity_specific_query_requires_direct_meaningful_coverage(self) -> None:
        result = answer_query(self.run_dir, "What have I said about Mecky's daughter?", limit=8, max_evidence=5)

        self.assertEqual(result.answer_status, AnswerStatus.INSUFFICIENT_EVIDENCE)
        self.assertEqual(result.evidence_used, ())
        self.assertEqual(result.citations, ())
        self.assertGreaterEqual(result.diagnostics.selected_evidence_count, 0)

    # Verify that multi-part queries are not marked supported when only one side of the concept is grounded.
    def test_multi_part_query_with_isolated_overlap_is_not_supported(self) -> None:
        result = answer_query(self.run_dir, "What have I said about burnout motivation late 2025?", limit=8, max_evidence=5)

        self.assertNotEqual(result.answer_status, AnswerStatus.SUPPORTED)
        self.assertIn(result.answer_status, {AnswerStatus.PARTIALLY_SUPPORTED, AnswerStatus.INSUFFICIENT_EVIDENCE})

    # Verify that evidence is marked partial when it covers only part of the requested topic.
    def test_partially_supported_when_query_is_only_partially_covered(self) -> None:
        result = answer_query(self.run_dir, "What have I said about burnout causes?", limit=8, max_evidence=5)

        self.assertEqual(result.answer_status, AnswerStatus.PARTIALLY_SUPPORTED)
        self.assertTrue(any("causes" in gap for gap in result.gaps))
        self.assertIn("partially answers", result.answer)

    # Verify that legitimate partial support survives when evidence covers one substantive query part but not another.
    def test_partial_support_survives_when_one_meaningful_query_part_is_covered(self) -> None:
        result = answer_query(self.run_dir, "What have I said about burnout motivation causes?", limit=8, max_evidence=5)

        self.assertEqual(result.answer_status, AnswerStatus.PARTIALLY_SUPPORTED)
        self.assertTrue(any(item.matched_terms for item in result.evidence_used))
        self.assertTrue(any("causes" in gap for gap in result.gaps))

    # Verify that strict mode remains the default and does not compose support across excerpts.
    def test_strict_mode_remains_default_for_conversational_memory_query(self) -> None:
        with patch("rag.answering.answer.retrieve_message_windows") as mocked_retrieve:
            mocked_retrieve.return_value = (
                self._stub_result(
                    result_rank=1,
                    conversation_id="chatgpt:conversation:conv-memory",
                    messages=(
                        self._stub_message(
                            "chatgpt:message:memory-1",
                            "user",
                            "Larry was sitting beside me while I was playing.",
                            "2025-11-01T10:00:00Z",
                        ),
                        self._stub_message(
                            "chatgpt:message:memory-2",
                            "user",
                            "The guitar riff was the part I kept repeating.",
                            "2025-11-01T10:01:00Z",
                        ),
                    ),
                ),
            )
            result = answer_query(self.run_dir, "What have I said about Larry's guitar playing?", limit=8, max_evidence=5)

        self.assertEqual(result.request.grounding_mode, "strict")
        self.assertEqual(result.answer_status, AnswerStatus.INSUFFICIENT_EVIDENCE)
        self.assertFalse(result.diagnostics.composition_used)

    # Verify that conversational-memory mode can qualify same-window distributed support as partial support.
    def test_conversational_memory_mode_uses_same_window_composed_support(self) -> None:
        with patch("rag.answering.answer.retrieve_message_windows") as mocked_retrieve:
            mocked_retrieve.return_value = (
                self._stub_result(
                    result_rank=1,
                    conversation_id="chatgpt:conversation:conv-memory",
                    messages=(
                        self._stub_message(
                            "chatgpt:message:memory-1",
                            "user",
                            "Larry was sitting beside me while I was playing.",
                            "2025-11-01T10:00:00Z",
                        ),
                        self._stub_message(
                            "chatgpt:message:memory-2",
                            "user",
                            "The guitar riff was the part I kept repeating.",
                            "2025-11-01T10:01:00Z",
                        ),
                    ),
                ),
            )
            result = answer_query(
                self.run_dir,
                "What have I said about Larry's guitar playing?",
                grounding_mode="conversational_memory",
                limit=8,
                max_evidence=5,
            )

        self.assertEqual(result.answer_status, AnswerStatus.PARTIALLY_SUPPORTED)
        self.assertTrue(result.diagnostics.composition_used)
        self.assertEqual(result.diagnostics.support_basis, "window_composed")
        self.assertEqual(result.diagnostics.supporting_excerpt_count, 2)
        self.assertEqual(result.diagnostics.user_excerpt_count, 2)
        self.assertEqual(result.diagnostics.coverage_terms, ("guitar", "larry", "playing"))
        self.assertGreaterEqual(result.diagnostics.coverage_ratio, 1.0)
        self.assertIn("partially answers the question", result.answer)

    # Verify that terms split across separate windows do not qualify for composed support.
    def test_conversational_memory_mode_forbids_cross_window_composition(self) -> None:
        with patch("rag.answering.answer.retrieve_message_windows") as mocked_retrieve:
            mocked_retrieve.return_value = (
                self._stub_result(
                    result_rank=1,
                    conversation_id="chatgpt:conversation:conv-memory",
                    messages=(
                        self._stub_message(
                            "chatgpt:message:window-a-1",
                            "user",
                            "Larry was sitting beside me while I was playing.",
                            "2025-11-01T10:00:00Z",
                        ),
                    ),
                ),
                self._stub_result(
                    result_rank=2,
                    conversation_id="chatgpt:conversation:conv-memory",
                    messages=(
                        self._stub_message(
                            "chatgpt:message:window-b-1",
                            "user",
                            "The guitar riff was the part I kept repeating.",
                            "2025-11-01T10:05:00Z",
                        ),
                    ),
                ),
            )
            result = answer_query(
                self.run_dir,
                "What have I said about Larry's guitar playing?",
                grounding_mode="conversational_memory",
                limit=8,
                max_evidence=5,
            )

        self.assertEqual(result.answer_status, AnswerStatus.INSUFFICIENT_EVIDENCE)
        self.assertFalse(result.diagnostics.composition_used)

    # Verify that assistant-only bridging cannot qualify composed support.
    def test_conversational_memory_mode_blocks_assistant_only_bridge(self) -> None:
        with patch("rag.answering.answer.retrieve_message_windows") as mocked_retrieve:
            mocked_retrieve.return_value = (
                self._stub_result(
                    result_rank=1,
                    conversation_id="chatgpt:conversation:conv-memory",
                    messages=(
                        self._stub_message(
                            "chatgpt:message:memory-user",
                            "user",
                            "Larry was sitting beside me yesterday.",
                            "2025-11-01T10:00:00Z",
                        ),
                        self._stub_message(
                            "chatgpt:message:memory-assistant",
                            "assistant",
                            "That sounds like Larry's guitar playing.",
                            "2025-11-01T10:01:00Z",
                        ),
                    ),
                ),
            )
            result = answer_query(
                self.run_dir,
                "What have I said about Larry's guitar playing?",
                grounding_mode="conversational_memory",
                limit=8,
                max_evidence=5,
            )

        self.assertEqual(result.answer_status, AnswerStatus.INSUFFICIENT_EVIDENCE)
        self.assertFalse(result.diagnostics.composition_used)

    # Verify that a single strict support item still yields supported even in conversational-memory mode.
    def test_conversational_memory_mode_preserves_single_excerpt_supported_case(self) -> None:
        result = answer_query(
            self.run_dir,
            "What have I said about burnout late 2025?",
            grounding_mode="conversational_memory",
            limit=8,
            max_evidence=5,
        )

        self.assertEqual(result.answer_status, AnswerStatus.SUPPORTED)
        self.assertEqual(result.diagnostics.support_basis, "single_excerpt")
        self.assertFalse(result.diagnostics.composition_used)

    # Verify that composed qualification enforces the configured coverage threshold.
    def test_conversational_memory_mode_enforces_coverage_threshold(self) -> None:
        with patch("rag.answering.answer.retrieve_message_windows") as mocked_retrieve:
            mocked_retrieve.return_value = (
                self._stub_result(
                    result_rank=1,
                    conversation_id="chatgpt:conversation:conv-memory",
                    messages=(
                        self._stub_message(
                            "chatgpt:message:memory-1",
                            "user",
                            "Larry was sitting beside me.",
                            "2025-11-01T10:00:00Z",
                        ),
                        self._stub_message(
                            "chatgpt:message:memory-2",
                            "user",
                            "The guitar riff was the part I kept repeating.",
                            "2025-11-01T10:01:00Z",
                        ),
                    ),
                ),
            )
            result = answer_query(
                self.run_dir,
                "What have I said about Larry's guitar playing style?",
                grounding_mode="conversational_memory",
                limit=8,
                max_evidence=5,
            )

        self.assertEqual(result.answer_status, AnswerStatus.INSUFFICIENT_EVIDENCE)
        self.assertFalse(result.diagnostics.composition_used)
        self.assertLess(result.diagnostics.coverage_ratio, 0.75)

    # Verify that retrieved candidates can exist while qualification still rejects them with accurate wording.
    def test_retrieved_candidates_can_exist_but_none_qualify(self) -> None:
        with patch("rag.answering.answer.retrieve_message_windows") as mocked_retrieve:
            mocked_retrieve.return_value = (
                self._stub_result(
                    result_rank=1,
                    conversation_id="chatgpt:conversation:conv-larry",
                    messages=(
                        self._stub_message(
                            "chatgpt:message:larry-1",
                            "user",
                            "Larry and I were sitting around while I was playing bass.",
                            "2025-11-01T10:00:00Z",
                        ),
                        self._stub_message(
                            "chatgpt:message:larry-2",
                            "user",
                            "I kept playing while Larry listened to the riff.",
                            "2025-11-01T10:01:00Z",
                        ),
                    ),
                ),
            )
            result = answer_query(self.run_dir, "What have I said about Larry's guitar playing?", limit=8, max_evidence=5)

        self.assertEqual(result.answer_status, AnswerStatus.INSUFFICIENT_EVIDENCE)
        self.assertEqual(result.evidence_used, ())
        self.assertEqual(result.diagnostics.selected_evidence_count, 2)
        self.assertEqual(result.diagnostics.qualified_evidence_count, 0)

    # Verify that the insufficiency gap distinguishes qualification failure from missing retrieval candidates.
    def test_gap_wording_distinguishes_retrieval_from_qualification_failure(self) -> None:
        with patch("rag.answering.answer.retrieve_message_windows") as mocked_retrieve:
            mocked_retrieve.return_value = (
                self._stub_result(
                    result_rank=1,
                    conversation_id="chatgpt:conversation:conv-larry",
                    messages=(
                        self._stub_message(
                            "chatgpt:message:larry-1",
                            "user",
                            "Larry and I were sitting around while I was playing bass.",
                            "2025-11-01T10:00:00Z",
                        ),
                    ),
                ),
            )
            result = answer_query(self.run_dir, "What have I said about Larry's guitar playing?", limit=8, max_evidence=5)

        self.assertEqual(
            result.gaps,
            ("Relevant retrieval candidates were found, but none directly supported the full combined query concept after evidence qualification.",),
        )

    # Verify that answer diagnostics expose the concrete rejection reasons for dropped evidence.
    def test_answer_diagnostics_show_rejection_reasons(self) -> None:
        with patch("rag.answering.answer.retrieve_message_windows") as mocked_retrieve:
            mocked_retrieve.return_value = (
                self._stub_result(
                    result_rank=1,
                    conversation_id="chatgpt:conversation:conv-larry",
                    messages=(
                        self._stub_message(
                            "chatgpt:message:larry-1",
                            "user",
                            "Larry and I were sitting around while I was playing bass.",
                            "2025-11-01T10:00:00Z",
                        ),
                    ),
                ),
            )
            result = answer_query(self.run_dir, "What have I said about Larry's guitar playing?", limit=8, max_evidence=5)

        self.assertEqual(len(result.diagnostics.rejected_evidence), 1)
        diagnostic = result.diagnostics.rejected_evidence[0]
        self.assertEqual(diagnostic.message_id, "chatgpt:message:larry-1")
        self.assertIn("combined_concept_not_supported", diagnostic.rejection_reasons)
        self.assertIn("guitar", diagnostic.missing_focus_terms)

    # Verify that speculative assistant text does not qualify as factual support for entity-specific questions.
    def test_speculative_assistant_text_does_not_qualify_as_entity_support(self) -> None:
        with patch("rag.answering.answer.retrieve_message_windows") as mocked_retrieve:
            mocked_retrieve.return_value = (
                self._stub_result(
                    result_rank=1,
                    conversation_id="chatgpt:conversation:conv-mecky",
                    messages=(
                        self._stub_message(
                            "chatgpt:message:mecky-user",
                            "user",
                            "I mentioned Mecky yesterday.",
                            "2025-11-01T10:00:00Z",
                        ),
                        self._stub_message(
                            "chatgpt:message:mecky-assistant",
                            "assistant",
                            "That is presumably someone's daughter.",
                            "2025-11-01T10:01:00Z",
                        ),
                    ),
                ),
            )
            result = answer_query(self.run_dir, "What have I said about Mecky's daughter?", limit=8, max_evidence=5)

        self.assertEqual(result.answer_status, AnswerStatus.INSUFFICIENT_EVIDENCE)
        self.assertEqual(result.evidence_used, ())
        self.assertEqual(result.citations, ())

    # Verify that conflicting evidence is surfaced as ambiguous instead of being smoothed over.
    def test_ambiguous_when_conflicting_evidence_is_selected(self) -> None:
        result = answer_query(self.run_dir, "What have I said about workload?", limit=8, max_evidence=5)

        self.assertEqual(result.answer_status, AnswerStatus.AMBIGUOUS)
        self.assertTrue(result.conflicts)
        self.assertIn("manageable", result.answer)
        self.assertIn("unmanageable", result.answer)

    # Verify that citations map back to the exact evidence items used in the answer.
    def test_citations_map_to_actual_evidence_used(self) -> None:
        result = answer_query(self.run_dir, "What have I said about burnout?", limit=8, max_evidence=5)

        evidence_message_ids = [item.citation.message_id for item in result.evidence_used]
        citation_message_ids = [citation.message_id for citation in result.citations]
        self.assertEqual(citation_message_ids, evidence_message_ids)

    # Verify that the JSON payload stays stable and serialization-friendly.
    def test_json_output_shape_is_stable(self) -> None:
        result = answer_query(self.run_dir, "What have I said about burnout?", limit=8, max_evidence=5)
        payload = json.loads(answer_result_json(result))

        self.assertEqual(payload["answer_status"], "supported")
        self.assertEqual(payload["retrieval_summary"]["retrieval_mode"], "relevance")
        self.assertEqual(payload["request"]["grounding_mode"], "strict")
        self.assertEqual(payload["evidence_used"][0]["citation"]["message_id"], "chatgpt:message:burnout-user-1")
        self.assertIn("gaps", payload)
        self.assertIn("conflicts", payload)
        self.assertIn("diagnostics", payload)

    # Verify that the template generator does not pull in unrelated content outside selected evidence.
    def test_answer_generator_stays_inside_selected_evidence(self) -> None:
        result = answer_query(self.run_dir, "What have I said about burnout?", limit=8, max_evidence=5)

        selected_excerpts = {item.citation.excerpt for item in result.evidence_used}
        self.assertIn("Burnout was getting worse in late 2025.", selected_excerpts)
        self.assertNotIn("manageable", result.answer)
        self.assertNotIn("marathon", result.answer)

    # Verify that the default Phase 3A path remains unchanged when LLM synthesis is not requested.
    def test_default_phase_3a_path_is_unchanged_without_llm(self) -> None:
        baseline = answer_query(self.run_dir, "What have I said about burnout?", limit=8, max_evidence=5)
        with patch("rag.answering.answer.synthesize_answer_with_llm") as mocked_synthesize:
            llm_disabled = answer_query(self.run_dir, "What have I said about burnout?", limit=8, max_evidence=5, llm=False)

        mocked_synthesize.assert_not_called()
        self.assertEqual(llm_disabled.answer_status, baseline.answer_status)
        self.assertEqual(llm_disabled.answer, baseline.answer)
        self.assertEqual(llm_disabled.citations, baseline.citations)

    # Verify that valid LLM synthesis only rewrites answer text and cited evidence, without changing status.
    def test_valid_llm_synthesis_preserves_status_and_traceable_citations(self) -> None:
        with patch("rag.answering.answer.synthesize_answer_with_llm") as mocked_synthesize:
            mocked_synthesize.return_value = type(
                "StubSynthesis",
                (),
                {
                    "answer_text": "Burnout was an ongoing concern that worsened in late 2025.",
                    "citation_ids": ("e1",),
                },
            )()
            result = answer_query(
                self.run_dir,
                "What have I said about burnout?",
                limit=8,
                max_evidence=5,
                llm=True,
                llm_model="gpt-5-mini",
            )

        self.assertEqual(result.answer_status, AnswerStatus.SUPPORTED)
        self.assertEqual(result.answer, "Burnout was an ongoing concern that worsened in late 2025.")
        self.assertEqual(len(result.citations), 1)
        self.assertEqual(result.citations[0].message_id, "chatgpt:message:burnout-user-1")
        self.assertGreaterEqual(len(result.evidence_used), 1)

    # Verify that invalid LLM synthesis falls back to the deterministic Phase 3A answer.
    def test_invalid_llm_output_falls_back_to_deterministic_answer(self) -> None:
        baseline = answer_query(self.run_dir, "What have I said about burnout?", limit=8, max_evidence=5)
        with patch("rag.answering.answer.synthesize_answer_with_llm", side_effect=ValueError("bad llm output")):
            result = answer_query(self.run_dir, "What have I said about burnout?", limit=8, max_evidence=5, llm=True)

        self.assertEqual(result.answer_status, baseline.answer_status)
        self.assertEqual(result.answer, baseline.answer)
        self.assertEqual(result.citations, baseline.citations)

    # Verify that the no-evidence path never tries to synthesize and stays insufficient_evidence.
    def test_no_evidence_case_skips_llm_and_stays_insufficient(self) -> None:
        with patch("rag.answering.answer.synthesize_answer_with_llm") as mocked_synthesize:
            result = answer_query(self.run_dir, "What have I said about marathon training?", limit=8, max_evidence=5, llm=True)

        mocked_synthesize.assert_not_called()
        self.assertEqual(result.answer_status, AnswerStatus.INSUFFICIENT_EVIDENCE)
        self.assertEqual(result.evidence_used, ())

    # Verify that constrained synthesis accepts valid structured output that cites only provided evidence IDs.
    def test_llm_synthesis_validates_structured_output_and_citations(self) -> None:
        baseline = answer_query(self.run_dir, "What have I said about burnout?", limit=8, max_evidence=5)
        request = LLMSynthesisRequest(
            query=baseline.query,
            answer_status=baseline.answer_status,
            evidence_items=baseline.evidence_used,
            gaps=baseline.gaps,
            conflicts=baseline.conflicts,
            model="gpt-5-mini",
        )
        citation_ids = [f"e{item.rank}" for item in baseline.evidence_used]
        with patch(
            "rag.answering.generator_llm._call_llm",
            return_value=json.dumps(
                {
                    "answer_text": "Burnout worsened in late 2025 and drained daily energy.",
                    "citation_ids": citation_ids,
                }
            ),
        ):
            synthesis = synthesize_answer_with_llm(request)

        self.assertEqual(synthesis.citation_ids, tuple(citation_ids))
        self.assertIn("late 2025", synthesis.answer_text)

    # Verify that synthesis rejects structured output that introduces unseen surface forms or bad citations.
    def test_llm_synthesis_rejects_invalid_structured_output(self) -> None:
        baseline = answer_query(self.run_dir, "What have I said about burnout?", limit=8, max_evidence=5)
        request = LLMSynthesisRequest(
            query=baseline.query,
            answer_status=baseline.answer_status,
            evidence_items=baseline.evidence_used,
            gaps=baseline.gaps,
            conflicts=baseline.conflicts,
            model="gpt-5-mini",
        )
        with patch(
            "rag.answering.generator_llm._call_llm",
            return_value=json.dumps(
                {
                    "answer_text": "Burnout worsened in 2026 after a conversation with Alice.",
                    "citation_ids": ["e9"],
                }
            ),
        ):
            with self.assertRaises(ValueError):
                synthesize_answer_with_llm(request)

    # Verify that answer_query passes the requested retrieval mode through to the retrieval layer.
    def test_retrieval_mode_is_passed_through(self) -> None:
        with patch("rag.answering.answer.retrieve_message_windows") as mocked_retrieve:
            mocked_retrieve.return_value = ()
            answer_query(self.run_dir, "burnout", retrieval_mode="newest", limit=3, max_evidence=2)

        self.assertEqual(mocked_retrieve.call_args.kwargs["mode"], "newest")
        self.assertEqual(mocked_retrieve.call_args.kwargs["limit"], 3)

    # Verify that the evidence cap truncates deterministically in ranked selection order.
    def test_evidence_cap_is_respected_deterministically(self) -> None:
        result = answer_query(self.run_dir, "What have I said about burnout?", limit=8, max_evidence=1)

        self.assertEqual(len(result.evidence_used), 1)
        self.assertEqual(result.evidence_used[0].citation.message_id, "chatgpt:message:burnout-user-1")

    # Build a small retrieval-result stub without changing real retrieval behavior.
    def _stub_result(self, *, result_rank: int, conversation_id: str, messages: tuple[dict[str, object], ...]) -> object:
        return type(
            "StubRankedResult",
            (),
            {
                "rank": result_rank,
                "result_id": f"stub-window-{result_rank}",
                "score": 1.5,
                "run_id": "answer-sample-run",
                "provider": "chatgpt",
                "conversation_id": conversation_id,
                "conversation_title": "Stub Conversation",
                "messages": messages,
                "match_basis": {"retrieval_mode": "relevance"},
            },
        )()

    # Build one canonical-looking message stub for answer-layer qualification tests.
    def _stub_message(
        self,
        message_id: str,
        author_role: str,
        text: str,
        created_at: str,
        conversation_id: str = "chatgpt:conversation:conv-mecky",
    ) -> dict[str, object]:
        return {
            "message_id": message_id,
            "provider": "chatgpt",
            "conversation_id": conversation_id,
            "author_role": author_role,
            "created_at": created_at,
            "text": text,
        }


if __name__ == "__main__":
    unittest.main()
