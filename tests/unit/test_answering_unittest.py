import json
import unittest
from pathlib import Path
from unittest.mock import patch

from rag.answering.answer import answer_query, answer_result_json
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
        self.assertEqual(payload["evidence_used"][0]["citation"]["message_id"], "chatgpt:message:burnout-user-1")
        self.assertIn("gaps", payload)
        self.assertIn("conflicts", payload)

    # Verify that the template generator does not pull in unrelated content outside selected evidence.
    def test_answer_generator_stays_inside_selected_evidence(self) -> None:
        result = answer_query(self.run_dir, "What have I said about burnout?", limit=8, max_evidence=5)

        selected_excerpts = {item.citation.excerpt for item in result.evidence_used}
        self.assertIn("Burnout was getting worse in late 2025.", selected_excerpts)
        self.assertNotIn("manageable", result.answer)
        self.assertNotIn("marathon", result.answer)

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
    ) -> dict[str, object]:
        return {
            "message_id": message_id,
            "provider": "chatgpt",
            "conversation_id": "chatgpt:conversation:conv-mecky",
            "author_role": author_role,
            "created_at": created_at,
            "text": text,
        }


if __name__ == "__main__":
    unittest.main()
