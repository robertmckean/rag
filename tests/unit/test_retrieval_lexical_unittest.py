import unittest
from pathlib import Path

from rag.retrieval.lexical import RetrievalFilters, retrieve_message_windows


# These tests cover the first message-first lexical retrieval slice and its result contract.
# The fixture run is designed to exercise ranking, window expansion, filtering, and tie-breaking.
# Assertions stay focused on observable behavior rather than internal helper implementation details.
class RetrievalLexicalTests(unittest.TestCase):
    # Point tests at the normalized run fixture used for all first-slice retrieval checks.
    def setUp(self) -> None:
        self.run_dir = Path("tests/fixtures/retrieval/sample_run")

    # Verify that lexical ranking returns contextual windows around the highest-scoring focal messages.
    def test_ranks_messages_and_returns_context_windows(self) -> None:
        results = retrieve_message_windows(self.run_dir, "resume", limit=2)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].provider, "chatgpt")
        self.assertEqual(results[0].conversation_id, "chatgpt:conversation:conv-resume")
        self.assertEqual(results[0].focal_message_id, "chatgpt:message:msg-user-1")
        self.assertEqual(results[0].window_start_sequence_index, 0)
        self.assertEqual(results[0].window_end_sequence_index, 3)
        self.assertEqual(
            [message["message_id"] for message in results[0].messages],
            [
                "chatgpt:message:msg-system-1",
                "chatgpt:message:msg-user-1",
                "chatgpt:message:msg-assistant-1",
                "chatgpt:message:msg-user-2",
            ],
        )

    # Verify that multiple high-scoring focal hits dedupe down to one identical conversation window.
    def test_dedupes_identical_windows(self) -> None:
        results = retrieve_message_windows(self.run_dir, "resume", limit=10)

        self.assertEqual(len(results), 1)

    # Verify that provider and date filters narrow the candidate pool before ranking.
    def test_applies_provider_and_date_filters(self) -> None:
        results = retrieve_message_windows(
            self.run_dir,
            "burnout",
            filters=RetrievalFilters(
                provider="claude",
                date_from="2025-03-10",
                date_to="2025-03-20",
            ),
        )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].provider, "claude")
        self.assertEqual(results[0].focal_message_id, "claude:message:msg-1")

    # Verify that author-role and conversation filters are honored.
    def test_applies_author_role_and_conversation_filters(self) -> None:
        results = retrieve_message_windows(
            self.run_dir,
            "leadership",
            filters=RetrievalFilters(
                conversation_id="chatgpt:conversation:conv-resume",
                author_role="assistant",
            ),
        )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].focal_message_id, "chatgpt:message:msg-assistant-1")

    # Verify that deterministic tie-breaking prefers the earlier timestamp when scores are equal.
    def test_uses_deterministic_tie_breaking(self) -> None:
        results = retrieve_message_windows(self.run_dir, "resume", limit=5)

        self.assertEqual(results[0].focal_message_id, "chatgpt:message:msg-user-1")
        self.assertEqual(results[0].match_basis["scoring_features"]["token_overlap"], 1)
        self.assertEqual(results[0].match_basis["scoring_features"]["title_overlap"], 1)


if __name__ == "__main__":
    unittest.main()
