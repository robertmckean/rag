import unittest
from pathlib import Path

from rag.retrieval.read_model import load_normalized_run, tokenize_query


# These tests lock down the retrieval read model built over immutable normalized artifacts.
# The sample run fixture is intentionally small so ordering and lookup behavior stay obvious.
# Searchable text normalization is validated here because every ranker depends on it.
class RetrievalReadModelTests(unittest.TestCase):
    # Point tests at one small normalized run fixture with both providers represented.
    def setUp(self) -> None:
        self.run_dir = Path("tests/fixtures/retrieval/sample_run")

    # Verify that one run loads conversations, messages, manifest, and lookup structures correctly.
    def test_loads_run_and_builds_lookup_structures(self) -> None:
        loaded_run = load_normalized_run(self.run_dir)

        self.assertEqual(loaded_run.run_id, "sample-run")
        self.assertEqual(len(loaded_run.conversation_by_id), 2)
        self.assertEqual(len(loaded_run.message_by_id), 7)
        self.assertIn("chatgpt:conversation:conv-resume", loaded_run.messages_by_conversation_id)
        self.assertEqual(
            [message["sequence_index"] for message in loaded_run.messages_by_conversation_id["chatgpt:conversation:conv-resume"]],
            [0, 1, 2, 3],
        )

    # Verify that searchable text is built from canonical message content and normalized lexically.
    def test_builds_normalized_searchable_text(self) -> None:
        loaded_run = load_normalized_run(self.run_dir)

        self.assertEqual(
            loaded_run.searchable_text_by_message_id["chatgpt:message:msg-assistant-1"],
            "we can rewrite your resume summary and leadership bullets",
        )
        self.assertEqual(tokenize_query("Resume summary!"), ("resume", "summary"))


if __name__ == "__main__":
    unittest.main()
