import json
import shutil
import unittest
from pathlib import Path

from rag.embeddings.store import EmbeddingRecord, write_embedding_records
from rag.retrieval.lexical import retrieve_message_windows


class _FakeQueryEmbeddingClient:
    # Return deterministic query vectors so semantic retrieval can be tested without live API calls.
    def embed_texts(self, texts: list[str], *, model: str) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            if text == "larry guitar playing":
                vectors.append([1.0, 0.0])
            elif text == "resume leadership":
                vectors.append([0.0, 1.0])
            else:
                vectors.append([0.1, 0.1])
        return vectors


class SemanticRetrievalTests(unittest.TestCase):
    # Copy the retrieval fixture run so semantic/hybrid tests can append messages and embedding artifacts safely.
    def setUp(self) -> None:
        self.source_run_dir = Path("tests/fixtures/retrieval/sample_run")
        self.tmp_root = Path("tests/_tmp_retrieval_semantic")
        if self.tmp_root.exists():
            shutil.rmtree(self.tmp_root, ignore_errors=True)
        self.run_dir = self.tmp_root / "sample_run"
        shutil.copytree(self.source_run_dir, self.run_dir)
        self._append_music_message()
        self._write_test_embeddings()

    # Remove temporary run copies and generated semantic artifacts.
    def tearDown(self) -> None:
        if self.tmp_root.exists():
            shutil.rmtree(self.tmp_root, ignore_errors=True)

    # Verify that semantic retrieval can surface a bass message for a guitar-playing query through embeddings.
    def test_semantic_retrieval_returns_expected_message(self) -> None:
        results = retrieve_message_windows(
            self.run_dir,
            "Larry guitar playing",
            channel="semantic",
            embedding_model="test-embedding",
            embedding_client=_FakeQueryEmbeddingClient(),
            limit=5,
        )

        self.assertEqual(results[0].focal_message_id, "claude:message:msg-music-1")
        self.assertEqual(results[0].match_basis["retrieval_channel"], "semantic")
        self.assertEqual(results[0].match_basis["retrieval_sources"], ["semantic"])
        self.assertIsNone(results[0].match_basis["scoring_features"]["bm25_score"])
        self.assertGreater(results[0].match_basis["scoring_features"]["semantic_similarity"], 0.99)

    # Verify that hybrid retrieval deduplicates overlap and preserves provenance from both channels.
    def test_hybrid_retrieval_merges_and_dedupes_overlap(self) -> None:
        results = retrieve_message_windows(
            self.run_dir,
            "resume leadership",
            channel="hybrid",
            embedding_model="test-embedding",
            embedding_client=_FakeQueryEmbeddingClient(),
            limit=5,
        )

        result_by_message_id = {result.focal_message_id: result for result in results}
        self.assertIn("chatgpt:message:msg-assistant-1", result_by_message_id)
        assistant_result = result_by_message_id["chatgpt:message:msg-assistant-1"]
        self.assertEqual(assistant_result.match_basis["retrieval_sources"], ["bm25", "semantic"])
        self.assertIsNotNone(assistant_result.match_basis["scoring_features"]["bm25_score"])
        self.assertIsNotNone(assistant_result.match_basis["scoring_features"]["semantic_similarity"])
        self.assertEqual(
            len([result for result in results if result.focal_message_id == "chatgpt:message:msg-assistant-1"]),
            1,
        )

    # Append one music-related message so semantic retrieval can cover a vocabulary-mismatch case.
    def _append_music_message(self) -> None:
        conversation = {
            "conversation_id": "claude:conversation:conv-music",
            "provider": "claude",
            "source_conversation_id": "conv-music",
            "title": "Music Notes",
            "summary": None,
            "created_at": "2025-03-16T12:00:00Z",
            "updated_at": "2025-03-16T12:00:00Z",
            "message_count": 1,
            "participant_summary": {"roles": ["user"], "authors": []},
            "source_artifact": {"root": "History_Claude", "conversation_file": "conversations.json", "sidecar_files": []},
            "source_metadata": {},
        }
        message = {
            "message_id": "claude:message:msg-music-1",
            "conversation_id": "claude:conversation:conv-music",
            "provider": "claude",
            "source_message_id": "msg-music-1",
            "parent_message_id": None,
            "sequence_index": 0,
            "author_role": "user",
            "author_name": None,
            "sender": "human",
            "created_at": "2025-03-16T12:00:00Z",
            "updated_at": None,
            "text": "Larry was playing the bass in the garage.",
            "content_blocks": [
                {"type": "text", "text": "Larry was playing the bass in the garage.", "start_timestamp": None, "stop_timestamp": None, "citations": None}
            ],
            "attachments": [],
            "source_artifact": {"conversation_file": "conversations.json", "raw_message_path": "chat_messages[99]"},
            "source_metadata": {},
        }
        (self.run_dir / "conversations.jsonl").write_text(
            (self.run_dir / "conversations.jsonl").read_text(encoding="utf-8") + json.dumps(conversation, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
        (self.run_dir / "messages.jsonl").write_text(
            (self.run_dir / "messages.jsonl").read_text(encoding="utf-8") + json.dumps(message, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )

    # Write a deterministic embeddings artifact for the copied retrieval run.
    def _write_test_embeddings(self) -> None:
        records = (
            EmbeddingRecord(
                run_id="sample-run",
                message_id="chatgpt:message:msg-user-1",
                conversation_id="chatgpt:conversation:conv-resume",
                provider="chatgpt",
                author_role="user",
                created_at="2025-03-01T09:01:00Z",
                sequence_index=1,
                embedding_model="test-embedding",
                embedding_dimensions=2,
                text="need help with my resume bullets",
                original_text_length=31,
                stored_text_length=31,
                truncation_occurred=False,
                original_token_count=6,
                stored_token_count=6,
                embedding=(0.0, 0.6),
            ),
            EmbeddingRecord(
                run_id="sample-run",
                message_id="chatgpt:message:msg-assistant-1",
                conversation_id="chatgpt:conversation:conv-resume",
                provider="chatgpt",
                author_role="assistant",
                created_at="2025-03-01T09:02:00Z",
                sequence_index=2,
                embedding_model="test-embedding",
                embedding_dimensions=2,
                text="we can rewrite your resume summary and leadership bullets",
                original_text_length=56,
                stored_text_length=56,
                truncation_occurred=False,
                original_token_count=9,
                stored_token_count=9,
                embedding=(0.0, 1.0),
            ),
            EmbeddingRecord(
                run_id="sample-run",
                message_id="chatgpt:message:msg-user-2",
                conversation_id="chatgpt:conversation:conv-resume",
                provider="chatgpt",
                author_role="user",
                created_at="2025-03-01T09:03:00Z",
                sequence_index=3,
                embedding_model="test-embedding",
                embedding_dimensions=2,
                text="focus on project leadership examples",
                original_text_length=36,
                stored_text_length=36,
                truncation_occurred=False,
                original_token_count=5,
                stored_token_count=5,
                embedding=(0.0, 0.9),
            ),
            EmbeddingRecord(
                run_id="sample-run",
                message_id="claude:message:msg-music-1",
                conversation_id="claude:conversation:conv-music",
                provider="claude",
                author_role="user",
                created_at="2025-03-16T12:00:00Z",
                sequence_index=0,
                embedding_model="test-embedding",
                embedding_dimensions=2,
                text="larry was playing the bass in the garage",
                original_text_length=41,
                stored_text_length=41,
                truncation_occurred=False,
                original_token_count=8,
                stored_token_count=8,
                embedding=(1.0, 0.0),
            ),
        )
        write_embedding_records(self.run_dir, records)


if __name__ == "__main__":
    unittest.main()
