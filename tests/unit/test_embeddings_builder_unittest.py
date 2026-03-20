import shutil
import unittest
from pathlib import Path

import json
from unittest.mock import patch

from rag.embeddings.builder import build_run_embeddings, prepare_text_for_embedding
from rag.embeddings.store import append_embedding_records, load_embedding_records


class _FakeEmbeddingClient:
    # Return deterministic vectors based on simple text signatures for artifact-generation tests.
    def embed_texts(self, texts: list[str], *, model: str) -> list[list[float]]:
        return [[float(index + 1), float(len(text.split()))] for index, text in enumerate(texts)]


class _RecordingEmbeddingClient:
    # Keep the exact texts passed to the embedding client so truncation can be asserted directly.
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed_texts(self, texts: list[str], *, model: str) -> list[list[float]]:
        self.calls.append(list(texts))
        return [[1.0, 1.0] for _ in texts]


class _StreamingObservationClient:
    # Observe how many records are already on disk each time a batch is sent to the embedding client.
    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.observed_counts: list[int] = []

    def embed_texts(self, texts: list[str], *, model: str) -> list[list[float]]:
        artifact_path = self.run_dir / "message_embeddings.jsonl"
        self.observed_counts.append(
            0 if not artifact_path.exists() else len([line for line in artifact_path.read_text(encoding="utf-8").splitlines() if line.strip()])
        )
        return [[1.0, 1.0] for _ in texts]


class EmbeddingBuilderTests(unittest.TestCase):
    # Point artifact-generation tests at the retrieval fixture run and isolate writes in a temp copy.
    def setUp(self) -> None:
        self.source_run_dir = Path("tests/fixtures/retrieval/sample_run")
        self.tmp_root = Path("tests/_tmp_embeddings_builder")
        if self.tmp_root.exists():
            shutil.rmtree(self.tmp_root, ignore_errors=True)
        self.run_dir = self.tmp_root / "sample_run"
        shutil.copytree(self.source_run_dir, self.run_dir)

    # Remove temporary run copies and generated embedding artifacts.
    def tearDown(self) -> None:
        if self.tmp_root.exists():
            shutil.rmtree(self.tmp_root, ignore_errors=True)

    # Verify that embedding generation writes one deterministic record per non-empty message.
    def test_builds_message_embedding_artifact(self) -> None:
        result = build_run_embeddings(
            self.run_dir,
            model="test-embedding",
            batch_size=2,
            embedding_client=_FakeEmbeddingClient(),
        )

        records = load_embedding_records(self.run_dir)
        self.assertEqual(result.record_count, 6)
        self.assertEqual(result.skipped_empty_count, 1)
        self.assertEqual(len(records), 6)
        self.assertTrue(result.artifact_path.exists())
        self.assertEqual(records[0].embedding_model, "test-embedding")
        self.assertEqual(records[0].embedding_dimensions, 2)
        self.assertEqual(records[0].message_id, "chatgpt:message:msg-user-1")
        self.assertEqual(records[0].text, "need help with my resume bullets")
        self.assertFalse(records[0].truncation_occurred)
        self.assertEqual(records[0].original_text_length, len(records[0].text))
        self.assertEqual(records[0].stored_text_length, len(records[0].text))
        self.assertEqual(records[-1].message_id, "claude:message:msg-3")

    # Verify that embedding batches are appended incrementally and the artifact exists during the run.
    def test_builds_embedding_artifact_in_streaming_batches(self) -> None:
        observing_client = _StreamingObservationClient(self.run_dir)

        result = build_run_embeddings(
            self.run_dir,
            model="test-embedding",
            batch_size=2,
            embedding_client=observing_client,
            progress_every_batches=1,
        )

        self.assertEqual(result.record_count, 6)
        self.assertEqual(observing_client.observed_counts, [0, 2, 4])
        self.assertEqual(len(load_embedding_records(self.run_dir)), 6)

    # Verify that append retries after a transient Windows-style file lock and still writes one batch once.
    def test_append_embedding_records_retries_transient_permission_error(self) -> None:
        build_run_embeddings(
            self.run_dir,
            model="test-embedding",
            batch_size=2,
            limit=1,
            embedding_client=_FakeEmbeddingClient(),
        )
        existing_record = load_embedding_records(self.run_dir)[0]
        original_open = Path.open
        open_calls = {"count": 0}

        def flaky_open(path_obj: Path, *args, **kwargs):
            mode = args[0] if args else kwargs.get("mode", "r")
            if path_obj == self.run_dir / "message_embeddings.jsonl" and mode == "ab":
                open_calls["count"] += 1
                if open_calls["count"] == 1:
                    raise PermissionError("transient file lock")
            return original_open(path_obj, *args, **kwargs)

        with patch("pathlib.Path.open", new=flaky_open):
            append_embedding_records(self.run_dir, (existing_record,))

        records = load_embedding_records(self.run_dir)
        self.assertEqual(open_calls["count"], 2)
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0].message_id, records[1].message_id)

    # Verify that short message text stays unchanged during embedding preprocessing.
    def test_prepare_text_for_embedding_leaves_short_text_unchanged(self) -> None:
        prepared = prepare_text_for_embedding("short text", model="test-embedding", max_tokens=10)

        self.assertEqual(prepared.text, "short text")
        self.assertFalse(prepared.truncation_occurred)
        self.assertEqual(prepared.original_text_length, 10)
        self.assertEqual(prepared.stored_text_length, 10)

    # Verify that overlong message text is truncated before the embedding client sees it.
    def test_build_run_embeddings_truncates_overlong_text_before_api_call(self) -> None:
        long_text = "a" * 50
        self._append_message(
            message_id="chatgpt:message:msg-long",
            conversation_id="chatgpt:conversation:conv-long",
            text=long_text,
        )
        recording_client = _RecordingEmbeddingClient()

        build_run_embeddings(
            self.run_dir,
            model="test-embedding",
            batch_size=20,
            max_tokens=10,
            embedding_client=recording_client,
        )

        records = load_embedding_records(self.run_dir)
        long_record = next(record for record in records if record.message_id == "chatgpt:message:msg-long")
        self.assertTrue(long_record.truncation_occurred)
        self.assertEqual(long_record.original_text_length, 50)
        self.assertEqual(long_record.stored_text_length, 10)
        self.assertEqual(long_record.text, "a" * 10)
        self.assertIn("a" * 10, recording_client.calls[-1])
        self.assertNotIn("a" * 50, recording_client.calls[-1])

    # Verify that limit mode embeds only the requested leading subset in deterministic order.
    def test_build_run_embeddings_respects_limit(self) -> None:
        result = build_run_embeddings(
            self.run_dir,
            model="test-embedding",
            batch_size=10,
            limit=2,
            embedding_client=_FakeEmbeddingClient(),
        )

        records = load_embedding_records(self.run_dir)
        self.assertEqual(result.selected_message_count, 2)
        self.assertEqual(result.record_count, 2)
        self.assertEqual([record.message_id for record in records], [
            "chatgpt:message:msg-user-1",
            "chatgpt:message:msg-assistant-1",
        ])
        self.assertEqual(records[0].subset_limit, 2)

    # Verify that resume skips already-embedded message/model pairs and appends only missing records.
    def test_build_run_embeddings_resumes_without_duplicates(self) -> None:
        initial = build_run_embeddings(
            self.run_dir,
            model="test-embedding",
            batch_size=2,
            limit=2,
            embedding_client=_FakeEmbeddingClient(),
        )
        resumed = build_run_embeddings(
            self.run_dir,
            model="test-embedding",
            batch_size=10,
            embedding_client=_FakeEmbeddingClient(),
        )

        records = load_embedding_records(self.run_dir)
        self.assertEqual(initial.record_count, 2)
        self.assertEqual(resumed.resumed_skip_count, 2)
        self.assertEqual(len(records), 6)
        self.assertEqual(len({record.message_id for record in records}), 6)

    # Verify that deterministic sampling with the same seed selects the same messages in the same order.
    def test_build_run_embeddings_sampling_is_deterministic(self) -> None:
        first_run_dir = self.tmp_root / "sample_run_first"
        second_run_dir = self.tmp_root / "sample_run_second"
        shutil.copytree(self.source_run_dir, first_run_dir)
        shutil.copytree(self.source_run_dir, second_run_dir)

        build_run_embeddings(
            first_run_dir,
            model="test-embedding",
            batch_size=10,
            sample=3,
            sample_seed=17,
            embedding_client=_FakeEmbeddingClient(),
        )
        build_run_embeddings(
            second_run_dir,
            model="test-embedding",
            batch_size=10,
            sample=3,
            sample_seed=17,
            embedding_client=_FakeEmbeddingClient(),
        )

        first_ids = [record.message_id for record in load_embedding_records(first_run_dir)]
        second_ids = [record.message_id for record in load_embedding_records(second_run_dir)]
        self.assertEqual(first_ids, second_ids)

    # Verify that tool-role and low-information acknowledgment messages are excluded from embedding generation.
    def test_embedding_generation_filters_tool_and_low_information_messages(self) -> None:
        self._append_message(
            message_id="chatgpt:message:msg-tool",
            conversation_id="chatgpt:conversation:conv-tool",
            text="Tool output should not be embedded.",
            author_role="tool",
        )
        self._append_message(
            message_id="chatgpt:message:msg-ack",
            conversation_id="chatgpt:conversation:conv-tool",
            text="Okay.",
        )
        self._append_message(
            message_id="chatgpt:message:msg-meaningful-short",
            conversation_id="chatgpt:conversation:conv-tool",
            text="Bass riff",
        )

        result = build_run_embeddings(
            self.run_dir,
            model="test-embedding",
            batch_size=20,
            embedding_client=_FakeEmbeddingClient(),
        )

        records = load_embedding_records(self.run_dir)
        message_ids = {record.message_id for record in records}
        self.assertNotIn("chatgpt:message:msg-tool", message_ids)
        self.assertNotIn("chatgpt:message:msg-ack", message_ids)
        self.assertIn("chatgpt:message:msg-meaningful-short", message_ids)
        self.assertEqual(result.filtered_tool_role_count, 1)
        self.assertEqual(result.filtered_low_information_count, 1)
        self.assertEqual(result.filtered_trivial_short_count, 0)

    # Verify that targeted conversation builds embed only messages from the requested conversation.
    def test_build_run_embeddings_can_target_one_conversation(self) -> None:
        result = build_run_embeddings(
            self.run_dir,
            model="test-embedding",
            batch_size=10,
            conversation_ids=("claude:conversation:conv-burnout",),
            embedding_client=_FakeEmbeddingClient(),
        )

        records = load_embedding_records(self.run_dir)
        self.assertEqual(result.targeted_match_count, 3)
        self.assertEqual(result.selected_message_count, 3)
        self.assertEqual({record.conversation_id for record in records}, {"claude:conversation:conv-burnout"})
        self.assertEqual(result.targeted_conversation_count, 1)

    # Verify that repeated targeted conversation builds resume safely without duplicate records.
    def test_targeted_conversation_build_resumes_without_duplicates(self) -> None:
        first = build_run_embeddings(
            self.run_dir,
            model="test-embedding",
            batch_size=10,
            conversation_ids=("claude:conversation:conv-burnout",),
            embedding_client=_FakeEmbeddingClient(),
        )
        second = build_run_embeddings(
            self.run_dir,
            model="test-embedding",
            batch_size=10,
            conversation_ids=("claude:conversation:conv-burnout",),
            embedding_client=_FakeEmbeddingClient(),
        )

        records = load_embedding_records(self.run_dir)
        self.assertEqual(first.record_count, 3)
        self.assertEqual(second.record_count, 0)
        self.assertEqual(second.resumed_skip_count, 3)
        self.assertEqual(len(records), 3)
        self.assertEqual(len({record.message_id for record in records}), 3)

    # Append one standalone user message for builder mutation tests.
    def _append_message(self, *, message_id: str, conversation_id: str, text: str, author_role: str = "user") -> None:
        conversation = {
            "conversation_id": conversation_id,
            "provider": "chatgpt",
            "source_conversation_id": conversation_id.rsplit(":", 1)[-1],
            "title": "Embedding Builder Fixture",
            "summary": None,
            "created_at": "2025-03-01T12:00:00Z",
            "updated_at": "2025-03-01T12:00:00Z",
            "message_count": 1,
            "participant_summary": {"roles": ["user"], "authors": []},
            "source_artifact": {"root": "History_ChatGPT", "conversation_file": "conversations-long.json", "sidecar_files": []},
            "source_metadata": {},
        }
        message = {
            "attachments": [],
            "author_name": None,
            "author_role": author_role,
            "content_blocks": [{"citations": None, "start_timestamp": None, "stop_timestamp": None, "text": text, "type": "text"}],
            "conversation_id": conversation_id,
            "created_at": "2025-03-01T12:00:00Z",
            "message_id": message_id,
            "parent_message_id": None,
            "provider": "chatgpt",
            "sender": None,
            "sequence_index": 0,
            "source_artifact": {"conversation_file": "conversations-long.json", "raw_message_path": f"mapping.{message_id}.message"},
            "source_message_id": message_id.rsplit(":", 1)[-1],
            "source_metadata": {"channel": "final", "content_type": "text", "flags": None, "status": "finished_successfully"},
            "text": text,
            "updated_at": None,
        }
        (self.run_dir / "conversations.jsonl").write_text(
            (self.run_dir / "conversations.jsonl").read_text(encoding="utf-8") + json.dumps(conversation, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
        (self.run_dir / "messages.jsonl").write_text(
            (self.run_dir / "messages.jsonl").read_text(encoding="utf-8") + json.dumps(message, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    unittest.main()
