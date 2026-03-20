import json
import shutil
import unittest
from pathlib import Path

from src.rag.normalize.chatgpt_run import write_chatgpt_normalized_run


# These integration tests validate the ChatGPT-only run writer end to end.
# The fixture bundle exercises shard handling, manifest notes, and canonical JSONL output.
# Deterministic output matters because graph linearization policy must produce stable reruns.
class ChatGPTRunIntegrationTests(unittest.TestCase):
    # Set up an isolated output directory so each test owns its generated artifacts.
    def setUp(self) -> None:
        self.fixture_input = Path("tests/fixtures/chatgpt/sample_export")
        self.output_root = Path("tests/_tmp_chatgpt_runs")
        self.run_id = "fixture-chatgpt-run"
        if self.output_root.exists():
            shutil.rmtree(self.output_root, ignore_errors=True)

    # Clean up the temporary output directory after each run-writer test.
    def tearDown(self) -> None:
        if self.output_root.exists():
            shutil.rmtree(self.output_root, ignore_errors=True)

    # Verify that the ChatGPT run writer emits expected files, counts, notes, and golden records.
    def test_writes_expected_files_and_golden_records(self) -> None:
        run_dir = write_chatgpt_normalized_run(
            self.fixture_input,
            self.output_root,
            run_id=self.run_id,
            created_at="2026-03-20T12:00:00Z",
        )

        conversations_path = run_dir / "conversations.jsonl"
        messages_path = run_dir / "messages.jsonl"
        manifest_path = run_dir / "manifest.json"

        self.assertTrue(conversations_path.exists())
        self.assertTrue(messages_path.exists())
        self.assertTrue(manifest_path.exists())

        conversation_lines = conversations_path.read_text(encoding="utf-8").splitlines()
        message_lines = messages_path.read_text(encoding="utf-8").splitlines()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        self.assertEqual(len(conversation_lines), 2)
        self.assertEqual(len(message_lines), 7)
        self.assertEqual(manifest["conversation_count"], 2)
        self.assertEqual(manifest["message_count"], 7)
        self.assertEqual(manifest["providers_present"], ["chatgpt"])
        self.assertIn("Only visible current_node chains are included.", manifest["notes"])
        self.assertIn("Branch nodes not on the visible chain are excluded.", manifest["notes"])
        self.assertIn("message=null nodes are excluded from canonical messages.", manifest["notes"])
        self.assertIn("current_node ancestor chain", manifest["graph_linearization_policy"])

        first_conversation = json.loads(conversation_lines[0])
        first_message = json.loads(message_lines[0])

        self.assertEqual(first_conversation["conversation_id"], "chatgpt:conversation:conv-1")
        self.assertEqual(first_conversation["source_metadata"]["default_model_slug"], "gpt-4o")
        self.assertEqual(first_message["message_id"], "chatgpt:message:msg-system-1")
        self.assertEqual(first_message["parent_message_id"], None)
        self.assertEqual(first_message["content_blocks"][0]["type"], "text")

    # Verify that rerunning the same input with the same run metadata is byte-stable.
    def test_output_is_deterministic_for_same_input(self) -> None:
        first_dir = write_chatgpt_normalized_run(
            self.fixture_input,
            self.output_root,
            run_id=self.run_id,
            created_at="2026-03-20T12:00:00Z",
        )
        first_conversations = (first_dir / "conversations.jsonl").read_text(encoding="utf-8")
        first_messages = (first_dir / "messages.jsonl").read_text(encoding="utf-8")
        first_manifest = (first_dir / "manifest.json").read_text(encoding="utf-8")

        shutil.rmtree(self.output_root, ignore_errors=True)

        second_dir = write_chatgpt_normalized_run(
            self.fixture_input,
            self.output_root,
            run_id=self.run_id,
            created_at="2026-03-20T12:00:00Z",
        )
        second_conversations = (second_dir / "conversations.jsonl").read_text(encoding="utf-8")
        second_messages = (second_dir / "messages.jsonl").read_text(encoding="utf-8")
        second_manifest = (second_dir / "manifest.json").read_text(encoding="utf-8")

        self.assertEqual(first_conversations, second_conversations)
        self.assertEqual(first_messages, second_messages)
        self.assertEqual(first_manifest, second_manifest)


if __name__ == "__main__":
    unittest.main()
