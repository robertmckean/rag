import json
import shutil
import unittest
from pathlib import Path

from src.rag.normalize.claude_run import write_claude_normalized_run


class ClaudeRunIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture_input = Path("tests/fixtures/claude/sample_bundle/conversations.json")
        self.output_root = Path("tests/_tmp_runs")
        self.run_id = "fixture-run"
        if self.output_root.exists():
            shutil.rmtree(self.output_root, ignore_errors=True)

    def tearDown(self) -> None:
        if self.output_root.exists():
            shutil.rmtree(self.output_root, ignore_errors=True)

    def test_writes_expected_files_and_golden_records(self) -> None:
        run_dir = write_claude_normalized_run(
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

        self.assertEqual(len(conversation_lines), 1)
        self.assertEqual(len(message_lines), 2)
        self.assertEqual(manifest["conversation_count"], 1)
        self.assertEqual(manifest["message_count"], 2)
        self.assertEqual(manifest["providers_present"], ["claude"])

        first_conversation = json.loads(conversation_lines[0])
        first_message = json.loads(message_lines[0])

        self.assertEqual(first_conversation["conversation_id"], "claude:conversation:conv-1")
        self.assertEqual(first_conversation["source_metadata"]["account_uuid"], "account-1")
        self.assertEqual(first_message["message_id"], "claude:message:msg-1")
        self.assertEqual(first_message["text"], "Hello from block one\n\nHello from block two")
        self.assertEqual(first_message["attachments"][0]["source_ref"], "att-1")

    def test_output_is_deterministic_for_same_input(self) -> None:
        first_dir = write_claude_normalized_run(
            self.fixture_input,
            self.output_root,
            run_id=self.run_id,
            created_at="2026-03-20T12:00:00Z",
        )
        first_conversations = (first_dir / "conversations.jsonl").read_text(encoding="utf-8")
        first_messages = (first_dir / "messages.jsonl").read_text(encoding="utf-8")
        first_manifest = (first_dir / "manifest.json").read_text(encoding="utf-8")

        shutil.rmtree(self.output_root, ignore_errors=True)

        second_dir = write_claude_normalized_run(
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
