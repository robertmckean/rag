import json
import shutil
import unittest
from pathlib import Path

from src.rag.normalize.combined_run import write_combined_normalized_run


class CombinedRunIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.chatgpt_input = Path("tests/fixtures/chatgpt/sample_export")
        self.claude_input = Path("tests/fixtures/claude/sample_bundle/conversations.json")
        self.output_root = Path("tests/_tmp_combined_runs")
        self.run_id = "fixture-combined-run"
        if self.output_root.exists():
            shutil.rmtree(self.output_root, ignore_errors=True)

    def tearDown(self) -> None:
        if self.output_root.exists():
            shutil.rmtree(self.output_root, ignore_errors=True)

    def test_writes_combined_outputs_and_preserves_provider_provenance(self) -> None:
        run_dir = write_combined_normalized_run(
            chatgpt_export_root=self.chatgpt_input,
            claude_conversations_path=self.claude_input,
            output_root=self.output_root,
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

        self.assertEqual(len(conversation_lines), 3)
        self.assertEqual(len(message_lines), 9)
        self.assertEqual(manifest["conversation_count"], 3)
        self.assertEqual(manifest["message_count"], 9)
        self.assertEqual(manifest["providers_present"], ["chatgpt", "claude"])
        self.assertEqual(
            manifest["counts_by_provider"],
            {
                "chatgpt": {"conversation_count": 2, "message_count": 7},
                "claude": {"conversation_count": 1, "message_count": 2},
            },
        )
        self.assertIn("ChatGPT includes only visible current_node chains.", manifest["notes"])
        self.assertIn(
            "Claude excludes sidecar streams: memories.json, projects.json, users.json.",
            manifest["notes"],
        )
        self.assertEqual(
            manifest["provider_policies"]["shared"]["source_metadata_policy"],
            "minimal_allowlist",
        )

        conversations = [json.loads(line) for line in conversation_lines]
        messages = [json.loads(line) for line in message_lines]

        providers_in_conversations = {record["provider"] for record in conversations}
        providers_in_messages = {record["provider"] for record in messages}
        self.assertEqual(providers_in_conversations, {"chatgpt", "claude"})
        self.assertEqual(providers_in_messages, {"chatgpt", "claude"})

        chatgpt_conversation = next(
            record for record in conversations if record["provider"] == "chatgpt"
        )
        claude_conversation = next(
            record for record in conversations if record["provider"] == "claude"
        )
        chatgpt_message = next(
            record for record in messages if record["provider"] == "chatgpt"
        )
        claude_message = next(
            record for record in messages if record["provider"] == "claude"
        )

        self.assertEqual(chatgpt_conversation["conversation_id"], "chatgpt:conversation:conv-1")
        self.assertEqual(chatgpt_conversation["source_metadata"]["default_model_slug"], "gpt-4o")
        self.assertEqual(claude_conversation["conversation_id"], "claude:conversation:conv-1")
        self.assertEqual(claude_conversation["source_metadata"]["account_uuid"], "account-1")
        self.assertEqual(chatgpt_message["message_id"], "chatgpt:message:msg-system-1")
        self.assertEqual(chatgpt_message["provider"], "chatgpt")
        self.assertEqual(claude_message["message_id"], "claude:message:msg-1")
        self.assertEqual(claude_message["provider"], "claude")

    def test_output_is_deterministic_for_same_input(self) -> None:
        first_dir = write_combined_normalized_run(
            chatgpt_export_root=self.chatgpt_input,
            claude_conversations_path=self.claude_input,
            output_root=self.output_root,
            run_id=self.run_id,
            created_at="2026-03-20T12:00:00Z",
        )
        first_conversations = (first_dir / "conversations.jsonl").read_text(encoding="utf-8")
        first_messages = (first_dir / "messages.jsonl").read_text(encoding="utf-8")
        first_manifest = (first_dir / "manifest.json").read_text(encoding="utf-8")

        shutil.rmtree(self.output_root, ignore_errors=True)

        second_dir = write_combined_normalized_run(
            chatgpt_export_root=self.chatgpt_input,
            claude_conversations_path=self.claude_input,
            output_root=self.output_root,
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
