import unittest
from pathlib import Path

from src.rag.normalize.chatgpt import (
    discover_chatgpt_conversation_shards,
    extract_chatgpt_records,
)


class ChatGPTExtractionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture_root = Path("tests/fixtures/chatgpt/sample_export")

    def test_discovers_sorted_shards(self) -> None:
        shard_paths = discover_chatgpt_conversation_shards(self.fixture_root)

        self.assertEqual(
            [path.name for path in shard_paths],
            ["conversations-000.json", "conversations-001.json"],
        )

    def test_extracts_conversations_across_shards(self) -> None:
        conversations, messages = extract_chatgpt_records(self.fixture_root)

        self.assertEqual(len(conversations), 2)
        self.assertEqual(len(messages), 7)
        self.assertEqual(conversations[0].conversation_id, "chatgpt:conversation:conv-1")
        self.assertEqual(conversations[1].conversation_id, "chatgpt:conversation:conv-2")
        self.assertEqual(conversations[0].source_artifact.conversation_file, "conversations-000.json")
        self.assertEqual(conversations[1].source_artifact.conversation_file, "conversations-001.json")

    def test_linearizes_current_node_chain_deterministically(self) -> None:
        conversations, messages = extract_chatgpt_records(self.fixture_root)

        first_conversation_messages = [
            message
            for message in messages
            if message.conversation_id == conversations[0].conversation_id
        ]
        self.assertEqual(
            [message.source_message_id for message in first_conversation_messages],
            [
                "msg-system-1",
                "msg-user-1",
                "msg-assistant-1",
                "msg-user-2",
                "msg-assistant-2",
            ],
        )
        self.assertEqual(
            [message.sequence_index for message in first_conversation_messages],
            [0, 1, 2, 3, 4],
        )

    def test_preserves_parent_message_ids_for_visible_chain(self) -> None:
        conversations, messages = extract_chatgpt_records(self.fixture_root)
        first_conversation_messages = [
            message
            for message in messages
            if message.conversation_id == conversations[0].conversation_id
        ]

        self.assertIsNone(first_conversation_messages[0].parent_message_id)
        self.assertEqual(
            first_conversation_messages[1].parent_message_id,
            first_conversation_messages[0].message_id,
        )
        self.assertEqual(
            first_conversation_messages[4].parent_message_id,
            first_conversation_messages[3].message_id,
        )

    def test_maps_content_parts_and_derives_text(self) -> None:
        _, messages = extract_chatgpt_records(self.fixture_root)

        assistant_message = next(
            message for message in messages if message.source_message_id == "msg-assistant-1"
        )
        self.assertEqual(assistant_message.content_blocks[0].type, "text")
        self.assertEqual(assistant_message.content_blocks[0].text, "Here is the generated image.")
        self.assertEqual(assistant_message.content_blocks[1].type, "image_asset_pointer")
        self.assertIsNone(assistant_message.content_blocks[1].text)
        self.assertEqual(assistant_message.text, "Here is the generated image.")

    def test_normalizes_timestamps_and_preserves_allowlisted_metadata(self) -> None:
        conversations, messages = extract_chatgpt_records(self.fixture_root)

        self.assertEqual(conversations[0].created_at, "2024-11-03T16:00:00Z")
        self.assertEqual(conversations[0].source_metadata.default_model_slug, "gpt-4o")
        self.assertEqual(conversations[0].source_metadata.is_starred, True)
        self.assertEqual(messages[0].created_at, "2024-11-03T16:00:00Z")
        self.assertEqual(messages[1].source_metadata.channel, "final")
        self.assertEqual(messages[1].source_metadata.content_type, "text")

    def test_preserves_attachment_references_without_copying_blobs(self) -> None:
        _, messages = extract_chatgpt_records(self.fixture_root)

        user_message = next(
            message for message in messages if message.source_message_id == "msg-user-1"
        )
        assistant_message = next(
            message for message in messages if message.source_message_id == "msg-assistant-1"
        )

        self.assertEqual(user_message.attachments[0].kind, "attachment")
        self.assertEqual(user_message.attachments[0].source_ref, "file-abc")
        self.assertEqual(user_message.attachments[0].path, "image.png")
        self.assertEqual(assistant_message.attachments[0].kind, "image_asset_pointer")
        self.assertEqual(assistant_message.attachments[0].source_ref, "file-service://file-inline-1")
        self.assertEqual(assistant_message.attachments[0].path, "file-inline-1")


if __name__ == "__main__":
    unittest.main()
