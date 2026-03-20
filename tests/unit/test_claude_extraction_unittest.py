import unittest
from pathlib import Path

from src.rag.normalize.claude import extract_claude_records


# These tests verify that Claude raw exports map cleanly into the canonical schema.
# Fixtures cover both the simple conversation file and a bundle with sidecar data present.
# The assertions stay close to phase-1 contract details such as role mapping and references.
class ClaudeExtractionTests(unittest.TestCase):
    # Point each test at the minimal fixtures needed to exercise extraction behavior.
    def setUp(self) -> None:
        self.fixture_path = Path("tests/fixtures/claude/sample_conversations.json")
        self.bundle_path = Path("tests/fixtures/claude/sample_bundle/conversations.json")

    # Verify that Claude conversations and messages are extracted with canonical ids and roles.
    def test_extracts_conversation_and_messages(self) -> None:
        conversations, messages = extract_claude_records(self.fixture_path)

        self.assertEqual(len(conversations), 1)
        self.assertEqual(len(messages), 2)
        self.assertEqual(conversations[0].conversation_id, "claude:conversation:conv-1")
        self.assertEqual(conversations[0].source_metadata.account_uuid, "account-1")
        self.assertEqual(conversations[0].message_count, 2)
        self.assertEqual(messages[0].author_role, "user")
        self.assertEqual(messages[1].author_role, "assistant")
        self.assertEqual(messages[0].sender, "human")
        self.assertEqual(messages[1].sender, "assistant")

    # Verify that content blocks remain authoritative and text is derived from those blocks.
    def test_message_content_blocks_are_authoritative_and_text_is_derived(self) -> None:
        _, messages = extract_claude_records(self.fixture_path)

        self.assertEqual(
            messages[0].text,
            "Hello from block one\n\nHello from block two",
        )
        self.assertEqual(messages[1].text, None)
        self.assertEqual(messages[0].content_blocks[0].type, "text")
        self.assertEqual(messages[1].content_blocks[1].type, "image")

    # Verify that conversation, message, and block timestamps normalize into canonical UTC strings.
    def test_timestamps_are_normalized_in_extracted_records(self) -> None:
        conversations, messages = extract_claude_records(self.fixture_path)

        self.assertEqual(conversations[0].created_at, "2025-02-27T13:53:04.968636Z")
        self.assertEqual(messages[0].created_at, "2025-02-27T13:53:05.581841Z")
        self.assertEqual(messages[0].content_blocks[0].start_timestamp, "2025-02-27T13:53:05.581841Z")

    # Verify that attachments and files are preserved as references without blob copying.
    def test_attachment_and_file_references_are_preserved_as_references_only(self) -> None:
        _, messages = extract_claude_records(self.bundle_path)

        refs = messages[0].attachments
        self.assertEqual(len(refs), 2)
        self.assertEqual(refs[0].kind, "attachment")
        self.assertEqual(refs[0].source_ref, "att-1")
        self.assertEqual(refs[0].path, None)
        self.assertEqual(refs[1].kind, "file")
        self.assertEqual(refs[1].source_ref, "file-1")
        self.assertEqual(messages[0].source_metadata.flags, ("visible",))

    # Verify that Claude sidecar files do not leak into canonical conversation or message streams.
    def test_sidecar_files_are_not_ingested_into_canonical_streams(self) -> None:
        conversations, messages = extract_claude_records(self.bundle_path)

        self.assertEqual(len(conversations), 1)
        self.assertEqual(len(messages), 2)
        self.assertEqual(conversations[0].source_artifact.conversation_file, "conversations.json")
        self.assertEqual(conversations[0].source_artifact.sidecar_files, ())


if __name__ == "__main__":
    unittest.main()
