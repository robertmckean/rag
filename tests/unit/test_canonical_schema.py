from src.rag.normalize.canonical_schema import (
    AttachmentReference,
    CanonicalConversation,
    CanonicalMessage,
    ContentBlock,
    ConversationSourceMetadata,
    MessageSourceArtifact,
    MessageSourceMetadata,
    ParticipantSummary,
    SourceArtifact,
    derive_text_from_blocks,
)


# These tests protect the canonical schema shape that JSONL writers serialize.
# The focus is on authoritative content blocks, derived text, and allowlisted metadata.
# Keeping the assertions narrow helps catch accidental schema drift quickly.

# Verify that only text blocks contribute to the derived plain-text field.
def test_derive_text_from_text_blocks_only() -> None:
    blocks = (
        ContentBlock(type="text", text="first"),
        ContentBlock(type="image", text="ignored"),
        ContentBlock(type="text", text="second"),
    )
    assert derive_text_from_blocks(blocks) == "first\n\nsecond"


# Verify that conversation metadata serializes with the expected allowlisted fields.
def test_conversation_to_dict_preserves_allowlisted_metadata_shape() -> None:
    conversation = CanonicalConversation(
        conversation_id="chatgpt:conversation:abc",
        provider="chatgpt",
        source_conversation_id="abc",
        title="Example",
        summary=None,
        created_at="2025-01-01T00:00:00Z",
        updated_at=None,
        message_count=2,
        participant_summary=ParticipantSummary(roles=("user", "assistant"), authors=()),
        source_artifact=SourceArtifact(
            root="chatgpt/History_ChatGPT",
            conversation_file="conversations-000.json",
            sidecar_files=("export_manifest.json",),
        ),
        source_metadata=ConversationSourceMetadata(
            conversation_origin="chat",
            is_archived=False,
            is_study_mode=False,
        ),
    )

    payload = conversation.to_dict()
    assert payload["source_metadata"]["conversation_origin"] == "chat"
    assert "account_uuid" in payload["source_metadata"]


# Verify that message serialization preserves nested content blocks and attachment references.
def test_message_to_dict_preserves_content_blocks_and_attachments() -> None:
    message = CanonicalMessage(
        message_id="claude:message:msg-1",
        conversation_id="claude:conversation:conv-1",
        provider="claude",
        source_message_id="msg-1",
        parent_message_id=None,
        sequence_index=0,
        sender="human",
        created_at="2025-02-27T13:53:05.581841Z",
        updated_at=None,
        text="hello",
        content_blocks=(ContentBlock(type="text", text="hello"),),
        attachments=(AttachmentReference(kind="file", path=None, source_ref="file-1"),),
        source_artifact=MessageSourceArtifact(
            conversation_file="History_Claude/conversations.json",
            raw_message_path="chat_messages[0]",
        ),
        source_metadata=MessageSourceMetadata(flags=()),
    )

    payload = message.to_dict()
    assert payload["content_blocks"][0]["text"] == "hello"
    assert payload["attachments"][0]["source_ref"] == "file-1"
