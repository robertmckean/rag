from src.rag.normalize.identifiers import make_conversation_id, make_message_id


# These tests lock down the canonical id shape used across all normalized outputs.
# Fallback ids matter because reruns must stay deterministic even when source ids are missing.
# The module is intentionally narrow, so each test covers one public behavior directly.

# Verify that source conversation ids are preserved when they exist.
def test_conversation_id_uses_source_id_when_present() -> None:
    assert make_conversation_id("chatgpt", "abc-123") == "chatgpt:conversation:abc-123"


# Verify that missing conversation ids fall back to a deterministic file-and-ordinal key.
def test_conversation_id_falls_back_deterministically() -> None:
    assert (
        make_conversation_id(
            "claude",
            None,
            conversation_file="History_Claude/conversations.json",
            ordinal=2,
        )
        == "claude:conversation:missing:History_Claude/conversations.json:2"
    )


# Verify that source message ids are preserved when they exist.
def test_message_id_uses_source_id_when_present() -> None:
    assert (
        make_message_id("claude", "msg-1", conversation_id="claude:conversation:conv-1")
        == "claude:message:msg-1"
    )


# Verify that missing message ids fall back to a deterministic local key.
def test_message_id_falls_back_deterministically() -> None:
    assert (
        make_message_id(
            "chatgpt",
            None,
            conversation_id="chatgpt:conversation:conv-1",
            stable_local_key="node-7",
        )
        == "chatgpt:message:missing:chatgpt:conversation:conv-1:node-7"
    )
