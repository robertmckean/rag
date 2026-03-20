from src.rag.normalize.identifiers import make_conversation_id, make_message_id


def test_conversation_id_uses_source_id_when_present() -> None:
    assert make_conversation_id("chatgpt", "abc-123") == "chatgpt:conversation:abc-123"


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


def test_message_id_uses_source_id_when_present() -> None:
    assert (
        make_message_id("claude", "msg-1", conversation_id="claude:conversation:conv-1")
        == "claude:message:msg-1"
    )


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
