"""Claude-only extraction into the canonical phase-1 schema."""

from __future__ import annotations

import json
from pathlib import Path

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
from src.rag.normalize.identifiers import make_conversation_id, make_message_id
from src.rag.normalize.timestamps import normalize_timestamp


def extract_claude_records(
    conversations_path: Path,
) -> tuple[list[CanonicalConversation], list[CanonicalMessage]]:
    """Load Claude conversations and map them into canonical records."""
    payload = json.loads(conversations_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Claude conversations export must be a JSON array.")

    source_root = _derive_source_root(conversations_path)
    source_file = str(conversations_path.relative_to(source_root))

    conversations: list[CanonicalConversation] = []
    messages: list[CanonicalMessage] = []

    for conversation_ordinal, raw_conversation in enumerate(payload):
        if not isinstance(raw_conversation, dict):
            continue

        source_conversation_id = _string_or_none(raw_conversation.get("uuid"))
        conversation_id = make_conversation_id(
            "claude",
            source_conversation_id,
            conversation_file=source_file,
            ordinal=conversation_ordinal,
        )

        raw_messages = raw_conversation.get("chat_messages")
        if not isinstance(raw_messages, list):
            raw_messages = []

        participant_roles = tuple(
            sorted(
                {
                    sender
                    for sender in (_string_or_none(item.get("sender")) for item in raw_messages if isinstance(item, dict))
                    if sender
                }
            )
        )

        conversations.append(
            CanonicalConversation(
                conversation_id=conversation_id,
                provider="claude",
                source_conversation_id=source_conversation_id or "",
                title=_blank_to_none(raw_conversation.get("name")),
                summary=_blank_to_none(raw_conversation.get("summary")),
                created_at=normalize_timestamp(raw_conversation.get("created_at")),
                updated_at=normalize_timestamp(raw_conversation.get("updated_at")),
                message_count=len(raw_messages),
                participant_summary=ParticipantSummary(roles=participant_roles, authors=()),
                source_artifact=SourceArtifact(
                    root=str(source_root.name),
                    conversation_file=source_file,
                    sidecar_files=(),
                ),
                source_metadata=ConversationSourceMetadata(
                    account_uuid=_extract_account_uuid(raw_conversation.get("account")),
                ),
            )
        )

        for sequence_index, raw_message in enumerate(raw_messages):
            if not isinstance(raw_message, dict):
                continue
            messages.append(
                _map_claude_message(
                    raw_message,
                    conversation_id=conversation_id,
                    source_file=source_file,
                    sequence_index=sequence_index,
                )
            )

    return conversations, messages


def _map_claude_message(
    raw_message: dict[str, object],
    *,
    conversation_id: str,
    source_file: str,
    sequence_index: int,
) -> CanonicalMessage:
    source_message_id = _string_or_none(raw_message.get("uuid"))
    message_id = make_message_id(
        "claude",
        source_message_id,
        conversation_id=conversation_id,
        stable_local_key=f"index-{sequence_index}",
    )

    raw_content = raw_message.get("content")
    content_blocks = _map_content_blocks(raw_message.get("content"))
    attachment_refs = _map_attachment_refs(raw_message)

    return CanonicalMessage(
        message_id=message_id,
        conversation_id=conversation_id,
        provider="claude",
        source_message_id=source_message_id or "",
        parent_message_id=None,
        sequence_index=sequence_index,
        author_role=_map_sender_to_author_role(raw_message.get("sender")),
        sender=_string_or_none(raw_message.get("sender")),
        created_at=normalize_timestamp(raw_message.get("created_at")),
        updated_at=normalize_timestamp(raw_message.get("updated_at")),
        text=derive_text_from_blocks(content_blocks),
        content_blocks=content_blocks,
        attachments=attachment_refs,
        source_artifact=MessageSourceArtifact(
            conversation_file=source_file,
            raw_message_path=f"chat_messages[{sequence_index}]",
        ),
        source_metadata=MessageSourceMetadata(
            flags=_collect_flags(raw_content),
        ),
    )


def _map_content_blocks(raw_content: object) -> tuple[ContentBlock, ...]:
    if not isinstance(raw_content, list):
        return ()

    blocks: list[ContentBlock] = []
    for item in raw_content:
        if not isinstance(item, dict):
            continue
        citations = item.get("citations")
        if isinstance(citations, list):
            citations_value: tuple[object, ...] | None = tuple(citations)
        else:
            citations_value = None
        blocks.append(
            ContentBlock(
                type=_string_or_none(item.get("type")) or "unknown",
                text=_blank_to_none(item.get("text")),
                start_timestamp=normalize_timestamp(item.get("start_timestamp")),
                stop_timestamp=normalize_timestamp(item.get("stop_timestamp")),
                citations=citations_value,
            )
        )
    return tuple(blocks)


def _map_attachment_refs(raw_message: dict[str, object]) -> tuple[AttachmentReference, ...]:
    refs: list[AttachmentReference] = []
    for key in ("attachments", "files"):
        raw_items = raw_message.get(key)
        if not isinstance(raw_items, list):
            continue
        for item in raw_items:
            refs.append(
                AttachmentReference(
                    kind=key[:-1] if key.endswith("s") else key,
                    path=None,
                    source_ref=_extract_attachment_source_ref(item),
                )
            )
    return tuple(refs)


def _extract_attachment_source_ref(item: object) -> str | None:
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        for key in ("uuid", "id", "file_id", "name", "file_name"):
            value = _string_or_none(item.get(key))
            if value:
                return value
    return None


def _collect_flags(raw_content: object) -> tuple[object, ...] | None:
    flags: list[object] = []
    if not isinstance(raw_content, list):
        return None
    for item in raw_content:
        if not isinstance(item, dict):
            continue
        flag_value = item.get("flags")
        if flag_value is None:
            continue
        if isinstance(flag_value, list):
            flags.extend(flag_value)
        else:
            flags.append(flag_value)
    return None if not flags else tuple(flags)


def _extract_account_uuid(account: object) -> str | None:
    if not isinstance(account, dict):
        return None
    return _string_or_none(account.get("uuid"))


def _map_sender_to_author_role(sender: object) -> str | None:
    value = _string_or_none(sender)
    if value == "human":
        return "user"
    if value == "assistant":
        return "assistant"
    return None


def _string_or_none(value: object) -> str | None:
    if isinstance(value, str):
        return value
    return None


def _blank_to_none(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def _derive_source_root(conversations_path: Path) -> Path:
    if conversations_path.parent.name:
        return conversations_path.parent
    return conversations_path.parent
