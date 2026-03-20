"""ChatGPT-only extraction into the canonical phase-1 schema."""

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


def extract_chatgpt_records(
    export_root: Path,
) -> tuple[list[CanonicalConversation], list[CanonicalMessage]]:
    """Load ChatGPT conversation shards and map them into canonical records."""
    shard_paths = discover_chatgpt_conversation_shards(export_root)

    conversations: list[CanonicalConversation] = []
    messages: list[CanonicalMessage] = []

    for shard_path in shard_paths:
        payload = json.loads(shard_path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError(f"ChatGPT shard must contain a JSON array: {shard_path}")

        source_file = str(shard_path.relative_to(export_root))
        for conversation_ordinal, raw_conversation in enumerate(payload):
            if not isinstance(raw_conversation, dict):
                continue

            conversation, conversation_messages = _map_conversation(
                raw_conversation,
                export_root=export_root,
                source_file=source_file,
                conversation_ordinal=conversation_ordinal,
            )
            conversations.append(conversation)
            messages.extend(conversation_messages)

    return conversations, messages


def discover_chatgpt_conversation_shards(export_root: Path) -> list[Path]:
    """Return the sorted conversation shard set for a ChatGPT export bundle."""
    shard_paths = sorted(export_root.glob("conversations-*.json"))
    if not shard_paths:
        raise FileNotFoundError(
            f"No ChatGPT conversation shards found under {export_root}"
        )
    return shard_paths


def _map_conversation(
    raw_conversation: dict[str, object],
    *,
    export_root: Path,
    source_file: str,
    conversation_ordinal: int,
) -> tuple[CanonicalConversation, list[CanonicalMessage]]:
    source_conversation_id = _string_or_none(
        raw_conversation.get("conversation_id") or raw_conversation.get("id")
    )
    conversation_id = make_conversation_id(
        "chatgpt",
        source_conversation_id,
        conversation_file=source_file,
        ordinal=conversation_ordinal,
    )

    mapping = raw_conversation.get("mapping")
    if not isinstance(mapping, dict):
        mapping = {}

    visible_nodes = _linearize_visible_conversation(mapping, raw_conversation.get("current_node"))
    canonical_messages = _map_messages(
        visible_nodes,
        conversation_id=conversation_id,
        source_file=source_file,
    )

    participant_roles = sorted(
        {
            message.author_role
            for message in canonical_messages
            if message.author_role
        }
    )
    participant_authors = sorted(
        {
            message.author_name
            for message in canonical_messages
            if message.author_name
        }
    )

    conversation = CanonicalConversation(
        conversation_id=conversation_id,
        provider="chatgpt",
        source_conversation_id=source_conversation_id or "",
        title=_blank_to_none(raw_conversation.get("title")),
        summary=None,
        created_at=normalize_timestamp(raw_conversation.get("create_time")),
        updated_at=normalize_timestamp(raw_conversation.get("update_time")),
        message_count=len(canonical_messages),
        participant_summary=ParticipantSummary(
            roles=tuple(participant_roles),
            authors=tuple(participant_authors),
        ),
        source_artifact=SourceArtifact(
            root=str(export_root.name),
            conversation_file=source_file,
            sidecar_files=(),
        ),
        source_metadata=ConversationSourceMetadata(
            conversation_origin=_string_or_none(raw_conversation.get("conversation_origin")),
            is_archived=_bool_or_none(raw_conversation.get("is_archived")),
            is_starred=_bool_or_none(raw_conversation.get("is_starred")),
            is_read_only=_bool_or_none(raw_conversation.get("is_read_only")),
            is_do_not_remember=_bool_or_none(raw_conversation.get("is_do_not_remember")),
            is_study_mode=_bool_or_none(raw_conversation.get("is_study_mode")),
            default_model_slug=_string_or_none(raw_conversation.get("default_model_slug")),
        ),
    )
    return conversation, canonical_messages


def _linearize_visible_conversation(
    mapping: dict[str, object],
    current_node_id: object,
) -> list[tuple[str, dict[str, object], dict[str, object]]]:
    if not isinstance(current_node_id, str) or current_node_id not in mapping:
        return []

    path: list[tuple[str, dict[str, object], dict[str, object]]] = []
    seen: set[str] = set()
    node_id = current_node_id

    while node_id and node_id not in seen and node_id in mapping:
        seen.add(node_id)
        raw_node = mapping.get(node_id)
        if not isinstance(raw_node, dict):
            break
        raw_message = raw_node.get("message")
        if isinstance(raw_message, dict):
            path.append((node_id, raw_node, raw_message))
        parent = raw_node.get("parent")
        node_id = parent if isinstance(parent, str) else None

    path.reverse()
    return path


def _map_messages(
    visible_nodes: list[tuple[str, dict[str, object], dict[str, object]]],
    *,
    conversation_id: str,
    source_file: str,
) -> list[CanonicalMessage]:
    messages: list[CanonicalMessage] = []
    canonical_ids_by_node: dict[str, str] = {}

    for sequence_index, (node_id, raw_node, raw_message) in enumerate(visible_nodes):
        source_message_id = _string_or_none(raw_message.get("id")) or node_id
        message_id = make_message_id(
            "chatgpt",
            source_message_id,
            conversation_id=conversation_id,
            stable_local_key=node_id,
        )

        parent_node_id = raw_node.get("parent")
        parent_message_id = (
            canonical_ids_by_node.get(parent_node_id)
            if isinstance(parent_node_id, str)
            else None
        )

        content_blocks, attachments = _map_content_and_attachments(raw_message)

        author = raw_message.get("author")
        metadata = raw_message.get("metadata")
        content = raw_message.get("content")

        message = CanonicalMessage(
            message_id=message_id,
            conversation_id=conversation_id,
            provider="chatgpt",
            source_message_id=source_message_id,
            parent_message_id=parent_message_id,
            sequence_index=sequence_index,
            author_role=_extract_author_role(author),
            author_name=_extract_author_name(author),
            created_at=normalize_timestamp(raw_message.get("create_time")),
            updated_at=normalize_timestamp(raw_message.get("update_time")),
            text=derive_text_from_blocks(content_blocks),
            content_blocks=content_blocks,
            attachments=attachments,
            source_artifact=MessageSourceArtifact(
                conversation_file=source_file,
                raw_message_path=f"mapping.{node_id}.message",
            ),
            source_metadata=MessageSourceMetadata(
                content_type=_extract_content_type(content),
                channel=_string_or_none(raw_message.get("channel")),
                status=_string_or_none(raw_message.get("status")),
            ),
        )
        messages.append(message)
        canonical_ids_by_node[node_id] = message_id

    return messages


def _map_content_and_attachments(
    raw_message: dict[str, object],
) -> tuple[tuple[ContentBlock, ...], tuple[AttachmentReference, ...]]:
    raw_content = raw_message.get("content")
    raw_metadata = raw_message.get("metadata")

    blocks: list[ContentBlock] = []
    attachments: list[AttachmentReference] = []

    if isinstance(raw_content, dict):
        content_type = _string_or_none(raw_content.get("content_type")) or "unknown"
        parts = raw_content.get("parts")
        if isinstance(parts, list):
            for part_index, part in enumerate(parts):
                if isinstance(part, str):
                    blocks.append(
                        ContentBlock(
                            type="text",
                            text=_blank_to_none(part),
                        )
                    )
                elif isinstance(part, dict):
                    part_type = _string_or_none(part.get("content_type")) or content_type
                    blocks.append(
                        ContentBlock(
                            type=part_type,
                            text=_blank_to_none(part.get("text")),
                        )
                    )
                    attachments.extend(
                        _map_part_attachment_refs(part, default_kind=part_type, part_index=part_index)
                    )

    if isinstance(raw_metadata, dict):
        raw_attachments = raw_metadata.get("attachments")
        if isinstance(raw_attachments, list):
            for item in raw_attachments:
                source_ref = _extract_attachment_source_ref(item)
                path = _extract_attachment_path(item)
                attachments.append(
                    AttachmentReference(
                        kind="attachment",
                        path=path,
                        source_ref=source_ref,
                    )
                )

    return tuple(blocks), tuple(attachments)


def _map_part_attachment_refs(
    part: dict[str, object],
    *,
    default_kind: str,
    part_index: int,
) -> list[AttachmentReference]:
    refs: list[AttachmentReference] = []
    source_ref = _string_or_none(part.get("asset_pointer"))
    if source_ref:
        refs.append(
            AttachmentReference(
                kind=default_kind or "attachment",
                path=_asset_pointer_to_path(source_ref),
                source_ref=source_ref,
            )
        )
        return refs

    file_id = _string_or_none(part.get("file_id"))
    if file_id:
        refs.append(
            AttachmentReference(
                kind=default_kind or "attachment",
                path=None,
                source_ref=file_id,
            )
        )
        return refs

    if part:
        refs.append(
            AttachmentReference(
                kind=default_kind or "attachment",
                path=None,
                source_ref=f"part-{part_index}",
            )
        )
    return refs


def _extract_author_role(author: object) -> str | None:
    if not isinstance(author, dict):
        return None
    return _string_or_none(author.get("role"))


def _extract_author_name(author: object) -> str | None:
    if not isinstance(author, dict):
        return None
    return _blank_to_none(author.get("name"))


def _extract_content_type(content: object) -> str | None:
    if not isinstance(content, dict):
        return None
    return _string_or_none(content.get("content_type"))


def _extract_attachment_source_ref(item: object) -> str | None:
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        for key in ("id", "file_id", "asset_pointer", "name"):
            value = _string_or_none(item.get(key))
            if value:
                return value
    return None


def _extract_attachment_path(item: object) -> str | None:
    if not isinstance(item, dict):
        return None
    for key in ("name", "file_name"):
        value = _string_or_none(item.get(key))
        if value:
            return value
    return None


def _asset_pointer_to_path(asset_pointer: str) -> str | None:
    prefix = "file-service://"
    if asset_pointer.startswith(prefix):
        return asset_pointer[len(prefix) :]
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


def _bool_or_none(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    return None
