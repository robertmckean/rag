"""Canonical phase-1 schema for normalized conversations and messages."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


CONVERSATION_SOURCE_METADATA_FIELDS = {
    "chatgpt": {
        "conversation_origin",
        "is_archived",
        "is_starred",
        "is_read_only",
        "is_do_not_remember",
        "is_study_mode",
        "default_model_slug",
    },
    "claude": {
        "account_uuid",
    },
}

MESSAGE_SOURCE_METADATA_FIELDS = {
    "chatgpt": {
        "content_type",
        "channel",
        "status",
    },
    "claude": {
        "flags",
    },
}


@dataclass(frozen=True)
class ParticipantSummary:
    roles: tuple[str, ...] = ()
    authors: tuple[str, ...] = ()


@dataclass(frozen=True)
class SourceArtifact:
    root: str
    conversation_file: str | None = None
    sidecar_files: tuple[str, ...] = ()


@dataclass(frozen=True)
class ConversationSourceMetadata:
    conversation_origin: str | None = None
    is_archived: bool | None = None
    is_starred: bool | None = None
    is_read_only: bool | None = None
    is_do_not_remember: bool | None = None
    is_study_mode: bool | None = None
    default_model_slug: str | None = None
    account_uuid: str | None = None


@dataclass(frozen=True)
class CanonicalConversation:
    conversation_id: str
    provider: str
    source_conversation_id: str
    title: str | None
    summary: str | None
    created_at: str | None
    updated_at: str | None
    message_count: int
    source_artifact: SourceArtifact
    source_metadata: ConversationSourceMetadata
    participant_summary: ParticipantSummary = field(default_factory=ParticipantSummary)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ContentBlock:
    type: str
    text: str | None = None
    start_timestamp: str | None = None
    stop_timestamp: str | None = None
    citations: tuple[object, ...] | None = None


@dataclass(frozen=True)
class AttachmentReference:
    kind: str
    path: str | None = None
    source_ref: str | None = None


@dataclass(frozen=True)
class MessageSourceArtifact:
    conversation_file: str | None = None
    raw_message_path: str | None = None


@dataclass(frozen=True)
class MessageSourceMetadata:
    content_type: str | None = None
    channel: str | None = None
    status: str | None = None
    flags: tuple[object, ...] | None = None


@dataclass(frozen=True)
class CanonicalMessage:
    message_id: str
    conversation_id: str
    provider: str
    source_message_id: str
    parent_message_id: str | None
    sequence_index: int
    content_blocks: tuple[ContentBlock, ...]
    source_artifact: MessageSourceArtifact
    source_metadata: MessageSourceMetadata
    author_role: str | None = None
    author_name: str | None = None
    sender: str | None = None
    created_at: str | None = None
    updated_at: str | None = None
    text: str | None = None
    attachments: tuple[AttachmentReference, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def derive_text_from_blocks(content_blocks: tuple[ContentBlock, ...]) -> str | None:
    """Flatten text blocks into a plain-text convenience field."""
    texts = [block.text for block in content_blocks if block.type == "text" and block.text]
    if not texts:
        return None
    return "\n\n".join(texts)
