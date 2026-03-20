"""Canonical phase-1 schema for normalized conversations and messages."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


# The schema keeps shared canonical fields first and provider metadata narrow.
# `content_blocks` is the authoritative message body representation in phase 1.
# `text` is only a derived convenience field for downstream inspection and filtering.
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
    # Participant modeling stays intentionally lightweight in phase 1.
    roles: tuple[str, ...] = ()
    authors: tuple[str, ...] = ()


@dataclass(frozen=True)
class SourceArtifact:
    # Provenance fields point back to the raw export bundle used for normalization.
    root: str
    conversation_file: str | None = None
    sidecar_files: tuple[str, ...] = ()


@dataclass(frozen=True)
class ConversationSourceMetadata:
    # Conversation metadata is constrained to the explicit phase-1 allowlist.
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

    # Convert dataclass records into the JSON-serializable shape used by JSONL writers.
    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ContentBlock:
    # Content blocks preserve structured body information across providers.
    type: str
    text: str | None = None
    start_timestamp: str | None = None
    stop_timestamp: str | None = None
    citations: tuple[object, ...] | None = None


@dataclass(frozen=True)
class AttachmentReference:
    # Attachments stay reference-only so canonical runs do not duplicate raw blobs.
    kind: str
    path: str | None = None
    source_ref: str | None = None


@dataclass(frozen=True)
class MessageSourceArtifact:
    # Message provenance captures both the source file and the raw path inside that file.
    conversation_file: str | None = None
    raw_message_path: str | None = None


@dataclass(frozen=True)
class MessageSourceMetadata:
    # Message metadata is limited to fields explicitly needed for phase-1 fidelity.
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

    # Convert dataclass records into the JSON-serializable shape used by JSONL writers.
    def to_dict(self) -> dict[str, object]:
        return asdict(self)


# Flatten only text blocks into the optional plain-text convenience field.
def derive_text_from_blocks(content_blocks: tuple[ContentBlock, ...]) -> str | None:
    """Flatten text blocks into a plain-text convenience field."""
    texts = [block.text for block in content_blocks if block.type == "text" and block.text]
    if not texts:
        return None
    return "\n\n".join(texts)
