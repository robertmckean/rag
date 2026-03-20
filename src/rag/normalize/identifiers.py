"""Canonical identifier helpers for normalized records."""

from __future__ import annotations


def make_conversation_id(
    provider: str,
    source_conversation_id: str | None,
    *,
    conversation_file: str | None = None,
    ordinal: int | None = None,
) -> str:
    """Return a canonical conversation identifier."""
    if source_conversation_id:
        return f"{provider}:conversation:{source_conversation_id}"

    stable_file = conversation_file or "unknown"
    stable_ordinal = 0 if ordinal is None else ordinal
    return f"{provider}:conversation:missing:{stable_file}:{stable_ordinal}"


def make_message_id(
    provider: str,
    source_message_id: str | None,
    *,
    conversation_id: str,
    stable_local_key: str | None = None,
) -> str:
    """Return a canonical message identifier."""
    if source_message_id:
        return f"{provider}:message:{source_message_id}"

    fallback_key = stable_local_key or "unknown"
    return f"{provider}:message:missing:{conversation_id}:{fallback_key}"
