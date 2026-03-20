"""Read model for loading one immutable normalized run into retrieval-friendly structures."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from rag.retrieval.utils import string_or_none


# The read model is intentionally thin and only builds structures needed by the first slice.
# Phase 2 works directly over conversations.jsonl, messages.jsonl, and manifest.json.
# Searchable text is derived once during load so ranking logic does not keep re-parsing blocks.


@dataclass(frozen=True)
class LoadedRun:
    run_id: str
    run_dir: Path
    manifest: dict[str, object]
    conversations_path: Path
    messages_path: Path
    manifest_path: Path
    conversation_by_id: dict[str, dict[str, object]]
    message_by_id: dict[str, dict[str, object]]
    messages_by_conversation_id: dict[str, tuple[dict[str, object], ...]]
    searchable_text_by_message_id: dict[str, str]


# Load one normalized run and build the lookup structures needed by lexical retrieval.
def load_normalized_run(run_dir: Path) -> LoadedRun:
    resolved_run_dir = run_dir.resolve()
    if not resolved_run_dir.exists() or not resolved_run_dir.is_dir():
        raise FileNotFoundError(f"Run directory not found: {resolved_run_dir}")

    conversations_path = resolved_run_dir / "conversations.jsonl"
    messages_path = resolved_run_dir / "messages.jsonl"
    manifest_path = resolved_run_dir / "manifest.json"
    _require_file(conversations_path)
    _require_file(messages_path)
    _require_file(manifest_path)

    conversations = _load_jsonl(conversations_path)
    messages = _load_jsonl(messages_path)
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Malformed JSON in {manifest_path}: {exc.msg}") from exc

    conversation_by_id: dict[str, dict[str, object]] = {}
    for conversation in conversations:
        conversation_id = string_or_none(conversation.get("conversation_id"))
        if not conversation_id:
            continue
        conversation_by_id[conversation_id] = conversation

    message_by_id: dict[str, dict[str, object]] = {}
    grouped_messages: dict[str, list[dict[str, object]]] = {}
    searchable_text_by_message_id: dict[str, str] = {}

    for message in messages:
        message_id = string_or_none(message.get("message_id"))
        conversation_id = string_or_none(message.get("conversation_id"))
        if not message_id or not conversation_id:
            continue

        message_by_id[message_id] = message
        grouped_messages.setdefault(conversation_id, []).append(message)
        searchable_text_by_message_id[message_id] = build_searchable_text(message)

    # Sequence order is the retrieval backbone for contextual windows.
    ordered_messages_by_conversation_id = {
        conversation_id: tuple(sorted(items, key=_message_order_key))
        for conversation_id, items in grouped_messages.items()
    }

    return LoadedRun(
        run_id=string_or_none(manifest.get("run_id")) or resolved_run_dir.name,
        run_dir=resolved_run_dir,
        manifest=manifest,
        conversations_path=conversations_path,
        messages_path=messages_path,
        manifest_path=manifest_path,
        conversation_by_id=conversation_by_id,
        message_by_id=message_by_id,
        messages_by_conversation_id=ordered_messages_by_conversation_id,
        searchable_text_by_message_id=searchable_text_by_message_id,
    )


# Build the normalized lexical text used for message ranking.
def build_searchable_text(message: dict[str, object]) -> str:
    direct_text = string_or_none(message.get("text"))
    if direct_text:
        # Phase 1 already derives text from content blocks, so prefer it and avoid duplicate terms.
        return normalize_lexical_text(direct_text)

    text_parts: list[str] = []
    content_blocks = message.get("content_blocks")
    if isinstance(content_blocks, list):
        for block in content_blocks:
            if not isinstance(block, dict):
                continue
            block_text = string_or_none(block.get("text"))
            if block_text:
                text_parts.append(block_text)

    collapsed = " ".join(part.strip() for part in text_parts if part.strip())
    return normalize_lexical_text(collapsed)


# Normalize text into the lowercase token space used by the lexical scorer.
def normalize_lexical_text(value: str) -> str:
    return " ".join(_tokenize(value))


# Tokenize text conservatively so ranking remains inspectable and deterministic.
def tokenize_query(value: str) -> tuple[str, ...]:
    return tuple(_tokenize(value))


# Load one JSON object per line from a JSONL file.
def _load_jsonl(path: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed JSONL in {path} at line {line_number}: {exc.msg}") from exc
            if isinstance(payload, dict):
                records.append(payload)
    return records


# Fail early when a required normalized artifact is missing from the run directory.
def _require_file(path: Path) -> None:
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Required run file not found: {path}")


# Sort messages by sequence index first and message id second for stable tie-breaking.
def _message_order_key(message: dict[str, object]) -> tuple[int, str]:
    sequence_index = message.get("sequence_index")
    if not isinstance(sequence_index, int):
        sequence_index = 0
    message_id = string_or_none(message.get("message_id")) or ""
    return sequence_index, message_id


# Extract lowercase alphanumeric tokens used by the simple lexical ranker.
def _tokenize(value: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", value.lower())
