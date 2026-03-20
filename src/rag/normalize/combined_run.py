"""Combined multi-provider normalized run writing."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from src.rag.normalize.canonical_schema import CanonicalConversation, CanonicalMessage
from src.rag.normalize.chatgpt import extract_chatgpt_records
from src.rag.normalize.claude import extract_claude_records
from src.rag.storage.jsonl_writer import write_jsonl


def write_combined_normalized_run(
    *,
    chatgpt_export_root: Path,
    claude_conversations_path: Path,
    output_root: Path,
    run_id: str | None = None,
    created_at: str | None = None,
) -> Path:
    """Extract both providers and write one combined normalized run folder."""
    chatgpt_conversations, chatgpt_messages = extract_chatgpt_records(chatgpt_export_root)
    claude_conversations, claude_messages = extract_claude_records(claude_conversations_path)

    all_conversations = sorted(
        [*chatgpt_conversations, *claude_conversations],
        key=_conversation_sort_key,
    )
    all_messages = sorted(
        [*chatgpt_messages, *claude_messages],
        key=_message_sort_key,
    )

    resolved_run_id = run_id or _default_run_id()
    resolved_created_at = created_at or _utc_now_iso()
    run_dir = output_root / resolved_run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    conversations_output = run_dir / "conversations.jsonl"
    messages_output = run_dir / "messages.jsonl"
    manifest_output = run_dir / "manifest.json"

    write_jsonl(conversations_output, [record.to_dict() for record in all_conversations])
    write_jsonl(messages_output, [record.to_dict() for record in all_messages])

    counts_by_provider = {
        "chatgpt": {
            "conversation_count": len(chatgpt_conversations),
            "message_count": len(chatgpt_messages),
        },
        "claude": {
            "conversation_count": len(claude_conversations),
            "message_count": len(claude_messages),
        },
    }

    manifest = {
        "run_id": resolved_run_id,
        "created_at": resolved_created_at,
        "providers_present": ["chatgpt", "claude"],
        "source_roots": {
            "chatgpt": str(chatgpt_export_root),
            "claude": str(claude_conversations_path.parent),
        },
        "conversation_count": len(all_conversations),
        "message_count": len(all_messages),
        "counts_by_provider": counts_by_provider,
        "notes": [
            "Combined phase-1 normalization run across ChatGPT and Claude.",
            "ChatGPT includes only visible current_node chains.",
            "ChatGPT excludes branch nodes not on the visible chain.",
            "ChatGPT excludes message=null nodes from canonical messages.",
            "Claude excludes sidecar streams: memories.json, projects.json, users.json.",
            "Attachment and file references are preserved without copying blobs.",
            "source_metadata is minimized to the phase-1 allowlist only.",
        ],
        "provider_policies": {
            "chatgpt": {
                "graph_linearization_policy": (
                    "Deterministic visible transcript order is reconstructed by walking "
                    "each conversation's current_node ancestor chain and reversing that path."
                ),
                "branch_exclusion": True,
                "null_message_node_exclusion": True,
                "sidecar_streams_ingested": [],
            },
            "claude": {
                "sidecar_streams_ingested": [],
                "sidecar_streams_excluded": ["memories.json", "projects.json", "users.json"],
            },
            "shared": {
                "attachment_blob_copying": False,
                "source_metadata_policy": "minimal_allowlist",
            },
        },
    }
    manifest_output.write_text(
        json.dumps(manifest, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    return run_dir


def _conversation_sort_key(record: CanonicalConversation) -> tuple[str, str]:
    return (record.provider, record.conversation_id)


def _message_sort_key(record: CanonicalMessage) -> tuple[str, str, int, str]:
    return (
        record.provider,
        record.conversation_id,
        record.sequence_index,
        record.message_id,
    )


def _default_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
