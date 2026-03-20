"""ChatGPT-only normalized run writing."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from src.rag.normalize.chatgpt import extract_chatgpt_records
from src.rag.storage.jsonl_writer import write_jsonl


def write_chatgpt_normalized_run(
    export_root: Path,
    output_root: Path,
    *,
    run_id: str | None = None,
    created_at: str | None = None,
) -> Path:
    """Extract ChatGPT records and write one normalized run folder."""
    canonical_conversations, canonical_messages = extract_chatgpt_records(export_root)

    resolved_run_id = run_id or _default_run_id()
    resolved_created_at = created_at or _utc_now_iso()
    run_dir = output_root / resolved_run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    conversations_output = run_dir / "conversations.jsonl"
    messages_output = run_dir / "messages.jsonl"
    manifest_output = run_dir / "manifest.json"

    write_jsonl(conversations_output, [record.to_dict() for record in canonical_conversations])
    write_jsonl(messages_output, [record.to_dict() for record in canonical_messages])

    manifest = {
        "run_id": resolved_run_id,
        "created_at": resolved_created_at,
        "providers_present": ["chatgpt"],
        "source_roots": [str(export_root)],
        "conversation_count": len(canonical_conversations),
        "message_count": len(canonical_messages),
        "graph_linearization_policy": (
            "Deterministic visible transcript order is reconstructed by walking "
            "each conversation's current_node ancestor chain and reversing that path."
        ),
        "notes": [
            "ChatGPT-only phase-1 normalization run.",
            "Only visible current_node chains are included.",
            "Branch nodes not on the visible chain are excluded.",
            "message=null nodes are excluded from canonical messages.",
            "Sidecar files are not ingested into canonical conversation or message streams.",
            "Attachment references are preserved without copying blobs.",
        ],
    }
    manifest_output.write_text(
        json.dumps(manifest, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    return run_dir


def _default_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
