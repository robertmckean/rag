"""File-based embedding artifacts tied to one normalized run directory."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

from rag.utils import string_or_none


EMBEDDINGS_ARTIFACT_NAME = "message_embeddings.jsonl"


@dataclass(frozen=True)
class EmbeddingRecord:
    run_id: str
    message_id: str
    conversation_id: str
    provider: str
    author_role: str | None
    created_at: str | None
    sequence_index: int
    embedding_model: str
    embedding_dimensions: int
    text: str
    embedding: tuple[float, ...]
    original_text_length: int
    stored_text_length: int
    truncation_occurred: bool
    original_token_count: int | None = None
    stored_token_count: int | None = None
    total_messages_in_run: int | None = None
    build_timestamp: str | None = None
    subset_offset: int | None = None
    subset_limit: int | None = None
    subset_sample_size: int | None = None
    subset_sample_seed: int | None = None

    # Convert the record into a JSON-serializable dict for artifact storage.
    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["embedding"] = list(self.embedding)
        return payload


# Resolve the deterministic embeddings artifact location for one normalized run.
def embeddings_artifact_path(run_dir: Path) -> Path:
    return run_dir / EMBEDDINGS_ARTIFACT_NAME


# Read one embedding record per line from the run-local embeddings artifact.
def load_embedding_records(run_dir: Path) -> tuple[EmbeddingRecord, ...]:
    artifact_path = embeddings_artifact_path(run_dir)
    if not artifact_path.exists() or not artifact_path.is_file():
        raise FileNotFoundError(f"Embeddings artifact not found: {artifact_path}")

    records: list[EmbeddingRecord] = []
    with artifact_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed embeddings JSONL in {artifact_path} at line {line_number}: {exc.msg}") from exc
            if not isinstance(payload, dict):
                continue
            embedding_value = payload.get("embedding")
            if not isinstance(embedding_value, list):
                raise ValueError(f"Invalid embedding payload in {artifact_path} at line {line_number}: missing embedding list")
            records.append(
                EmbeddingRecord(
                    run_id=string_or_none(payload.get("run_id")) or "",
                    message_id=string_or_none(payload.get("message_id")) or "",
                    conversation_id=string_or_none(payload.get("conversation_id")) or "",
                    provider=string_or_none(payload.get("provider")) or "unknown",
                    author_role=string_or_none(payload.get("author_role")),
                    created_at=string_or_none(payload.get("created_at")),
                    sequence_index=payload.get("sequence_index") if isinstance(payload.get("sequence_index"), int) else 0,
                    embedding_model=string_or_none(payload.get("embedding_model")) or "",
                    embedding_dimensions=payload.get("embedding_dimensions")
                    if isinstance(payload.get("embedding_dimensions"), int)
                    else len(embedding_value),
                    text=string_or_none(payload.get("text")) or "",
                    original_text_length=payload.get("original_text_length")
                    if isinstance(payload.get("original_text_length"), int)
                    else len(string_or_none(payload.get("text")) or ""),
                    stored_text_length=payload.get("stored_text_length")
                    if isinstance(payload.get("stored_text_length"), int)
                    else len(string_or_none(payload.get("text")) or ""),
                    truncation_occurred=bool(payload.get("truncation_occurred", False)),
                    original_token_count=payload.get("original_token_count")
                    if isinstance(payload.get("original_token_count"), int)
                    else None,
                    stored_token_count=payload.get("stored_token_count")
                    if isinstance(payload.get("stored_token_count"), int)
                    else None,
                    total_messages_in_run=payload.get("total_messages_in_run")
                    if isinstance(payload.get("total_messages_in_run"), int)
                    else None,
                    build_timestamp=string_or_none(payload.get("build_timestamp")),
                    subset_offset=payload.get("subset_offset")
                    if isinstance(payload.get("subset_offset"), int)
                    else None,
                    subset_limit=payload.get("subset_limit")
                    if isinstance(payload.get("subset_limit"), int)
                    else None,
                    subset_sample_size=payload.get("subset_sample_size")
                    if isinstance(payload.get("subset_sample_size"), int)
                    else None,
                    subset_sample_seed=payload.get("subset_sample_seed")
                    if isinstance(payload.get("subset_sample_seed"), int)
                    else None,
                    embedding=tuple(float(value) for value in embedding_value),
                )
            )
    return tuple(records)


# Write embedding records atomically: write to a temp file first, then rename over the real artifact.
def write_embedding_records_atomic(run_dir: Path, records: tuple[EmbeddingRecord, ...]) -> Path:
    artifact_path = embeddings_artifact_path(run_dir)
    tmp_path = artifact_path.parent / (EMBEDDINGS_ARTIFACT_NAME + ".tmp")
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    with tmp_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_dict(), ensure_ascii=True, sort_keys=True))
            handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    if artifact_path.exists():
        artifact_path.unlink()
    tmp_path.rename(artifact_path)
    return artifact_path
