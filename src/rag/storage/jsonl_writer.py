"""Utilities for writing deterministic JSONL files."""

from __future__ import annotations

import json
from pathlib import Path


def write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    """Write records to JSONL with deterministic formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True, sort_keys=True))
            handle.write("\n")
