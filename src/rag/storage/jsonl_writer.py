"""Utilities for writing deterministic JSONL files."""

from __future__ import annotations

import json
from pathlib import Path


# JSONL is the canonical on-disk format for phase-1 normalized records.
# Keys are sorted so repeated runs over the same data produce stable output lines.
# The writer also owns directory creation so callers do not duplicate filesystem setup.

# Write one JSON object per line using stable formatting rules.
def write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    """Write records to JSONL with deterministic formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True, sort_keys=True))
            handle.write("\n")
