"""Small retrieval helpers shared across the Phase 2 read and ranking layers."""

from __future__ import annotations


# Normalize optional scalar values to trimmed strings where retrieval expects them.
def string_or_none(value: object) -> str | None:
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    return None
