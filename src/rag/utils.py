"""Shared helpers used across multiple rag modules."""

from __future__ import annotations


# Normalize optional scalar values to trimmed strings where modules expect them.
def string_or_none(value: object) -> str | None:
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    return None
