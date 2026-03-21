"""Small retrieval helpers shared across the Phase 2 read and ranking layers."""

from __future__ import annotations

# Re-export from the shared location so existing imports continue to work.
from rag.utils import string_or_none

__all__ = ["string_or_none"]
