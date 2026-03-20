"""Retrieval package for Phase 2 read-only access over normalized runs."""

# Retrieval code reads immutable phase-1 artifacts rather than changing their schema.
# The package starts with lexical ranking and contextual windows only.
# Keeping retrieval separate from normalization prevents Phase 2 from mutating Phase 1.
