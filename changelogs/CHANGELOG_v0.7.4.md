# v0.7.4 — Entity-scoped narrative timeline (Phase 10B)

## Changes

### Entity-scoped narrative (Phase 10B)
- Added `_phase_mentions_entity()`: whole-word case-insensitive check against
  phase label and description.
- Added `_filter_narrative_for_entity()`: filters narrative phases to those
  mentioning the target entity, preserves chronological order, and builds
  scoped transitions (keeps originals between adjacent retained phases,
  creates gap transitions for non-adjacent retained phases).
- Extended `_answer_entity_scoped()` to include a Timeline section showing
  filtered phases with descriptions and transitions.
- Edge cases handled: no matching phases, single phase, entity in label only.

## Test coverage
- 384 tests passing (18 new: 7 phase filtering, 3 transition filtering,
  8 scoped timeline answer tests — including cross-entity leakage,
  determinism, edge cases).

## Real corpus validation
- `--answer "what happened with Marc"` → 8 Marc-only phases with 7
  transitions, no unrelated phases.
- `--answer "what happened with Benz"` → 5 Benz-only phases with 4
  transitions, gap transition where intermediate phases were filtered.
- Both outputs clearly more useful than 10A-only (occurrence counts +
  mentions) — now a full entity-scoped chronological timeline.
