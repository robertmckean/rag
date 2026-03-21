# v0.7.6 — Role-aware phase descriptions (Phase 12A)

## Changes

### Role-aware narrative construction (Phase 12A)
- Phase descriptions now place user-role excerpts before assistant-role
  excerpts, preserving relative order within each group.
- Phase label entity selection now prefers entities from user-role items;
  falls back to all entities only when no user items exist.
- Assistant-only phases produce unchanged output.
- No changes to phase grouping, transitions, gaps, or retrieval.

## Test coverage
- 415 tests passing (9 new: 5 role-aware description ordering,
  4 role-aware label entity selection).

## Real corpus validation
- `--answer "what happened with Marc"` — in mixed phases (e.g. phase 2),
  user content now leads: `[user] "Just remembering what Marc told me..."`
  appears before `[assistant] "Exactly — and let's decode that..."`.
- Labels like "Marc, Pinn" derived from user-mentioned entities, not
  assistant analysis.
- Assistant-only phases (e.g. phase 5) unchanged.
