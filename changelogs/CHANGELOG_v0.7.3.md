# v0.7.3 — Entity-scoped query routing (Phase 10A)

## Changes

### Entity-scoped queries (Phase 10A)
- Added `ENTITY_SCOPED` intent to deterministic query router.
- Entity detection via whole-word case-insensitive matching against
  PatternReport.entities; highest occurrence count breaks ties.
- ENTITY_SCOPED triggers only when both a known entity name and a question
  pattern word (what, how, when, happened, with, etc.) are present.
- Scoped answer filters clusters, entity_cluster_links, and temporal_bursts
  to the target entity only — no cross-entity leakage.
- Answer includes: occurrence count, active period, relevant clusters,
  cross-cluster bridge info, temporal bursts, and key mention excerpts.
- Existing intent behavior (ENTITY, THEME, CROSS_TOPIC, TEMPORAL, TIMELINE)
  is unchanged.
- Updated `_answer_unknown` help text to list entity-scoped queries.

### Roadmap
- Checked off all 10A items in `TO_DO.md`.
- Updated current state: entity-scoped routing works, 366 tests passing.

## Test coverage
- 366 tests passing (24 new: 6 entity detection, 7 intent classification,
  11 entity-scoped answer formatting — including filtering, leakage,
  determinism, and fallback tests).

## Real corpus validation
- `--answer "what happened with Marc"` → Marc-only: 7 occurrences, 3 bursts,
  3 key mentions, no unrelated entity data.
- `--answer "what happened with Benz"` → Benz-only: 4 occurrences, 3 bursts,
  3 key mentions, no unrelated entity data.
- `--answer "who are the main people"` → unchanged generic ENTITY listing.
