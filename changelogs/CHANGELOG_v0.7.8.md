# v0.7.8 — Temporal position comparison (Phase 13B)

## Changes

### Deterministic temporal comparison (Phase 13B)
- New `ThinkingEvolution` frozen dataclass: entity, chronologically ordered
  positions, and detected shift signals.
- `build_thinking_evolution(entity, positions)` sorts by date ascending
  (undated last), compares adjacent pairs via three heuristics.
- `collect_positions_for_entity(all_positions, entity)` filters by
  `Position.entity` match with case-insensitive whole-word text fallback.
- Shift detection heuristics (conservative, no LLM):
  1. Explicit self-revision markers (changed my mind, was wrong, no longer,
     used to think).
  2. Negation change (don't/not/never introduced or removed).
  3. Sentiment-bearing term change (positive-to-negative or vice versa).
- No shifts produced for stable/consistent positions.
- No router integration, no contradiction classification yet.

## Test coverage
- 462 tests passing (21 new: 3 sort, 3 stable, 2 negation, 4 explicit
  revision, 2 sentiment, 4 entity scoping, 1 determinism, 2 serialization).

## Real corpus validation
- "what happened with Marc" — 1 position extracted, no shifts (single
  position, correct behavior).
- "what did I realize about myself" — 2 positions across 6 months, no
  shifts (stable, correct — no negation or sentiment change).
- Shift heuristics fire accurately in unit tests; no false positives on
  real corpus data.
- Bottleneck is upstream position yield (Phase 13A), not comparison logic.
