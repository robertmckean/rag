# v0.7.9 — Evolution router + contradiction detection (Phase 13C, 14A)

## Changes

### Evolution router integration (Phase 13C)
- New `EVOLUTION` intent in query router, triggered by evolution keywords
  ("thinking", "journey", "evolve", "path", "progression", etc.).
- Entity + evolution keyword → entity-scoped evolution; strong keyword
  alone → topic-level evolution.
- `_answer_evolution()` formats three cases:
  - Case A (<2 positions): "Insufficient evidence to determine evolution."
  - Case B (2+ stable): "Position appears stable over time."
  - Case C (2+ with shifts): "Possible evolution detected."
- Conservative language throughout — no overclaiming.

### Contradiction/change signal detection (Phase 14A)
- New `Contradiction` frozen dataclass: entity, earlier/later Position,
  signal (reason string), date_range.
- `detect_contradictions(entity, positions)` compares adjacent chronological
  positions via three heuristics:
  1. Explicit self-revision markers (changed my mind, was wrong, no longer,
     used to think).
  2. Negation introduced or removed.
  3. Sentiment-bearing term reversal (positive ↔ negative).
- Reuses existing `_detect_shift` infrastructure — no logic duplication.
- No router integration yet (Phase 14C).

## Test coverage
- 494 tests passing (32 new: 14 Phase 13C router, 18 Phase 14A contradiction).

## Real corpus validation
- "how did my thinking about Marc change" — routes to EVOLUTION, correctly
  reports insufficient evidence.
- "how did my thinking about myself evolve" — 2 stable positions across
  6 months, no false shift claims.
- "what was my path to shadow work" — routes to EVOLUTION (topic-level).
- Broad sweep (10 queries, 11 positions): 0 false contradictions detected.
- Bottleneck is upstream position density, not detection logic.
