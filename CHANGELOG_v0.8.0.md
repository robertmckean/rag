# Changelog — v0.8.0

Released: 2026-03-21

## Phase 14B — Change type classification

- Added `change_type` field to `Contradiction` model with four deterministic
  categories: `reversal`, `softening`, `strengthening`, `evolution`
- Added marker-strength table (0–5 scale) for stance-marker intensity comparison
- Classification priority: explicit self-revision → sentiment reversal →
  negation + strength comparison → marker strength fallback → evolution default
- Added `_classify_change_type()` deterministic classifier in `positions.py`
- 12 new tests covering all classification categories and determinism

## Phase 14C — Contradiction router integration

- Added `CONTRADICTION` intent to query router with keyword/phrase detection
  (contradict, reverse, soften, strengthen, grew, growth, clearer, changed mind)
- Priority ordering: CONTRADICTION → EVOLUTION → ENTITY_SCOPED → keyword scoring
- Three-case answer formatting: insufficient evidence / no signal / detection
  with change-type details
- 15 new router tests for intent classification and answer formatting
- Real corpus validation: correct routing for contradiction, growth, and
  mind-change queries

## Test count

521 tests passing (up from 494).
