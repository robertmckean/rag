# v0.7.7 — Role-aware summary + position extraction (Phase 12B, 13A)

## Changes

### Role-aware narrative summary (Phase 12B)
- `_generate_summary()` now reports "X of Y phase(s) grounded in user content"
  in the narrative summary string.
- Detection uses `[user]` tag presence in phase description.
- No changes to phase grouping, transitions, gaps, or retrieval.

### Deterministic position extraction (Phase 13A)
- New module `src/rag/narrative/positions.py` with `Position` frozen dataclass
  and `extract_positions(phase)` function.
- 18 stance markers: 10 core (I think, I believe, I decided, I realized,
  I concluded, my view is, I was wrong about, I changed my mind, I no longer,
  I used to think) + 8 naturalistic (I don't think, I don't believe, I feel
  like, I noticed, I learned, I understand, I'm sure, I figured out).
- Extracts user-role positions only; assistant segments are ignored.
- Unicode smart-quote normalization (U+2018/2019/02BC) for apostrophe matching.
- Longest-first regex ordering prevents partial matches.
- Entity association from extracted sentence only (no inference).
- No router integration, no temporal comparison, no contradiction detection.

## Test coverage
- 441 tests passing (21 new: 5 Phase 12B summary, 16 Phase 13A positions).

## Real corpus validation
- "what happened with Marc" — extracted `[i don't think] I don't think it went
  very well.` from user content (smart-quote apostrophe handled correctly).
- "shadow work" — extracted `[i understand] I understand what the shadow is.`
- "what did I realize about myself" — 2 positions: `[i noticed]`, `[i think]`.
- Assistant commentary excluded across all queries.
- Yield is conservative (~1-2 positions per narrative) — appropriate for v1.
