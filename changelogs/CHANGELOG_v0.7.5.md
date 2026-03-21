# v0.7.5 — Assistant restatement dedup (Phase 11)

## Changes

### Assistant restatement detection and downweighting (Phase 11)
- Added `is_assistant_restatement(assistant_text, user_texts)` in
  `src/rag/retrieval/types.py`: deterministic token-overlap comparison after
  lowercasing, punctuation stripping, and stopword removal.
- Threshold: 60% of assistant content tokens must overlap a single nearby
  user message to classify as restatement.
- Added `get_nearby_user_texts()`: extracts user messages within a 5-message
  radius of the focal assistant message in the same conversation.
- Applied `ASSISTANT_RESTATEMENT_FACTOR` (0.5x) in both BM25 and semantic
  scoring paths via lazy per-conversation index lookup.
- Assistant restatements are downweighted, not removed — they remain
  available but rank below the original user signal.
- Non-restatement assistant messages (new analysis, new information) are
  unaffected.

## Test coverage
- 406 tests passing (22 new: 3 tokenization, 11 restatement detection,
  4 nearby user text extraction, 4 factor/constant verification).

## Real corpus validation
- 207 assistant restatements detected across chatgpt-live-check corpus.
- "code style preferences inline comments" — top 8 results all user messages;
  assistant restatements pushed below user originals.
- "confession about Marc no sex relationship" — top 7 results all user
  messages; assistant messages that add new analysis preserved at reasonable
  ranks (8, 11); pure restatements pushed down.
- "Marc zero chance Pinn" — top 4 user messages; assistant content at rank 5
  adds new analysis (not flagged as restatement).
