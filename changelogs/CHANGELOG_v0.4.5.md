# v0.4.5 - Retrieval quality: short-message filter and focal-visible window dedup

Date: 2026-03-21

## Implements

- adds a minimum token count filter (SEMANTIC_MIN_TOKEN_COUNT = 4) to semantic
  candidate generation so ultra-short messages like "Larry." no longer rank
  deceptively high on cosine similarity alone
- replaces exact window-key deduplication with focal-visible dedup: a candidate
  is skipped when its focal message sequence index already falls within an
  accepted window range in the same conversation
- updates three retrieval test assertions to reflect correct dedup behavior on
  the small fixture conversation (4 messages fit in one window)

## Release meaning

This release fixes the two most visible retrieval quality problems observed
during live semantic and hybrid testing: short-fragment noise in semantic results
and near-duplicate overlapping windows from dense conversation threads. No
scoring formulas, embedding artifacts, or public CLI/API contracts are changed.

## Validation

- PYTHONPATH=src python -m unittest discover -s tests -p "test_*.py"
- 114 tests discovered, 114 pass, zero failures
- Live retrieval tested with "what was my path to shadow work" across all three
  channels (bm25, semantic, hybrid) confirming improved result diversity
