# v0.4.6 - Retrieval quality pass: voice preference, cross-provider dedup, meta-commentary suppression

Date: 2026-03-21

## Implements

- user-voice preference: applies USER_VOICE_BOOST (1.25) to user-authored focal
  messages and ASSISTANT_VOICE_FACTOR (0.8) to assistant messages in both BM25 and
  semantic candidate scoring, so the user's own words rank above assistant reactions
- cross-provider dedup: collapses near-duplicate results where the same user text
  was sent to both ChatGPT and Claude by comparing the first 100 characters of
  normalized focal message text against already-accepted results
- assistant meta-commentary suppression: applies an additional
  ASSISTANT_META_COMMENTARY_FACTOR (0.6) to assistant messages whose text opens
  with process/reaction filler phrases (e.g., "I'd be happy to", "Your formatted
  text file is ready"), down-ranking without hard-filtering
- adds is_assistant_meta_commentary() detector with a narrow prefix-match rule
  covering 35 common assistant opener patterns
- adds test for cross-provider dedup with multi-provider fixture
- adds test for meta-commentary detection covering both positive and negative cases
- updates existing test assertions for voice boost ranking changes

## Benchmark results (BM25, top 10, 5 queries)

- user/assistant split: 49/1 (up from ~42/8 pre-v0.4.6)
- cross-provider duplicates eliminated: 5 pairs across 3 queries reduced to 0
- meta-commentary assistant messages in top 10: 0 (1 substantive assistant remains)
- the sole surviving assistant result is analytical synthesis, not filler

## Release meaning

This is the retrieval quality baseline freeze. The three changes form a coherent
pass that addresses ranking bias, evidence diversity, and noise suppression. The
remaining retrieval gaps (e.g., trivial entity-match queries ranking on keyword
overlap alone) are lexical relevance boundaries that require synthesis-layer
evaluation rather than further retrieval tuning.

## Validation

- PYTHONPATH=src python -m unittest discover -s tests -p "test_*.py"
- 89 tests pass, zero failures
- Live benchmark across 5 queries confirms improved diversity and signal quality
