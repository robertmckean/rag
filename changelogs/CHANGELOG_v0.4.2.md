# v0.4.2 - Last pure BM25 baseline and qualification diagnostics

Date: 2026-03-20

## Implements

- adds `rag.cli.answer --debug-qualification` for answer-layer qualification
  inspection
- exposes per-window qualification details for strict and
  `conversational_memory` grounding checks
- makes conversational-memory qualification failures auditable on real runs

## Release meaning

This release marks the last pure-BM25 retrieval baseline before hybrid
retrieval work.

- current retrieval remains BM25 lexical retrieval
- grounded answering and qualification are stable and testable on top of that
  retrieval layer
- the next planned retrieval milestone is BM25 + embeddings, not more ad hoc
  BM25 tweaking

## Documented limitation

- BM25 relies on lexical overlap and fails on vocabulary mismatch
- real example:
  - query: `What have I said about Larry's guitar playing?`
  - real retrieved evidence includes Larry and `playing the bass`
  - strict and conversational-memory grounding both correctly remain
    `insufficient_evidence` because `guitar` is not lexically grounded
- see `docs/known_limitations.md`

## Validation

- focused answer and answer-CLI tests pass
- full unittest discovery passes
- strict, conversational-memory, and debug qualification CLI paths were
  exercised against the real `combined-live-check` run
