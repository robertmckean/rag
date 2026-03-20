# v0.4.3 - Add hybrid retrieval baseline and harden embedding builds

Date: 2026-03-20

## Implements

- adds Phase 4A semantic retrieval over run-local message embeddings
- adds hybrid retrieval that merges BM25 and semantic candidates with explicit
  provenance and score visibility
- adds `rag.cli.build_embeddings` for file-based message-level embedding
  artifacts
- adds subset, sampling, resume, targeted conversation/message selection, and
  low-information filtering for embedding builds
- hardens JSONL artifact append behavior on Windows with batch-local writes,
  explicit flush and `fsync`, and retry on transient `PermissionError`

## Release meaning

This release is the first hybrid retrieval baseline for the repository.

- BM25 remains the lexical baseline
- semantic retrieval is additive and file-based
- hybrid retrieval broadens recall without changing grounded-answer
  qualification rules
- Phase 3 grounding behavior remains strict and deterministic on top of the
  broader retrieval layer

## Known boundary

- semantic and hybrid quality depend on artifact coverage
- the earlier poor Larry semantic result was traced to incomplete embedding
  coverage rather than a proven semantic-ranking failure
- BM25, semantic, and hybrid retrieval are now all inspectable from the CLI
  with explicit provenance and per-channel scores

## Validation

- focused embedding-builder tests pass
- focused semantic/hybrid retrieval tests pass
- full unittest discovery passes
- live-corpus validation confirmed that semantic retrieval becomes relevant once
  the Larry-related messages are actually embedded
