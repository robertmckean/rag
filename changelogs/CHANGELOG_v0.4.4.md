# v0.4.4 - Decouple runtime embedding imports and harden artifact writes

Date: 2026-03-21

## Implements

- moves the shared embedding client and default model contract out of
  `builder.py` into `src/rag/embeddings/client.py`
- keeps query-time retrieval and related CLI paths from importing build-time
  embedding orchestration code
- refactors deterministic answering internals into focused submodules while
  preserving `rag.answering.answer` as the public API surface
- replaces streaming embedding artifact appends with a buffered atomic write so
  completed builds publish a clean JSONL artifact in one rename step
- updates embedding and semantic retrieval tests to match the new artifact write
  behavior
- adds a structured documentation layout under `docs/` plus a placeholder run
  audit and versioned architecture decision note

## Release meaning

This release reduces coupling between build-time embedding code and query-time
retrieval code while making embedding artifact publication more reliable on
Windows.

## Validation

- `python -m unittest tests.unit.test_embeddings_builder_unittest tests.unit.test_retrieval_semantic_unittest`
- `python -m unittest tests.unit.test_answering_unittest`
- `python -m unittest discover -s tests -p "test*_unittest.py"`
