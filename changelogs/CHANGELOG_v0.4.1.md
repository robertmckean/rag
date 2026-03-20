# v0.4.1 - Conversational-memory grounding mode and LLM synthesis

Date: 2026-03-20

## Implements

- adds opt-in `conversational_memory` grounding mode for same-window local evidence composition
- adds constrained LLM-backed answer synthesis via `--llm` flag (Phase 3B)
- adds answer diagnostics with per-item rejection reasons and qualification traceability
- distinguishes retrieval miss from qualification failure in gap wording
- adds `--grounding-mode` and `--llm` / `--llm-model` CLI arguments
- adds Phase 3C conversational-memory design spec

## Conversational-memory mode

- opt-in via `--grounding-mode conversational_memory`
- composition is bounded to one retrieved window within one conversation
- requires at least one user-authored excerpt in the composing set
- requires 75% coverage of meaningful query terms across contributing excerpts
- distributed composition maps to `partially_supported` by default
- `supported` remains reserved for single-excerpt grounding

## LLM synthesis (Phase 3B)

- constrained rewrite layer on top of deterministic Phase 3A outputs
- validates citation references against provided evidence
- rejects synthesis output that introduces unseen entities or dates
- falls back to deterministic answer on any LLM or validation failure

## Intentionally excludes

- Phase 1 normalization contract changes
- embeddings or vector storage
- UI work
- cross-run retrieval
- semantic ranking beyond lexical improvements
- retrieval-level deduplication of near-duplicate messages

## Known limitations

- retrieval still returns near-duplicate messages as separate results
- conversational-memory mode requires explicit opt-in
- LLM synthesis requires the OpenAI Python SDK installed separately

## Validation

- 72 tests pass across unit and integration suites
- answer CLI smoke-tested for strict and conversational_memory modes
- LLM synthesis validation tested with mock adapter
