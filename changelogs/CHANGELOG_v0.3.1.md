# v0.3.1 - Retrieval Documentation And Release Guidance Refresh

Date: 2026-03-20

## Implements

- refreshes repository guidance to reflect the current Phase 2 retrieval state
- updates top-level documentation so retrieval modes and current repo structure are explicit
- replaces placeholder project-summary guidance with a concrete repo summary
- hardens retrieval CLI release validation guidance for missing and malformed run artifacts

## Intentionally Excludes

- Phase 1 normalization contract changes
- embeddings
- vector storage
- UI work
- answer generation
- cross-run retrieval
- semantic ranking beyond the existing lexical retrieval layer

## Known Limitations

- retrieval remains local and run-scoped to one normalized run directory at a time
- timeline mode is a compact chronology view and does not return contextual message windows
- attachment handling remains reference-only from the underlying normalized records
- local workspace warnings from denied directories such as `tmpeq_ll7va/` are outside the release scope

## Validation

- broad unittest suite run from the configured `drum310` environment
- retrieval CLI smoke-tested for normal timeline output
- retrieval CLI failure path checked for a missing run directory
