# AGENTS.md

This file provides guidance to Codex and other coding agents when working in
this repository.

## Project Goal

Build a personal RAG system over exported ChatGPT and Claude histories. The
current codebase ingests raw exports into canonical normalized records and now
adds a local, inspectable lexical retrieval layer over those normalized runs.

## Project Status

- Active milestone: Phase 2 retrieval stabilization and iteration
- Phase 1 normalization is complete and its output contract is frozen
- Phase 2 currently supports BM25 lexical retrieval, contextual window results,
  timeline exploration, query normalization, and chronological retrieval modes
- Prefer the current code in `src/rag/` and the run artifacts under
  `data/normalized/runs/` over older notes or ad hoc local files

## Environment

- Use the intended local Python environment for this repo
- In this workspace, retrieval validation is typically run with:
  `$env:PYTHONPATH='src'; python -m unittest ...`
- Prefer PowerShell-native commands and repo-relative paths
- Raw exports live under `data/raw/`
- Immutable normalized run artifacts live under `data/normalized/runs/<run_id>/`

## Repo Structure

- `src/rag/normalize/`
  Phase 1 normalization, provider extraction, and run writing
- `src/rag/retrieval/`
  Phase 2 retrieval read model and lexical ranking logic
- `src/rag/cli/`
  CLI entry points for inspection, normalization, analysis, and retrieval
- `data/raw/`
  immutable provider export inputs
- `data/normalized/runs/`
  immutable normalized outputs per run
- `tests/unit/`
  focused module tests
- `tests/integration/`
  end-to-end normalization and CLI behavior checks
- `docs/`
  schema, policy, rerun, testing, and phase objective documents

## Phase Boundaries

- Phase 1 scope is frozen:
  canonical schema, provider extraction, JSONL outputs, manifests, and docs
- Do not change Phase 1 output fields or artifact layout unless the user
  explicitly approves a contract change
- Phase 2 works against existing normalized artifacts only
- Current Phase 2 non-goals:
  embeddings, vector DBs, UI, answer generation, cross-run retrieval, and
  semantic ranking beyond explicit lexical improvements

## Working Style

- Investigate the exact code path that produces the behavior before proposing changes
- Prefer narrow, reversible edits inside the current phase boundary
- Do not rename files, move packages, or refactor structure unless the task
  explicitly requires it
- Keep retrieval changes local to `src/rag/retrieval/` and `src/rag/cli/`
  unless shared code truly needs adjustment
- When adding retrieval behavior, preserve inspectability:
  scoring details, ordering basis, and provenance should remain obvious

## Validation

- After retrieval changes, run the smallest useful retrieval tests first
- If the change touches shared behavior, run the broader unittest suite after
  the focused checks
- Prefer explicit CLI smoke tests for user-facing retrieval behavior
- If a command depends on package resolution, run it in the intended local
  environment and set `PYTHONPATH=src` when needed

## Dependencies

- Prefer standard-library solutions unless a dependency is already justified
- Do not add retrieval infrastructure dependencies just to improve ranking
- Keep lexical improvements local and inspectable before considering anything
  embedding-related

## Documentation And Comments

- Keep `docs/phase-2-retrieval-objectives.md` aligned with implemented retrieval behavior
- Update documentation when behavior changes become user-visible
- Add comments where retrieval flow, ranking logic, chronology rules, or
  provenance handling would otherwise be ambiguous
- Do not perform broad comment-only rewrites unless explicitly requested

## Review Priorities

- Protect the Phase 1 normalization contract
- Catch retrieval regressions in ranking order, chronology handling, filters,
  provenance, and CLI behavior
- Prioritize deterministic behavior, readable failure modes, and stale repo
  guidance over stylistic cleanup
