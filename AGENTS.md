# AGENTS.md

This file provides guidance to Codex and other coding agents when working in
this repository.

## Project Goal

Build a personal RAG system over exported ChatGPT and Claude histories. The
current codebase ingests raw exports into canonical normalized records, adds a
local lexical retrieval layer, supports deterministic grounded answers plus
deterministic answer evaluation, reconstructs grounded narratives from
retrieved evidence, and extracts recurring-entity patterns across narratives.

## Project Status

- Active milestone: Phase 13C evolution router integration (see TO_DO.md)
- Phase 1 normalization is complete and its output contract is frozen
- Phase 2 currently supports BM25 lexical retrieval, semantic retrieval over
  run-local message embeddings, hybrid retrieval, contextual window results,
  timeline exploration, query normalization, and chronological retrieval modes
- Phase 3A currently supports deterministic grounded answers, answer-status
  classification, citation assembly, and a benchmark eval harness
- Phase 3A supports an opt-in `conversational_memory` grounding mode for
  same-window local evidence composition
- Phase 3B supports constrained LLM-backed answer synthesis via `--llm`
- Phase 6 supports grounded narrative reconstruction from retrieved evidence
  with configurable phase grouping, transition detection, gap detection, and
  limitation reporting
- Phase 7 supports recurring-entity pattern extraction across narratives with
  explicit alias normalization, content snippets, deterministic topic clustering
  via agglomerative term-overlap merging, and deterministic output
- Hybrid retrieval is now the active retrieval extension path; BM25 remains the
  lexical baseline and semantic retrieval is additive rather than a replacement
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
- `src/rag/answering/`
  Phase 3A grounded answer models and deterministic answer pipeline
- `src/rag/eval/`
  deterministic benchmark metrics and eval runner for grounded answers
- `src/rag/narrative/`
  Phase 6 grounded narrative reconstruction from retrieved evidence
- `src/rag/patterns/`
  Phase 7 recurring-entity pattern extraction with alias normalization and
  deterministic topic clustering
- `src/rag/cli/`
  CLI entry points for inspection, normalization, analysis, retrieval, answers,
  evals, narrative reconstruction, and pattern extraction
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
- Phase 3A builds on Phase 2 retrieval without changing retrieval semantics
- Phase 6 works against Phase 2 retrieval + Phase 3A evidence qualification
- Phase 7 works against Phase 6 narrative reconstructions
- Current non-goals:
  vector DBs, UI, cross-run retrieval, rerankers, chatbot behavior, and
  semantic answer-grounding by LLM

## Working Style

- Investigate the exact code path that produces the behavior before proposing changes
- Prefer narrow, reversible edits inside the current phase boundary
- Do not rename files, move packages, or refactor structure unless the task
  explicitly requires it
- Keep retrieval changes local to `src/rag/retrieval/` and `src/rag/cli/`
  unless shared code truly needs adjustment
- Keep answer/eval changes local to `src/rag/answering/`, `src/rag/eval/`,
  and `src/rag/cli/` unless shared code truly needs adjustment
- When adding retrieval behavior, preserve inspectability:
  scoring details, ordering basis, and provenance should remain obvious
- When adding grounded-answer behavior, preserve deterministic evidence
  qualification, explicit status rules, and citation traceability

## Validation

- After retrieval changes, run the smallest useful retrieval tests first
- After grounded-answer or eval changes, run focused answer/eval tests first
- If the change touches shared behavior, run the broader unittest suite after
  the focused checks
- Prefer explicit CLI smoke tests for user-facing retrieval behavior
- Prefer explicit CLI smoke tests for user-facing answer/eval behavior
- If a command depends on package resolution, run it in the intended local
  environment and set `PYTHONPATH=src` when needed

## Dependencies

- Prefer standard-library solutions unless a dependency is already justified
- Do not add retrieval infrastructure dependencies just to improve ranking
- Keep embedding retrieval file-based and inspectable before considering
  heavier search infrastructure

## Documentation And Comments

- Keep `docs/phase-2-retrieval-objectives.md` aligned with implemented retrieval behavior
- Keep Phase 3A answer/eval docs aligned with the current deterministic answer contract
- Update documentation when behavior changes become user-visible
- Add comments where retrieval flow, ranking logic, chronology rules, or
  provenance handling would otherwise be ambiguous
- Do not perform broad comment-only rewrites unless explicitly requested

## Review Priorities

- Protect the Phase 1 normalization contract
- Catch retrieval regressions in ranking order, chronology handling, filters,
  provenance, and CLI behavior
- Catch answer/eval regressions in evidence qualification, status assignment,
  citation integrity, and deterministic reporting
- Prioritize deterministic behavior, readable failure modes, and stale repo
  guidance over stylistic cleanup
