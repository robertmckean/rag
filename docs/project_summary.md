# rag

Personal RAG system built from exported ChatGPT and Claude histories.

## Current Scope

The repository now has three active layers:
- Phase 1 normalization:
  raw export inspection, canonical conversation/message schema, provider-specific extraction, deterministic JSONL outputs, and immutable run manifests
- Phase 2 retrieval:
  local lexical retrieval over one normalized run using BM25 scoring, contextual message windows, chronological retrieval modes, and timeline exploration
- Phase 3A grounded answers:
  deterministic answer generation, explicit answer status, citation assembly,
  deterministic benchmark evaluation, and an opt-in `conversational_memory`
  grounding mode for same-window local evidence composition

## Release Baseline

The current release line is the last pure-BM25 retrieval baseline before hybrid
retrieval work.

- active retrieval method: BM25 lexical retrieval
- answering/grounding: stable and deterministic on top of retrieval
- next planned retrieval step: hybrid BM25 + embeddings

Known example of the current lexical boundary:
- query: `What have I said about Larry's guitar playing?`
- real retrieved evidence mentions Larry and `playing the bass`
- lexical grounding correctly remains insufficient because `guitar` is not
  present in the retrieved evidence

See `docs/known_limitations.md` for the documented failure mode.

## Source Of Truth

Normalized outputs are immutable per run under:

```text
data/normalized/runs/<run_id>/
  conversations.jsonl
  messages.jsonl
  manifest.json
```

Phase 2 retrieval works directly against those artifacts and does not mutate them.

## Current Retrieval Modes

- `relevance`
- `newest`
- `oldest`
- `relevance_recency`
- `timeline`

`timeline` is a compact oldest-first chronology view across conversations. The
other modes return contextual message windows around focal matches.

## Current Non-Goals

- embeddings
- vector databases
- UI implementation
- cross-run retrieval
- semantic search beyond the current lexical retrieval layer
- LLM-backed answer generation
- autonomous follow-up questions
- answer verification loops

Near-term next step after this baseline:
- introduce embeddings as a second retrieval channel without removing BM25

## Current Answering/Eval Scope

- `rag.cli.answer` produces deterministic grounded answers for a single run
- `rag.cli.answer` supports `--grounding-mode strict|conversational_memory`
- answer statuses are:
  - `supported`
  - `partially_supported`
  - `ambiguous`
  - `insufficient_evidence`
- `rag.cli.eval` runs a benchmark query bank against the real answer pipeline
- eval output reports status accuracy, failure types, and per-case results
