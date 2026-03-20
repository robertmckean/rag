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

## Phase 4A Hybrid Retrieval

The retrieval layer now supports three channels over a single normalized run:
- `bm25`
- `semantic`
- `hybrid`

Phase 4A keeps the system file-based and local:
- one embedding artifact per normalized run
- one embedding per normalized message
- cosine similarity over stored vectors
- no vector database
- no ANN index
- no reranker

Hybrid retrieval is a union of BM25 and semantic candidates with explicit
provenance:
- `bm25`
- `semantic`
- `both`

Grounding remains unchanged:
- retrieval can broaden recall
- answer qualification and status assignment remain strict

## Current Retrieval Boundary

Hybrid retrieval is now implemented, but BM25 remains the lexical baseline and
semantic retrieval only helps when the embedding artifact actually covers the
relevant messages.

Known example of the lexical boundary:
- query: `What have I said about Larry's guitar playing?`
- real retrieved evidence mentions Larry and `playing the bass`
- BM25 alone can miss semantically related phrasing when wording diverges
- semantic and hybrid retrieval improve only after the Larry-related messages
  are actually embedded into the run-local artifact

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

- vector databases
- UI implementation
- cross-run retrieval
- ANN indexing
- reranking
- autonomous follow-up questions
- answer verification loops

Near-term next step after this baseline:
- improve embedding-build robustness and retrieval-quality evaluation without
  replacing BM25

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
