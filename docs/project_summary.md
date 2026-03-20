# rag

Personal RAG system built from exported ChatGPT and Claude histories.

## Current Scope

The repository has two active layers:
- Phase 1 normalization:
  raw export inspection, canonical conversation/message schema, provider-specific extraction, deterministic JSONL outputs, and immutable run manifests
- Phase 2 retrieval:
  local lexical retrieval over one normalized run using BM25 scoring, contextual message windows, chronological retrieval modes, and timeline exploration

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
- answer-generation layer
- cross-run retrieval
- semantic search beyond the current lexical retrieval layer
