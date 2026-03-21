# Project File Map — v0.4.3 Snapshot

Generated: 2026-03-21

## Summary

- 64 Python files across src/ and tests/
- ~9,200 lines of Python code
- 87 tests (unit + integration)
- 8 major modules: normalize, storage, inspection, analysis, retrieval, embeddings, answering, eval
- 10 CLI commands
- 15+ documentation files

---

## 1. Normalization (Phase 1 — frozen)

Ingests raw ChatGPT and Claude exports, produces canonical JSONL artifacts per run.

| File | Lines | Purpose |
|------|-------|---------|
| `src/rag/normalize/__init__.py` | 5 | Package init |
| `src/rag/normalize/canonical_schema.py` | 149 | Frozen dataclasses for conversations and messages |
| `src/rag/normalize/identifiers.py` | 40 | Builds canonical IDs like `chatgpt:conversation:...` |
| `src/rag/normalize/timestamps.py` | 50 | Normalizes provider timestamps to UTC ISO-8601 |
| `src/rag/normalize/chatgpt.py` | 405 | ChatGPT extraction from sharded conversations JSON |
| `src/rag/normalize/claude.py` | 263 | Claude extraction from conversations.json export |
| `src/rag/normalize/chatgpt_run.py` | 77 | Writes ChatGPT-only normalized run with manifest |
| `src/rag/normalize/claude_run.py` | 70 | Writes Claude-only normalized run with manifest |
| `src/rag/normalize/combined_run.py` | 137 | Multi-provider run writer, merges both extractors |

Output: `data/normalized/runs/<run_id>/` containing `conversations.jsonl`, `messages.jsonl`, `manifest.json`

---

## 2. Storage

| File | Lines | Purpose |
|------|-------|---------|
| `src/rag/storage/__init__.py` | 5 | Package init |
| `src/rag/storage/jsonl_writer.py` | 20 | Deterministic JSONL writing with sorted keys |

---

## 3. Inspection

| File | Lines | Purpose |
|------|-------|---------|
| `src/rag/inspection/__init__.py` | 5 | Package init |
| `src/rag/inspection/inventory.py` | 72 | Checks raw export folders for readiness |

---

## 4. Analysis

| File | Lines | Purpose |
|------|-------|---------|
| `src/rag/analysis/__init__.py` | 5 | Package init |
| `src/rag/analysis/message_quality.py` | 314 | Reports empty, low-signal, and malformed messages |

---

## 5. Retrieval (Phase 2)

Loads normalized runs and ranks messages by BM25, semantic similarity, or hybrid.

| File | Lines | Purpose |
|------|-------|---------|
| `src/rag/retrieval/__init__.py` | 5 | Package init |
| `src/rag/retrieval/read_model.py` | 158 | Loads normalized JSONL into `LoadedRun` dataclass with lookup dicts and searchable text |
| `src/rag/retrieval/utils.py` | 11 | `string_or_none` helper |
| `src/rag/retrieval/lexical.py` | 1038 | **Largest file.** BM25 scoring, semantic retrieval, hybrid merge, query parsing, 5 retrieval modes, contextual window building, timeline view, recency boost, filter matching |

Key constants in `lexical.py`: `BM25_K1=1.5`, `BM25_B=0.75`, `EXACT_PHRASE_BOOST=1.5`, `TITLE_TERM_BOOST=0.35`, `RECENCY_BOOST_MAX=0.35`, `RETRIEVAL_CHANNELS=("bm25","semantic","hybrid")`

---

## 6. Embeddings (Phase 4A)

Builds and stores message-level embedding vectors for semantic retrieval.

| File | Lines | Purpose |
|------|-------|---------|
| `src/rag/embeddings/__init__.py` | 2 | Package init |
| `src/rag/embeddings/builder.py` | 469 | Embedding generation: filters messages, batches API calls, buffers in memory, writes atomically |
| `src/rag/embeddings/store.py` | 195 | JSONL artifact read/write for `EmbeddingRecord`, atomic write via temp file + rename |
| `src/rag/embeddings/similarity.py` | 18 | Cosine similarity with zero-vector safety |

Output: `data/normalized/runs/<run_id>/message_embeddings.jsonl` (gitignored, generated per machine)

Eligibility filters (embed-time only): empty text, tool role, low-information acknowledgments, trivially short

---

## 7. Answering (Phase 3A/3B)

Deterministic grounded answers with evidence qualification and status classification.

| File | Lines | Purpose |
|------|-------|---------|
| `src/rag/answering/models.py` | 156 | `AnswerResult`, `EvidenceItem`, `Citation`, `AnswerStatus` enum, diagnostics dataclasses |
| `src/rag/answering/answer.py` | 1120 | **Second largest file.** Evidence qualification, status assignment, composed support, gap wording, debug payloads, LLM synthesis orchestration |
| `src/rag/answering/generator_llm.py` | 206 | Constrained OpenAI LLM rewrite with entity/date hallucination detection, falls back to deterministic on any failure |

Answer statuses: `supported`, `partially_supported`, `ambiguous`, `insufficient_evidence`

Grounding modes: `strict` (default), `conversational_memory` (opt-in, same-window composition)

---

## 8. Evaluation (Phase 3A)

Benchmarks answer quality against a hand-built query bank.

| File | Lines | Purpose |
|------|-------|---------|
| `src/rag/eval/models.py` | 80 | `EvalCase`, `EvalCaseResult`, `FailureType` enum |
| `src/rag/eval/metrics.py` | 204 | Rule-based failure classification and evidence qualification checks |
| `src/rag/eval/runner.py` | 157 | Runs answer pipeline against benchmark bank, produces per-case reports |

---

## 9. CLI Commands

All CLIs follow the same pattern: `python -m rag.cli.<name> --run-dir ... [options]`

| File | Lines | Command | Purpose |
|------|-------|---------|---------|
| `src/rag/cli/normalize_chatgpt.py` | 55 | `normalize_chatgpt` | ChatGPT-only normalization |
| `src/rag/cli/normalize_claude.py` | 55 | `normalize_claude` | Claude-only normalization |
| `src/rag/cli/normalize_all.py` | 62 | `normalize_all` | Combined multi-provider normalization |
| `src/rag/cli/inspect_exports.py` | 63 | `inspect_exports` | Check raw export readiness |
| `src/rag/cli/analyze_messages.py` | 56 | `analyze_messages` | Message quality analysis |
| `src/rag/cli/retrieve.py` | 235 | `retrieve` | Search with `--channel bm25\|semantic\|hybrid`, `--mode relevance\|newest\|oldest\|relevance_recency\|timeline` |
| `src/rag/cli/build_embeddings.py` | 90 | `build_embeddings` | Generate embedding artifact with `--model`, `--batch-size`, `--conversation-id` |
| `src/rag/cli/answer.py` | 139 | `answer` | Grounded answer with `--grounding-mode`, `--debug-qualification`, `--llm` |
| `src/rag/cli/eval.py` | 68 | `eval` | Run benchmark eval against query bank |

---

## 10. Configuration

| File | Lines | Purpose |
|------|-------|---------|
| `src/rag/config.py` | 54 | Project paths, data directories, runtime settings |
| `src/rag/__init__.py` | 5 | Package init |
| `requirements.txt` | 72 | Python dependencies (openai, pandas, numpy, torch, tiktoken, etc.) |
| `.gitignore` | 38 | Excludes data, caches, embedding artifacts, IDE files |

---

## 11. Tests

### Unit Tests

| File | Lines | What it tests |
|------|-------|---------------|
| `tests/unit/test_identifiers.py` | 44 | Canonical ID generation |
| `tests/unit/test_timestamps.py` | 28 | Timestamp normalization |
| `tests/unit/test_canonical_schema.py` | 82 | Schema dataclass construction |
| `tests/unit/test_jsonl_writer.py` | 22 | JSONL writing |
| `tests/unit/test_inventory.py` | 36 | Export readiness checks |
| `tests/unit/test_chatgpt_extraction_unittest.py` | 125 | ChatGPT extraction logic |
| `tests/unit/test_claude_extraction_unittest.py` | 74 | Claude extraction logic |
| `tests/unit/test_retrieval_lexical_unittest.py` | 413 | BM25 ranking, modes, filters, query parsing, timeline (27 tests) |
| `tests/unit/test_retrieval_read_model_unittest.py` | 40 | Run loading into read model |
| `tests/unit/test_retrieval_semantic_unittest.py` | 204 | Semantic retrieval, hybrid merge, provenance tracking |
| `tests/unit/test_retrieve_cli_unittest.py` | 178 | Retrieval CLI argument parsing and output |
| `tests/unit/test_embeddings_builder_unittest.py` | 334 | Embedding build, filtering, targeting, atomic write, sampling |
| `tests/unit/test_answering_unittest.py` | 566 | Evidence qualification, status assignment, LLM synthesis, composed support |
| `tests/unit/test_answer_cli_unittest.py` | 189 | Answer CLI flags, debug qualification, JSON output |
| `tests/unit/test_eval_metrics_unittest.py` | 149 | Eval metrics and failure classification |
| `tests/unit/test_eval_runner_unittest.py` | 90 | Eval runner benchmark execution |
| `tests/unit/test_unittest_discovery_unittest.py` | 37 | Test discovery metadata |

### Integration Tests

| File | Lines | What it tests |
|------|-------|---------------|
| `tests/integration/test_chatgpt_run_unittest.py` | 96 | End-to-end ChatGPT normalization |
| `tests/integration/test_claude_run_unittest.py` | 92 | End-to-end Claude normalization |
| `tests/integration/test_combined_run_unittest.py` | 133 | End-to-end combined normalization |
| `tests/integration/test_inspect_exports_cli.py` | 26 | Export inspection CLI |

### Test Fixtures

| Directory | Contents |
|-----------|----------|
| `tests/fixtures/chatgpt/` | Sharded sample ChatGPT exports (conversations-000.json, conversations-001.json) |
| `tests/fixtures/claude/` | Sample Claude export (conversations.json) plus full bundle |
| `tests/fixtures/retrieval/sample_run/` | 2 conversations, 7 messages — used by retrieval and embedding tests |
| `tests/fixtures/answering/sample_run/` | Normalized run for answer qualification tests |
| `tests/fixtures/eval/query_bank.json` | Benchmark eval cases |

---

## 12. Documentation

| File | Purpose |
|------|---------|
| `README.md` (266 lines) | Project overview, CLI examples, phase status |
| `AGENTS.md` (126 lines) | Coding agent policy: structure, boundaries, working style |
| `CLAUDE.md` (6 lines) | Points to AGENTS.md |
| `docs/project_summary.md` | Scope overview of all active layers |
| `docs/known_limitations.md` | BM25 vocabulary mismatch, retrieval boundary |
| `docs/phase-2-retrieval-objectives.md` | Phase 2 design objectives |
| `docs/phase_3_grounded_answers.md` | Phase 3A grounded answer design |
| `docs/schemas/canonical_schema.md` | Canonical schema spec |
| `docs/testing.md` | Test runner notes |
| `docs/phase1_policy.md` | Phase 1 normalization policy |
| `docs/rerun_guide.md` | How to rerun normalization |

## 13. Changelogs

| File | Release |
|------|---------|
| `changelogs/CHANGELOG_v0.1.0.md` | Initial normalization |
| `changelogs/CHANGELOG_v0.3.1.md` | BM25, query normalization, modes |
| `changelogs/CHANGELOG_v0.3.2.md` | Timeline mode fix |
| `changelogs/CHANGELOG_v0.4.0.md` | Phase 3A grounded answers |
| `changelogs/CHANGELOG_v0.4.1.md` | Conversational memory, LLM synthesis |
| `changelogs/CHANGELOG_v0.4.2.md` | Qualification diagnostics, BM25 baseline |
| `changelogs/CHANGELOG_v0.4.3.md` | Hybrid retrieval, embedding builds |

## 14. Planning

| File | Purpose |
|------|---------|
| `planning/phase_3c_conversational_memory_spec.md` | Conversational memory grounding mode design |
| `planning/phase_4a_hybrid_retrieval.md` | Hybrid retrieval (BM25 + embeddings) design |
| `planning/Phase_4B_Result_DeDuplication.md` | Result deduplication planning |

---

## Module Dependency Flow

```
raw exports
    ↓
normalize/ → storage/jsonl_writer → data/normalized/runs/<run_id>/
    ↓
retrieval/read_model → loads JSONL into memory
    ↓
retrieval/lexical → BM25 scoring + window building
    ↓
embeddings/builder → calls OpenAI API → embeddings/store → message_embeddings.jsonl
    ↓
retrieval/lexical → semantic scoring + hybrid merge (imports from embeddings/)
    ↓
answering/answer → evidence qualification + status classification
    ↓
answering/generator_llm → optional LLM rewrite (Phase 3B)
    ↓
eval/ → benchmark answer quality
    ↓
cli/ → user-facing entry points for all of the above
```

---

## Known Structural Issues (from code review)

1. `lexical.py` (1038 lines) has too many responsibilities — BM25, semantic, hybrid, parsing, windows, modes should be separate files
2. `answer.py` (1120 lines) similarly overloaded — qualification, status, diagnostics, composition should split
3. `_safe_print` duplicated in 4 files — should be one shared utility
4. `string_or_none` in `retrieval/utils.py` but imported by embeddings — should be in `rag/utils.py`
5. No `__init__.py` in some test directories — `unittest discover` may miss tests without explicit module names
