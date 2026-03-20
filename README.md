# rag

Personal RAG system built from exported ChatGPT and Claude histories.

## Current Layout

- `src/rag/`: project package and future CLI, inspection, parsing, normalization, and storage modules
- `data/raw/`: immutable source exports grouped by provider
- `data/normalized/runs/`: timestamped normalization outputs for each ingest run
- `docs/`: project documentation and schema notes
- `tests/`: fixtures, unit tests, and integration tests
- `tools/`: local PowerShell and launcher helpers

## Environment

The local workflow is PowerShell-first and uses the Conda environment named in `.conda-env.txt`.

Current project configuration lives in `src/rag/config.py`.

For Python validation in this repo, the practical command pattern is:

```powershell
$env:PYTHONPATH='src'; python -m unittest ...
```

`tools/run_in_env.ps1` exists for script-path entry points, but it may be blocked in constrained PowerShell sessions during Conda activation.

## Phase 1 Pipeline

Raw inputs:
- `data/raw/chatgpt/History_ChatGPT/`
- `data/raw/claude/History_Claude/conversations.json`

Inspection CLI:
```powershell
@'
from src.rag.cli.inspect_exports import main
raise SystemExit(main(['--input', 'data/raw']))
'@ | python -
```

Claude-only normalization:
```powershell
@'
from src.rag.cli.normalize_claude import main
raise SystemExit(main([
  '--input', 'data/raw/claude/History_Claude/conversations.json',
  '--output-root', 'data/normalized/runs',
  '--run-id', 'claude-run'
]))
'@ | python -
```

ChatGPT-only normalization:
```powershell
@'
from src.rag.cli.normalize_chatgpt import main
raise SystemExit(main([
  '--input-root', 'data/raw/chatgpt/History_ChatGPT',
  '--output-root', 'data/normalized/runs',
  '--run-id', 'chatgpt-run'
]))
'@ | python -
```

Combined normalization:
```powershell
@'
from src.rag.cli.normalize_all import main
raise SystemExit(main([
  '--chatgpt-input-root', 'data/raw/chatgpt/History_ChatGPT',
  '--claude-input', 'data/raw/claude/History_Claude/conversations.json',
  '--output-root', 'data/normalized/runs',
  '--run-id', 'combined-run'
]))
'@ | python -
```

Normalized output layout:
```text
data/normalized/runs/<run_id>/
  conversations.jsonl
  messages.jsonl
  manifest.json
```

Additional phase-1 docs:
- `docs/schemas/canonical_schema.md`
- `docs/phase1_policy.md`
- `docs/rerun_guide.md`
- `docs/testing.md`

Known phase-1 limitations:
- canonical output may preserve low-signal structural records such as empty system, tool, or shell messages
- this is expected phase-1 behavior and not treated as an extraction failure
- ChatGPT normalization includes only the visible `current_node` chain and excludes alternate branches
- attachment handling is reference-only
- retrieval-oriented filtering is deferred to a later phase

## Phase 2 Retrieval

Current retrieval capabilities:
- BM25-based lexical message ranking
- contextual message-window retrieval
- retrieval modes:
  - `relevance`
  - `newest`
  - `oldest`
  - `relevance_recency`
  - `timeline`
- query normalization with quoted-phrase support
- timeline exploration across conversations within one normalized run

Current retrieval boundary:
- retrieval is still lexical only
- BM25 is the active retrieval method
- there is no embedding or semantic retrieval channel in this version

Known lexical limitation:
- vocabulary mismatch can fail even when the topic is relevant
- real example:
  - query: `What have I said about Larry's guitar playing?`
  - retrieved run evidence includes Larry plus `playing the bass`
  - qualification correctly remains `insufficient_evidence` because `guitar`
    is not grounded in the retrieved evidence
- see `docs/known_limitations.md`

Next retrieval milestone:
- keep BM25 as the baseline lexical channel
- add embeddings as a second retrieval channel
- use this pure-BM25 release as the regression baseline for hybrid retrieval

Retrieval CLI:
```powershell
$env:PYTHONPATH='src'; @'
from rag.cli.retrieve import main
raise SystemExit(main([
  '--run-dir', 'data/normalized/runs/<run_id>',
  '--query', 'project',
  '--mode', 'timeline',
  '--limit', '10'
]))
'@ | python -
```

Timeline mode returns compact chronological focal-message entries across conversations.
The other retrieval modes return contextual message windows.

## Phase 3A Grounded Answers

Current grounded-answer capabilities:
- deterministic answer generation over a single normalized run
- explicit grounding modes:
  - `strict` (default)
  - `conversational_memory` for same-window local evidence composition
- explicit answer status:
  - `supported`
  - `partially_supported`
  - `ambiguous`
  - `insufficient_evidence`
- bounded evidence selection and citation assembly
- deterministic answer evaluation against a benchmark query bank

Grounded answer CLI:
```powershell
$env:PYTHONPATH='src'; @'
from rag.cli.answer import main
raise SystemExit(main([
  '--run-dir', 'data/normalized/runs/<run_id>',
  '--query', 'What have I said about burnout?',
  '--retrieval-mode', 'relevance',
  '--grounding-mode', 'strict',
  '--limit', '8',
  '--max-evidence', '5'
]))
'@ | python -
```

Eval CLI:
```powershell
$env:PYTHONPATH='src'; @'
from rag.cli.eval import main
raise SystemExit(main([
  '--run-dir', 'data/normalized/runs/<run_id>',
  '--bench', 'tests/fixtures/eval/query_bank.json'
]))
'@ | python -
```

Grounded-answer constraints:
- retrieval remains the source of available evidence
- answer generation is deterministic and template-based
- conversational-memory composition is opt-in and stays within one retrieved window
- no LLM calls, embeddings, vector DBs, or cross-run federation in this phase

Current release status:
- retrieval plus grounding is stable and testable end to end
- Phase 3 grounded answering is functionally complete on top of the current
  pure-BM25 retrieval layer
