# rag

Personal RAG system built from exported ChatGPT and Claude histories.

## Phase 1 Layout

- `src/rag/`: project package and future CLI, inspection, parsing, normalization, and storage modules
- `data/raw/`: immutable source exports grouped by provider
- `data/normalized/runs/`: timestamped normalization outputs for each ingest run
- `docs/`: project documentation and schema notes
- `tests/`: fixtures, unit tests, and integration tests
- `tools/`: local PowerShell and launcher helpers

## Environment

The local workflow is PowerShell-first and uses the Conda environment named in `.conda-env.txt`.

Current project configuration lives in `src/rag/config.py`. The repo-root `config.py` remains as a temporary compatibility shim.

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
