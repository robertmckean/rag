# v0.1.0 - Phase 1 Normalization Pipeline

Date: 2026-03-20

## Implements

- ChatGPT and Claude raw export inspection
- canonical conversation/message schema
- provider-specific extraction for ChatGPT and Claude
- deterministic JSONL outputs and manifest writing
- combined multi-provider normalization orchestration
- documentation for schema, policy, rerun flow, and testing
- read-only normalized-message quality analysis

## Intentionally Excludes

- retrieval
- indexing
- embeddings
- vector storage
- search
- retrieval-oriented message filtering
- attachment blob copying

## Known Limitations

- canonical output may preserve low-signal structural records
- empty system/tool messages and empty shell records may appear in canonical output
- this is expected phase-1 behavior, not a sampled extraction failure
- ChatGPT normalization includes only the visible `current_node` chain; alternate branches are excluded
- attachment handling is reference-only

## Rerun Commands

Inspect readiness:

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
  '--run-id', 'claude-rerun'
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
  '--run-id', 'chatgpt-rerun'
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
  '--run-id', 'combined-rerun'
]))
'@ | python -
```
