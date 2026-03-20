# Rerun Guide

## 1. Place New Raw Exports

Put provider exports in these locations:
- ChatGPT bundle under `data/raw/chatgpt/History_ChatGPT/`
- Claude conversation export at `data/raw/claude/History_Claude/conversations.json`

Keep raw exports immutable once placed.

## 2. Inspect Readiness

```powershell
@'
from src.rag.cli.inspect_exports import main
raise SystemExit(main(['--input', 'data/raw']))
'@ | python -
```

This reports:
- provider folder status
- non-hidden file counts
- top-level entries
- discovered files

## 3. Run Provider-Specific Normalization

Claude only:

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

ChatGPT only:

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

## 4. Run Combined Normalization

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

## 5. Review Outputs

Each run writes:

```text
data/normalized/runs/<run_id>/
  conversations.jsonl
  messages.jsonl
  manifest.json
```

Review `manifest.json` first to confirm:
- providers present
- counts
- exclusions applied
- source roots

When reviewing outputs, expect some low-signal structural records in `messages.jsonl`.
Examples include:
- empty system/tool scaffold messages from ChatGPT
- empty shell records where the provider export itself contains no meaningful text

This is expected phase-1 behavior. Retrieval-focused filtering is deferred to a later phase.

## 6. Run Tests

Current executable test command:

```powershell
python -m unittest tests.integration.test_combined_run_unittest tests.integration.test_chatgpt_run_unittest tests.integration.test_claude_run_unittest tests.unit.test_chatgpt_extraction_unittest tests.unit.test_claude_extraction_unittest
```
