# Testing Notes

## Current Test Runner

The implemented phase-1 pipeline is currently exercised with Python `unittest`.

Executable command:

```powershell
python -m unittest tests.integration.test_combined_run_unittest tests.integration.test_chatgpt_run_unittest tests.integration.test_claude_run_unittest tests.unit.test_chatgpt_extraction_unittest tests.unit.test_claude_extraction_unittest
```

## Pytest Status

There are earlier test files in the repo that follow a pytest style, but this environment does not currently have `pytest` installed.

So the practical phase-1 validation path is:
- use `unittest` for the implemented extraction and run-writing pipeline
- add `pytest` only if you intentionally want to standardize the whole suite around it

## Validation Strategy

Phase 1 validation currently covers:
- raw input inspection
- Claude extraction
- ChatGPT extraction
- Claude-only run writing
- ChatGPT-only run writing
- combined multi-provider run writing

For code changes, prefer the smallest useful `unittest` command that covers the touched slice before rerunning the full combined suite.
