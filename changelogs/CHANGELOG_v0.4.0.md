## v0.4.0 - 2026-03-20

Scope:
- add Phase 3A grounded answers and a deterministic evaluation harness on top of the frozen normalization and retrieval layers

Included:
- add deterministic grounded-answer models and answer pipeline
- add `rag.cli.answer` with terminal rendering, JSON output, and bounded evidence selection
- tighten answer evidence qualification for entity-specific and multi-part queries
- add deterministic benchmark eval models, metrics, runner, and `rag.cli.eval`
- add compact answer and eval fixture banks plus focused unit tests
- keep Phase 1 normalized artifacts and Phase 2 retrieval contracts unchanged
- include current Phase 3 planning notes under `docs/`

Validation:
- `PYTHONPATH=src python -m unittest discover -s tests -v`
