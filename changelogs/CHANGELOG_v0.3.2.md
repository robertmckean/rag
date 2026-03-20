## v0.3.2 - 2026-03-20

Scope:
- stabilize the Phase 2 retrieval API and test workflow without changing Phase 1 artifacts

Included:
- make standard `unittest` discovery work under `tests/`
- reject `mode="timeline"` at the window-oriented library API boundary with explicit guidance
- keep CLI `--mode timeline` behavior aligned with the dedicated timeline path
- remove dead excerpt logic and extract shared retrieval string normalization
- add focused tests for missing timestamps, invalid timeline mode, CLI `--json-out`, and discovery
- ignore interrupted test temp directories more explicitly
- remove the unused repo-root `config.py` compatibility shim

Validation:
- `PYTHONPATH=src python -m unittest discover -s tests -v`
