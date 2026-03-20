"""Central project paths and runtime settings."""

from pathlib import Path


# All repo paths are derived from this file location to keep local execution stable.
# Centralizing path constants avoids subtle mismatches across CLI and normalization modules.
# The config module is also the compatibility target for the repo-root shim.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_DIR = Path(__file__).resolve().parent
SRC_DIR = PACKAGE_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
NORMALIZED_DATA_DIR = DATA_DIR / "normalized"
NORMALIZED_RUNS_DIR = NORMALIZED_DATA_DIR / "runs"
DOCS_DIR = PROJECT_ROOT / "docs"
MODELS_DIR = PROJECT_ROOT / "models"
FILES_DIR = PROJECT_ROOT / "files"
RESULTS_DIR = PROJECT_ROOT / "results"

PROJECT_NAME = PROJECT_ROOT.name
DEFAULT_INPUT_PATH = RAW_DATA_DIR
DEFAULT_OUTPUT_PATH = NORMALIZED_RUNS_DIR


# Create the writable directories that phase-1 commands expect to exist.
# This keeps path setup behavior consistent across scripts and tests.
def ensure_project_dirs() -> None:
    """Create common writable directories used by the project."""
    for path in (
        RAW_DATA_DIR,
        NORMALIZED_RUNS_DIR,
        DOCS_DIR,
        MODELS_DIR,
        FILES_DIR,
        RESULTS_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)
