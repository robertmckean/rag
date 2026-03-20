"""Central project paths and runtime settings."""

from pathlib import Path


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
