"""Temporary compatibility shim; src.rag.config is the canonical source."""

from src.rag.config import (
    DATA_DIR,
    DEFAULT_INPUT_PATH,
    DEFAULT_OUTPUT_PATH,
    DOCS_DIR,
    FILES_DIR,
    MODELS_DIR,
    NORMALIZED_DATA_DIR,
    NORMALIZED_RUNS_DIR,
    PACKAGE_DIR,
    PROJECT_NAME,
    PROJECT_ROOT,
    RAW_DATA_DIR,
    RESULTS_DIR,
    SRC_DIR,
    ensure_project_dirs,
)

__all__ = [
    "DATA_DIR",
    "DEFAULT_INPUT_PATH",
    "DEFAULT_OUTPUT_PATH",
    "DOCS_DIR",
    "FILES_DIR",
    "MODELS_DIR",
    "NORMALIZED_DATA_DIR",
    "NORMALIZED_RUNS_DIR",
    "PACKAGE_DIR",
    "PROJECT_NAME",
    "PROJECT_ROOT",
    "RAW_DATA_DIR",
    "RESULTS_DIR",
    "SRC_DIR",
    "ensure_project_dirs",
]
