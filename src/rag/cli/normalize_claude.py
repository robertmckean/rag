"""CLI for Claude-only normalized run writing."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.rag.config import NORMALIZED_RUNS_DIR
from src.rag.normalize.claude_run import write_claude_normalized_run


# This wrapper limits itself to argument parsing and handing control to the run writer.
# The CLI exists so local runs can target a specific export file and output root.
# All Claude extraction and manifest behavior remains in the library layer.

# Build the parser for the Claude-only normalization command.
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Write a Claude-only normalized run from the raw conversations export."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to Claude conversations.json.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=NORMALIZED_RUNS_DIR,
        help="Directory beneath which the run_id folder will be created.",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Optional explicit run id for deterministic output paths.",
    )
    return parser


# Parse CLI arguments, write the run, and print the created run directory.
def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_dir = write_claude_normalized_run(
        args.input.resolve(),
        args.output_root.resolve(),
        run_id=args.run_id,
    )
    print(f"run_dir: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
