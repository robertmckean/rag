"""CLI for ChatGPT-only normalized run writing."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.rag.config import NORMALIZED_RUNS_DIR
from src.rag.normalize.chatgpt_run import write_chatgpt_normalized_run


# This wrapper limits itself to argument parsing and handing control to the run writer.
# ChatGPT-specific graph handling stays in the normalization module, not the CLI.
# The output is a single run directory whose path is printed for follow-up inspection.

# Build the parser for the ChatGPT-only normalization command.
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Write a ChatGPT-only normalized run from the raw conversation shard export."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        required=True,
        help="Path to the ChatGPT export root containing conversations-*.json shards.",
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
    run_dir = write_chatgpt_normalized_run(
        args.input_root.resolve(),
        args.output_root.resolve(),
        run_id=args.run_id,
    )
    print(f"run_dir: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
