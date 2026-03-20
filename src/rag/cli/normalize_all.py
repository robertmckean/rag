"""CLI for combined multi-provider normalized run writing."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.rag.config import NORMALIZED_RUNS_DIR
from src.rag.normalize.combined_run import write_combined_normalized_run


# This CLI coordinates one combined run without duplicating provider extraction logic.
# Argument parsing is kept minimal so orchestration behavior stays in the run writer.
# Printing the final run directory gives callers one stable location to inspect afterward.

# Build the parser for the combined normalization command.
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Write a combined ChatGPT and Claude normalized run."
    )
    parser.add_argument(
        "--chatgpt-input-root",
        type=Path,
        required=True,
        help="Path to the ChatGPT export root containing conversations-*.json shards.",
    )
    parser.add_argument(
        "--claude-input",
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


# Parse CLI arguments, write the combined run, and print the created run directory.
def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_dir = write_combined_normalized_run(
        chatgpt_export_root=args.chatgpt_input_root.resolve(),
        claude_conversations_path=args.claude_input.resolve(),
        output_root=args.output_root.resolve(),
        run_id=args.run_id,
    )
    print(f"run_dir: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
