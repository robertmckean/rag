"""CLI for Claude-only normalized run writing."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.rag.config import NORMALIZED_RUNS_DIR
from src.rag.normalize.claude_run import write_claude_normalized_run


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
