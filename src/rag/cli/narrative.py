"""CLI for narrative reconstruction over one normalized run."""

from __future__ import annotations

import argparse
from pathlib import Path

from rag.cli.utils import safe_print, safe_print_error
from rag.narrative.builder import (
    DEFAULT_GAP_THRESHOLD_DAYS,
    DEFAULT_NARRATIVE_LIMIT,
    DEFAULT_NARRATIVE_MAX_EVIDENCE,
    DEFAULT_PHASE_WINDOW_DAYS,
    build_narrative_from_run,
)
from rag.narrative.renderer import render_debug, render_json, render_text


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reconstruct a grounded narrative from one normalized run."
    )
    parser.add_argument("--run-dir", type=Path, required=True, help="Path to one normalized run directory.")
    parser.add_argument("--query", type=str, required=True, help="Natural-language question to reconstruct.")
    parser.add_argument(
        "--retrieval-mode",
        choices=("relevance", "newest", "oldest"),
        default="relevance",
        help="Retrieval ordering mode.",
    )
    parser.add_argument(
        "--channel",
        choices=("bm25", "semantic", "hybrid"),
        default="bm25",
        help="Retrieval channel.",
    )
    parser.add_argument("--limit", type=int, default=DEFAULT_NARRATIVE_LIMIT, help="Maximum retrieval results.")
    parser.add_argument("--max-evidence", type=int, default=DEFAULT_NARRATIVE_MAX_EVIDENCE, help="Maximum evidence items.")
    parser.add_argument(
        "--phase-window-days",
        type=int,
        default=DEFAULT_PHASE_WINDOW_DAYS,
        help="Maximum days between evidence items in the same phase.",
    )
    parser.add_argument(
        "--gap-threshold-days",
        type=int,
        default=DEFAULT_GAP_THRESHOLD_DAYS,
        help="Minimum gap in days to report as a temporal gap.",
    )
    parser.add_argument(
        "--format",
        choices=("json", "text", "debug"),
        default="json",
        help="Output format.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        narrative = build_narrative_from_run(
            args.run_dir.resolve(),
            args.query,
            retrieval_mode=args.retrieval_mode,
            channel=args.channel,
            limit=args.limit,
            max_evidence=args.max_evidence,
            phase_window_days=args.phase_window_days,
            gap_threshold_days=args.gap_threshold_days,
        )
    except (FileNotFoundError, OSError, ValueError) as exc:
        safe_print_error(f"error: {exc}")
        return 2

    if args.format == "json":
        safe_print(render_json(narrative))
    elif args.format == "text":
        safe_print(render_text(narrative))
    elif args.format == "debug":
        safe_print(render_debug(narrative))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
