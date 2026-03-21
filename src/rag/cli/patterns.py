"""CLI for recurring-entity pattern extraction over one normalized run."""

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
from rag.patterns.extractor import extract_recurring_entities
from rag.patterns.renderer import render_json, render_text
from rag.patterns.router import route_answer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract recurring-entity patterns from narrative reconstructions."
    )
    parser.add_argument("--run-dir", type=Path, required=True, help="Path to one normalized run directory.")
    parser.add_argument("--queries", type=str, nargs="+", required=True, help="One or more natural-language queries.")
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
    parser.add_argument("--limit", type=int, default=DEFAULT_NARRATIVE_LIMIT, help="Maximum retrieval results per query.")
    parser.add_argument("--max-evidence", type=int, default=DEFAULT_NARRATIVE_MAX_EVIDENCE, help="Maximum evidence items per query.")
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
        choices=("json", "text"),
        default="json",
        help="Output format.",
    )
    parser.add_argument(
        "--answer",
        type=str,
        default=None,
        help="Route a question to the appropriate data and produce a concise answer.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_dir = args.run_dir.resolve()

    narratives = []
    for query in args.queries:
        try:
            narrative = build_narrative_from_run(
                run_dir,
                query,
                retrieval_mode=args.retrieval_mode,
                channel=args.channel,
                limit=args.limit,
                max_evidence=args.max_evidence,
                phase_window_days=args.phase_window_days,
                gap_threshold_days=args.gap_threshold_days,
            )
            narratives.append(narrative)
        except (FileNotFoundError, OSError, ValueError) as exc:
            safe_print_error(f"error ({query}): {exc}")
            return 2

    report = extract_recurring_entities(narratives)

    if args.answer:
        safe_print(route_answer(args.answer, report, narratives))
        return 0

    if args.format == "json":
        safe_print(render_json(report))
    elif args.format == "text":
        safe_print(render_text(report))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
