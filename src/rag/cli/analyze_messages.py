"""CLI for read-only normalized message quality analysis."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.rag.analysis.message_quality import (
    analyze_message_quality,
    render_message_quality_report,
    write_message_quality_report,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze normalized messages.jsonl for empty and low-signal records."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Normalized run directory containing messages.jsonl.",
    )
    parser.add_argument(
        "--write-json",
        action="store_true",
        help="Write analysis/message_quality_report.json under the run directory.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_dir = args.run_dir.resolve()
    messages_path = run_dir / "messages.jsonl"
    report = analyze_message_quality(messages_path)
    print(render_message_quality_report(report))

    if args.write_json:
        output_path = run_dir / "analysis" / "message_quality_report.json"
        write_message_quality_report(report, output_path)
        print("")
        print(f"json_report: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
