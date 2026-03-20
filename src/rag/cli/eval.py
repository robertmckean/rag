"""CLI for deterministic grounded-answer benchmark evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from rag.eval.runner import eval_report_json, render_eval_report, run_benchmark


# The eval CLI exercises the current answer pipeline rather than a mocked variant.
# Terminal output stays concise so failures are easy to scan during iteration.
# JSON output preserves per-case detail for later analysis or diffing.


# Build the parser for the deterministic eval CLI.
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run deterministic grounded-answer evaluation against one normalized run.")
    parser.add_argument("--run-dir", type=Path, required=True, help="Path to one normalized run directory.")
    parser.add_argument("--bench", type=Path, required=True, help="Path to the benchmark query bank JSON file.")
    parser.add_argument("--json", action="store_true", help="Print the eval report as structured JSON.")
    parser.add_argument("--json-out", type=Path, default=None, help="Optional path for structured JSON output.")
    parser.add_argument("--fail-only", action="store_true", help="Show only failed cases in terminal output.")
    return parser


# Parse CLI arguments, run the benchmark, print the report, and optionally write JSON.
def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary, case_results = run_benchmark(args.run_dir.resolve(), args.bench.resolve())
    except (FileNotFoundError, OSError, ValueError) as exc:
        _safe_print_error(f"error: {exc}")
        return 2

    report_json = eval_report_json(summary, case_results)
    if args.json:
        _safe_print(report_json.rstrip())
    else:
        _safe_print(render_eval_report(summary, case_results, fail_only=args.fail_only))

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(report_json, encoding="utf-8")
        if not args.json:
            _safe_print("")
            _safe_print(f"json_out: {args.json_out.resolve()}")

    return 0


# Print terminal output with replacement fallback for consoles that cannot encode some text.
def _safe_print(value: str) -> None:
    encoding = sys.stdout.encoding or "utf-8"
    safe_value = value.encode(encoding, errors="replace").decode(encoding, errors="replace")
    print(safe_value)


# Print CLI errors to stderr with the same encoding fallback used for normal output.
def _safe_print_error(value: str) -> None:
    encoding = sys.stderr.encoding or "utf-8"
    safe_value = value.encode(encoding, errors="replace").decode(encoding, errors="replace")
    print(safe_value, file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
