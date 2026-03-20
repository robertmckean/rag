"""CLI for deterministic grounded answers over one normalized run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rag.answering.answer import (
    ANSWER_RETRIEVAL_MODES,
    GROUNDING_MODES,
    answer_query,
    answer_result_json,
    qualification_debug_payload,
    render_qualification_debug,
    render_answer_result,
)


# The answer CLI is a review surface for grounded answers rather than a chat interface.
# Retrieval stays authoritative for evidence collection, and the answer layer only summarizes bounded evidence.
# JSON output is first-class so the same result object can be inspected in tests and scripts.


# Build the parser for the grounded answer CLI.
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a deterministic grounded answer from one normalized run.")
    parser.add_argument("--run-dir", type=Path, required=True, help="Path to one normalized run directory.")
    parser.add_argument("--query", type=str, required=True, help="Natural-language question to answer.")
    parser.add_argument(
        "--retrieval-mode",
        choices=ANSWER_RETRIEVAL_MODES,
        default="relevance",
        help="Retrieval ordering mode used to gather evidence.",
    )
    parser.add_argument(
        "--grounding-mode",
        choices=GROUNDING_MODES,
        default="strict",
        help="Grounding policy used to qualify evidence after retrieval.",
    )
    parser.add_argument("--limit", type=int, default=8, help="Maximum number of retrieval results to inspect.")
    parser.add_argument("--max-evidence", type=int, default=5, help="Maximum number of evidence items to keep.")
    parser.add_argument("--llm", action="store_true", help="Enable constrained LLM-backed answer synthesis.")
    parser.add_argument("--llm-model", type=str, default=None, help="Optional OpenAI model override for --llm mode.")
    parser.add_argument("--debug-qualification", action="store_true", help="Print detailed evidence-qualification diagnostics.")
    parser.add_argument("--json", action="store_true", help="Print the answer result as structured JSON.")
    parser.add_argument("--json-out", type=Path, default=None, help="Optional path for structured JSON output.")
    return parser


# Parse arguments, generate a grounded answer, print it, and optionally write JSON to disk.
def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = answer_query(
            args.run_dir.resolve(),
            args.query,
            retrieval_mode=args.retrieval_mode,
            grounding_mode=args.grounding_mode,
            limit=args.limit,
            max_evidence=args.max_evidence,
            llm=args.llm,
            llm_model=args.llm_model,
        )
    except (FileNotFoundError, OSError, ValueError) as exc:
        _safe_print_error(f"error: {exc}")
        return 2

    debug_payload = None
    if args.debug_qualification:
        try:
            debug_payload = qualification_debug_payload(
                args.run_dir.resolve(),
                args.query,
                retrieval_mode=args.retrieval_mode,
                grounding_mode=args.grounding_mode,
                limit=args.limit,
                max_evidence=args.max_evidence,
            )
        except (FileNotFoundError, OSError, ValueError) as exc:
            _safe_print_error(f"error: {exc}")
            return 2

    json_payload = answer_result_json(result)
    if args.json:
        if debug_payload is None:
            _safe_print(json_payload.rstrip())
        else:
            combined_payload = {
                "answer_result": result.to_dict(),
                "qualification_debug": debug_payload,
            }
            _safe_print(json.dumps(combined_payload, ensure_ascii=True, indent=2, sort_keys=True))
    else:
        _safe_print(render_answer_result(result))
        if debug_payload is not None:
            _safe_print("")
            _safe_print(render_qualification_debug(debug_payload))

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        if debug_payload is None:
            args.json_out.write_text(json_payload, encoding="utf-8")
        else:
            combined_payload = json.dumps(
                {
                    "answer_result": result.to_dict(),
                    "qualification_debug": debug_payload,
                },
                ensure_ascii=True,
                indent=2,
                sort_keys=True,
            )
            args.json_out.write_text(combined_payload + "\n", encoding="utf-8")
        if not args.json:
            _safe_print("")
            _safe_print(f"json_out: {args.json_out.resolve()}")

    return 0


# Print terminal output with replacement fallback for consoles that cannot encode some evidence text.
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
