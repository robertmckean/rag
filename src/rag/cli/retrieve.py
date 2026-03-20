"""CLI for the first lexical Phase 2 retrieval slice over one normalized run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from rag.retrieval.lexical import RetrievalFilters, retrieve_message_windows


# The CLI is a debugging surface for the first retrieval slice rather than a full product interface.
# It prints compact windowed results for humans and can also emit JSON for inspection.
# The command reads one immutable run folder and never mutates canonical artifacts.


# Build the parser for the lexical retrieval CLI.
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run lexical retrieval against one normalized run and return message windows."
    )
    parser.add_argument("--run-dir", type=Path, required=True, help="Path to one normalized run directory.")
    parser.add_argument("--query", type=str, required=True, help="Query text to rank against normalized messages.")
    parser.add_argument("--limit", type=int, default=10, help="Maximum number of retrieval results to return.")
    parser.add_argument("--provider", type=str, default=None, help="Optional provider filter.")
    parser.add_argument("--conversation-id", type=str, default=None, help="Optional conversation filter.")
    parser.add_argument("--from", dest="date_from", type=str, default=None, help="Optional inclusive lower date bound.")
    parser.add_argument("--to", dest="date_to", type=str, default=None, help="Optional inclusive upper date bound.")
    parser.add_argument("--author-role", type=str, default=None, help="Optional canonical author role filter.")
    parser.add_argument("--json-out", type=Path, default=None, help="Optional path for structured JSON results.")
    return parser


# Parse CLI arguments, execute retrieval, print a readable summary, and optionally write JSON.
def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    filters = RetrievalFilters(
        provider=args.provider,
        conversation_id=args.conversation_id,
        date_from=args.date_from,
        date_to=args.date_to,
        author_role=args.author_role,
    )
    results = retrieve_message_windows(
        args.run_dir.resolve(),
        args.query,
        limit=args.limit,
        filters=filters,
    )
    _safe_print(render_results_summary(args.run_dir.resolve(), args.query, filters, results))

    if args.json_out:
        payload = {
            "run_dir": str(args.run_dir.resolve()),
            "query": args.query,
            "filters": filters.__dict__,
            "result_count": len(results),
            "results": [result.to_dict() for result in results],
        }
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _safe_print("")
        _safe_print(f"json_out: {args.json_out.resolve()}")

    return 0


# Render the retrieval results in a compact terminal-friendly summary.
def render_results_summary(
    run_dir: Path,
    query: str,
    filters: RetrievalFilters,
    results: tuple[object, ...],
) -> str:
    lines = [
        "Retrieval Results",
        f"run_dir: {run_dir}",
        f"query: {query}",
        f"filters: {render_filters(filters)}",
        f"result_count: {len(results)}",
    ]

    for result in results:
        lines.append("")
        lines.append(
            f"[{result.rank}] score={result.score:.2f} provider={result.provider} conversation={result.conversation_id}"
        )
        lines.append(f"  title: {result.conversation_title}")
        lines.append(
            f"  focal_message_id: {result.focal_message_id} sequence_index={result.focal_sequence_index}"
        )
        lines.append(
            f"  matched_terms: {', '.join(result.match_basis['matched_text_terms']) or '(none)'}"
        )
        lines.append(f"  excerpt: {result.match_basis['focal_match_excerpt']}")
        lines.append("  window_messages:")
        for message in result.messages:
            marker = "*" if message.get("message_id") == result.focal_message_id else "-"
            snippet = _message_snippet(message)
            lines.append(
                f"    {marker} seq={message.get('sequence_index')} role={message.get('author_role')} "
                f"created_at={message.get('created_at')} id={message.get('message_id')} text={snippet}"
            )

    return "\n".join(lines)


# Render the active filter set compactly for the terminal summary.
def render_filters(filters: RetrievalFilters) -> str:
    values = {
        "provider": filters.provider,
        "conversation_id": filters.conversation_id,
        "date_from": filters.date_from,
        "date_to": filters.date_to,
        "author_role": filters.author_role,
    }
    rendered = [f"{key}={value}" for key, value in values.items() if value is not None]
    return ", ".join(rendered) if rendered else "(none)"


# Build a short one-line message preview for terminal result output.
def _message_snippet(message: dict[str, object]) -> str | None:
    text = message.get("text")
    if isinstance(text, str) and text.strip():
        snippet = text.strip()
        return snippet[:117] + "..." if len(snippet) > 120 else snippet
    return None


# Print terminal output with replacement fallback for consoles that cannot encode some message text.
def _safe_print(value: str) -> None:
    encoding = sys.stdout.encoding or "utf-8"
    safe_value = value.encode(encoding, errors="replace").decode(encoding, errors="replace")
    print(safe_value)


if __name__ == "__main__":
    raise SystemExit(main())
