"""CLI for the first lexical Phase 2 retrieval slice over one normalized run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rag.cli.utils import safe_print, safe_print_error
from rag.embeddings.client import DEFAULT_EMBEDDING_MODEL
from rag.retrieval.lexical import (
    CLI_RETRIEVAL_MODES,
    RETRIEVAL_CHANNELS,
    RetrievalFilters,
    retrieve_message_timeline,
    retrieve_message_windows,
)


# The CLI is a debugging surface for the first retrieval slice rather than a full product interface.
# Window modes show local conversation context around a focal message.
# Timeline mode shows flat chronological entries across conversations for topic-evolution browsing.


# Build the parser for the lexical retrieval CLI.
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run lexical retrieval against one normalized run and return message windows."
    )
    parser.add_argument("--run-dir", type=Path, required=True, help="Path to one normalized run directory.")
    parser.add_argument("--query", type=str, required=True, help="Query text to rank against normalized messages.")
    parser.add_argument("--limit", type=int, default=10, help="Maximum number of retrieval results to return.")
    parser.add_argument("--mode", choices=CLI_RETRIEVAL_MODES, default="relevance", help="Retrieval ordering mode.")
    parser.add_argument(
        "--channel",
        choices=RETRIEVAL_CHANNELS,
        default="bm25",
        help="Retrieval channel: existing BM25, semantic embeddings, or hybrid union.",
    )
    parser.add_argument("--provider", type=str, default=None, help="Optional provider filter.")
    parser.add_argument("--conversation-id", type=str, default=None, help="Optional conversation filter.")
    parser.add_argument("--from", dest="date_from", type=str, default=None, help="Optional inclusive lower date bound.")
    parser.add_argument("--to", dest="date_to", type=str, default=None, help="Optional inclusive upper date bound.")
    parser.add_argument("--author-role", type=str, default=None, help="Optional canonical author role filter.")
    parser.add_argument("--embedding-model", type=str, default=DEFAULT_EMBEDDING_MODEL, help="Embedding model for semantic and hybrid retrieval.")
    parser.add_argument("--semantic-top-k", type=int, default=None, help="Semantic candidate pool size before window expansion.")
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
    try:
        if args.mode == "timeline" and args.channel != "bm25":
            raise ValueError("Semantic and hybrid retrieval are not supported with --mode timeline in Phase 4A.")
        results = retrieve_message_windows(
            args.run_dir.resolve(),
            args.query,
            limit=args.limit,
            mode=args.mode,
            filters=filters,
            channel=args.channel,
            semantic_top_k=args.semantic_top_k,
            embedding_model=args.embedding_model,
        ) if args.mode != "timeline" else retrieve_message_timeline(
            args.run_dir.resolve(),
            args.query,
            limit=args.limit,
            filters=filters,
        )
    except (FileNotFoundError, OSError, ValueError) as exc:
        safe_print_error(f"error: {exc}")
        return 2

    safe_print(render_results_summary(args.run_dir.resolve(), args.query, args.mode, args.channel, filters, results))

    if args.json_out:
        payload = {
            "run_dir": str(args.run_dir.resolve()),
            "query": args.query,
            "mode": args.mode,
            "channel": args.channel,
            "embedding_model": args.embedding_model,
            "semantic_top_k": args.semantic_top_k,
            "filters": filters.__dict__,
            "result_count": len(results),
            "results": [result.to_dict() for result in results],
        }
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        safe_print("")
        safe_print(f"json_out: {args.json_out.resolve()}")

    return 0


# Render the retrieval results in a compact terminal-friendly summary.
def render_results_summary(
    run_dir: Path,
    query: str,
    mode: str,
    channel: str,
    filters: RetrievalFilters,
    results: tuple[object, ...],
) -> str:
    lines = [
        "Retrieval Results",
        f"run_dir: {run_dir}",
        f"query: {query}",
        f"mode: {mode}",
        f"channel: {channel}",
        f"filters: {render_filters(filters)}",
        f"result_count: {len(results)}",
    ]
    if not results:
        lines.append("no matching results")
        return "\n".join(lines)

    for result in results:
        lines.append("")
        if mode == "timeline":
            lines.extend(_render_timeline_result(result))
        else:
            lines.extend(_render_window_result(result))

    return "\n".join(lines)


# Render one standard contextual window result.
def _render_window_result(result: object) -> list[str]:
    lines = [
        "  view: contextual_window",
        f"[{result.rank}] score={result.score:.2f} provider={result.provider} conversation={result.conversation_id}",
        f"  title: {result.conversation_title}",
        f"  focal_message_id: {result.focal_message_id} sequence_index={result.focal_sequence_index}",
        "  ranking: "
        f"mode={result.match_basis.get('retrieval_mode')} "
        f"channel={result.match_basis.get('retrieval_channel')} "
        f"sources={result.match_basis.get('retrieval_sources')} "
        f"relevance_score={result.match_basis['scoring_features'].get('relevance_score')} "
        f"bm25_score={result.match_basis['scoring_features'].get('bm25_score')} "
        f"semantic_similarity={result.match_basis['scoring_features'].get('semantic_similarity')} "
        f"recency_boost={result.match_basis['scoring_features'].get('recency_boost')} "
        f"basis={result.match_basis['scoring_features'].get('chronological_rank_basis')}",
        "  query_inputs: "
        f"terms={result.match_basis.get('scoring_terms')} "
        f"phrases={result.match_basis.get('normalized_phrase_targets')}",
        f"  matched_terms: {', '.join(result.match_basis['matched_text_terms']) or '(none)'}",
        f"  excerpt: {result.match_basis['focal_match_excerpt']}",
        "  window_messages:",
    ]
    for message in result.messages:
        marker = "*" if message.get("message_id") == result.focal_message_id else "-"
        snippet = _message_snippet(message)
        lines.append(
            f"    {marker} seq={message.get('sequence_index')} role={message.get('author_role')} "
            f"created_at={message.get('created_at')} id={message.get('message_id')} text={snippet}"
        )
    return lines


# Render one compact cross-conversation timeline result.
def _render_timeline_result(result: object) -> list[str]:
    return [
        "  view: timeline_compact",
        f"[{result.rank}] created_at={result.focal_created_at} score={result.score:.2f} provider={result.provider}",
        f"  conversation: {result.conversation_id}",
        f"  title: {result.conversation_title}",
        f"  focal_message_id: {result.focal_message_id} sequence_index={result.focal_sequence_index}",
        f"  author_role: {result.author_role}",
        "  ranking: "
        f"mode={result.match_basis.get('retrieval_mode')} "
        f"sources={result.match_basis.get('retrieval_sources')} "
        f"basis={result.match_basis['scoring_features'].get('chronological_rank_basis')} "
        f"relevance_score={result.match_basis['scoring_features'].get('relevance_score')} "
        f"bm25_score={result.match_basis['scoring_features'].get('bm25_score')} "
        f"semantic_similarity={result.match_basis['scoring_features'].get('semantic_similarity')}",
        "  query_inputs: "
        f"terms={result.match_basis.get('scoring_terms')} "
        f"phrases={result.match_basis.get('normalized_phrase_targets')}",
        f"  matched_terms: {', '.join(result.match_basis['matched_text_terms']) or '(none)'}",
        f"  excerpt: {result.focal_excerpt}",
        f"  source_message_path: {result.provenance['source_message_paths'][0]}",
    ]


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


if __name__ == "__main__":
    raise SystemExit(main())
