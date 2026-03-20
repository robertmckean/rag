"""CLI for building message-level embeddings for one normalized run."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from rag.embeddings.builder import (
    DEFAULT_EMBEDDING_BATCH_SIZE,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_SAMPLE_SEED,
    build_run_embeddings,
)


# Build the parser for the embeddings artifact generation CLI.
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build message embeddings for one normalized run.")
    parser.add_argument("--run-dir", type=Path, required=True, help="Path to one normalized run directory.")
    parser.add_argument("--model", type=str, default=DEFAULT_EMBEDDING_MODEL, help="Embedding model name.")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_EMBEDDING_BATCH_SIZE, help="Embedding batch size. Start with 5 or 10 for the first live validation of a new run.")
    parser.add_argument("--limit", type=int, default=None, help="Embed only the first N eligible messages after offset.")
    parser.add_argument("--offset", type=int, default=0, help="Skip the first N eligible messages before embedding.")
    parser.add_argument("--sample", type=int, default=None, help="Embed a deterministic random sample of N eligible messages.")
    parser.add_argument("--sample-seed", type=int, default=DEFAULT_SAMPLE_SEED, help="Deterministic seed for --sample.")
    parser.add_argument("--conversation-id", action="append", default=None, help="Restrict embedding generation to one conversation id. Repeatable.")
    parser.add_argument("--message-id", action="append", default=None, help="Restrict embedding generation to one message id. Repeatable.")
    return parser


# Execute embedding generation and print a compact build summary.
def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = build_run_embeddings(
            args.run_dir.resolve(),
            model=args.model,
            batch_size=args.batch_size,
            limit=args.limit,
            offset=args.offset,
            sample=args.sample,
            sample_seed=args.sample_seed,
            conversation_ids=tuple(args.conversation_id or ()),
            message_ids=tuple(args.message_id or ()),
        )
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        _safe_print_error(f"error: {exc}")
        return 2

    _safe_print("Embedding Build")
    _safe_print(f"run_dir: {args.run_dir.resolve()}")
    _safe_print(f"run_id: {result.run_id}")
    _safe_print(f"model: {result.model}")
    _safe_print(f"batch_size: {result.batch_size}")
    _safe_print(f"total_messages_in_run: {result.total_messages_in_run}")
    _safe_print(f"selected_message_count: {result.selected_message_count}")
    _safe_print(f"resumed_skip_count: {result.resumed_skip_count}")
    _safe_print(f"targeted_match_count: {result.targeted_match_count}")
    _safe_print(f"subset_offset: {result.subset_offset}")
    _safe_print(f"subset_limit: {result.subset_limit}")
    _safe_print(f"subset_sample_size: {result.subset_sample_size}")
    _safe_print(f"subset_sample_seed: {result.subset_sample_seed}")
    _safe_print(f"targeted_conversation_count: {result.targeted_conversation_count}")
    _safe_print(f"targeted_message_id_count: {result.targeted_message_id_count}")
    _safe_print(f"record_count: {result.record_count}")
    _safe_print(f"skipped_empty_count: {result.skipped_empty_count}")
    _safe_print(f"filtered_tool_role_count: {result.filtered_tool_role_count}")
    _safe_print(f"filtered_low_information_count: {result.filtered_low_information_count}")
    _safe_print(f"filtered_trivial_short_count: {result.filtered_trivial_short_count}")
    _safe_print(f"artifact_path: {result.artifact_path}")
    return 0


# Print terminal output safely when the console cannot encode some text.
def _safe_print(value: str) -> None:
    encoding = sys.stdout.encoding or "utf-8"
    safe_value = value.encode(encoding, errors="replace").decode(encoding, errors="replace")
    print(safe_value)


# Print CLI errors safely to stderr.
def _safe_print_error(value: str) -> None:
    encoding = sys.stderr.encoding or "utf-8"
    safe_value = value.encode(encoding, errors="replace").decode(encoding, errors="replace")
    print(safe_value, file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
