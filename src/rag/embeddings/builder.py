"""Embedding builder for one immutable normalized run."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import random
from typing import Protocol
import sys

from rag.retrieval.read_model import load_normalized_run
from rag.retrieval.read_model import tokenize_query
from rag.retrieval.utils import string_or_none

from rag.embeddings.store import (
    EmbeddingRecord,
    append_embedding_records,
    existing_embedding_keys,
    initialize_embedding_records_file,
)


DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_EMBEDDING_BATCH_SIZE = 100
DEFAULT_EMBEDDING_MAX_TOKENS = 8000
DEFAULT_PROGRESS_EVERY_BATCHES = 1
DEFAULT_SAMPLE_SEED = 0
LOW_INFORMATION_ACKNOWLEDGMENTS = frozenset(
    {
        "cool",
        "got it",
        "k",
        "kk",
        "no",
        "ok",
        "okay",
        "sounds good",
        "sure",
        "thank you",
        "thanks",
        "yes",
        "yep",
    }
)


class EmbeddingClient(Protocol):
    # Embed a batch of texts with one configured model.
    def embed_texts(self, texts: list[str], *, model: str) -> list[list[float]]:
        ...


@dataclass(frozen=True)
class EmbeddingBuildResult:
    run_id: str
    model: str
    batch_size: int
    artifact_path: Path
    record_count: int
    skipped_empty_count: int
    total_messages_in_run: int
    selected_message_count: int
    resumed_skip_count: int
    targeted_match_count: int
    subset_offset: int | None
    subset_limit: int | None
    subset_sample_size: int | None
    subset_sample_seed: int | None
    targeted_conversation_count: int
    targeted_message_id_count: int
    filtered_tool_role_count: int
    filtered_low_information_count: int
    filtered_trivial_short_count: int


@dataclass(frozen=True)
class PreparedEmbeddingText:
    text: str
    original_text_length: int
    stored_text_length: int
    truncation_occurred: bool
    original_token_count: int | None
    stored_token_count: int | None


class OpenAIEmbeddingClient:
    # Call the modern OpenAI embeddings API through the current Python SDK.
    def embed_texts(self, texts: list[str], *, model: str) -> list[list[float]]:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("OpenAI Python SDK is required for embedding generation.") from exc
        client = OpenAI()
        response = client.embeddings.create(model=model, input=texts)
        return [list(item.embedding) for item in response.data]


# Build message-level embeddings for one normalized run and write them back into that run directory.
def build_run_embeddings(
    run_dir: Path,
    *,
    model: str = DEFAULT_EMBEDDING_MODEL,
    batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE,
    max_tokens: int = DEFAULT_EMBEDDING_MAX_TOKENS,
    embedding_client: EmbeddingClient | None = None,
    progress_every_batches: int = DEFAULT_PROGRESS_EVERY_BATCHES,
    limit: int | None = None,
    offset: int = 0,
    sample: int | None = None,
    sample_seed: int = DEFAULT_SAMPLE_SEED,
    conversation_ids: tuple[str, ...] = (),
    message_ids: tuple[str, ...] = (),
) -> EmbeddingBuildResult:
    loaded_run = load_normalized_run(run_dir)
    client = embedding_client or OpenAIEmbeddingClient()

    messages = sorted(
        loaded_run.message_by_id.values(),
        key=lambda message: (
            string_or_none(message.get("conversation_id")) or "",
            message.get("sequence_index") if isinstance(message.get("sequence_index"), int) else 0,
            string_or_none(message.get("message_id")) or "",
        ),
    )
    skipped_empty_count = 0
    filtered_tool_role_count = 0
    filtered_low_information_count = 0
    filtered_trivial_short_count = 0
    eligible_messages: list[dict[str, object]] = []
    for message in messages:
        message_id = string_or_none(message.get("message_id")) or ""
        if not message_id:
            continue
        searchable_text = loaded_run.searchable_text_by_message_id.get(message_id, "")
        filter_reason = _embedding_filter_reason(message, searchable_text)
        if filter_reason == "empty_text":
            skipped_empty_count += 1
            continue
        if filter_reason == "tool_role":
            filtered_tool_role_count += 1
            continue
        if filter_reason == "low_information_ack":
            filtered_low_information_count += 1
            continue
        if filter_reason == "trivially_short":
            filtered_trivial_short_count += 1
            continue
        eligible_messages.append(message)
    total_messages_in_run = len(eligible_messages)
    targeted_messages = _apply_targeted_selection(
        eligible_messages,
        conversation_ids=conversation_ids,
        message_ids=message_ids,
    )
    targeted_match_count = len(targeted_messages)
    selected_messages = _select_messages(
        targeted_messages,
        offset=offset,
        limit=limit,
        sample=sample,
        sample_seed=sample_seed,
    )
    existing_keys = existing_embedding_keys(loaded_run.run_dir)
    messages_to_embed = [
        message
        for message in selected_messages
        if ((string_or_none(message.get("message_id")) or ""), model) not in existing_keys
    ]
    resumed_skip_count = len(selected_messages) - len(messages_to_embed)
    selected_message_count = len(selected_messages)
    build_timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    _safe_print(f"Embedding {selected_message_count} selected messages out of {total_messages_in_run} eligible messages")
    if conversation_ids or message_ids:
        _safe_print(
            "targeted_selection: "
            f"matched_messages={targeted_match_count} "
            f"conversation_filters={len(conversation_ids)} "
            f"message_id_filters={len(message_ids)}"
        )
    _safe_print(
        "filtered_counts: "
        f"empty_text={skipped_empty_count} "
        f"tool_role={filtered_tool_role_count} "
        f"low_information={filtered_low_information_count} "
        f"trivially_short={filtered_trivial_short_count}"
    )
    if batch_size < 20 and selected_message_count >= 1000:
        _safe_print("warning: batch_size < 20 on a large build will be slow; validate first, then increase it")

    pending_messages: list[dict[str, object]] = []
    pending_texts: list[PreparedEmbeddingText] = []
    artifact_path = loaded_run.run_dir / "message_embeddings.jsonl"
    if not artifact_path.exists():
        artifact_path = initialize_embedding_records_file(loaded_run.run_dir)
    embedded_count = 0
    batch_count = 0

    for message in messages_to_embed:
        message_id = string_or_none(message.get("message_id")) or ""
        searchable_text = loaded_run.searchable_text_by_message_id.get(message_id, "")
        prepared_text = prepare_text_for_embedding(searchable_text, model=model, max_tokens=max_tokens)
        pending_messages.append(message)
        pending_texts.append(prepared_text)
        if len(pending_texts) >= batch_size:
            batch_records = _embed_batch(
                loaded_run.run_id,
                pending_messages,
                pending_texts,
                model,
                client,
                total_messages_in_run=total_messages_in_run,
                build_timestamp=build_timestamp,
                subset_offset=offset or None,
                subset_limit=limit,
                subset_sample_size=sample,
                subset_sample_seed=sample_seed if sample is not None else None,
            )
            append_embedding_records(loaded_run.run_dir, tuple(batch_records))
            embedded_count += len(batch_records)
            batch_count += 1
            _maybe_log_progress(embedded_count, len(messages_to_embed), batch_count, progress_every_batches)
            pending_messages = []
            pending_texts = []

    if pending_texts:
        batch_records = _embed_batch(
            loaded_run.run_id,
            pending_messages,
            pending_texts,
            model,
            client,
            total_messages_in_run=total_messages_in_run,
            build_timestamp=build_timestamp,
            subset_offset=offset or None,
            subset_limit=limit,
            subset_sample_size=sample,
            subset_sample_seed=sample_seed if sample is not None else None,
        )
        append_embedding_records(loaded_run.run_dir, tuple(batch_records))
        embedded_count += len(batch_records)
        batch_count += 1
        _maybe_log_progress(embedded_count, len(messages_to_embed), batch_count, progress_every_batches)

    return EmbeddingBuildResult(
        run_id=loaded_run.run_id,
        model=model,
        batch_size=batch_size,
        artifact_path=artifact_path,
        record_count=embedded_count,
        skipped_empty_count=skipped_empty_count,
        total_messages_in_run=total_messages_in_run,
        selected_message_count=selected_message_count,
        resumed_skip_count=resumed_skip_count,
        targeted_match_count=targeted_match_count,
        subset_offset=offset or None,
        subset_limit=limit,
        subset_sample_size=sample,
        subset_sample_seed=sample_seed if sample is not None else None,
        targeted_conversation_count=len(conversation_ids),
        targeted_message_id_count=len(message_ids),
        filtered_tool_role_count=filtered_tool_role_count,
        filtered_low_information_count=filtered_low_information_count,
        filtered_trivial_short_count=filtered_trivial_short_count,
    )


# Embed one deterministic batch of normalized message texts.
def _embed_batch(
    run_id: str,
    messages: list[dict[str, object]],
    prepared_texts: list[PreparedEmbeddingText],
    model: str,
    embedding_client: EmbeddingClient,
    *,
    total_messages_in_run: int,
    build_timestamp: str,
    subset_offset: int | None,
    subset_limit: int | None,
    subset_sample_size: int | None,
    subset_sample_seed: int | None,
) -> list[EmbeddingRecord]:
    embeddings = embedding_client.embed_texts([prepared_text.text for prepared_text in prepared_texts], model=model)
    if len(embeddings) != len(messages):
        raise ValueError("Embedding client returned a different number of vectors than requested texts.")

    records: list[EmbeddingRecord] = []
    for message, prepared_text, embedding in zip(messages, prepared_texts, embeddings):
        records.append(
            EmbeddingRecord(
                run_id=run_id,
                message_id=string_or_none(message.get("message_id")) or "",
                conversation_id=string_or_none(message.get("conversation_id")) or "",
                provider=string_or_none(message.get("provider")) or "unknown",
                author_role=string_or_none(message.get("author_role")),
                created_at=string_or_none(message.get("created_at")),
                sequence_index=message.get("sequence_index") if isinstance(message.get("sequence_index"), int) else 0,
                embedding_model=model,
                embedding_dimensions=len(embedding),
                text=prepared_text.text,
                original_text_length=prepared_text.original_text_length,
                stored_text_length=prepared_text.stored_text_length,
                truncation_occurred=prepared_text.truncation_occurred,
                original_token_count=prepared_text.original_token_count,
                stored_token_count=prepared_text.stored_token_count,
                total_messages_in_run=total_messages_in_run,
                build_timestamp=build_timestamp,
                subset_offset=subset_offset,
                subset_limit=subset_limit,
                subset_sample_size=subset_sample_size,
                subset_sample_seed=subset_sample_seed,
                embedding=tuple(float(value) for value in embedding),
            )
        )
    return records


# Prepare one message text for embedding by truncating safely below the model token limit.
def prepare_text_for_embedding(
    text: str,
    *,
    model: str,
    max_tokens: int = DEFAULT_EMBEDDING_MAX_TOKENS,
) -> PreparedEmbeddingText:
    tokenizer = _load_tokenizer(model)
    original_text_length = len(text)

    if tokenizer is not None:
        encoded = tokenizer.encode(text)
        original_token_count = len(encoded)
        if original_token_count <= max_tokens:
            return PreparedEmbeddingText(
                text=text,
                original_text_length=original_text_length,
                stored_text_length=original_text_length,
                truncation_occurred=False,
                original_token_count=original_token_count,
                stored_token_count=original_token_count,
            )
        truncated_text = tokenizer.decode(encoded[:max_tokens])
        return PreparedEmbeddingText(
            text=truncated_text,
            original_text_length=original_text_length,
            stored_text_length=len(truncated_text),
            truncation_occurred=True,
            original_token_count=original_token_count,
            stored_token_count=len(tokenizer.encode(truncated_text)),
        )

    # Fallback: treat each character as one token proxy so truncation stays safely below the API limit.
    if original_text_length <= max_tokens:
        return PreparedEmbeddingText(
            text=text,
            original_text_length=original_text_length,
            stored_text_length=original_text_length,
            truncation_occurred=False,
            original_token_count=original_text_length,
            stored_token_count=original_text_length,
        )
    truncated_text = text[:max_tokens]
    return PreparedEmbeddingText(
        text=truncated_text,
        original_text_length=original_text_length,
        stored_text_length=len(truncated_text),
        truncation_occurred=True,
        original_token_count=original_text_length,
        stored_token_count=len(truncated_text),
    )


# Load a tokenizer for the configured embedding model when tiktoken is available locally.
def _load_tokenizer(model: str) -> object | None:
    try:
        import tiktoken
    except ImportError:
        return None
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        return tiktoken.get_encoding("cl100k_base")


# Print coarse-grained embedding progress every N batches so long runs are visible.
def _maybe_log_progress(
    embedded_count: int,
    total_messages_to_embed: int,
    batch_count: int,
    progress_every_batches: int,
) -> None:
    if progress_every_batches <= 0:
        return
    if batch_count % progress_every_batches != 0 and embedded_count != total_messages_to_embed:
        return
    _safe_print(f"Embedded {embedded_count} / {total_messages_to_embed} messages")


# Print builder progress with the same safe encoding fallback used by other CLIs.
def _safe_print(value: str) -> None:
    encoding = sys.stdout.encoding or "utf-8"
    safe_value = value.encode(encoding, errors="replace").decode(encoding, errors="replace")
    print(safe_value)


# Select a deterministic subset before batching so quick validation builds are cheap.
def _select_messages(
    messages: list[dict[str, object]],
    *,
    offset: int,
    limit: int | None,
    sample: int | None,
    sample_seed: int,
) -> list[dict[str, object]]:
    selected = list(messages[max(offset, 0) :])
    if limit is not None:
        selected = selected[: max(limit, 0)]
    if sample is not None:
        sample_size = min(max(sample, 0), len(selected))
        rng = random.Random(sample_seed)
        indexed_messages = list(enumerate(selected))
        sampled = rng.sample(indexed_messages, sample_size)
        sampled.sort(key=lambda item: item[0])
        selected = [message for _, message in sampled]
    return selected


# Restrict embedding generation to explicitly requested conversations and/or messages.
def _apply_targeted_selection(
    messages: list[dict[str, object]],
    *,
    conversation_ids: tuple[str, ...],
    message_ids: tuple[str, ...],
) -> list[dict[str, object]]:
    if not conversation_ids and not message_ids:
        return list(messages)

    conversation_id_set = {value for value in conversation_ids if value}
    message_id_set = {value for value in message_ids if value}
    selected: list[dict[str, object]] = []
    for message in messages:
        conversation_id = string_or_none(message.get("conversation_id")) or ""
        message_id = string_or_none(message.get("message_id")) or ""
        if conversation_id in conversation_id_set or message_id in message_id_set:
            selected.append(message)
    return selected


# Decide whether a message should be excluded from embedding generation only.
def _embedding_filter_reason(message: dict[str, object], searchable_text: str) -> str | None:
    if not searchable_text:
        return "empty_text"

    author_role = string_or_none(message.get("author_role"))
    if author_role == "tool":
        return "tool_role"

    normalized_text = searchable_text.strip()
    if normalized_text in LOW_INFORMATION_ACKNOWLEDGMENTS:
        return "low_information_ack"

    tokens = tokenize_query(normalized_text)
    if len(tokens) == 1 and len(normalized_text) <= 2:
        return "trivially_short"

    return None
