"""Minimal lexical retrieval over normalized message records with contextual window expansion."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from rag.retrieval.read_model import LoadedRun, load_normalized_run, normalize_lexical_text, tokenize_query


# Ranking is message-first because message text is the narrowest useful relevance signal.
# Returned results are contextual windows so humans and LLMs can see local conversation flow.
# The contract keeps scoring details explicit so retrieval behavior is easy to debug.


@dataclass(frozen=True)
class RetrievalFilters:
    provider: str | None = None
    conversation_id: str | None = None
    date_from: str | None = None
    date_to: str | None = None
    author_role: str | None = None


@dataclass(frozen=True)
class RankedResult:
    rank: int
    score: float
    result_id: str
    run_id: str
    provider: str
    conversation_id: str
    conversation_title: str | None
    conversation_created_at: str | None
    conversation_updated_at: str | None
    focal_message_id: str
    focal_created_at: str | None
    focal_sequence_index: int
    window_start_sequence_index: int
    window_end_sequence_index: int
    messages: tuple[dict[str, object], ...]
    match_basis: dict[str, object]
    provenance: dict[str, object]

    # Convert the retrieval result into a JSON-serializable dict for CLI output.
    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class _Candidate:
    score: float
    provider: str
    conversation_id: str
    message_id: str
    created_at: str | None
    match_basis: dict[str, object]


# Load a run directory and execute lexical retrieval in one call.
def retrieve_message_windows(
    run_dir: Path,
    query: str,
    *,
    limit: int = 10,
    window_radius: int = 2,
    filters: RetrievalFilters | None = None,
) -> tuple[RankedResult, ...]:
    loaded_run = load_normalized_run(run_dir)
    return search_loaded_run(
        loaded_run,
        query,
        limit=limit,
        window_radius=window_radius,
        filters=filters,
    )


# Execute lexical retrieval against a loaded run and return contextual message windows.
def search_loaded_run(
    loaded_run: LoadedRun,
    query: str,
    *,
    limit: int = 10,
    window_radius: int = 2,
    filters: RetrievalFilters | None = None,
) -> tuple[RankedResult, ...]:
    resolved_filters = filters or RetrievalFilters()
    normalized_query = normalize_lexical_text(query)
    query_tokens = tokenize_query(query)
    if not query_tokens:
        return ()

    candidates = _rank_candidates(loaded_run, normalized_query, query_tokens, resolved_filters)
    ordered_candidates = sorted(candidates, key=_candidate_sort_key)

    results: list[RankedResult] = []
    seen_window_keys: set[tuple[str, int, int]] = set()

    for candidate in ordered_candidates:
        result = _build_window_result(
            loaded_run,
            candidate,
            window_radius=window_radius,
            rank=len(results) + 1,
        )
        window_key = (
            result.conversation_id,
            result.window_start_sequence_index,
            result.window_end_sequence_index,
        )
        if window_key in seen_window_keys:
            continue
        seen_window_keys.add(window_key)
        results.append(result)
        if len(results) >= limit:
            break

    return tuple(results)


# Rank message-level candidates using simple lexical and metadata-aware signals.
def _rank_candidates(
    loaded_run: LoadedRun,
    normalized_query: str,
    query_tokens: tuple[str, ...],
    filters: RetrievalFilters,
) -> list[_Candidate]:
    candidates: list[_Candidate] = []
    query_token_set = set(query_tokens)

    for message_id, message in loaded_run.message_by_id.items():
        if not _message_matches_filters(message, filters):
            continue

        searchable_text = loaded_run.searchable_text_by_message_id.get(message_id, "")
        if not searchable_text:
            continue

        searchable_token_set = set(tokenize_query(searchable_text))
        overlap_tokens = tuple(sorted(query_token_set & searchable_token_set))
        exact_phrase_match = normalized_query in searchable_text

        conversation = loaded_run.conversation_by_id.get(_string_or_none(message.get("conversation_id")) or "")
        conversation_title = _string_or_none(conversation.get("title")) if conversation else None
        normalized_title = normalize_lexical_text(conversation_title or "")
        title_overlap_tokens = tuple(sorted(query_token_set & set(tokenize_query(normalized_title))))

        token_overlap_score = float(len(overlap_tokens))
        exact_phrase_boost = 3.0 if exact_phrase_match else 0.0
        title_boost = 0.5 * len(title_overlap_tokens)

        # Title overlap can boost a real message hit, but it should not create a hit by itself.
        if token_overlap_score <= 0.0 and exact_phrase_boost <= 0.0:
            continue

        score = token_overlap_score + exact_phrase_boost + title_boost
        if score <= 0.0:
            continue

        candidates.append(
            _Candidate(
                score=score,
                provider=_string_or_none(message.get("provider")) or "unknown",
                conversation_id=_string_or_none(message.get("conversation_id")) or "",
                message_id=message_id,
                created_at=_string_or_none(message.get("created_at")),
                match_basis={
                    "matched_text_terms": list(overlap_tokens),
                    "matched_metadata_filters": _matched_filters_dict(filters),
                    "scoring_features": {
                        "token_overlap": len(overlap_tokens),
                        "exact_phrase_match": exact_phrase_match,
                        "exact_phrase_boost": exact_phrase_boost,
                        "title_overlap": len(title_overlap_tokens),
                        "title_boost": title_boost,
                    },
                    "focal_match_excerpt": _build_excerpt(message, overlap_tokens),
                },
            )
        )

    return candidates


# Expand one ranked focal message into the default conversation window result.
def _build_window_result(
    loaded_run: LoadedRun,
    candidate: _Candidate,
    *,
    window_radius: int,
    rank: int,
) -> RankedResult:
    focal_message = loaded_run.message_by_id[candidate.message_id]
    conversation = loaded_run.conversation_by_id[candidate.conversation_id]
    ordered_messages = loaded_run.messages_by_conversation_id[candidate.conversation_id]
    focal_index = _find_message_position(ordered_messages, candidate.message_id)

    start_index = max(0, focal_index - window_radius)
    end_index = min(len(ordered_messages) - 1, focal_index + window_radius)
    window_messages = ordered_messages[start_index : end_index + 1]

    start_sequence = _message_sequence_index(window_messages[0])
    end_sequence = _message_sequence_index(window_messages[-1])
    focal_sequence = _message_sequence_index(focal_message)
    result_id = f"{loaded_run.run_id}:{candidate.message_id}:{start_sequence}:{end_sequence}"

    return RankedResult(
        rank=rank,
        score=candidate.score,
        result_id=result_id,
        run_id=loaded_run.run_id,
        provider=candidate.provider,
        conversation_id=candidate.conversation_id,
        conversation_title=_string_or_none(conversation.get("title")),
        conversation_created_at=_string_or_none(conversation.get("created_at")),
        conversation_updated_at=_string_or_none(conversation.get("updated_at")),
        focal_message_id=candidate.message_id,
        focal_created_at=_string_or_none(focal_message.get("created_at")),
        focal_sequence_index=focal_sequence,
        window_start_sequence_index=start_sequence,
        window_end_sequence_index=end_sequence,
        messages=tuple(window_messages),
        match_basis=candidate.match_basis,
        provenance={
            "conversations_jsonl_path": str(loaded_run.conversations_path),
            "messages_jsonl_path": str(loaded_run.messages_path),
            "manifest_json_path": str(loaded_run.manifest_path),
            "source_conversation_file": _source_conversation_file(conversation),
            "source_message_paths": [
                _source_message_path(message)
                for message in window_messages
            ],
        },
    )


# Apply the minimal first-slice metadata filters against one message.
def _message_matches_filters(message: dict[str, object], filters: RetrievalFilters) -> bool:
    if filters.provider and _string_or_none(message.get("provider")) != filters.provider:
        return False
    if filters.conversation_id and _string_or_none(message.get("conversation_id")) != filters.conversation_id:
        return False
    if filters.author_role and _string_or_none(message.get("author_role")) != filters.author_role:
        return False
    if not _message_matches_date_filters(message, filters.date_from, filters.date_to):
        return False
    return True


# Apply date filters using canonical UTC timestamps already present in phase-1 messages.
def _message_matches_date_filters(
    message: dict[str, object],
    date_from: str | None,
    date_to: str | None,
) -> bool:
    if not date_from and not date_to:
        return True

    created_at = _string_or_none(message.get("created_at"))
    if not created_at:
        return False

    created_value = _parse_iso_timestamp(created_at)
    if date_from and created_value < _parse_date_boundary(date_from, end_of_day=False):
        return False
    if date_to and created_value > _parse_date_boundary(date_to, end_of_day=True):
        return False
    return True


# Build the matched-filter payload returned with each result for debugging.
def _matched_filters_dict(filters: RetrievalFilters) -> dict[str, object]:
    matched: dict[str, object] = {}
    if filters.provider:
        matched["provider"] = filters.provider
    if filters.conversation_id:
        matched["conversation_id"] = filters.conversation_id
    if filters.date_from:
        matched["date_from"] = filters.date_from
    if filters.date_to:
        matched["date_to"] = filters.date_to
    if filters.author_role:
        matched["author_role"] = filters.author_role
    return matched


# Build a short excerpt from the focal message text for ranking inspection output.
def _build_excerpt(message: dict[str, object], overlap_tokens: tuple[str, ...]) -> str | None:
    text = _string_or_none(message.get("text")) or ""
    if not text:
        return None
    excerpt = text.strip()
    if len(excerpt) > 160:
        excerpt = excerpt[:157] + "..."
    if overlap_tokens:
        return excerpt
    return excerpt


# Sort candidates by score first and then by stable chronological/id tie-breakers.
def _candidate_sort_key(candidate: _Candidate) -> tuple[float, str, str]:
    return (-candidate.score, candidate.created_at or "", candidate.message_id)


# Find a focal message inside its ordered conversation stream.
def _find_message_position(messages: tuple[dict[str, object], ...], message_id: str) -> int:
    for index, message in enumerate(messages):
        if _string_or_none(message.get("message_id")) == message_id:
            return index
    return 0


# Read the sequence index from one canonical message.
def _message_sequence_index(message: dict[str, object]) -> int:
    value = message.get("sequence_index")
    if isinstance(value, int):
        return value
    return 0


# Read the source conversation file used to build one canonical conversation record.
def _source_conversation_file(conversation: dict[str, object]) -> str | None:
    source_artifact = conversation.get("source_artifact")
    if isinstance(source_artifact, dict):
        return _string_or_none(source_artifact.get("conversation_file"))
    return None


# Read the raw message path used to build one canonical message record.
def _source_message_path(message: dict[str, object]) -> str | None:
    source_artifact = message.get("source_artifact")
    if isinstance(source_artifact, dict):
        return _string_or_none(source_artifact.get("raw_message_path"))
    return None


# Parse canonical ISO timestamps produced by Phase 1.
def _parse_iso_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


# Parse either YYYY-MM-DD or full ISO timestamps for CLI date filters.
def _parse_date_boundary(value: str, *, end_of_day: bool) -> datetime:
    normalized_value = value.strip()
    if len(normalized_value) == 10:
        suffix = "T23:59:59.999999Z" if end_of_day else "T00:00:00Z"
        normalized_value = f"{normalized_value}{suffix}"
    parsed = datetime.fromisoformat(normalized_value.replace("Z", "+00:00"))
    return parsed.astimezone(timezone.utc)


# Normalize optional scalar values to strings where retrieval expects them.
def _string_or_none(value: object) -> str | None:
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    return None
