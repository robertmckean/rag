"""Lexical retrieval orchestrator over normalized message records with contextual and timeline views.

This module is the public API surface for retrieval. Consumers should import from here.
Internal scoring, query parsing, semantic ranking, and type definitions live in submodules.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from rag.embeddings.client import DEFAULT_EMBEDDING_MODEL
from rag.retrieval.read_model import LoadedRun, load_normalized_run, normalize_lexical_text, tokenize_query
from rag.retrieval.utils import string_or_none

# --- Re-exports for downstream consumers (do not remove) ---
from rag.retrieval.types import (  # noqa: F401
    BM25_K1,
    BM25_B,
    EXACT_PHRASE_BOOST,
    TITLE_TERM_BOOST,
    RECENCY_BOOST_MAX,
    USER_VOICE_BOOST,
    ASSISTANT_VOICE_FACTOR,
    ASSISTANT_META_COMMENTARY_FACTOR,
    is_assistant_meta_commentary,
    WINDOW_RETRIEVAL_MODES,
    TIMELINE_RETRIEVAL_MODE,
    CLI_RETRIEVAL_MODES,
    RETRIEVAL_CHANNELS,
    STOPWORD_FILTER_ENABLED,
    STOPWORDS,
    RetrievalFilters,
    RankedResult,
    TimelineResult,
    ParsedQuery,
    _Candidate,
)
from rag.retrieval.query import parse_query  # noqa: F401
from rag.retrieval.scoring import (
    build_bm25_scorer,
    score_message_bm25,
    apply_retrieval_mode,
    sort_candidates as _sort_candidates,
    candidate_sort_key as _candidate_sort_key,
    with_mode_details as _with_mode_details,
)
from rag.retrieval.semantic import (
    rank_semantic_candidates as _rank_semantic_candidates,
    merge_hybrid_candidates as _merge_hybrid_candidates,
)


# Ranking is message-first because message text is the narrowest useful relevance signal.
# The default retrieval modes return contextual windows so humans and LLMs can see local conversation flow.
# Timeline mode is a separate flat chronology view for topic evolution across conversations.


# Load a run directory and execute lexical retrieval in one call.
def retrieve_message_windows(
    run_dir: Path,
    query: str,
    *,
    limit: int = 10,
    window_radius: int = 2,
    mode: str = "relevance",
    filters: RetrievalFilters | None = None,
    channel: str = "bm25",
    semantic_top_k: int | None = None,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    embedding_client: object | None = None,
    ) -> tuple[RankedResult, ...]:
    if mode == TIMELINE_RETRIEVAL_MODE:
        raise ValueError("Timeline mode requires retrieve_message_timeline() or the CLI --mode timeline path.")
    loaded_run = load_normalized_run(run_dir)
    return search_loaded_run(
        loaded_run,
        query,
        limit=limit,
        window_radius=window_radius,
        mode=mode,
        filters=filters,
        channel=channel,
        semantic_top_k=semantic_top_k,
        embedding_model=embedding_model,
        embedding_client=embedding_client,
    )


# Load a run directory and execute timeline retrieval in one call.
def retrieve_message_timeline(
    run_dir: Path,
    query: str,
    *,
    limit: int = 10,
    filters: RetrievalFilters | None = None,
) -> tuple[TimelineResult, ...]:
    loaded_run = load_normalized_run(run_dir)
    return search_loaded_run_timeline(
        loaded_run,
        query,
        limit=limit,
        filters=filters,
    )


# Execute lexical retrieval against a loaded run and return contextual message windows.
def search_loaded_run(
    loaded_run: LoadedRun,
    query: str,
    *,
    limit: int = 10,
    window_radius: int = 2,
    mode: str = "relevance",
    filters: RetrievalFilters | None = None,
    channel: str = "bm25",
    semantic_top_k: int | None = None,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    embedding_client: object | None = None,
) -> tuple[RankedResult, ...]:
    resolved_filters = filters or RetrievalFilters()
    parsed_query = parse_query(query)
    if channel not in RETRIEVAL_CHANNELS:
        raise ValueError(f"Unsupported retrieval channel: {channel}")
    if mode == TIMELINE_RETRIEVAL_MODE:
        raise ValueError("Timeline mode is not supported by search_loaded_run(); use search_loaded_run_timeline().")
    if mode not in WINDOW_RETRIEVAL_MODES:
        raise ValueError(f"Unsupported retrieval mode: {mode}")
    if channel == "bm25" and not parsed_query.scoring_terms:
        return ()
    if mode == TIMELINE_RETRIEVAL_MODE and channel != "bm25":
        raise ValueError("Semantic and hybrid retrieval are not supported for timeline mode in Phase 4A.")

    candidates = _retrieve_candidates(
        loaded_run,
        parsed_query,
        resolved_filters,
        channel=channel,
        semantic_top_k=semantic_top_k or limit,
        embedding_model=embedding_model,
        embedding_client=embedding_client,
    )
    ordered_candidates = apply_retrieval_mode(candidates, mode)

    results: list[RankedResult] = []
    accepted_windows: dict[str, list[tuple[int, int]]] = {}
    accepted_focal_texts: list[str] = []

    for candidate in ordered_candidates:
        result = _build_window_result(
            loaded_run,
            candidate,
            window_radius=window_radius,
            mode=mode,
            rank=len(results) + 1,
            channel=channel,
        )
        if _focal_already_visible(result, accepted_windows):
            continue
        focal_text = _get_focal_text(loaded_run, candidate.message_id)
        if _focal_text_is_near_duplicate(focal_text, accepted_focal_texts):
            continue
        accepted_windows.setdefault(result.conversation_id, []).append(
            (result.window_start_sequence_index, result.window_end_sequence_index)
        )
        if focal_text:
            accepted_focal_texts.append(focal_text)
        results.append(result)
        if len(results) >= limit:
            break

    return tuple(results)


# Skip a candidate whose focal message is already visible inside an accepted window.
# This prevents near-duplicate windows from dense conversation threads without rejecting
# legitimately distinct windows whose edges happen to share a few context messages.
def _focal_already_visible(
    result: RankedResult,
    accepted_windows: dict[str, list[tuple[int, int]]],
) -> bool:
    prior_ranges = accepted_windows.get(result.conversation_id)
    if not prior_ranges:
        return False
    focal = result.focal_sequence_index
    for accepted_start, accepted_end in prior_ranges:
        if accepted_start <= focal <= accepted_end:
            return True
    return False


# Extract the normalized focal message text for cross-provider dedup comparison.
def _get_focal_text(loaded_run: LoadedRun, message_id: str) -> str:
    message = loaded_run.message_by_id.get(message_id)
    if not message:
        return ""
    text = string_or_none(message.get("text")) or ""
    return normalize_lexical_text(text)


# Minimum prefix length for near-duplicate detection.  Two focal messages whose normalized
# text shares at least this many leading characters are treated as cross-provider duplicates.
# 100 characters is long enough to avoid false positives on short generic phrases while
# catching the verbatim user messages that appear in both ChatGPT and Claude exports.
CROSS_PROVIDER_DEDUP_PREFIX_LENGTH = 100


# Return True when the focal message text is a near-duplicate of any already-accepted result.
# Near-duplicate means the first N characters of normalized text match, which catches the
# common pattern of the same user message sent to both ChatGPT and Claude.
def _focal_text_is_near_duplicate(focal_text: str, accepted_texts: list[str]) -> bool:
    if not focal_text or len(focal_text) < CROSS_PROVIDER_DEDUP_PREFIX_LENGTH:
        return False
    prefix = focal_text[:CROSS_PROVIDER_DEDUP_PREFIX_LENGTH]
    for accepted in accepted_texts:
        if accepted[:CROSS_PROVIDER_DEDUP_PREFIX_LENGTH] == prefix:
            return True
    return False


# Execute lexical retrieval against a loaded run and return focal-message timeline results.
def search_loaded_run_timeline(
    loaded_run: LoadedRun,
    query: str,
    *,
    limit: int = 10,
    filters: RetrievalFilters | None = None,
) -> tuple[TimelineResult, ...]:
    resolved_filters = filters or RetrievalFilters()
    parsed_query = parse_query(query)
    if not parsed_query.scoring_terms:
        return ()

    candidates = _rank_candidates(loaded_run, parsed_query, resolved_filters)
    ordered_candidates = _sort_candidates(candidates, mode="timeline")

    results: list[TimelineResult] = []
    seen_message_ids: set[str] = set()
    for candidate in ordered_candidates:
        # Timeline mode is focal-message-oriented, so dedupe only by focal message id.
        if candidate.message_id in seen_message_ids:
            continue
        seen_message_ids.add(candidate.message_id)
        results.append(_build_timeline_result(loaded_run, candidate, rank=len(results) + 1))
        if len(results) >= limit:
            break
    return tuple(results)


# --- Candidate ranking (BM25 lexical path) ---


# Rank message-level candidates using simple lexical and metadata-aware signals.
def _rank_candidates(
    loaded_run: LoadedRun,
    parsed_query: ParsedQuery,
    filters: RetrievalFilters,
) -> list[_Candidate]:
    candidates: list[_Candidate] = []
    scorer = build_bm25_scorer(loaded_run, parsed_query.scoring_terms)
    query_token_set = set(parsed_query.scoring_terms)

    for message_id, message in loaded_run.message_by_id.items():
        if not _message_matches_filters(message, filters):
            continue

        searchable_text = loaded_run.searchable_text_by_message_id.get(message_id, "")
        if not searchable_text:
            continue

        bm25_score, term_contributions = score_message_bm25(scorer, message_id)
        overlap_tokens = tuple(contribution["term"] for contribution in term_contributions)
        matched_phrase_targets = tuple(
            phrase_target for phrase_target in parsed_query.normalized_phrase_targets if phrase_target in searchable_text
        )
        exact_phrase_match = bool(matched_phrase_targets)

        conversation = loaded_run.conversation_by_id.get(string_or_none(message.get("conversation_id")) or "")
        conversation_title = string_or_none(conversation.get("title")) if conversation else None
        normalized_title = normalize_lexical_text(conversation_title or "")
        title_overlap_tokens = tuple(sorted(query_token_set & set(tokenize_query(normalized_title))))

        exact_phrase_boost = EXACT_PHRASE_BOOST * len(matched_phrase_targets)
        title_boost = TITLE_TERM_BOOST * len(title_overlap_tokens)

        # Title overlap can boost a real message hit, but it should not create a hit by itself.
        if bm25_score <= 0.0 and exact_phrase_boost <= 0.0:
            continue

        raw_score = bm25_score + exact_phrase_boost + title_boost
        if raw_score <= 0.0:
            continue

        author_role = string_or_none(message.get("author_role"))
        voice_factor = USER_VOICE_BOOST if author_role == "user" else ASSISTANT_VOICE_FACTOR
        if author_role == "assistant" and is_assistant_meta_commentary(string_or_none(message.get("text")) or ""):
            voice_factor *= ASSISTANT_META_COMMENTARY_FACTOR
        score = raw_score * voice_factor

        candidates.append(
            _Candidate(
                score=score,
                base_score=score,
                recency_boost=0.0,
                bm25_score=round(bm25_score, 6),
                semantic_similarity=None,
                retrieval_sources=("bm25",),
                provider=string_or_none(message.get("provider")) or "unknown",
                conversation_id=string_or_none(message.get("conversation_id")) or "",
                message_id=message_id,
                created_at=string_or_none(message.get("created_at")),
                created_at_value=_parse_iso_timestamp(string_or_none(message.get("created_at")) or "")
                if string_or_none(message.get("created_at"))
                else None,
                match_basis={
                    "raw_query": parsed_query.raw_query,
                    "normalized_query_terms": list(parsed_query.normalized_query_terms),
                    "scoring_terms": list(parsed_query.scoring_terms),
                    "quoted_phrases": list(parsed_query.quoted_phrases),
                    "normalized_phrase_targets": list(parsed_query.normalized_phrase_targets),
                    "stopword_filter_enabled": STOPWORD_FILTER_ENABLED,
                    "stopword_filtered_terms": list(parsed_query.stopword_filtered_terms),
                    "matched_text_terms": list(overlap_tokens),
                    "matched_phrase_targets": list(matched_phrase_targets),
                    "matched_metadata_filters": _matched_filters_dict(filters),
                    "scoring_features": {
                        "query_terms": list(parsed_query.scoring_terms),
                        "bm25_k1": BM25_K1,
                        "bm25_b": BM25_B,
                        "document_length": scorer.document_length_by_message_id.get(message_id, 0),
                        "average_document_length": round(scorer.average_document_length, 6),
                        "bm25_term_contributions": term_contributions,
                        "bm25_score": round(bm25_score, 6),
                        "exact_phrase_match": exact_phrase_match,
                        "matched_phrase_count": len(matched_phrase_targets),
                        "exact_phrase_boost": exact_phrase_boost,
                        "title_overlap": len(title_overlap_tokens),
                        "title_boost": title_boost,
                        "retrieval_mode": "relevance",
                        "recency_boost": 0.0,
                        "chronological_rank_basis": "relevance_score_desc",
                        "relevance_score": round(score, 6),
                        "ranking_score": round(score, 6),
                        "final_score": round(score, 6),
                        "bm25_score": round(score, 6),
                        "semantic_similarity": None,
                    },
                    "focal_match_excerpt": _build_excerpt(message),
                    "retrieval_sources": ["bm25"],
                },
            )
        )

    return candidates


# --- Channel routing ---


# Choose the requested retrieval channel while preserving the existing lexical path.
def _retrieve_candidates(
    loaded_run: LoadedRun,
    parsed_query: ParsedQuery,
    filters: RetrievalFilters,
    *,
    channel: str,
    semantic_top_k: int,
    embedding_model: str,
    embedding_client: object | None,
) -> list[_Candidate]:
    bm25_candidates = _rank_candidates(loaded_run, parsed_query, filters) if parsed_query.scoring_terms else []
    if channel == "bm25":
        return bm25_candidates

    semantic_candidates = _rank_semantic_candidates(
        loaded_run,
        parsed_query,
        filters,
        limit=semantic_top_k,
        embedding_model=embedding_model,
        embedding_client=embedding_client,
        message_matches_filters_fn=_message_matches_filters,
    )
    if channel == "semantic":
        return semantic_candidates
    return _merge_hybrid_candidates(bm25_candidates, semantic_candidates)


# --- Window and timeline result builders ---


# Expand one ranked focal message into the default conversation window result.
def _build_window_result(
    loaded_run: LoadedRun,
    candidate: _Candidate,
    *,
    window_radius: int,
    mode: str,
    rank: int,
    channel: str,
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
        conversation_title=string_or_none(conversation.get("title")),
        conversation_created_at=string_or_none(conversation.get("created_at")),
        conversation_updated_at=string_or_none(conversation.get("updated_at")),
        focal_message_id=candidate.message_id,
        focal_created_at=string_or_none(focal_message.get("created_at")),
        focal_sequence_index=focal_sequence,
        window_start_sequence_index=start_sequence,
        window_end_sequence_index=end_sequence,
        messages=tuple(window_messages),
        match_basis={
            **candidate.match_basis,
            "retrieval_mode": mode,
            "result_view": "contextual_window",
            "retrieval_channel": channel,
        },
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


# Build one compact focal-message result for cross-conversation timeline exploration.
def _build_timeline_result(
    loaded_run: LoadedRun,
    candidate: _Candidate,
    *,
    rank: int,
) -> TimelineResult:
    focal_message = loaded_run.message_by_id[candidate.message_id]
    conversation = loaded_run.conversation_by_id[candidate.conversation_id]
    match_basis = {
        **candidate.match_basis,
        "retrieval_mode": "timeline",
        "result_view": "timeline_compact",
    }
    scoring_features = dict(match_basis.get("scoring_features", {}))
    scoring_features["chronological_rank_basis"] = "created_at_asc_across_conversations"
    match_basis["scoring_features"] = scoring_features

    return TimelineResult(
        rank=rank,
        score=candidate.base_score,
        run_id=loaded_run.run_id,
        provider=candidate.provider,
        conversation_id=candidate.conversation_id,
        conversation_title=string_or_none(conversation.get("title")),
        focal_message_id=candidate.message_id,
        focal_created_at=string_or_none(focal_message.get("created_at")),
        focal_sequence_index=_message_sequence_index(focal_message),
        author_role=string_or_none(focal_message.get("author_role")),
        focal_excerpt=_build_excerpt(focal_message),
        match_basis=match_basis,
        provenance={
            "conversations_jsonl_path": str(loaded_run.conversations_path),
            "messages_jsonl_path": str(loaded_run.messages_path),
            "manifest_json_path": str(loaded_run.manifest_path),
            "source_conversation_file": _source_conversation_file(conversation),
            "source_message_paths": [_source_message_path(focal_message)],
        },
    )


# --- Filtering helpers ---


# Apply the minimal first-slice metadata filters against one message.
def _message_matches_filters(message: dict[str, object], filters: RetrievalFilters) -> bool:
    if filters.provider and string_or_none(message.get("provider")) != filters.provider:
        return False
    if filters.conversation_id and string_or_none(message.get("conversation_id")) != filters.conversation_id:
        return False
    if filters.author_role and string_or_none(message.get("author_role")) != filters.author_role:
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

    created_at = string_or_none(message.get("created_at"))
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


# --- Small helpers ---


# Build a short excerpt from the focal message text for ranking inspection output.
def _build_excerpt(message: dict[str, object]) -> str | None:
    text = string_or_none(message.get("text")) or ""
    if not text:
        return None
    excerpt = text.strip()
    if len(excerpt) > 160:
        excerpt = excerpt[:157] + "..."
    return excerpt


# Find a focal message inside its ordered conversation stream.
def _find_message_position(messages: tuple[dict[str, object], ...], message_id: str) -> int:
    for index, message in enumerate(messages):
        if string_or_none(message.get("message_id")) == message_id:
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
        return string_or_none(source_artifact.get("conversation_file"))
    return None


# Read the raw message path used to build one canonical message record.
def _source_message_path(message: dict[str, object]) -> str | None:
    source_artifact = message.get("source_artifact")
    if isinstance(source_artifact, dict):
        return string_or_none(source_artifact.get("raw_message_path"))
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
