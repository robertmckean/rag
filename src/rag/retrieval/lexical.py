"""Minimal lexical retrieval over normalized message records with contextual and timeline views."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from rag.retrieval.read_model import LoadedRun, load_normalized_run, normalize_lexical_text, tokenize_query
from rag.retrieval.utils import string_or_none


# Ranking is message-first because message text is the narrowest useful relevance signal.
# The default retrieval modes return contextual windows so humans and LLMs can see local conversation flow.
# Timeline mode is a separate flat chronology view for topic evolution across conversations.


# BM25 constants stay explicit so ranking can be tuned without changing the retrieval contract.
BM25_K1 = 1.5
BM25_B = 0.75
EXACT_PHRASE_BOOST = 1.5
TITLE_TERM_BOOST = 0.35
RECENCY_BOOST_MAX = 0.35
WINDOW_RETRIEVAL_MODES = ("relevance", "newest", "oldest", "relevance_recency")
TIMELINE_RETRIEVAL_MODE = "timeline"
CLI_RETRIEVAL_MODES = WINDOW_RETRIEVAL_MODES + (TIMELINE_RETRIEVAL_MODE,)
STOPWORD_FILTER_ENABLED = False
STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "how",
        "i",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "show",
        "that",
        "the",
        "to",
        "was",
        "what",
        "where",
        "with",
    }
)


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
class TimelineResult:
    # Timeline results intentionally stay compact because this mode is for flat chronology browsing.
    # Reusing the windowed result shape would force unused window fields into timeline-only output.
    # The separate type is therefore intentional rather than an incomplete refactor.
    rank: int
    score: float
    run_id: str
    provider: str
    conversation_id: str
    conversation_title: str | None
    focal_message_id: str
    focal_created_at: str | None
    focal_sequence_index: int
    author_role: str | None
    focal_excerpt: str | None
    match_basis: dict[str, object]
    provenance: dict[str, object]

    # Convert the timeline result into a JSON-serializable dict for CLI output.
    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ParsedQuery:
    raw_query: str
    normalized_query_terms: tuple[str, ...]
    scoring_terms: tuple[str, ...]
    stopword_filtered_terms: tuple[str, ...]
    quoted_phrases: tuple[str, ...]
    normalized_phrase_targets: tuple[str, ...]


@dataclass(frozen=True)
class _Candidate:
    score: float
    base_score: float
    recency_boost: float
    provider: str
    conversation_id: str
    message_id: str
    created_at: str | None
    created_at_value: datetime | None
    match_basis: dict[str, object]


# Hold the per-run statistics and scoring helpers for BM25 message ranking.
@dataclass(frozen=True)
class _BM25Scorer:
    query_terms: tuple[str, ...]
    document_count: int
    average_document_length: float
    document_frequency_by_term: dict[str, int]
    document_length_by_message_id: dict[str, int]
    term_frequency_by_message_id: dict[str, Counter[str]]


# Build the BM25 scorer over the already-loaded normalized run.
def _build_bm25_scorer(loaded_run: LoadedRun, query_terms: tuple[str, ...]) -> _BM25Scorer:
    document_frequency_by_term: dict[str, int] = {}
    document_length_by_message_id: dict[str, int] = {}
    term_frequency_by_message_id: dict[str, Counter[str]] = {}

    for message_id, searchable_text in loaded_run.searchable_text_by_message_id.items():
        tokens = tokenize_query(searchable_text)
        term_counts = Counter(tokens)
        term_frequency_by_message_id[message_id] = term_counts
        document_length_by_message_id[message_id] = len(tokens)

        for term in set(tokens):
            document_frequency_by_term[term] = document_frequency_by_term.get(term, 0) + 1

    document_count = len(term_frequency_by_message_id)
    total_document_length = sum(document_length_by_message_id.values())
    average_document_length = (total_document_length / document_count) if document_count else 0.0

    return _BM25Scorer(
        query_terms=query_terms,
        document_count=document_count,
        average_document_length=average_document_length,
        document_frequency_by_term=document_frequency_by_term,
        document_length_by_message_id=document_length_by_message_id,
        term_frequency_by_message_id=term_frequency_by_message_id,
    )


# Score one message with BM25 and return both the score and per-term contribution details.
def _score_message_bm25(
    scorer: _BM25Scorer,
    message_id: str,
) -> tuple[float, list[dict[str, object]]]:
    document_length = scorer.document_length_by_message_id.get(message_id, 0)
    term_frequencies = scorer.term_frequency_by_message_id.get(message_id, Counter())
    if not term_frequencies:
        return 0.0, []

    contributions: list[dict[str, object]] = []
    total_score = 0.0

    for term in scorer.query_terms:
        term_frequency = term_frequencies.get(term, 0)
        if term_frequency <= 0:
            continue

        document_frequency = scorer.document_frequency_by_term.get(term, 0)
        inverse_document_frequency = math.log(
            1.0 + ((scorer.document_count - document_frequency + 0.5) / (document_frequency + 0.5))
        )
        average_document_length = scorer.average_document_length or 1.0
        length_normalizer = BM25_K1 * (1.0 - BM25_B + BM25_B * (document_length / average_document_length))
        numerator = term_frequency * (BM25_K1 + 1.0)
        denominator = term_frequency + length_normalizer
        contribution = inverse_document_frequency * (numerator / denominator)
        total_score += contribution
        contributions.append(
            {
                "term": term,
                "term_frequency": term_frequency,
                "document_frequency": document_frequency,
                "inverse_document_frequency": round(inverse_document_frequency, 6),
                "contribution": round(contribution, 6),
            }
        )

    return total_score, contributions


# Load a run directory and execute lexical retrieval in one call.
def retrieve_message_windows(
    run_dir: Path,
    query: str,
    *,
    limit: int = 10,
    window_radius: int = 2,
    mode: str = "relevance",
    filters: RetrievalFilters | None = None,
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
) -> tuple[RankedResult, ...]:
    resolved_filters = filters or RetrievalFilters()
    parsed_query = parse_query(query)
    if not parsed_query.scoring_terms:
        return ()
    if mode == TIMELINE_RETRIEVAL_MODE:
        raise ValueError("Timeline mode is not supported by search_loaded_run(); use search_loaded_run_timeline().")
    if mode not in WINDOW_RETRIEVAL_MODES:
        raise ValueError(f"Unsupported retrieval mode: {mode}")

    candidates = _rank_candidates(loaded_run, parsed_query, resolved_filters)
    ordered_candidates = _apply_retrieval_mode(candidates, mode)

    results: list[RankedResult] = []
    seen_window_keys: set[tuple[str, int, int]] = set()

    for candidate in ordered_candidates:
        result = _build_window_result(
            loaded_run,
            candidate,
            window_radius=window_radius,
            mode=mode,
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


# Rank message-level candidates using simple lexical and metadata-aware signals.
def _rank_candidates(
    loaded_run: LoadedRun,
    parsed_query: ParsedQuery,
    filters: RetrievalFilters,
) -> list[_Candidate]:
    candidates: list[_Candidate] = []
    scorer = _build_bm25_scorer(loaded_run, parsed_query.scoring_terms)
    query_token_set = set(parsed_query.scoring_terms)

    for message_id, message in loaded_run.message_by_id.items():
        if not _message_matches_filters(message, filters):
            continue

        searchable_text = loaded_run.searchable_text_by_message_id.get(message_id, "")
        if not searchable_text:
            continue

        bm25_score, term_contributions = _score_message_bm25(scorer, message_id)
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

        score = bm25_score + exact_phrase_boost + title_boost
        if score <= 0.0:
            continue

        candidates.append(
            _Candidate(
                score=score,
                base_score=score,
                recency_boost=0.0,
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
                    },
                    "focal_match_excerpt": _build_excerpt(message),
                },
            )
        )

    return candidates


# Parse the raw query into normalized scoring terms and exact phrase targets.
def parse_query(raw_query: str) -> ParsedQuery:
    normalized_raw_query = " ".join(raw_query.split())
    quoted_phrases = tuple(match.group(1).strip() for match in _quoted_phrase_matches(normalized_raw_query) if match.group(1).strip())
    phrase_targets = tuple(
        normalized_phrase for normalized_phrase in (
            normalize_lexical_text(phrase) for phrase in quoted_phrases
        )
        if normalized_phrase
    )

    all_query_terms = tuple(_unique_preserving_order(tokenize_query(normalized_raw_query)))
    stopword_filtered_terms = tuple(term for term in all_query_terms if term not in STOPWORDS)
    scoring_terms = stopword_filtered_terms if STOPWORD_FILTER_ENABLED and stopword_filtered_terms else all_query_terms

    return ParsedQuery(
        raw_query=raw_query,
        normalized_query_terms=all_query_terms,
        scoring_terms=scoring_terms,
        stopword_filtered_terms=stopword_filtered_terms,
        quoted_phrases=quoted_phrases,
        normalized_phrase_targets=phrase_targets,
    )


# Apply the requested retrieval mode while keeping the candidate pool unchanged.
def _apply_retrieval_mode(candidates: list[_Candidate], mode: str) -> list[_Candidate]:
    if mode == "relevance":
        return _sort_candidates(
            [_with_mode_details(candidate, mode=mode, ranking_score=candidate.base_score, recency_boost=0.0) for candidate in candidates],
            mode=mode,
        )
    if mode == "newest":
        return _sort_candidates(
            [_with_mode_details(candidate, mode=mode, ranking_score=candidate.base_score, recency_boost=0.0) for candidate in candidates],
            mode=mode,
        )
    if mode == "oldest":
        return _sort_candidates(
            [_with_mode_details(candidate, mode=mode, ranking_score=candidate.base_score, recency_boost=0.0) for candidate in candidates],
            mode=mode,
        )
    if mode == "relevance_recency":
        return _apply_relevance_recency_mode(candidates)
    raise ValueError(f"Unsupported retrieval mode: {mode}")


# Apply a modest recency boost on top of lexical relevance without changing the candidate pool.
def _apply_relevance_recency_mode(candidates: list[_Candidate]) -> list[_Candidate]:
    timestamp_values = [candidate.created_at_value for candidate in candidates if candidate.created_at_value is not None]
    if not timestamp_values:
        scored_candidates = [
            _with_mode_details(candidate, mode="relevance_recency", ranking_score=candidate.base_score, recency_boost=0.0)
            for candidate in candidates
        ]
        return sorted(scored_candidates, key=_candidate_sort_key)

    min_timestamp = min(timestamp_values)
    max_timestamp = max(timestamp_values)
    span_seconds = (max_timestamp - min_timestamp).total_seconds()

    scored_candidates: list[_Candidate] = []
    for candidate in candidates:
        recency_boost = 0.0
        if candidate.created_at_value is not None and span_seconds > 0.0:
            recency_position = (candidate.created_at_value - min_timestamp).total_seconds() / span_seconds
            recency_boost = RECENCY_BOOST_MAX * recency_position
        scored_candidates.append(
            _with_mode_details(
                candidate,
                mode="relevance_recency",
                ranking_score=candidate.base_score + recency_boost,
                recency_boost=recency_boost,
            )
        )

    return sorted(scored_candidates, key=_candidate_sort_key)


# Sort candidates for each retrieval view while keeping chronological behavior explicit.
def _sort_candidates(candidates: list[_Candidate], *, mode: str) -> list[_Candidate]:
    if mode == "newest":
        return sorted(candidates, key=lambda candidate: _candidate_sort_key_chronological(candidate, descending=True))
    if mode in {"oldest", TIMELINE_RETRIEVAL_MODE}:
        return sorted(candidates, key=lambda candidate: _candidate_sort_key_chronological(candidate, descending=False))
    return sorted(candidates, key=_candidate_sort_key)


# Stamp the active retrieval mode details onto one candidate without changing its lexical inputs.
def _with_mode_details(
    candidate: _Candidate,
    *,
    mode: str,
    ranking_score: float,
    recency_boost: float,
) -> _Candidate:
    scoring_features = dict(candidate.match_basis.get("scoring_features", {}))
    scoring_features["retrieval_mode"] = mode
    scoring_features["recency_boost"] = round(recency_boost, 6)
    scoring_features["relevance_score"] = round(candidate.base_score, 6)
    scoring_features["ranking_score"] = round(ranking_score, 6)
    scoring_features["final_score"] = round(ranking_score, 6)
    scoring_features["chronological_rank_basis"] = _chronological_rank_basis(mode)

    match_basis = dict(candidate.match_basis)
    match_basis["scoring_features"] = scoring_features

    return _Candidate(
        score=ranking_score,
        base_score=candidate.base_score,
        recency_boost=recency_boost,
        provider=candidate.provider,
        conversation_id=candidate.conversation_id,
        message_id=candidate.message_id,
        created_at=candidate.created_at,
        created_at_value=candidate.created_at_value,
        match_basis=match_basis,
    )


# Expand one ranked focal message into the default conversation window result.
def _build_window_result(
    loaded_run: LoadedRun,
    candidate: _Candidate,
    *,
    window_radius: int,
    mode: str,
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


# Build a short excerpt from the focal message text for ranking inspection output.
def _build_excerpt(message: dict[str, object]) -> str | None:
    text = string_or_none(message.get("text")) or ""
    if not text:
        return None
    excerpt = text.strip()
    if len(excerpt) > 160:
        excerpt = excerpt[:157] + "..."
    return excerpt


# Find quoted phrases in the raw query without changing the surrounding tokenization rules.
def _quoted_phrase_matches(value: str) -> tuple[re.Match[str], ...]:
    return tuple(re.finditer(r'"([^"]+)"', value))


# Sort candidates by score first and then by stable chronological/id tie-breakers.
def _candidate_sort_key(candidate: _Candidate) -> tuple[float, str, str]:
    return (-candidate.score, candidate.created_at or "", candidate.message_id)


# Sort candidates chronologically while keeping lexical score as a stable secondary tie-breaker.
def _candidate_sort_key_chronological(candidate: _Candidate, *, descending: bool) -> tuple[float, float, str]:
    if candidate.created_at_value is None:
        timestamp = float("-inf") if descending else float("inf")
    else:
        timestamp = candidate.created_at_value.timestamp()
    return ((-timestamp) if descending else timestamp, -candidate.base_score, candidate.message_id)


# Render a short label describing which field currently controls chronological ordering.
def _chronological_rank_basis(mode: str) -> str:
    if mode == "newest":
        return "created_at_desc"
    if mode == "oldest":
        return "created_at_asc"
    if mode == "relevance_recency":
        return "relevance_score_desc_plus_recency_boost"
    return "relevance_score_desc"


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


# Remove duplicate tokens while preserving the user query's left-to-right order.
def _unique_preserving_order(values: tuple[str, ...]) -> tuple[str, ...]:
    seen: set[str] = set()
    unique_values: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        unique_values.append(value)
    return tuple(unique_values)
