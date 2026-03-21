"""BM25 scoring, sort keys, and retrieval mode application for the retrieval layer."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

from rag.retrieval.read_model import LoadedRun, tokenize_query
from rag.retrieval.types import (
    BM25_K1,
    BM25_B,
    RECENCY_BOOST_MAX,
    _Candidate,
)


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
def build_bm25_scorer(loaded_run: LoadedRun, query_terms: tuple[str, ...]) -> _BM25Scorer:
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
def score_message_bm25(
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


# Sort candidates by score first and then by stable chronological/id tie-breakers.
def candidate_sort_key(candidate: _Candidate) -> tuple[float, str, str]:
    return (-candidate.score, candidate.created_at or "", candidate.message_id)


# Sort hybrid candidates by overlap first, then by strongest available score, then stable chronology/id.
def hybrid_candidate_sort_key(candidate: _Candidate) -> tuple[float, float, str, str]:
    return (-len(candidate.retrieval_sources), -candidate.base_score, candidate.created_at or "", candidate.message_id)


# Sort candidates chronologically while keeping lexical score as a stable secondary tie-breaker.
def candidate_sort_key_chronological(candidate: _Candidate, *, descending: bool) -> tuple[float, float, str]:
    if candidate.created_at_value is None:
        timestamp = float("-inf") if descending else float("inf")
    else:
        timestamp = candidate.created_at_value.timestamp()
    return ((-timestamp) if descending else timestamp, -candidate.base_score, candidate.message_id)


# Render a short label describing which field currently controls chronological ordering.
def chronological_rank_basis(mode: str) -> str:
    if mode == "newest":
        return "created_at_desc"
    if mode == "oldest":
        return "created_at_asc"
    if mode == "relevance_recency":
        return "relevance_score_desc_plus_recency_boost"
    return "relevance_score_desc"


# Sort candidates for each retrieval view while keeping chronological behavior explicit.
def sort_candidates(candidates: list[_Candidate], *, mode: str) -> list[_Candidate]:
    if mode == "newest":
        return sorted(candidates, key=lambda candidate: candidate_sort_key_chronological(candidate, descending=True))
    if mode in {"oldest", "timeline"}:
        return sorted(candidates, key=lambda candidate: candidate_sort_key_chronological(candidate, descending=False))
    return sorted(candidates, key=candidate_sort_key)


# Stamp the active retrieval mode details onto one candidate without changing its lexical inputs.
def with_mode_details(
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
    scoring_features["chronological_rank_basis"] = chronological_rank_basis(mode)
    scoring_features["bm25_score"] = candidate.bm25_score
    scoring_features["semantic_similarity"] = candidate.semantic_similarity

    match_basis = dict(candidate.match_basis)
    match_basis["scoring_features"] = scoring_features
    match_basis["retrieval_sources"] = list(candidate.retrieval_sources)

    return _Candidate(
        score=ranking_score,
        base_score=candidate.base_score,
        recency_boost=recency_boost,
        bm25_score=candidate.bm25_score,
        semantic_similarity=candidate.semantic_similarity,
        retrieval_sources=candidate.retrieval_sources,
        provider=candidate.provider,
        conversation_id=candidate.conversation_id,
        message_id=candidate.message_id,
        created_at=candidate.created_at,
        created_at_value=candidate.created_at_value,
        match_basis=match_basis,
    )


# Apply the requested retrieval mode while keeping the candidate pool unchanged.
def apply_retrieval_mode(candidates: list[_Candidate], mode: str) -> list[_Candidate]:
    if mode == "relevance":
        return sort_candidates(
            [with_mode_details(candidate, mode=mode, ranking_score=candidate.base_score, recency_boost=0.0) for candidate in candidates],
            mode=mode,
        )
    if mode == "newest":
        return sort_candidates(
            [with_mode_details(candidate, mode=mode, ranking_score=candidate.base_score, recency_boost=0.0) for candidate in candidates],
            mode=mode,
        )
    if mode == "oldest":
        return sort_candidates(
            [with_mode_details(candidate, mode=mode, ranking_score=candidate.base_score, recency_boost=0.0) for candidate in candidates],
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
            with_mode_details(candidate, mode="relevance_recency", ranking_score=candidate.base_score, recency_boost=0.0)
            for candidate in candidates
        ]
        return sorted(scored_candidates, key=candidate_sort_key)

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
            with_mode_details(
                candidate,
                mode="relevance_recency",
                ranking_score=candidate.base_score + recency_boost,
                recency_boost=recency_boost,
            )
        )

    return sorted(scored_candidates, key=candidate_sort_key)
