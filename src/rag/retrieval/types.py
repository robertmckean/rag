"""Shared types and constants for the retrieval layer."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime


# BM25 constants stay explicit so ranking can be tuned without changing the retrieval contract.
BM25_K1 = 1.5
BM25_B = 0.75
EXACT_PHRASE_BOOST = 1.5
TITLE_TERM_BOOST = 0.35
RECENCY_BOOST_MAX = 0.35
WINDOW_RETRIEVAL_MODES = ("relevance", "newest", "oldest", "relevance_recency")
TIMELINE_RETRIEVAL_MODE = "timeline"
CLI_RETRIEVAL_MODES = WINDOW_RETRIEVAL_MODES + (TIMELINE_RETRIEVAL_MODE,)
RETRIEVAL_CHANNELS = ("bm25", "semantic", "hybrid")
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
    bm25_score: float | None
    semantic_similarity: float | None
    retrieval_sources: tuple[str, ...]
    provider: str
    conversation_id: str
    message_id: str
    created_at: str | None
    created_at_value: datetime | None
    match_basis: dict[str, object]
