"""Shared types and constants for the retrieval layer."""

from __future__ import annotations

import string

from dataclasses import asdict, dataclass
from datetime import datetime


# BM25 constants stay explicit so ranking can be tuned without changing the retrieval contract.
BM25_K1 = 1.5
BM25_B = 0.75
EXACT_PHRASE_BOOST = 1.5
TITLE_TERM_BOOST = 0.35
RECENCY_BOOST_MAX = 0.35
USER_VOICE_BOOST = 1.25
ASSISTANT_VOICE_FACTOR = 0.8
ASSISTANT_META_COMMENTARY_FACTOR = 0.6
ASSISTANT_RESTATEMENT_FACTOR = 0.5
RESTATEMENT_OVERLAP_THRESHOLD = 0.60
RESTATEMENT_WINDOW_RADIUS = 5

# Opening phrases that signal assistant process text, reaction filler, or delivery mechanics.
# Matching is case-insensitive against the first 80 characters of the message text.
# Keep this list narrow: only patterns that are reliably low-information openers.
ASSISTANT_META_PREFIXES = (
    "i'd be happy to",
    "i'd be glad to",
    "i would be happy to",
    "sure, i can",
    "sure! i can",
    "sure, let me",
    "sure! let me",
    "of course! ",
    "of course, ",
    "absolutely! ",
    "absolutely, ",
    "great question",
    "great! let me",
    "that's a great",
    "that's a really great",
    "certainly! ",
    "certainly, ",
    "here's what i",
    "here is what i",
    "let me help you",
    "your formatted text file is ready",
    "your file is ready",
    "here's the updated",
    "here is the updated",
    "here's your updated",
    "i've updated",
    "i've created",
    "i've generated",
    "here are some",
    "here are a few",
    "based on what you've shared",
    "based on our conversation",
    "based on everything you've shared",
    "thank you for sharing",
    "thanks for sharing",
)


# Return True when the message text opens with a meta-commentary pattern.
def is_assistant_meta_commentary(text: str) -> bool:
    if not text:
        return False
    prefix = text[:80].lower().lstrip()
    return any(prefix.startswith(pattern) for pattern in ASSISTANT_META_PREFIXES)


_RESTATEMENT_STOPWORDS: frozenset[str] = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by", "can",
    "do", "does", "did", "for", "from", "had", "has", "have", "he", "her",
    "him", "his", "how", "i", "if", "in", "into", "is", "it", "its", "just",
    "me", "my", "no", "not", "of", "on", "or", "our", "out", "she", "so",
    "than", "that", "the", "their", "them", "then", "there", "they", "this",
    "to", "too", "up", "us", "very", "was", "we", "were", "what", "when",
    "where", "which", "who", "will", "with", "would", "you", "your",
})

_PUNCTUATION_TABLE = str.maketrans("", "", string.punctuation)


def _content_tokens(text: str) -> set[str]:
    """Lowercase, strip punctuation, tokenize, remove stopwords."""
    words = text.lower().translate(_PUNCTUATION_TABLE).split()
    return {w for w in words if w and w not in _RESTATEMENT_STOPWORDS}


def is_assistant_restatement(assistant_text: str, user_texts: list[str]) -> bool:
    """Return True if *assistant_text* substantially overlaps any single user text.

    Overlap is measured as the fraction of the assistant's content tokens that
    also appear in a nearby user message.  If that fraction exceeds
    ``RESTATEMENT_OVERLAP_THRESHOLD`` for any user text, the assistant message
    is classified as a restatement.
    """
    assistant_tokens = _content_tokens(assistant_text)
    if not assistant_tokens:
        return False

    for user_text in user_texts:
        user_tokens = _content_tokens(user_text)
        if not user_tokens:
            continue
        overlap = len(assistant_tokens & user_tokens)
        ratio = overlap / len(assistant_tokens)
        if ratio >= RESTATEMENT_OVERLAP_THRESHOLD:
            return True

    return False


def get_nearby_user_texts(
    ordered_messages: tuple[dict[str, object], ...],
    focal_index: int,
) -> list[str]:
    """Return text of user-role messages within RESTATEMENT_WINDOW_RADIUS of *focal_index*."""
    start = max(0, focal_index - RESTATEMENT_WINDOW_RADIUS)
    end = min(len(ordered_messages), focal_index + RESTATEMENT_WINDOW_RADIUS + 1)
    texts: list[str] = []
    for i in range(start, end):
        if i == focal_index:
            continue
        msg = ordered_messages[i]
        if (msg.get("author_role") or "") == "user":
            text = msg.get("text") or ""
            if text:
                texts.append(str(text))
    return texts


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
