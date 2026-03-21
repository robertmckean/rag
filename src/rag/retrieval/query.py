"""Query parsing for the retrieval layer."""

from __future__ import annotations

import re

from rag.retrieval.read_model import normalize_lexical_text, tokenize_query
from rag.retrieval.types import ParsedQuery, STOPWORD_FILTER_ENABLED, STOPWORDS


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


# Find quoted phrases in the raw query without changing the surrounding tokenization rules.
def _quoted_phrase_matches(value: str) -> tuple[re.Match[str], ...]:
    return tuple(re.finditer(r'"([^"]+)"', value))


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
