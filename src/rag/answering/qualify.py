"""Evidence qualification, focus term logic, and qualification helpers for the answering layer."""

from __future__ import annotations

import re
from dataclasses import dataclass

from rag.answering.models import (
    Citation,
    EvidenceItem,
)
from rag.retrieval.lexical import parse_query
from rag.retrieval.read_model import tokenize_query
from rag.retrieval.utils import string_or_none


# Phase 3A constants for evidence qualification.
ANSWER_RETRIEVAL_MODES = ("relevance", "newest", "oldest")
GROUNDING_MODES = ("strict", "conversational_memory")
DEFAULT_GROUNDING_MODE = "strict"
DEFAULT_EVIDENCE_CAP = 5
MAX_EVIDENCE_PER_RESULT = 2
MAX_EXCERPT_LENGTH = 180
MIN_SUPPORT_SCORE = 0.5
COMPOSED_SUPPORT_COVERAGE_THRESHOLD = 0.75
QUESTION_STOPWORDS = frozenset(
    {
        "about",
        "did",
        "find",
        "have",
        "i",
        "me",
        "my",
        "said",
        "say",
        "show",
        "tell",
        "what",
        "where",
        "you",
    }
)
MIN_MEANINGFUL_FOCUS_TERM_LENGTH = 2
POSITIVE_CUES = frozenset({"better", "eased", "improved", "improving", "manageable", "resolved", "stable"})
NEGATIVE_CUES = frozenset({"draining", "exhausted", "harder", "tired", "unmanageable", "worse", "worsening"})
SPECULATION_CUES = frozenset({"maybe", "perhaps", "presumably", "probably", "someone", "somebody", "unclear"})


@dataclass(frozen=True)
class _QualificationOutcome:
    evidence_items: tuple[EvidenceItem, ...]
    support_basis: str | None
    composition_used: bool
    supporting_excerpt_count: int
    user_excerpt_count: int
    assistant_excerpt_count: int
    coverage_terms: tuple[str, ...]
    coverage_ratio: float
    window_id: str | None
    status_reason: str | None


# Extract the most topical query terms for answer-status reasoning instead of reusing raw retrieval tokens directly.
def focus_terms(query: str) -> tuple[str, ...]:
    parsed = parse_query(query)
    candidates = parsed.scoring_terms or parsed.stopword_filtered_terms or parsed.normalized_query_terms
    # Ignore one-character tokenization artifacts such as possessive "'s" fragments.
    filtered = tuple(
        term
        for term in candidates
        if term not in QUESTION_STOPWORDS and len(term) >= MIN_MEANINGFUL_FOCUS_TERM_LENGTH
    )
    return filtered or candidates


# Build one bounded excerpt for answer evidence and citations.
def message_excerpt(message: dict[str, object]) -> str | None:
    text = string_or_none(message.get("text"))
    if not text:
        return None
    excerpt = " ".join(text.split())
    if len(excerpt) > MAX_EXCERPT_LENGTH:
        excerpt = excerpt[: MAX_EXCERPT_LENGTH - 3] + "..."
    return excerpt


# Find the focus-term overlap between a candidate excerpt and the user query focus terms.
def matched_focus_terms(excerpt: str, focus_terms_value: tuple[str, ...]) -> tuple[str, ...]:
    excerpt_tokens = set(tokenize_query(excerpt))
    return tuple(term for term in focus_terms_value if term in excerpt_tokens)


# Pick messages from one retrieval window in the order that gives the answer layer the most useful local evidence.
def ordered_support_messages(
    messages: tuple[dict[str, object], ...],
    focus_terms_value: tuple[str, ...],
) -> tuple[dict[str, object], ...]:
    indexed_messages = list(enumerate(messages))

    # Skip windows that do not contain any topical evidence for the query.
    focus_matching_items = [
        item for item in indexed_messages
        if message_excerpt(item[1]) and matched_focus_terms(message_excerpt(item[1]) or "", focus_terms_value)
    ]
    if focus_terms_value and not focus_matching_items:
        return ()
    primary_focus_index, primary_focus_message = focus_matching_items[0]
    primary_focus_role = string_or_none(primary_focus_message.get("author_role"))

    # Prefer the focal message first, then other topical matches, then nearby same-author context.
    def sort_key(item: tuple[int, dict[str, object]]) -> tuple[int, int, str]:
        index, message = item
        distance = abs(index - primary_focus_index)
        role = string_or_none(message.get("author_role")) or ""
        role_priority = 0 if role == "user" else 1 if role == "assistant" else 2
        message_id = string_or_none(message.get("message_id")) or ""
        excerpt = message_excerpt(message) or ""
        has_focus_match = bool(matched_focus_terms(excerpt, focus_terms_value))
        same_author_family = bool(primary_focus_role) and role == primary_focus_role
        if message_id == string_or_none(primary_focus_message.get("message_id")):
            return (-1, role_priority, message_id)
        if has_focus_match:
            return (0, distance, message_id)
        if same_author_family:
            return (1, distance + role_priority, message_id)
        return (2, distance + role_priority, message_id)

    ordered = [message for _, message in sorted(indexed_messages, key=sort_key)]
    filtered: list[dict[str, object]] = []
    for message in ordered:
        excerpt = message_excerpt(message)
        if not excerpt:
            continue
        has_focus_match = bool(matched_focus_terms(excerpt, focus_terms_value))
        same_author_family = bool(primary_focus_role) and string_or_none(message.get("author_role")) == primary_focus_role
        if has_focus_match or same_author_family:
            filtered.append(message)
    return tuple(filtered)


# Select compact evidence items from ranked retrieval windows in stable rank order.
def select_evidence(results: tuple[object, ...], query: str, *, max_evidence: int) -> tuple[EvidenceItem, ...]:
    focus_terms_value = focus_terms(query)
    evidence_items: list[EvidenceItem] = []
    seen_message_ids: set[str] = set()

    for result in results:
        messages = tuple(getattr(result, "messages", ()))
        ordered_messages = ordered_support_messages(messages, focus_terms_value)
        kept_for_result = 0
        for message in ordered_messages:
            message_id = string_or_none(message.get("message_id"))
            excerpt = message_excerpt(message)
            if not message_id or not excerpt or message_id in seen_message_ids:
                continue
            seen_message_ids.add(message_id)
            evidence_items.append(
                EvidenceItem(
                    rank=len(evidence_items) + 1,
                    source_result_rank=int(getattr(result, "rank", len(evidence_items) + 1)),
                    window_id=string_or_none(getattr(result, "result_id", None)) or f"window:{getattr(result, 'rank', len(evidence_items) + 1)}",
                    score=float(getattr(result, "score", 0.0)),
                    retrieval_mode=string_or_none(getattr(result, "match_basis", {}).get("retrieval_mode")) or "relevance",
                    author_role=string_or_none(message.get("author_role")),
                    matched_terms=tuple(matched_focus_terms(excerpt, focus_terms_value)),
                    citation=Citation(
                        provider=string_or_none(message.get("provider")) or string_or_none(getattr(result, "provider")) or "unknown",
                        conversation_id=string_or_none(message.get("conversation_id"))
                        or string_or_none(getattr(result, "conversation_id"))
                        or "",
                        title=string_or_none(getattr(result, "conversation_title", None)),
                        message_id=message_id,
                        created_at=string_or_none(message.get("created_at")),
                        excerpt=excerpt,
                    ),
                )
            )
            kept_for_result += 1
            if kept_for_result >= MAX_EVIDENCE_PER_RESULT:
                break
            if len(evidence_items) >= max_evidence:
                return tuple(evidence_items)

    return tuple(evidence_items)


# Drop selected evidence that does not directly address any meaningful normalized query term.
def qualify_evidence_items(
    query: str,
    evidence_items: tuple[EvidenceItem, ...],
    *,
    grounding_mode: str,
) -> _QualificationOutcome:
    focus_terms_value = focus_terms(query)
    qualified_items: list[EvidenceItem] = []
    for item in evidence_items:
        if is_qualified_evidence_item(item, focus_terms_value, evidence_items):
            qualified_items.append(item)
    ordered_items = re_rank_evidence_items(tuple(qualified_items))
    requires_combined = requires_combined_focus_support(query, focus_terms_value)
    if has_combined_focus_support(ordered_items, focus_terms_value):
        return qualification_outcome_from_items(
            ordered_items,
            support_basis="single_excerpt",
            composition_used=False,
            coverage_terms=tuple(sorted({term for item in ordered_items for term in item.matched_terms})),
            coverage_ratio=coverage_ratio(tuple(sorted({term for item in ordered_items for term in item.matched_terms})), focus_terms_value),
            window_id=ordered_items[0].window_id if ordered_items else None,
            status_reason="single_excerpt_support",
        )
    if grounding_mode == "strict":
        if requires_combined:
            return empty_qualification_outcome(status_reason="combined_concept_not_supported")
        return qualification_outcome_from_items(
            ordered_items,
            support_basis="single_excerpt" if ordered_items else None,
            composition_used=False,
            coverage_terms=tuple(sorted({term for item in ordered_items for term in item.matched_terms})),
            coverage_ratio=coverage_ratio(tuple(sorted({term for item in ordered_items for term in item.matched_terms})), focus_terms_value),
            window_id=ordered_items[0].window_id if ordered_items else None,
            status_reason="strict_partial_support" if ordered_items else "no_qualified_evidence",
        )

    from rag.answering.compose import qualify_window_composed_support
    composed_outcome = qualify_window_composed_support(query, evidence_items)
    if composed_outcome is not None:
        return composed_outcome
    if requires_combined:
        return empty_qualification_outcome(status_reason="combined_concept_not_supported")
    return qualification_outcome_from_items(
        ordered_items,
        support_basis="single_excerpt" if ordered_items else None,
        composition_used=False,
        coverage_terms=tuple(sorted({term for item in ordered_items for term in item.matched_terms})),
        coverage_ratio=coverage_ratio(tuple(sorted({term for item in ordered_items for term in item.matched_terms})), focus_terms_value),
        window_id=ordered_items[0].window_id if ordered_items else None,
        status_reason="strict_partial_support" if ordered_items else "no_qualified_evidence",
    )


# Re-rank evidence items deterministically after qualification narrows the candidate set.
def re_rank_evidence_items(evidence_items: tuple[EvidenceItem, ...]) -> tuple[EvidenceItem, ...]:
    ordered_items = sorted(
        evidence_items,
        key=lambda item: (
            -len(item.matched_terms),
            1 if item.author_role == "assistant" else 0,
            -item.score,
            item.rank,
        ),
    )
    return tuple(
        EvidenceItem(
            rank=index,
            source_result_rank=item.source_result_rank,
            window_id=item.window_id,
            score=item.score,
            retrieval_mode=item.retrieval_mode,
            author_role=item.author_role,
            matched_terms=item.matched_terms,
            citation=item.citation,
        )
        for index, item in enumerate(ordered_items, start=1)
    )


# Build a stable empty qualification outcome when no evidence survives the current grounding policy.
def empty_qualification_outcome(*, status_reason: str) -> _QualificationOutcome:
    return _QualificationOutcome(
        evidence_items=(),
        support_basis=None,
        composition_used=False,
        supporting_excerpt_count=0,
        user_excerpt_count=0,
        assistant_excerpt_count=0,
        coverage_terms=(),
        coverage_ratio=0.0,
        window_id=None,
        status_reason=status_reason,
    )


# Convert a set of qualifying items into the standard qualification outcome shape.
def qualification_outcome_from_items(
    evidence_items: tuple[EvidenceItem, ...],
    *,
    support_basis: str | None,
    composition_used: bool,
    coverage_terms: tuple[str, ...],
    coverage_ratio_value: float | None = None,
    coverage_ratio: float = 0.0,
    window_id: str | None,
    status_reason: str,
) -> _QualificationOutcome:
    final_coverage_ratio = coverage_ratio_value if coverage_ratio_value is not None else coverage_ratio
    user_excerpt_count = sum(1 for item in evidence_items if item.author_role == "user")
    assistant_excerpt_count = sum(1 for item in evidence_items if item.author_role == "assistant")
    window_ids = {item.window_id for item in evidence_items if item.window_id}
    return _QualificationOutcome(
        evidence_items=evidence_items,
        support_basis=support_basis,
        composition_used=composition_used,
        supporting_excerpt_count=len(evidence_items),
        user_excerpt_count=user_excerpt_count,
        assistant_excerpt_count=assistant_excerpt_count,
        coverage_terms=coverage_terms,
        coverage_ratio=final_coverage_ratio,
        window_id=window_id if len(window_ids) <= 1 else None,
        status_reason=status_reason,
    )


# Treat an item as a composition contributor only when it adds direct grounded term coverage inside one window.
def is_window_contributing_item(item: EvidenceItem) -> bool:
    if item.score < MIN_SUPPORT_SCORE:
        return False
    if not item.matched_terms:
        return False
    if item.author_role == "assistant" and is_speculative_excerpt(item.citation.excerpt):
        return False
    return True


# Compute the ratio of meaningful query-term coverage supplied by one support unit.
def coverage_ratio(coverage_terms: tuple[str, ...], focus_terms_value: tuple[str, ...]) -> float:
    if not focus_terms_value:
        return 0.0
    return len(set(coverage_terms) & set(focus_terms_value)) / len(set(focus_terms_value))


# Treat an evidence item as strong only when it has score and topical overlap that can support a grounded answer.
def is_strong_evidence_item(item: EvidenceItem, focus_terms_value: tuple[str, ...]) -> bool:
    if item.score < MIN_SUPPORT_SCORE:
        return False
    if not focus_terms_value:
        return True
    return bool(item.matched_terms)


# Treat an evidence item as qualified only when it directly covers a meaningful query term.
def is_qualified_evidence_item(
    item: EvidenceItem,
    focus_terms_value: tuple[str, ...],
    evidence_items: tuple[EvidenceItem, ...],
) -> bool:
    if not focus_terms_value:
        return item.score >= MIN_SUPPORT_SCORE
    if not item.matched_terms:
        return False
    if len(focus_terms_value) >= 2 and item.author_role == "assistant":
        if is_speculative_excerpt(item.citation.excerpt):
            return False
        if not has_grounding_peer(item, evidence_items):
            return False
    return True


# Treat one multi-part query as fully supported only when a single strong item covers the combined concept.
def has_combined_focus_support(evidence_items: tuple[EvidenceItem, ...], focus_terms_value: tuple[str, ...]) -> bool:
    if len(focus_terms_value) < 2:
        return bool(evidence_items)
    for item in evidence_items:
        if all(term in item.matched_terms for term in focus_terms_value):
            return True
    return False


# Require direct combined-concept support for possessive entity queries such as "X's daughter".
def requires_combined_focus_support(query: str, focus_terms_value: tuple[str, ...]) -> bool:
    if len(focus_terms_value) < 2:
        return False
    normalized_query = query.lower()
    return bool(re.search(r"\b[a-z0-9]+'s\s+[a-z0-9]+", normalized_query))


# Reject assistant speculation and require nearby non-assistant grounding for assistant factual support.
def has_grounding_peer(item: EvidenceItem, evidence_items: tuple[EvidenceItem, ...]) -> bool:
    for peer in evidence_items:
        if peer is item:
            continue
        if peer.source_result_rank != item.source_result_rank:
            continue
        if peer.author_role == "assistant":
            continue
        if is_speculative_excerpt(peer.citation.excerpt):
            continue
        if set(peer.matched_terms) >= set(item.matched_terms):
            return True
    return False


# Treat hedged assistant language as non-supporting for entity-specific factual questions.
def is_speculative_excerpt(excerpt: str) -> bool:
    excerpt_tokens = set(tokenize_query(excerpt))
    return bool(excerpt_tokens & SPECULATION_CUES)


# Deduplicate citations by message id while preserving evidence order.
def collect_citations(evidence_items: tuple[EvidenceItem, ...]) -> tuple[Citation, ...]:
    citations: list[Citation] = []
    seen_message_ids: set[str] = set()
    for item in evidence_items:
        if item.citation.message_id in seen_message_ids:
            continue
        seen_message_ids.add(item.citation.message_id)
        citations.append(item.citation)
    return tuple(citations)
