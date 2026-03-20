"""Deterministic grounded answering over one normalized run using Phase 2 retrieval."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from rag.answering.models import (
    AnswerRequest,
    AnswerResult,
    AnswerStatus,
    Citation,
    EvidenceItem,
    RetrievalSummary,
)
from rag.retrieval.lexical import RetrievalFilters, parse_query, retrieve_message_windows
from rag.retrieval.read_model import tokenize_query
from rag.retrieval.utils import string_or_none


# Phase 3A answers are retrieval-bounded summaries rather than a new search system.
# Evidence is selected from existing retrieval windows and then capped deterministically.
# Status is classified before answer text generation so the generator cannot smooth over uncertainty.

ANSWER_RETRIEVAL_MODES = ("relevance", "newest", "oldest")
DEFAULT_EVIDENCE_CAP = 5
MAX_EVIDENCE_PER_RESULT = 2
MAX_EXCERPT_LENGTH = 180
MIN_SUPPORT_SCORE = 0.5
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
class _StatusDecision:
    status: AnswerStatus
    gaps: tuple[str, ...]
    conflicts: tuple[str, ...]
    focus_terms: tuple[str, ...]


# Answer one natural-language query against a single normalized run using existing retrieval behavior.
def answer_query(
    run_dir: Path,
    query: str,
    *,
    retrieval_mode: str = "relevance",
    limit: int = 8,
    max_evidence: int = DEFAULT_EVIDENCE_CAP,
    filters: RetrievalFilters | None = None,
) -> AnswerResult:
    if retrieval_mode not in ANSWER_RETRIEVAL_MODES:
        raise ValueError(
            f"Unsupported answer retrieval mode: {retrieval_mode}. Supported modes: {', '.join(ANSWER_RETRIEVAL_MODES)}."
        )

    request = AnswerRequest(
        run_dir=str(run_dir.resolve()),
        query=query,
        retrieval_mode=retrieval_mode,
        limit=limit,
        max_evidence=max_evidence,
    )
    retrieval_results = retrieve_message_windows(
        run_dir.resolve(),
        query,
        limit=limit,
        mode=retrieval_mode,
        filters=filters,
    )
    selected_evidence = _select_evidence(retrieval_results, query, max_evidence=max_evidence)
    qualified_evidence = _qualify_evidence_items(query, selected_evidence)
    decision = _classify_answer_status(query, qualified_evidence)
    citations = _collect_citations(qualified_evidence)
    retrieval_summary = RetrievalSummary(
        run_id=_infer_run_id(run_dir, retrieval_results),
        retrieval_mode=retrieval_mode,
        retrieval_limit=limit,
        retrieved_result_count=len(retrieval_results),
        evidence_used_count=len(qualified_evidence),
    )
    answer_text = _generate_answer_text(decision, qualified_evidence)
    return AnswerResult(
        request=request,
        query=query,
        answer_status=decision.status,
        answer=answer_text,
        evidence_used=qualified_evidence,
        gaps=decision.gaps,
        conflicts=decision.conflicts,
        citations=citations,
        retrieval_summary=retrieval_summary,
    )


# Render one answer result into a compact review-friendly terminal summary.
def render_answer_result(result: AnswerResult) -> str:
    lines = [
        "Grounded Answer",
        f"query: {result.query}",
        f"retrieval_mode: {result.retrieval_summary.retrieval_mode}",
        f"answer_status: {result.answer_status.value}",
        f"answer: {result.answer}",
    ]
    if result.gaps:
        lines.append("gaps:")
        for gap in result.gaps:
            lines.append(f"  - {gap}")
    if result.conflicts:
        lines.append("conflicts:")
        for conflict in result.conflicts:
            lines.append(f"  - {conflict}")
    lines.append("citations:")
    if not result.citations:
        lines.append("  - (none)")
    else:
        for citation in result.citations:
            lines.append(
                "  - "
                f"{citation.provider} conversation={citation.conversation_id} "
                f"message={citation.message_id} created_at={citation.created_at} "
                f"excerpt={citation.excerpt}"
            )
    lines.append(
        "retrieval_summary: "
        f"run_id={result.retrieval_summary.run_id} "
        f"retrieved_result_count={result.retrieval_summary.retrieved_result_count} "
        f"evidence_used_count={result.retrieval_summary.evidence_used_count}"
    )
    return "\n".join(lines)


# Serialize one answer result deterministically for JSON output or tests.
def answer_result_json(result: AnswerResult) -> str:
    return json.dumps(result.to_dict(), ensure_ascii=True, indent=2, sort_keys=True) + "\n"


# Select compact evidence items from ranked retrieval windows in stable rank order.
def _select_evidence(results: tuple[object, ...], query: str, *, max_evidence: int) -> tuple[EvidenceItem, ...]:
    focus_terms = _focus_terms(query)
    evidence_items: list[EvidenceItem] = []
    seen_message_ids: set[str] = set()

    for result in results:
        messages = tuple(getattr(result, "messages", ()))
        ordered_messages = _ordered_support_messages(messages, focus_terms)
        kept_for_result = 0
        for message in ordered_messages:
            message_id = string_or_none(message.get("message_id"))
            excerpt = _message_excerpt(message)
            if not message_id or not excerpt or message_id in seen_message_ids:
                continue
            seen_message_ids.add(message_id)
            evidence_items.append(
                EvidenceItem(
                    rank=len(evidence_items) + 1,
                    source_result_rank=int(getattr(result, "rank", len(evidence_items) + 1)),
                    score=float(getattr(result, "score", 0.0)),
                    retrieval_mode=string_or_none(getattr(result, "match_basis", {}).get("retrieval_mode")) or "relevance",
                    author_role=string_or_none(message.get("author_role")),
                    matched_terms=tuple(_matched_focus_terms(excerpt, focus_terms)),
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


# Classify the answer status deterministically before any answer prose is generated.
def _classify_answer_status(query: str, evidence_items: tuple[EvidenceItem, ...]) -> _StatusDecision:
    focus_terms = _focus_terms(query)
    if not evidence_items:
        return _StatusDecision(
            status=AnswerStatus.INSUFFICIENT_EVIDENCE,
            gaps=("No relevant retrieval evidence was available for this question.",),
            conflicts=(),
            focus_terms=focus_terms,
        )

    strong_items = tuple(item for item in evidence_items if _is_strong_evidence_item(item, focus_terms))
    if not strong_items:
        return _StatusDecision(
            status=AnswerStatus.INSUFFICIENT_EVIDENCE,
            gaps=("The retrieved evidence was too weak or too indirect to support an answer.",),
            conflicts=(),
            focus_terms=focus_terms,
        )

    conflicts = _detect_conflicts(strong_items, focus_terms)
    if conflicts:
        return _StatusDecision(
            status=AnswerStatus.AMBIGUOUS,
            gaps=(),
            conflicts=conflicts,
            focus_terms=focus_terms,
        )

    covered_terms = {term for item in strong_items for term in item.matched_terms}
    uncovered_terms = tuple(term for term in focus_terms if term not in covered_terms)
    combined_focus_supported = _has_combined_focus_support(strong_items, focus_terms)
    requires_combined_support = _requires_combined_focus_support(query, focus_terms)

    narrow_single_item_case = len(focus_terms) <= 1 and len(strong_items) == 1
    if len(focus_terms) >= 2:
        if combined_focus_supported and not uncovered_terms:
            return _StatusDecision(
                status=AnswerStatus.SUPPORTED,
                gaps=(),
                conflicts=(),
                focus_terms=focus_terms,
            )
        if requires_combined_support and not combined_focus_supported:
            return _StatusDecision(
                status=AnswerStatus.INSUFFICIENT_EVIDENCE,
                gaps=("The retrieved evidence does not support the combined query concept directly.",),
                conflicts=(),
                focus_terms=focus_terms,
            )
    elif not uncovered_terms and (len(strong_items) >= 2 or narrow_single_item_case):
        return _StatusDecision(
            status=AnswerStatus.SUPPORTED,
            gaps=(),
            conflicts=(),
            focus_terms=focus_terms,
        )

    gaps: list[str] = []
    if uncovered_terms:
        gaps.append(f"The retrieved evidence does not directly cover: {', '.join(uncovered_terms)}.")
    if len(strong_items) == 1 and not narrow_single_item_case:
        gaps.append("Only one strong evidence item was available for this question.")
    if not gaps:
        gaps.append("The retrieved evidence is relevant but too thin for a fully supported answer.")
    return _StatusDecision(
        status=AnswerStatus.PARTIALLY_SUPPORTED,
        gaps=tuple(gaps),
        conflicts=(),
        focus_terms=focus_terms,
    )


# Drop selected evidence that does not directly address any meaningful normalized query term.
def _qualify_evidence_items(query: str, evidence_items: tuple[EvidenceItem, ...]) -> tuple[EvidenceItem, ...]:
    focus_terms = _focus_terms(query)
    qualified_items: list[EvidenceItem] = []
    for item in evidence_items:
        if _is_qualified_evidence_item(item, focus_terms, evidence_items):
            qualified_items.append(item)
    if _requires_combined_focus_support(query, focus_terms) and not _has_combined_focus_support(tuple(qualified_items), focus_terms):
        return ()
    ordered_items = sorted(
        qualified_items,
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
            score=item.score,
            retrieval_mode=item.retrieval_mode,
            author_role=item.author_role,
            matched_terms=item.matched_terms,
            citation=item.citation,
        )
        for index, item in enumerate(ordered_items, start=1)
    )


# Generate deterministic grounded answer prose from the selected evidence and precomputed status.
def _generate_answer_text(decision: _StatusDecision, evidence_items: tuple[EvidenceItem, ...]) -> str:
    primary_excerpt = evidence_items[0].citation.excerpt if evidence_items else None
    secondary_excerpt = evidence_items[1].citation.excerpt if len(evidence_items) > 1 else None

    if decision.status is AnswerStatus.INSUFFICIENT_EVIDENCE:
        return "I did not find enough relevant evidence in this run to answer that question confidently."
    if decision.status is AnswerStatus.AMBIGUOUS:
        conflicting_bits = []
        if primary_excerpt:
            conflicting_bits.append(f'"{primary_excerpt}"')
        if secondary_excerpt:
            conflicting_bits.append(f'"{secondary_excerpt}"')
        if conflicting_bits:
            return (
                "The retrieved evidence is mixed. "
                + "Conflicting relevant excerpts include "
                + " and ".join(conflicting_bits)
                + "."
            )
        return "The retrieved evidence is mixed and does not support a single unqualified answer."
    if decision.status is AnswerStatus.PARTIALLY_SUPPORTED:
        if primary_excerpt:
            return (
                "The retrieved evidence only partially answers the question. "
                f'The strongest relevant excerpt says "{primary_excerpt}".'
            )
        return "The retrieved evidence is relevant but too limited to justify a full answer."
    if primary_excerpt and secondary_excerpt:
        return (
            "Based on the retrieved conversation evidence, the strongest supporting excerpts say "
            f'"{primary_excerpt}" and "{secondary_excerpt}".'
        )
    if primary_excerpt:
        return (
            "Based on the retrieved conversation evidence, the strongest supporting excerpt says "
            f'"{primary_excerpt}".'
        )
    return "Based on the retrieved conversation evidence, the available support is limited but relevant."


# Deduplicate citations by message id while preserving evidence order.
def _collect_citations(evidence_items: tuple[EvidenceItem, ...]) -> tuple[Citation, ...]:
    citations: list[Citation] = []
    seen_message_ids: set[str] = set()
    for item in evidence_items:
        if item.citation.message_id in seen_message_ids:
            continue
        seen_message_ids.add(item.citation.message_id)
        citations.append(item.citation)
    return tuple(citations)


# Extract the most topical query terms for answer-status reasoning instead of reusing raw retrieval tokens directly.
def _focus_terms(query: str) -> tuple[str, ...]:
    parsed = parse_query(query)
    candidates = parsed.stopword_filtered_terms or parsed.normalized_query_terms
    # Ignore one-character tokenization artifacts such as possessive "'s" fragments.
    filtered = tuple(
        term
        for term in candidates
        if term not in QUESTION_STOPWORDS and len(term) >= MIN_MEANINGFUL_FOCUS_TERM_LENGTH
    )
    return filtered or candidates


# Pick messages from one retrieval window in the order that gives the answer layer the most useful local evidence.
def _ordered_support_messages(
    messages: tuple[dict[str, object], ...],
    focus_terms: tuple[str, ...],
) -> tuple[dict[str, object], ...]:
    indexed_messages = list(enumerate(messages))

    # Skip windows that do not contain any topical evidence for the query.
    focus_matching_items = [
        item for item in indexed_messages
        if _message_excerpt(item[1]) and _matched_focus_terms(_message_excerpt(item[1]) or "", focus_terms)
    ]
    if focus_terms and not focus_matching_items:
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
        excerpt = _message_excerpt(message) or ""
        has_focus_match = bool(_matched_focus_terms(excerpt, focus_terms))
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
        excerpt = _message_excerpt(message)
        if not excerpt:
            continue
        has_focus_match = bool(_matched_focus_terms(excerpt, focus_terms))
        same_author_family = bool(primary_focus_role) and string_or_none(message.get("author_role")) == primary_focus_role
        if has_focus_match or same_author_family:
            filtered.append(message)
    return tuple(filtered)


# Build one bounded excerpt for answer evidence and citations.
def _message_excerpt(message: dict[str, object]) -> str | None:
    text = string_or_none(message.get("text"))
    if not text:
        return None
    excerpt = " ".join(text.split())
    if len(excerpt) > MAX_EXCERPT_LENGTH:
        excerpt = excerpt[: MAX_EXCERPT_LENGTH - 3] + "..."
    return excerpt


# Find the focus-term overlap between a candidate excerpt and the user query focus terms.
def _matched_focus_terms(excerpt: str, focus_terms: tuple[str, ...]) -> tuple[str, ...]:
    excerpt_tokens = set(tokenize_query(excerpt))
    return tuple(term for term in focus_terms if term in excerpt_tokens)


# Treat an evidence item as strong only when it has score and topical overlap that can support a grounded answer.
def _is_strong_evidence_item(item: EvidenceItem, focus_terms: tuple[str, ...]) -> bool:
    if item.score < MIN_SUPPORT_SCORE:
        return False
    if not focus_terms:
        return True
    return bool(item.matched_terms)


# Treat an evidence item as qualified only when it directly covers a meaningful query term.
def _is_qualified_evidence_item(
    item: EvidenceItem,
    focus_terms: tuple[str, ...],
    evidence_items: tuple[EvidenceItem, ...],
) -> bool:
    if not focus_terms:
        return item.score >= MIN_SUPPORT_SCORE
    if not item.matched_terms:
        return False
    if len(focus_terms) >= 2 and item.author_role == "assistant":
        if _is_speculative_excerpt(item.citation.excerpt):
            return False
        if not _has_grounding_peer(item, evidence_items):
            return False
    return True


# Treat one multi-part query as fully supported only when a single strong item covers the combined concept.
def _has_combined_focus_support(evidence_items: tuple[EvidenceItem, ...], focus_terms: tuple[str, ...]) -> bool:
    if len(focus_terms) < 2:
        return bool(evidence_items)
    for item in evidence_items:
        if all(term in item.matched_terms for term in focus_terms):
            return True
    return False


# Require direct combined-concept support for possessive entity queries such as "X's daughter".
def _requires_combined_focus_support(query: str, focus_terms: tuple[str, ...]) -> bool:
    if len(focus_terms) < 2:
        return False
    normalized_query = query.lower()
    return bool(re.search(r"\b[a-z0-9]+'s\s+[a-z0-9]+", normalized_query))


# Reject assistant speculation and require nearby non-assistant grounding for assistant factual support.
def _has_grounding_peer(item: EvidenceItem, evidence_items: tuple[EvidenceItem, ...]) -> bool:
    for peer in evidence_items:
        if peer is item:
            continue
        if peer.source_result_rank != item.source_result_rank:
            continue
        if peer.author_role == "assistant":
            continue
        if _is_speculative_excerpt(peer.citation.excerpt):
            continue
        if set(peer.matched_terms) >= set(item.matched_terms):
            return True
    return False


# Treat hedged assistant language as non-supporting for entity-specific factual questions.
def _is_speculative_excerpt(excerpt: str) -> bool:
    excerpt_tokens = set(tokenize_query(excerpt))
    return bool(excerpt_tokens & SPECULATION_CUES)


# Detect direct positive/negative tension across selected excerpts so status does not hide conflicts.
def _detect_conflicts(evidence_items: tuple[EvidenceItem, ...], focus_terms: tuple[str, ...]) -> tuple[str, ...]:
    positive_excerpts: list[str] = []
    negative_excerpts: list[str] = []

    for item in evidence_items:
        excerpt_tokens = set(tokenize_query(item.citation.excerpt))
        if focus_terms and not any(term in excerpt_tokens for term in focus_terms):
            continue
        if excerpt_tokens & POSITIVE_CUES:
            positive_excerpts.append(item.citation.excerpt)
        if excerpt_tokens & NEGATIVE_CUES:
            negative_excerpts.append(item.citation.excerpt)

    if positive_excerpts and negative_excerpts:
        return (
            f'Positive and negative evidence both appear, including "{positive_excerpts[0]}" and "{negative_excerpts[0]}".',
        )
    return ()


# Read the run id from retrieval results when available and fall back to the run directory name otherwise.
def _infer_run_id(run_dir: Path, retrieval_results: tuple[object, ...]) -> str:
    if retrieval_results:
        run_id = string_or_none(getattr(retrieval_results[0], "run_id", None))
        if run_id:
            return run_id
    return run_dir.resolve().name
