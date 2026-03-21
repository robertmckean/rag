"""Composed window support logic for conversational-memory grounding mode."""

from __future__ import annotations

from rag.answering.models import EvidenceItem
from rag.answering.qualify import (
    COMPOSED_SUPPORT_COVERAGE_THRESHOLD,
    MIN_SUPPORT_SCORE,
    _QualificationOutcome,
    coverage_ratio,
    focus_terms,
    has_combined_focus_support,
    is_qualified_evidence_item,
    is_speculative_excerpt,
    is_window_contributing_item,
    qualification_outcome_from_items,
    re_rank_evidence_items,
)
from rag.answering.diagnostics_types import _WindowQualificationDebug


# Qualify one window for conversational-memory composition using only same-window lexical coverage.
def qualify_window_composed_support(
    query: str,
    evidence_items: tuple[EvidenceItem, ...],
) -> _QualificationOutcome | None:
    focus_terms_value = focus_terms(query)
    if not evidence_items or not focus_terms_value:
        return None

    grouped_items: dict[str, list[EvidenceItem]] = {}
    for item in evidence_items:
        window_key = item.window_id or f"window:{item.source_result_rank}"
        grouped_items.setdefault(window_key, []).append(item)

    sorted_window_keys = sorted(
        grouped_items,
        key=lambda window_key: min(item.source_result_rank for item in grouped_items[window_key]),
    )
    for window_key in sorted_window_keys:
        window_items = tuple(grouped_items[window_key])
        analysis = analyze_window_composition(window_items, focus_terms_value)
        if not analysis.conversational_memory_qualification_passed:
            continue
        return qualification_outcome_from_items(
            re_rank_evidence_items(analysis.contributing_items),
            support_basis="window_composed",
            composition_used=True,
            coverage_terms=analysis.coverage_terms,
            coverage_ratio=analysis.coverage_ratio,
            window_id=window_key,
            status_reason="window_composed_support",
        )
    return None


# Analyze one selected retrieval window so the CLI can explain strict and conversational-memory qualification outcomes.
def analyze_window_composition(
    window_items: tuple[EvidenceItem, ...],
    focus_terms_value: tuple[str, ...],
) -> _WindowQualificationDebug:
    strict_qualified_items = tuple(
        item for item in window_items
        if is_qualified_evidence_item(item, focus_terms_value, window_items)
    )
    strict_passed = has_combined_focus_support(strict_qualified_items, focus_terms_value) if focus_terms_value else bool(strict_qualified_items)

    contributors = tuple(item for item in window_items if is_window_contributing_item(item))
    coverage_terms = tuple(sorted({term for item in contributors for term in item.matched_terms}))
    coverage_ratio_value = coverage_ratio(coverage_terms, focus_terms_value)
    user_contributors = tuple(item for item in contributors if item.author_role == "user")
    assistant_contributors = tuple(item for item in contributors if item.author_role == "assistant")
    user_coverage_terms = tuple(sorted({term for item in user_contributors for term in item.matched_terms}))
    user_authored_coverage_met_threshold = coverage_ratio(user_coverage_terms, focus_terms_value) >= COMPOSED_SUPPORT_COVERAGE_THRESHOLD

    failure_reasons: list[str] = []
    if not contributors:
        failure_reasons.append("no_direct_lexical_matches")
    if len(contributors) < 2 and not strict_passed:
        failure_reasons.append("only_one_contributing_excerpt")
    if strict_passed:
        conversational_memory_passed = False
    else:
        conversational_memory_passed = (
            coverage_ratio_value >= COMPOSED_SUPPORT_COVERAGE_THRESHOLD
            and len(user_contributors) >= 1
            and len(contributors) >= 2
            and user_authored_coverage_met_threshold
        )
    if coverage_ratio_value < COMPOSED_SUPPORT_COVERAGE_THRESHOLD:
        failure_reasons.append("coverage_below_threshold")
    if len(user_contributors) < 1:
        failure_reasons.append("no_user_authored_contributor")
    if not user_authored_coverage_met_threshold:
        failure_reasons.append("user_only_coverage_below_threshold")
    if focus_terms_value and not strict_passed:
        failure_reasons.append("strict_combined_concept_not_supported")
    if not strict_passed and not conversational_memory_passed:
        failure_reasons.append("no_window_level_composed_support")

    first_item = window_items[0] if window_items else None
    return _WindowQualificationDebug(
        candidate_rank=first_item.source_result_rank if first_item else 0,
        conversation_id=first_item.citation.conversation_id if first_item else "",
        focal_message_id=first_item.citation.message_id if first_item else None,
        window_id=first_item.window_id if first_item else None,
        strict_qualification_passed=strict_passed,
        conversational_memory_qualification_passed=conversational_memory_passed,
        contributing_items=contributors,
        coverage_terms=coverage_terms,
        coverage_ratio=coverage_ratio_value,
        user_excerpt_count=len(user_contributors),
        assistant_excerpt_count=len(assistant_contributors),
        user_authored_coverage_met_threshold=user_authored_coverage_met_threshold,
        failure_reasons=tuple(dict.fromkeys(failure_reasons)),
    )
