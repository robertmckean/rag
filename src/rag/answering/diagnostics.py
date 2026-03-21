"""Debug payloads, qualification diagnostics, and render/debug helpers for the answering layer."""

from __future__ import annotations

from pathlib import Path

from rag.answering.compose import analyze_window_composition
from rag.answering.diagnostics_types import _WindowQualificationDebug
from rag.answering.models import (
    AnswerDiagnostics,
    EvidenceItem,
    EvidenceQualificationDiagnostic,
)
from rag.answering.qualify import (
    COMPOSED_SUPPORT_COVERAGE_THRESHOLD,
    DEFAULT_EVIDENCE_CAP,
    DEFAULT_GROUNDING_MODE,
    MIN_SUPPORT_SCORE,
    _QualificationOutcome,
    focus_terms,
    has_combined_focus_support,
    has_grounding_peer,
    is_speculative_excerpt,
    qualify_evidence_items,
    requires_combined_focus_support,
    select_evidence,
)
from rag.retrieval.lexical import RetrievalFilters, parse_query, retrieve_message_windows
from rag.retrieval.utils import string_or_none


# Record which selected evidence items were dropped and why so grounded-answer failures stay inspectable.
def build_answer_diagnostics(
    query: str,
    selected_evidence: tuple[EvidenceItem, ...],
    qualification: _QualificationOutcome,
) -> AnswerDiagnostics:
    focus_terms_value = focus_terms(query)
    qualified_message_ids = {item.citation.message_id for item in qualification.evidence_items}
    rejected_items: list[EvidenceQualificationDiagnostic] = []
    requires_combined = requires_combined_focus_support(query, focus_terms_value)

    for item in selected_evidence:
        if item.citation.message_id in qualified_message_ids:
            continue
        missing_focus_terms = tuple(term for term in focus_terms_value if term not in item.matched_terms)
        rejection_reasons: list[str] = []
        if not item.matched_terms:
            rejection_reasons.append("insufficient_directness")
        if item.score < MIN_SUPPORT_SCORE:
            rejection_reasons.append("filtered_by_threshold")
        if item.author_role == "assistant":
            if is_speculative_excerpt(item.citation.excerpt):
                rejection_reasons.append("assistant_authored_speculation")
            if not has_grounding_peer(item, selected_evidence):
                rejection_reasons.append("missing_non_assistant_grounding")
        if qualification.composition_used and qualification.window_id and item.window_id != qualification.window_id:
            rejection_reasons.append("outside_selected_support_window")
        if requires_combined and not has_combined_focus_support((item,), focus_terms_value):
            rejection_reasons.append("combined_concept_not_supported")
        if not rejection_reasons and item.matched_terms:
            rejection_reasons.append("window_coverage_below_threshold")
        if not rejection_reasons:
            rejection_reasons.append("filtered_by_group_qualification")
        rejected_items.append(
            EvidenceQualificationDiagnostic(
                message_id=item.citation.message_id,
                source_result_rank=item.source_result_rank,
                author_role=item.author_role,
                matched_terms=item.matched_terms,
                missing_focus_terms=missing_focus_terms,
                rejection_reasons=tuple(rejection_reasons),
                excerpt=item.citation.excerpt,
            )
        )

    return AnswerDiagnostics(
        focus_terms=focus_terms_value,
        selected_evidence_count=len(selected_evidence),
        qualified_evidence_count=len(qualification.evidence_items),
        support_basis=qualification.support_basis,
        composition_used=qualification.composition_used,
        supporting_excerpt_count=qualification.supporting_excerpt_count,
        user_excerpt_count=qualification.user_excerpt_count,
        assistant_excerpt_count=qualification.assistant_excerpt_count,
        coverage_terms=qualification.coverage_terms,
        coverage_ratio=qualification.coverage_ratio,
        window_id=qualification.window_id,
        status_reason=qualification.status_reason,
        rejected_evidence=tuple(rejected_items),
    )


# Build a structured qualification-debug payload without changing the normal answer contract.
def qualification_debug_payload(
    run_dir: Path,
    query: str,
    *,
    retrieval_mode: str = "relevance",
    grounding_mode: str = DEFAULT_GROUNDING_MODE,
    limit: int = 8,
    max_evidence: int = DEFAULT_EVIDENCE_CAP,
    filters: RetrievalFilters | None = None,
) -> dict[str, object]:
    parsed_query = parse_query(query)
    focus_terms_value = focus_terms(query)
    retrieval_results = retrieve_message_windows(
        run_dir.resolve(),
        query,
        limit=limit,
        mode=retrieval_mode,
        filters=filters,
    )
    selected_evidence = select_evidence(retrieval_results, query, max_evidence=max_evidence)
    qualification = qualify_evidence_items(query, selected_evidence, grounding_mode=grounding_mode)
    window_debug = build_window_debug_rows(
        retrieval_results,
        selected_evidence,
        focus_terms_value,
    )
    return {
        "query": query,
        "parsed_query_terms": list(parsed_query.normalized_query_terms),
        "scoring_terms": list(focus_terms_value),
        "grounding_mode": grounding_mode,
        "conversational_memory_threshold": COMPOSED_SUPPORT_COVERAGE_THRESHOLD,
        "retrieved_candidate_count": len(retrieval_results),
        "evidence_qualified_candidate_count": len(qualification.evidence_items),
        "windows": [
            {
                "candidate_rank": row.candidate_rank,
                "conversation_id": row.conversation_id,
                "focal_message_id": row.focal_message_id,
                "window_id": row.window_id,
                "strict_qualification_passed": row.strict_qualification_passed,
                "conversational_memory_qualification_passed": row.conversational_memory_qualification_passed,
                "contributing_items": [
                    {
                        "message_id": item.citation.message_id,
                        "author_role": item.author_role,
                        "matched_scoring_terms": list(item.matched_terms),
                        "excerpt": item.citation.excerpt,
                    }
                    for item in row.contributing_items
                ],
                "union_coverage_terms": list(row.coverage_terms),
                "coverage_ratio": row.coverage_ratio,
                "contributing_excerpt_count": len(row.contributing_items),
                "user_excerpt_count": row.user_excerpt_count,
                "assistant_excerpt_count": row.assistant_excerpt_count,
                "user_authored_coverage_met_threshold": row.user_authored_coverage_met_threshold,
                "failure_reasons": list(row.failure_reasons),
            }
            for row in window_debug
        ],
    }


# Render a human-readable qualification-debug report for CLI inspection.
def render_qualification_debug(payload: dict[str, object]) -> str:
    lines = [
        "Qualification Debug",
        f"parsed_query_terms: {payload['parsed_query_terms']}",
        f"scoring_terms: {payload['scoring_terms']}",
        f"grounding_mode: {payload['grounding_mode']}",
        f"conversational_memory_threshold: {payload['conversational_memory_threshold']}",
        f"retrieved_candidate_count: {payload['retrieved_candidate_count']}",
        f"evidence_qualified_candidate_count: {payload['evidence_qualified_candidate_count']}",
    ]
    windows = payload.get("windows", [])
    if isinstance(windows, list):
        for row in windows:
            lines.append("")
            lines.append(
                "candidate: "
                f"rank={row['candidate_rank']} "
                f"conversation_id={row['conversation_id']} "
                f"focal_message_id={row['focal_message_id']} "
                f"window_id={row['window_id']}"
            )
            lines.append(f"  strict_qualification_passed: {row['strict_qualification_passed']}")
            lines.append(f"  conversational_memory_qualification_passed: {row['conversational_memory_qualification_passed']}")
            lines.append(f"  union_coverage_terms: {row['union_coverage_terms']}")
            lines.append(f"  coverage_ratio: {row['coverage_ratio']}")
            lines.append(f"  contributing_excerpt_count: {row['contributing_excerpt_count']}")
            lines.append(f"  user_excerpt_count: {row['user_excerpt_count']}")
            lines.append(f"  assistant_excerpt_count: {row['assistant_excerpt_count']}")
            lines.append(f"  user_authored_coverage_met_threshold: {row['user_authored_coverage_met_threshold']}")
            lines.append(f"  failure_reasons: {row['failure_reasons']}")
            contributing_items = row.get("contributing_items", [])
            if contributing_items:
                lines.append("  contributing_items:")
                for item in contributing_items:
                    lines.append(
                        "    - "
                        f"message_id={item['message_id']} "
                        f"author_role={item['author_role']} "
                        f"matched_scoring_terms={item['matched_scoring_terms']} "
                        f"excerpt={item['excerpt']}"
                    )
    return "\n".join(lines)


# Build per-window debug rows in retrieval rank order so CLI output matches the real candidate order.
def build_window_debug_rows(
    retrieval_results: tuple[object, ...],
    selected_evidence: tuple[EvidenceItem, ...],
    focus_terms_value: tuple[str, ...],
) -> tuple[_WindowQualificationDebug, ...]:
    debug_rows: list[_WindowQualificationDebug] = []
    for result in retrieval_results:
        window_rank = int(getattr(result, "rank", 0))
        window_items = tuple(item for item in selected_evidence if item.source_result_rank == window_rank)
        if not window_items:
            debug_rows.append(
                _WindowQualificationDebug(
                    candidate_rank=window_rank,
                    conversation_id=string_or_none(getattr(result, "conversation_id", None)) or "",
                    focal_message_id=string_or_none(getattr(result, "focal_message_id", None)),
                    window_id=string_or_none(getattr(result, "result_id", None)) or f"window:{window_rank}",
                    strict_qualification_passed=False,
                    conversational_memory_qualification_passed=False,
                    contributing_items=(),
                    coverage_terms=(),
                    coverage_ratio=0.0,
                    user_excerpt_count=0,
                    assistant_excerpt_count=0,
                    user_authored_coverage_met_threshold=False,
                    failure_reasons=("no_direct_lexical_matches", "no_window_level_composed_support"),
                )
            )
            continue
        analyzed = analyze_window_composition(window_items, focus_terms_value)
        debug_rows.append(
            _WindowQualificationDebug(
                candidate_rank=window_rank,
                conversation_id=string_or_none(getattr(result, "conversation_id", None)) or analyzed.conversation_id,
                focal_message_id=string_or_none(getattr(result, "focal_message_id", None)) or analyzed.focal_message_id,
                window_id=string_or_none(getattr(result, "result_id", None)) or analyzed.window_id,
                strict_qualification_passed=analyzed.strict_qualification_passed,
                conversational_memory_qualification_passed=analyzed.conversational_memory_qualification_passed,
                contributing_items=analyzed.contributing_items,
                coverage_terms=analyzed.coverage_terms,
                coverage_ratio=analyzed.coverage_ratio,
                user_excerpt_count=analyzed.user_excerpt_count,
                assistant_excerpt_count=analyzed.assistant_excerpt_count,
                user_authored_coverage_met_threshold=analyzed.user_authored_coverage_met_threshold,
                failure_reasons=analyzed.failure_reasons,
            )
        )
    return tuple(debug_rows)
