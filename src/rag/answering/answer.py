"""Deterministic grounded answering over one normalized run using Phase 2 retrieval."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from rag.answering.generator_llm import LLMSynthesisRequest, synthesize_answer_with_llm
from rag.answering.models import (
    AnswerDiagnostics,
    AnswerRequest,
    AnswerResult,
    AnswerStatus,
    Citation,
    EvidenceQualificationDiagnostic,
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
class _StatusDecision:
    status: AnswerStatus
    gaps: tuple[str, ...]
    conflicts: tuple[str, ...]
    focus_terms: tuple[str, ...]


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


@dataclass(frozen=True)
class _WindowQualificationDebug:
    candidate_rank: int
    conversation_id: str
    focal_message_id: str | None
    window_id: str | None
    strict_qualification_passed: bool
    conversational_memory_qualification_passed: bool
    contributing_items: tuple[EvidenceItem, ...]
    coverage_terms: tuple[str, ...]
    coverage_ratio: float
    user_excerpt_count: int
    assistant_excerpt_count: int
    user_authored_coverage_met_threshold: bool
    failure_reasons: tuple[str, ...]


# Answer one natural-language query against a single normalized run using existing retrieval behavior.
def answer_query(
    run_dir: Path,
    query: str,
    *,
    retrieval_mode: str = "relevance",
    grounding_mode: str = DEFAULT_GROUNDING_MODE,
    limit: int = 8,
    max_evidence: int = DEFAULT_EVIDENCE_CAP,
    filters: RetrievalFilters | None = None,
    llm: bool = False,
    llm_model: str | None = None,
) -> AnswerResult:
    if retrieval_mode not in ANSWER_RETRIEVAL_MODES:
        raise ValueError(
            f"Unsupported answer retrieval mode: {retrieval_mode}. Supported modes: {', '.join(ANSWER_RETRIEVAL_MODES)}."
        )
    if grounding_mode not in GROUNDING_MODES:
        raise ValueError(
            f"Unsupported grounding mode: {grounding_mode}. Supported modes: {', '.join(GROUNDING_MODES)}."
        )

    request = AnswerRequest(
        run_dir=str(run_dir.resolve()),
        query=query,
        retrieval_mode=retrieval_mode,
        grounding_mode=grounding_mode,
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
    qualification = _qualify_evidence_items(query, selected_evidence, grounding_mode=grounding_mode)
    diagnostics = _build_answer_diagnostics(query, selected_evidence, qualification)
    decision = _classify_answer_status(query, qualification)
    decision = _refine_insufficient_evidence_wording(decision, diagnostics)
    citations = _collect_citations(qualification.evidence_items)
    retrieval_summary = RetrievalSummary(
        run_id=_infer_run_id(run_dir, retrieval_results),
        retrieval_mode=retrieval_mode,
        retrieval_limit=limit,
        retrieved_result_count=len(retrieval_results),
        evidence_used_count=len(qualification.evidence_items),
    )
    answer_text = _generate_answer_text(decision, qualification.evidence_items, diagnostics)
    result = AnswerResult(
        request=request,
        query=query,
        answer_status=decision.status,
        answer=answer_text,
        evidence_used=qualification.evidence_items,
        gaps=decision.gaps,
        conflicts=decision.conflicts,
        citations=citations,
        retrieval_summary=retrieval_summary,
        diagnostics=diagnostics,
    )
    return _apply_optional_llm_synthesis(
        result,
        llm=llm,
        llm_model=llm_model,
    )


# Render one answer result into a compact review-friendly terminal summary.
def render_answer_result(result: AnswerResult) -> str:
    lines = [
        "Grounded Answer",
        f"query: {result.query}",
        f"retrieval_mode: {result.retrieval_summary.retrieval_mode}",
        f"grounding_mode: {result.request.grounding_mode}",
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
    if result.diagnostics.support_basis:
        lines.append(
            "diagnostics: "
            f"support_basis={result.diagnostics.support_basis} "
            f"composition_used={result.diagnostics.composition_used} "
            f"coverage_ratio={result.diagnostics.coverage_ratio}"
        )
    return "\n".join(lines)


# Serialize one answer result deterministically for JSON output or tests.
def answer_result_json(result: AnswerResult) -> str:
    return json.dumps(result.to_dict(), ensure_ascii=True, indent=2, sort_keys=True) + "\n"


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
    focus_terms = _focus_terms(query)
    retrieval_results = retrieve_message_windows(
        run_dir.resolve(),
        query,
        limit=limit,
        mode=retrieval_mode,
        filters=filters,
    )
    selected_evidence = _select_evidence(retrieval_results, query, max_evidence=max_evidence)
    qualification = _qualify_evidence_items(query, selected_evidence, grounding_mode=grounding_mode)
    window_debug = _build_window_debug_rows(
        retrieval_results,
        selected_evidence,
        focus_terms,
    )
    return {
        "query": query,
        "parsed_query_terms": list(parsed_query.normalized_query_terms),
        "scoring_terms": list(focus_terms),
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


# Optionally rewrite the deterministic answer text with constrained LLM synthesis after Phase 3A is complete.
def _apply_optional_llm_synthesis(
    result: AnswerResult,
    *,
    llm: bool,
    llm_model: str | None,
) -> AnswerResult:
    if not llm:
        return result
    if not result.evidence_used:
        return result

    try:
        synthesis = synthesize_answer_with_llm(
            LLMSynthesisRequest(
                query=result.query,
                answer_status=result.answer_status,
                evidence_items=result.evidence_used,
                gaps=result.gaps,
                conflicts=result.conflicts,
                model=llm_model,
            )
        )
    except Exception:
        return result

    citations_by_id = {f"e{item.rank}": item.citation for item in result.evidence_used}
    synthesized_citations = tuple(
        citations_by_id[citation_id]
        for citation_id in synthesis.citation_ids
        if citation_id in citations_by_id
    )
    return AnswerResult(
        request=result.request,
        query=result.query,
        answer_status=result.answer_status,
        answer=synthesis.answer_text,
        evidence_used=result.evidence_used,
        gaps=result.gaps,
        conflicts=result.conflicts,
        citations=synthesized_citations or result.citations,
        retrieval_summary=result.retrieval_summary,
        diagnostics=result.diagnostics,
    )


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
                    window_id=string_or_none(getattr(result, "result_id", None)) or f"window:{getattr(result, 'rank', len(evidence_items) + 1)}",
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


# Record which selected evidence items were dropped and why so grounded-answer failures stay inspectable.
def _build_answer_diagnostics(
    query: str,
    selected_evidence: tuple[EvidenceItem, ...],
    qualification: _QualificationOutcome,
) -> AnswerDiagnostics:
    focus_terms = _focus_terms(query)
    qualified_message_ids = {item.citation.message_id for item in qualification.evidence_items}
    rejected_items: list[EvidenceQualificationDiagnostic] = []
    requires_combined_support = _requires_combined_focus_support(query, focus_terms)

    for item in selected_evidence:
        if item.citation.message_id in qualified_message_ids:
            continue
        missing_focus_terms = tuple(term for term in focus_terms if term not in item.matched_terms)
        rejection_reasons: list[str] = []
        if not item.matched_terms:
            rejection_reasons.append("insufficient_directness")
        if item.score < MIN_SUPPORT_SCORE:
            rejection_reasons.append("filtered_by_threshold")
        if item.author_role == "assistant":
            if _is_speculative_excerpt(item.citation.excerpt):
                rejection_reasons.append("assistant_authored_speculation")
            if not _has_grounding_peer(item, selected_evidence):
                rejection_reasons.append("missing_non_assistant_grounding")
        if qualification.composition_used and qualification.window_id and item.window_id != qualification.window_id:
            rejection_reasons.append("outside_selected_support_window")
        if requires_combined_support and not _has_combined_focus_support((item,), focus_terms):
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
        focus_terms=focus_terms,
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


# Distinguish between retrieval miss and qualification failure so insufficiency gaps stay accurate.
def _refine_insufficient_evidence_wording(
    decision: _StatusDecision,
    diagnostics: AnswerDiagnostics,
) -> _StatusDecision:
    if decision.status is not AnswerStatus.INSUFFICIENT_EVIDENCE:
        return decision
    if diagnostics.selected_evidence_count <= 0 or diagnostics.qualified_evidence_count > 0:
        return decision
    if diagnostics.composition_used:
        gap = "Relevant retrieval candidates were found, but the composed support stayed too weak to justify a grounded answer."
    elif any("combined_concept_not_supported" in item.rejection_reasons for item in diagnostics.rejected_evidence):
        gap = "Relevant retrieval candidates were found, but none directly supported the full combined query concept after evidence qualification."
    else:
        gap = "Relevant retrieval candidates were found, but none met the evidence qualification rules for this question."
    return _StatusDecision(
        status=decision.status,
        gaps=(gap,),
        conflicts=decision.conflicts,
        focus_terms=decision.focus_terms,
    )


# Classify the answer status deterministically before any answer prose is generated.
def _classify_answer_status(query: str, qualification: _QualificationOutcome) -> _StatusDecision:
    focus_terms = _focus_terms(query)
    evidence_items = qualification.evidence_items
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

    if qualification.composition_used:
        gaps = ("Support is distributed across multiple nearby excerpts within one retrieved conversation window.",)
        return _StatusDecision(
            status=AnswerStatus.PARTIALLY_SUPPORTED,
            gaps=gaps,
            conflicts=(),
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
def _qualify_evidence_items(
    query: str,
    evidence_items: tuple[EvidenceItem, ...],
    *,
    grounding_mode: str,
) -> _QualificationOutcome:
    focus_terms = _focus_terms(query)
    qualified_items: list[EvidenceItem] = []
    for item in evidence_items:
        if _is_qualified_evidence_item(item, focus_terms, evidence_items):
            qualified_items.append(item)
    ordered_items = _re_rank_evidence_items(tuple(qualified_items))
    requires_combined_support = _requires_combined_focus_support(query, focus_terms)
    if _has_combined_focus_support(ordered_items, focus_terms):
        return _qualification_outcome_from_items(
            ordered_items,
            support_basis="single_excerpt",
            composition_used=False,
            coverage_terms=tuple(sorted({term for item in ordered_items for term in item.matched_terms})),
            coverage_ratio=_coverage_ratio(tuple(sorted({term for item in ordered_items for term in item.matched_terms})), focus_terms),
            window_id=ordered_items[0].window_id if ordered_items else None,
            status_reason="single_excerpt_support",
        )
    if grounding_mode == "strict":
        if requires_combined_support:
            return _empty_qualification_outcome(status_reason="combined_concept_not_supported")
        return _qualification_outcome_from_items(
            ordered_items,
            support_basis="single_excerpt" if ordered_items else None,
            composition_used=False,
            coverage_terms=tuple(sorted({term for item in ordered_items for term in item.matched_terms})),
            coverage_ratio=_coverage_ratio(tuple(sorted({term for item in ordered_items for term in item.matched_terms})), focus_terms),
            window_id=ordered_items[0].window_id if ordered_items else None,
            status_reason="strict_partial_support" if ordered_items else "no_qualified_evidence",
        )

    composed_outcome = _qualify_window_composed_support(query, evidence_items)
    if composed_outcome is not None:
        return composed_outcome
    if requires_combined_support:
        return _empty_qualification_outcome(status_reason="combined_concept_not_supported")
    return _qualification_outcome_from_items(
        ordered_items,
        support_basis="single_excerpt" if ordered_items else None,
        composition_used=False,
        coverage_terms=tuple(sorted({term for item in ordered_items for term in item.matched_terms})),
        coverage_ratio=_coverage_ratio(tuple(sorted({term for item in ordered_items for term in item.matched_terms})), focus_terms),
        window_id=ordered_items[0].window_id if ordered_items else None,
        status_reason="strict_partial_support" if ordered_items else "no_qualified_evidence",
    )


# Generate deterministic grounded answer prose from the selected evidence and precomputed status.
def _generate_answer_text(
    decision: _StatusDecision,
    evidence_items: tuple[EvidenceItem, ...],
    diagnostics: AnswerDiagnostics,
) -> str:
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
        if diagnostics.composition_used and primary_excerpt and secondary_excerpt:
            return (
                "The retrieved evidence only partially answers the question. "
                "Support is distributed across multiple nearby excerpts within one conversation window, including "
                f'"{primary_excerpt}" and "{secondary_excerpt}".'
            )
        if diagnostics.composition_used and primary_excerpt:
            return (
                "The retrieved evidence only partially answers the question. "
                "Support is distributed across nearby excerpts within one conversation window, including "
                f'"{primary_excerpt}".'
            )
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


# Re-rank evidence items deterministically after qualification narrows the candidate set.
def _re_rank_evidence_items(evidence_items: tuple[EvidenceItem, ...]) -> tuple[EvidenceItem, ...]:
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
def _empty_qualification_outcome(*, status_reason: str) -> _QualificationOutcome:
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
def _qualification_outcome_from_items(
    evidence_items: tuple[EvidenceItem, ...],
    *,
    support_basis: str | None,
    composition_used: bool,
    coverage_terms: tuple[str, ...],
    coverage_ratio: float,
    window_id: str | None,
    status_reason: str,
) -> _QualificationOutcome:
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
        coverage_ratio=coverage_ratio,
        window_id=window_id if len(window_ids) <= 1 else None,
        status_reason=status_reason,
    )


# Qualify one window for conversational-memory composition using only same-window lexical coverage.
def _qualify_window_composed_support(
    query: str,
    evidence_items: tuple[EvidenceItem, ...],
) -> _QualificationOutcome | None:
    focus_terms = _focus_terms(query)
    if not evidence_items or not focus_terms:
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
        analysis = _analyze_window_composition(window_items, focus_terms)
        if not analysis.conversational_memory_qualification_passed:
            continue
        return _qualification_outcome_from_items(
            _re_rank_evidence_items(analysis.contributing_items),
            support_basis="window_composed",
            composition_used=True,
            coverage_terms=analysis.coverage_terms,
            coverage_ratio=analysis.coverage_ratio,
            window_id=window_key,
            status_reason="window_composed_support",
        )
    return None


# Treat an item as a composition contributor only when it adds direct grounded term coverage inside one window.
def _is_window_contributing_item(item: EvidenceItem) -> bool:
    if item.score < MIN_SUPPORT_SCORE:
        return False
    if not item.matched_terms:
        return False
    if item.author_role == "assistant" and _is_speculative_excerpt(item.citation.excerpt):
        return False
    return True


# Compute the ratio of meaningful query-term coverage supplied by one support unit.
def _coverage_ratio(coverage_terms: tuple[str, ...], focus_terms: tuple[str, ...]) -> float:
    if not focus_terms:
        return 0.0
    return len(set(coverage_terms) & set(focus_terms)) / len(set(focus_terms))


# Analyze one selected retrieval window so the CLI can explain strict and conversational-memory qualification outcomes.
def _analyze_window_composition(
    window_items: tuple[EvidenceItem, ...],
    focus_terms: tuple[str, ...],
) -> _WindowQualificationDebug:
    strict_qualified_items = tuple(
        item for item in window_items
        if _is_qualified_evidence_item(item, focus_terms, window_items)
    )
    strict_passed = _has_combined_focus_support(strict_qualified_items, focus_terms) if focus_terms else bool(strict_qualified_items)

    contributors = tuple(item for item in window_items if _is_window_contributing_item(item))
    coverage_terms = tuple(sorted({term for item in contributors for term in item.matched_terms}))
    coverage_ratio = _coverage_ratio(coverage_terms, focus_terms)
    user_contributors = tuple(item for item in contributors if item.author_role == "user")
    assistant_contributors = tuple(item for item in contributors if item.author_role == "assistant")
    user_coverage_terms = tuple(sorted({term for item in user_contributors for term in item.matched_terms}))
    user_authored_coverage_met_threshold = _coverage_ratio(user_coverage_terms, focus_terms) >= COMPOSED_SUPPORT_COVERAGE_THRESHOLD

    failure_reasons: list[str] = []
    if not contributors:
        failure_reasons.append("no_direct_lexical_matches")
    if len(contributors) < 2 and not strict_passed:
        failure_reasons.append("only_one_contributing_excerpt")
    if strict_passed:
        conversational_memory_passed = False
    else:
        conversational_memory_passed = (
            coverage_ratio >= COMPOSED_SUPPORT_COVERAGE_THRESHOLD
            and len(user_contributors) >= 1
            and len(contributors) >= 2
            and user_authored_coverage_met_threshold
        )
    if coverage_ratio < COMPOSED_SUPPORT_COVERAGE_THRESHOLD:
        failure_reasons.append("coverage_below_threshold")
    if len(user_contributors) < 1:
        failure_reasons.append("no_user_authored_contributor")
    if not user_authored_coverage_met_threshold:
        failure_reasons.append("user_only_coverage_below_threshold")
    if focus_terms and not strict_passed:
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
        coverage_ratio=coverage_ratio,
        user_excerpt_count=len(user_contributors),
        assistant_excerpt_count=len(assistant_contributors),
        user_authored_coverage_met_threshold=user_authored_coverage_met_threshold,
        failure_reasons=tuple(dict.fromkeys(failure_reasons)),
    )


# Build per-window debug rows in retrieval rank order so CLI output matches the real candidate order.
def _build_window_debug_rows(
    retrieval_results: tuple[object, ...],
    selected_evidence: tuple[EvidenceItem, ...],
    focus_terms: tuple[str, ...],
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
        analyzed = _analyze_window_composition(window_items, focus_terms)
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
    candidates = parsed.scoring_terms or parsed.stopword_filtered_terms or parsed.normalized_query_terms
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
