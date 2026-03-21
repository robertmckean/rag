"""Deterministic grounded answering orchestrator over one normalized run using Phase 2 retrieval.

This module is the public API surface for answering. Consumers should import from here.
Internal qualification, status, composition, and diagnostics live in submodules.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from rag.answering.generator_llm import LLMSynthesisRequest, synthesize_answer_with_llm
from rag.answering.models import (
    AnswerDiagnostics,
    AnswerRequest,
    AnswerResult,
    AnswerStatus,
    Citation,
    EvidenceItem,
    EvidenceQualificationDiagnostic,
    RetrievalSummary,
)
from rag.retrieval.lexical import RetrievalFilters, retrieve_message_windows
from rag.retrieval.utils import string_or_none

# --- Re-exports for downstream consumers (do not remove) ---
from rag.answering.qualify import (  # noqa: F401
    ANSWER_RETRIEVAL_MODES,
    GROUNDING_MODES,
    DEFAULT_GROUNDING_MODE,
    DEFAULT_EVIDENCE_CAP,
    collect_citations as _collect_citations,
    focus_terms as _focus_terms,
    qualify_evidence_items as _qualify_evidence_items,
    select_evidence as _select_evidence,
)
from rag.answering.status import (  # noqa: F401
    classify_answer_status as _classify_answer_status,
    generate_answer_text as _generate_answer_text,
    refine_insufficient_evidence_wording as _refine_insufficient_evidence_wording,
)
from rag.answering.diagnostics import (  # noqa: F401
    build_answer_diagnostics as _build_answer_diagnostics,
    qualification_debug_payload,
    render_qualification_debug,
)

logger = logging.getLogger(__name__)


# Phase 3A answers are retrieval-bounded summaries rather than a new search system.
# Evidence is selected from existing retrieval windows and then capped deterministically.
# Status is classified before answer text generation so the generator cannot smooth over uncertainty.


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
    hybrid: bool = False,
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
        llm=llm or hybrid,
        llm_model=llm_model,
        hybrid=hybrid,
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


# Optionally rewrite the deterministic answer text with constrained LLM synthesis after Phase 3A is complete.
def _apply_optional_llm_synthesis(
    result: AnswerResult,
    *,
    llm: bool,
    llm_model: str | None,
    hybrid: bool = False,
) -> AnswerResult:
    if not llm:
        return result
    if not result.evidence_used:
        return result

    mode_label = "hybrid" if hybrid else "llm"
    try:
        synthesis = synthesize_answer_with_llm(
            LLMSynthesisRequest(
                query=result.query,
                answer_status=result.answer_status,
                evidence_items=result.evidence_used,
                gaps=result.gaps,
                conflicts=result.conflicts,
                model=llm_model,
                hybrid=hybrid,
            )
        )
    except Exception as exc:
        logger.warning("LLM synthesis (%s) failed, falling back to deterministic answer: %s", mode_label, exc)
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


# Read the run id from retrieval results when available and fall back to the run directory name otherwise.
def _infer_run_id(run_dir: Path, retrieval_results: tuple[object, ...]) -> str:
    if retrieval_results:
        run_id = string_or_none(getattr(retrieval_results[0], "run_id", None))
        if run_id:
            return run_id
    return run_dir.resolve().name
