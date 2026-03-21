"""Serialization-friendly answer models for the first grounded answer slice."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum


# These models keep the answer contract explicit and reviewable.
# Phase 3A is deterministic, so the result object records the exact evidence used.
# The fields stay compact because this layer is for bounded grounded answers, not chat behavior.


class AnswerStatus(str, Enum):
    SUPPORTED = "supported"
    PARTIALLY_SUPPORTED = "partially_supported"
    AMBIGUOUS = "ambiguous"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


@dataclass(frozen=True)
class Citation:
    provider: str
    conversation_id: str
    title: str | None
    message_id: str
    created_at: str | None
    excerpt: str

    # Convert one citation into a JSON-friendly payload.
    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class EvidenceItem:
    rank: int
    source_result_rank: int
    window_id: str | None
    score: float
    retrieval_mode: str
    author_role: str | None
    matched_terms: tuple[str, ...]
    citation: Citation

    # Convert one selected evidence item into a JSON-friendly payload.
    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["citation"] = self.citation.to_dict()
        return payload


@dataclass(frozen=True)
class AnswerRequest:
    run_dir: str
    query: str
    retrieval_mode: str
    grounding_mode: str
    limit: int
    max_evidence: int

    # Convert the request settings into a JSON-friendly payload.
    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class RetrievalSummary:
    run_id: str
    retrieval_mode: str
    retrieval_limit: int
    retrieved_result_count: int
    evidence_used_count: int

    # Convert retrieval summary details into a JSON-friendly payload.
    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class EvidenceQualificationDiagnostic:
    message_id: str
    source_result_rank: int
    author_role: str | None
    matched_terms: tuple[str, ...]
    missing_focus_terms: tuple[str, ...]
    rejection_reasons: tuple[str, ...]
    excerpt: str

    # Convert one evidence-qualification diagnostic into a JSON-friendly payload.
    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class AnswerDiagnostics:
    focus_terms: tuple[str, ...]
    selected_evidence_count: int
    qualified_evidence_count: int
    support_basis: str | None
    composition_used: bool
    supporting_excerpt_count: int
    user_excerpt_count: int
    assistant_excerpt_count: int
    coverage_terms: tuple[str, ...]
    coverage_ratio: float
    window_id: str | None
    status_reason: str | None
    rejected_evidence: tuple[EvidenceQualificationDiagnostic, ...]

    # Convert answer-layer diagnostics into a JSON-friendly payload.
    def to_dict(self) -> dict[str, object]:
        return {
            "focus_terms": list(self.focus_terms),
            "selected_evidence_count": self.selected_evidence_count,
            "qualified_evidence_count": self.qualified_evidence_count,
            "support_basis": self.support_basis,
            "composition_used": self.composition_used,
            "supporting_excerpt_count": self.supporting_excerpt_count,
            "user_excerpt_count": self.user_excerpt_count,
            "assistant_excerpt_count": self.assistant_excerpt_count,
            "coverage_terms": list(self.coverage_terms),
            "coverage_ratio": self.coverage_ratio,
            "window_id": self.window_id,
            "status_reason": self.status_reason,
            "rejected_evidence": [item.to_dict() for item in self.rejected_evidence],
        }


@dataclass(frozen=True)
class AnswerResult:
    request: AnswerRequest
    query: str
    answer_status: AnswerStatus
    answer: str
    evidence_used: tuple[EvidenceItem, ...]
    gaps: tuple[str, ...]
    conflicts: tuple[str, ...]
    citations: tuple[Citation, ...]
    retrieval_summary: RetrievalSummary
    diagnostics: AnswerDiagnostics
    synthesis_mode: str = "deterministic"

    # Convert the full answer result into a stable JSON-friendly payload.
    def to_dict(self) -> dict[str, object]:
        return {
            "request": self.request.to_dict(),
            "query": self.query,
            "answer_status": self.answer_status.value,
            "answer": self.answer,
            "synthesis_mode": self.synthesis_mode,
            "evidence_used": [item.to_dict() for item in self.evidence_used],
            "gaps": list(self.gaps),
            "conflicts": list(self.conflicts),
            "citations": [citation.to_dict() for citation in self.citations],
            "retrieval_summary": self.retrieval_summary.to_dict(),
            "diagnostics": self.diagnostics.to_dict(),
        }
