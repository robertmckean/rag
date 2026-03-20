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

    # Convert the full answer result into a stable JSON-friendly payload.
    def to_dict(self) -> dict[str, object]:
        return {
            "request": self.request.to_dict(),
            "query": self.query,
            "answer_status": self.answer_status.value,
            "answer": self.answer,
            "evidence_used": [item.to_dict() for item in self.evidence_used],
            "gaps": list(self.gaps),
            "conflicts": list(self.conflicts),
            "citations": [citation.to_dict() for citation in self.citations],
            "retrieval_summary": self.retrieval_summary.to_dict(),
        }
