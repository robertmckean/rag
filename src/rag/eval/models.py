"""Deterministic evaluation models for the Phase 3A grounded-answer harness."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum


# These models stay specific to the current RAG answer pipeline.
# The eval harness records expected behavior, observed behavior, and classified failures.
# JSON-friendly dataclasses keep reports readable and easy to diff between runs.


class FailureType(str, Enum):
    FALSE_SUPPORTED = "FALSE_SUPPORTED"
    FALSE_PARTIAL = "FALSE_PARTIAL"
    FALSE_AMBIGUOUS = "FALSE_AMBIGUOUS"
    MISSED_SUPPORT = "MISSED_SUPPORT"
    MISSED_PARTIAL = "MISSED_PARTIAL"
    MISSED_AMBIGUITY = "MISSED_AMBIGUITY"
    ASSISTANT_LEAKAGE = "ASSISTANT_LEAKAGE"
    TOPICAL_DRIFT = "TOPICAL_DRIFT"
    MULTIPART_COVERAGE_FAILURE = "MULTIPART_COVERAGE_FAILURE"
    CITATION_INTEGRITY_FAILURE = "CITATION_INTEGRITY_FAILURE"
    CITATION_COUNT_FAILURE = "CITATION_COUNT_FAILURE"
    FORBIDDEN_TERM_VIOLATION = "FORBIDDEN_TERM_VIOLATION"


@dataclass(frozen=True)
class EvalCase:
    id: str
    query: str
    retrieval_mode: str
    expected_status: str
    expected_terms_any: tuple[str, ...]
    expected_terms_all: tuple[str, ...]
    forbidden_terms: tuple[str, ...]
    expected_min_citations: int | None
    expected_max_citations: int | None
    notes: str | None = None

    # Convert one benchmark case into a JSON-friendly payload.
    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class EvalCaseResult:
    case_id: str
    query: str
    expected_status: str
    actual_status: str
    passed: bool
    failure_types: tuple[FailureType, ...]
    metrics: dict[str, bool]
    citation_count: int
    evidence_used_count: int
    answer: str
    notes: str | None = None

    # Convert one evaluated case result into a JSON-friendly payload.
    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["failure_types"] = [failure_type.value for failure_type in self.failure_types]
        return payload


@dataclass(frozen=True)
class EvalSummary:
    run_id: str
    bench_path: str
    bench_cases: int
    passed: int
    failed: int
    status_accuracy: float
    failure_type_counts: dict[str, int]

    # Convert the aggregate summary into a JSON-friendly payload.
    def to_dict(self) -> dict[str, object]:
        return asdict(self)
