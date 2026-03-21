"""Shared diagnostic dataclasses used by both compose and diagnostics modules."""

from __future__ import annotations

from dataclasses import dataclass

from rag.answering.models import EvidenceItem


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
