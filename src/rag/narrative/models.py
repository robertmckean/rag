"""Schema definitions for narrative reconstruction outputs.

Every field is frozen and serializable.  Support levels follow Phase 5 grounding
rules: single-evidence claims may be ``supported``; any cross-evidence
composition is at most ``partially_supported``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class NarrativePhase:
    """One episode in the reconstructed timeline."""

    label: str
    description: str
    evidence_ids: tuple[str, ...]
    date_range: str | None
    support: str  # "supported" | "partially_supported"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class NarrativeTransition:
    """An observed shift between two adjacent phases."""

    from_phase: str
    to_phase: str
    description: str
    evidence_ids: tuple[str, ...]
    support: str  # always "partially_supported"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class NarrativeGap:
    """Missing temporal or coverage continuity in the evidence."""

    description: str
    reason: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class NarrativeLimitation:
    """Data-quality limitation that affects interpretation confidence."""

    description: str
    kind: str  # "truncated_excerpt" | "null_timestamp" | "ambiguous_ordering"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class NarrativeReconstruction:
    """Complete narrative reconstruction for one query."""

    query: str
    summary: str
    timeline: tuple[NarrativePhase, ...]
    transitions: tuple[NarrativeTransition, ...]
    gaps: tuple[NarrativeGap, ...]
    limitations: tuple[NarrativeLimitation, ...]
    evidence_count: int

    def to_dict(self) -> dict[str, object]:
        return {
            "query": self.query,
            "summary": self.summary,
            "timeline": [p.to_dict() for p in self.timeline],
            "transitions": [t.to_dict() for t in self.transitions],
            "gaps": [g.to_dict() for g in self.gaps],
            "limitations": [lim.to_dict() for lim in self.limitations],
            "evidence_count": self.evidence_count,
        }
