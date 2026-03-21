"""Schema definitions for recurring-entity pattern extraction.

Every field is frozen and serializable.  This is the minimum vertical slice
for Phase 7: entity recurrence only.  Topic clusters and trajectory detection
are deferred to later steps.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class EntityOccurrence:
    """One occurrence of a named entity in a single evidence item."""

    evidence_id: str
    created_at: str | None  # ISO-8601 or None, matches Citation.created_at
    excerpt: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class RecurringEntity:
    """A named entity that appears across multiple evidence items."""

    name: str
    occurrences: tuple[EntityOccurrence, ...]
    occurrence_count: int

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "occurrences": [occ.to_dict() for occ in self.occurrences],
            "occurrence_count": self.occurrence_count,
        }


@dataclass(frozen=True)
class PatternReport:
    """Complete pattern extraction report for one query."""

    query: str
    entities: tuple[RecurringEntity, ...]
    evidence_count: int

    def to_dict(self) -> dict[str, object]:
        return {
            "query": self.query,
            "entities": [e.to_dict() for e in self.entities],
            "evidence_count": self.evidence_count,
        }
