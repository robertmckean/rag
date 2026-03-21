"""Schema definitions for pattern extraction: recurring entities and topic clusters.

Every field is frozen and serializable.  Trajectory detection is deferred.
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
class TopicCluster:
    """A group of narrative phases sharing a coherent topic."""

    label: str
    phase_labels: tuple[str, ...]
    evidence_ids: tuple[str, ...]
    date_range: str | None
    key_entities: tuple[str, ...]
    key_terms: tuple[str, ...]
    phase_count: int

    def to_dict(self) -> dict[str, object]:
        return {
            "label": self.label,
            "phase_labels": list(self.phase_labels),
            "evidence_ids": list(self.evidence_ids),
            "date_range": self.date_range,
            "key_entities": list(self.key_entities),
            "key_terms": list(self.key_terms),
            "phase_count": self.phase_count,
        }


@dataclass(frozen=True)
class PatternReport:
    """Complete pattern extraction report for one query set."""

    query: str
    entities: tuple[RecurringEntity, ...]
    clusters: tuple[TopicCluster, ...]
    evidence_count: int

    def to_dict(self) -> dict[str, object]:
        return {
            "query": self.query,
            "entities": [e.to_dict() for e in self.entities],
            "clusters": [c.to_dict() for c in self.clusters],
            "evidence_count": self.evidence_count,
        }
