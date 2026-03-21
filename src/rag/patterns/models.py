"""Schema definitions for pattern extraction.

Recurring entities, topic clusters, cross-cluster entity links, and temporal
bursts.  Every field is frozen and serializable.
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
class EntityClusterLink:
    """An entity that appears across multiple topic clusters."""

    entity: str
    cluster_labels: tuple[str, ...]
    cluster_count: int
    total_phase_count: int

    def to_dict(self) -> dict[str, object]:
        return {
            "entity": self.entity,
            "cluster_labels": list(self.cluster_labels),
            "cluster_count": self.cluster_count,
            "total_phase_count": self.total_phase_count,
        }


@dataclass(frozen=True)
class TemporalBurst:
    """A period with unusually dense phase activity."""

    date_range: str
    phase_labels: tuple[str, ...]
    entities: tuple[str, ...]
    burst_size: int

    def to_dict(self) -> dict[str, object]:
        return {
            "date_range": self.date_range,
            "phase_labels": list(self.phase_labels),
            "entities": list(self.entities),
            "burst_size": self.burst_size,
        }


@dataclass(frozen=True)
class PatternReport:
    """Complete pattern extraction report for one query set."""

    query: str
    entities: tuple[RecurringEntity, ...]
    clusters: tuple[TopicCluster, ...]
    entity_cluster_links: tuple[EntityClusterLink, ...]
    temporal_bursts: tuple[TemporalBurst, ...]
    evidence_count: int

    def to_dict(self) -> dict[str, object]:
        return {
            "query": self.query,
            "entities": [e.to_dict() for e in self.entities],
            "clusters": [c.to_dict() for c in self.clusters],
            "entity_cluster_links": [l.to_dict() for l in self.entity_cluster_links],
            "temporal_bursts": [b.to_dict() for b in self.temporal_bursts],
            "evidence_count": self.evidence_count,
        }
