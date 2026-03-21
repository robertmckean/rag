"""Deterministic pattern extraction from narrative reconstructions.

Extracts recurring entities and topic clusters from one or more
``NarrativeReconstruction`` objects.  Entity aliases are merged via an
explicit static map.  Topic clusters are built by agglomerative term-overlap
merging across narrative phases.  No LLM inference or embedding-based
clustering.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Sequence

from rag.narrative.builder import content_terms_from_text, entity_terms_from_text
from rag.narrative.models import NarrativePhase, NarrativeReconstruction
from rag.patterns.aliases import ENTITY_ALIASES, canonicalize_entity
from rag.patterns.models import (
    EntityClusterLink,
    EntityOccurrence,
    PatternReport,
    RecurringEntity,
    TemporalBurst,
    TopicCluster,
)

_CLUSTER_SIMILARITY_THRESHOLD = 0.10

# Terms that appear in phase descriptions due to formatting, not content.
_DESCRIPTION_NOISE = frozenset({"user", "assistant", "unknown"})

_SNIPPET_MAX = 120


def _first_date_from_range(date_range: str | None) -> str | None:
    """Extract the first YYYY-MM-DD from a phase date_range string."""
    if not date_range or len(date_range) < 10:
        return None
    return date_range[:10]


def _extract_snippet(entity: str, description: str) -> str | None:
    """Find a content snippet centered on *entity* in a phase description.

    The description uses pipe-separated segments like:
        ``(date) [role] "excerpt text" | (date) [role] "excerpt text"``

    Returns a ~120-char window around the first mention, or None if not found.
    """
    # Search for the canonical name and any known aliases.
    search_names = {entity}
    for variant, canonical in ENTITY_ALIASES.items():
        if canonical == entity:
            search_names.add(variant)
    # Also check if entity itself is a variant.
    canonical_of = ENTITY_ALIASES.get(entity)
    if canonical_of:
        search_names.add(canonical_of)

    # Find the first segment containing any of the search names.
    segments = description.split(" | ")
    for segment in segments:
        for name in search_names:
            idx = segment.find(name)
            if idx == -1:
                continue
            # Strip the leading (date) [role] prefix to get raw text.
            text = re.sub(r"^\([^)]*\)\s*\[[^\]]*\]\s*\"?", "", segment)
            text = text.rstrip('"')
            # Find entity position in the cleaned text.
            text_idx = text.find(name)
            if text_idx == -1:
                # Entity was in the prefix; use text from the start.
                text_idx = 0
            # Window: center on the entity mention.
            half = _SNIPPET_MAX // 2
            start = max(0, text_idx - half)
            end = min(len(text), text_idx + len(name) + half)
            snippet = text[start:end].strip()
            if start > 0:
                snippet = "..." + snippet
            if end < len(text):
                snippet = snippet + "..."
            return snippet
    return None


def _occurrence_from_phase(entity: str, phase: NarrativePhase) -> EntityOccurrence:
    """Build one EntityOccurrence grounded in a narrative phase."""
    snippet = _extract_snippet(entity, phase.description)
    return EntityOccurrence(
        evidence_id=phase.evidence_ids[0] if phase.evidence_ids else "",
        created_at=_first_date_from_range(phase.date_range),
        excerpt=snippet if snippet else phase.label,
    )


def extract_recurring_entities(
    narratives: Sequence[NarrativeReconstruction],
) -> PatternReport:
    """Extract named entities recurring across 2+ distinct narrative phases.

    Parameters
    ----------
    narratives:
        One or more narrative reconstructions to scan.

    Returns
    -------
    PatternReport with only recurring entities (2+ phase appearances).
    """
    if not narratives:
        return PatternReport(query="", entities=(), clusters=(), entity_cluster_links=(), temporal_bursts=(), evidence_count=0)

    # Track entity -> list of (phase, narrative_index) to deduplicate same phase.
    entity_phases: dict[str, list[tuple[NarrativePhase, int]]] = defaultdict(list)

    for narr_idx, narrative in enumerate(narratives):
        for phase in narrative.timeline:
            raw_entities = entity_terms_from_text(phase.description)
            # Canonicalize unconditionally — explicit aliases are authoritative.
            seen_canonical: set[str] = set()
            for ent in raw_entities:
                canonical = canonicalize_entity(ent)
                if canonical in seen_canonical:
                    continue
                seen_canonical.add(canonical)
                entity_phases[canonical].append((phase, narr_idx))

    # Deduplicate: same entity in same phase (by evidence_ids) counts once.
    recurring: list[RecurringEntity] = []
    for name, phase_list in entity_phases.items():
        seen_phase_ids: set[tuple[str, ...]] = set()
        unique_occurrences: list[EntityOccurrence] = []
        for phase, _narr_idx in phase_list:
            if phase.evidence_ids in seen_phase_ids:
                continue
            seen_phase_ids.add(phase.evidence_ids)
            unique_occurrences.append(_occurrence_from_phase(name, phase))

        if len(unique_occurrences) < 2:
            continue

        # Sort occurrences: by created_at (nulls last), then evidence_id.
        unique_occurrences.sort(
            key=lambda o: (0 if o.created_at else 1, o.created_at or "", o.evidence_id),
        )

        recurring.append(RecurringEntity(
            name=name,
            occurrences=tuple(unique_occurrences),
            occurrence_count=len(unique_occurrences),
        ))

    # Deterministic ordering: by occurrence_count desc, then name asc.
    recurring.sort(key=lambda e: (-e.occurrence_count, e.name))

    # Query: join unique queries from all narratives.
    queries = []
    seen_queries: set[str] = set()
    for n in narratives:
        if n.query not in seen_queries:
            queries.append(n.query)
            seen_queries.add(n.query)
    query = "; ".join(queries)

    total_evidence = sum(n.evidence_count for n in narratives)

    # Extract topic clusters from all phases across narratives.
    all_phases: list[NarrativePhase] = []
    query_term_set: set[str] = set()
    for n in narratives:
        for term in n.query.lower().split():
            if len(term) >= 4:
                query_term_set.add(term)
        for phase in n.timeline:
            if phase not in all_phases:
                all_phases.append(phase)

    clusters = _extract_topic_clusters(all_phases, query_term_set)

    entity_cluster_links = _extract_entity_cluster_links(tuple(recurring), tuple(clusters))
    temporal_bursts = _extract_temporal_bursts(all_phases)

    return PatternReport(
        query=query,
        entities=tuple(recurring),
        clusters=tuple(clusters),
        entity_cluster_links=tuple(entity_cluster_links),
        temporal_bursts=tuple(temporal_bursts),
        evidence_count=total_evidence,
    )


# ---------------------------------------------------------------------------
# Topic clustering
# ---------------------------------------------------------------------------


def _phase_term_set(phase: NarrativePhase, query_terms: set[str]) -> set[str]:
    """Build a combined content + entity term set for a phase, excluding query terms."""
    content = content_terms_from_text(phase.description)
    entities = {e.lower() for e in entity_terms_from_text(phase.description)}
    return (content | entities) - query_terms - _DESCRIPTION_NOISE


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _cluster_date_range(phases: list[NarrativePhase]) -> str | None:
    """Compute date range across cluster member phases."""
    dates: list[str] = []
    for phase in phases:
        if phase.date_range:
            # Extract first and last dates from "YYYY-MM-DD" or "YYYY-MM-DD to YYYY-MM-DD".
            if " to " in phase.date_range:
                parts = phase.date_range.split(" to ")
                dates.append(parts[0][:10])
                dates.append(parts[-1][:10])
            elif len(phase.date_range) >= 10:
                dates.append(phase.date_range[:10])
    if not dates:
        return None
    dates.sort()
    if dates[0] == dates[-1]:
        return dates[0]
    return f"{dates[0]} to {dates[-1]}"


def _build_cluster_label(
    phases: list[NarrativePhase],
    query_terms: set[str],
) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
    """Build a deterministic label, key_entities, and key_terms for a cluster."""
    entity_counts: dict[str, int] = defaultdict(int)
    term_counts: dict[str, int] = defaultdict(int)

    for phase in phases:
        for ent in entity_terms_from_text(phase.description):
            canonical = canonicalize_entity(ent)
            entity_counts[canonical] += 1
        for term in content_terms_from_text(phase.description):
            if term not in query_terms and term not in _DESCRIPTION_NOISE:
                term_counts[term] += 1

    top_entities = sorted(entity_counts.items(), key=lambda x: (-x[1], x[0]))[:2]
    key_entities = tuple(e[0] for e in top_entities)

    # Exclude entity names (lowercased) from key_terms to avoid redundancy.
    entity_name_lower = {e.lower() for e in key_entities}
    top_terms = sorted(
        ((t, c) for t, c in term_counts.items() if t not in entity_name_lower),
        key=lambda x: (-x[1], x[0]),
    )[:2]
    key_terms = tuple(t[0] for t in top_terms)

    label_parts: list[str] = []
    if key_entities:
        label_parts.append(", ".join(key_entities))
    if key_terms:
        label_parts.append(", ".join(key_terms))
    label = ": ".join(label_parts) if label_parts else "Cluster"

    return label, key_entities, key_terms


def _extract_topic_clusters(
    phases: list[NarrativePhase],
    query_terms: set[str],
) -> list[TopicCluster]:
    """Agglomerative term-overlap clustering over narrative phases."""
    if len(phases) < 2:
        return []

    # Compute term sets per phase.
    term_sets = [_phase_term_set(p, query_terms) for p in phases]

    # Each cluster is a list of phase indices.
    clusters: list[list[int]] = [[i] for i in range(len(phases))]
    # Merged term sets per cluster.
    cluster_terms: list[set[str]] = [ts.copy() for ts in term_sets]

    # Greedy agglomerative merge.
    while True:
        best_sim = 0.0
        best_i = -1
        best_j = -1

        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                sim = _jaccard(cluster_terms[i], cluster_terms[j])
                if sim > best_sim:
                    best_sim = sim
                    best_i = i
                    best_j = j

        if best_sim < _CLUSTER_SIMILARITY_THRESHOLD:
            break

        # Merge j into i.
        clusters[best_i].extend(clusters[best_j])
        cluster_terms[best_i] |= cluster_terms[best_j]
        del clusters[best_j]
        del cluster_terms[best_j]

    # Build TopicCluster objects for clusters with 2+ phases.
    result: list[TopicCluster] = []
    for cluster_indices in clusters:
        if len(cluster_indices) < 2:
            continue

        member_phases = [phases[i] for i in cluster_indices]
        label, key_entities, key_terms = _build_cluster_label(member_phases, query_terms)

        all_evidence_ids: list[str] = []
        seen_eids: set[str] = set()
        for phase in member_phases:
            for eid in phase.evidence_ids:
                if eid not in seen_eids:
                    all_evidence_ids.append(eid)
                    seen_eids.add(eid)

        result.append(TopicCluster(
            label=label,
            phase_labels=tuple(p.label for p in member_phases),
            evidence_ids=tuple(all_evidence_ids),
            date_range=_cluster_date_range(member_phases),
            key_entities=key_entities,
            key_terms=key_terms,
            phase_count=len(member_phases),
        ))

    # Deterministic ordering: by phase_count desc, then label asc.
    result.sort(key=lambda c: (-c.phase_count, c.label))
    return result


# ---------------------------------------------------------------------------
# Cross-cluster entity links
# ---------------------------------------------------------------------------


def _extract_entity_cluster_links(
    entities: tuple[RecurringEntity, ...],
    clusters: tuple[TopicCluster, ...],
) -> list[EntityClusterLink]:
    """Identify entities that appear in 2+ topic clusters."""
    if len(clusters) < 2:
        return []

    entity_to_clusters: dict[str, list[TopicCluster]] = defaultdict(list)
    for cluster in clusters:
        for ent in cluster.key_entities:
            entity_to_clusters[ent].append(cluster)

    links: list[EntityClusterLink] = []
    for ent_name, ent_clusters in entity_to_clusters.items():
        if len(ent_clusters) < 2:
            continue
        total_phases = sum(c.phase_count for c in ent_clusters)
        links.append(EntityClusterLink(
            entity=ent_name,
            cluster_labels=tuple(c.label for c in ent_clusters),
            cluster_count=len(ent_clusters),
            total_phase_count=total_phases,
        ))

    # Deterministic ordering: by cluster_count desc, then entity name asc.
    links.sort(key=lambda l: (-l.cluster_count, l.entity))
    return links


# ---------------------------------------------------------------------------
# Temporal burst detection
# ---------------------------------------------------------------------------

_BURST_WINDOW_DAYS = 7
_BURST_MIN_PHASES = 3


def _approx_days(date_a: str, date_b: str) -> int:
    """Approximate day distance between two YYYY-MM-DD strings."""
    try:
        ya, ma, da = int(date_a[:4]), int(date_a[5:7]), int(date_a[8:10])
        yb, mb, db = int(date_b[:4]), int(date_b[5:7]), int(date_b[8:10])
        jd_a = ya * 365 + ma * 30 + da
        jd_b = yb * 365 + mb * 30 + db
        return abs(jd_b - jd_a)
    except (ValueError, IndexError):
        return 999


def _extract_temporal_bursts(
    phases: list[NarrativePhase],
) -> list[TemporalBurst]:
    """Identify periods with 3+ phases within a 7-day window."""
    # Collect phases with usable dates.
    dated: list[tuple[str, NarrativePhase]] = []
    for phase in phases:
        date = _first_date_from_range(phase.date_range)
        if date:
            dated.append((date, phase))

    if len(dated) < _BURST_MIN_PHASES:
        return []

    dated.sort(key=lambda x: x[0])

    bursts: list[TemporalBurst] = []
    i = 0
    while i < len(dated):
        # Expand window from position i.
        j = i + 1
        while j < len(dated) and _approx_days(dated[i][0], dated[j][0]) <= _BURST_WINDOW_DAYS:
            j += 1
        window = dated[i:j]
        if len(window) >= _BURST_MIN_PHASES:
            phase_labels = tuple(p.label for _, p in window)
            entities: set[str] = set()
            for _, p in window:
                for ent in entity_terms_from_text(p.description):
                    entities.add(canonicalize_entity(ent))
            date_range = window[0][0] if window[0][0] == window[-1][0] else f"{window[0][0]} to {window[-1][0]}"
            bursts.append(TemporalBurst(
                date_range=date_range,
                phase_labels=phase_labels,
                entities=tuple(sorted(entities)),
                burst_size=len(window),
            ))
            # Skip past this burst to avoid overlapping bursts.
            i = j
        else:
            i += 1

    # Deterministic ordering: by burst_size desc, then date_range asc.
    bursts.sort(key=lambda b: (-b.burst_size, b.date_range))
    return bursts
