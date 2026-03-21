"""Deterministic recurring-entity extraction from narrative reconstructions.

Scans one or more ``NarrativeReconstruction`` objects and identifies named
entities that appear across 2+ distinct narrative phases.  Applies a static
alias map to unconditionally merge known spelling variants (e.g. Mark → Marc).
Explicit aliases are authoritative — co-occurrence in the same phase does not
block the merge.  No fuzzy matching, LLM inference, or automatic alias
discovery.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Sequence

from rag.narrative.builder import entity_terms_from_text
from rag.narrative.models import NarrativePhase, NarrativeReconstruction
from rag.patterns.aliases import canonicalize_entity
from rag.patterns.models import EntityOccurrence, PatternReport, RecurringEntity


def _first_date_from_range(date_range: str | None) -> str | None:
    """Extract the first YYYY-MM-DD from a phase date_range string."""
    if not date_range or len(date_range) < 10:
        return None
    return date_range[:10]


def _occurrence_from_phase(entity: str, phase: NarrativePhase) -> EntityOccurrence:
    """Build one EntityOccurrence grounded in a narrative phase."""
    return EntityOccurrence(
        evidence_id=phase.evidence_ids[0] if phase.evidence_ids else "",
        created_at=_first_date_from_range(phase.date_range),
        excerpt=phase.label,
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
        return PatternReport(query="", entities=(), evidence_count=0)

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

    return PatternReport(
        query=query,
        entities=tuple(recurring),
        evidence_count=total_evidence,
    )
