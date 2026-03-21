"""Deterministic query routing over pattern extraction outputs.

Maps user questions to the appropriate subset of PatternReport and
NarrativeReconstruction data, then produces concise grounded answers.
No LLM, no new analytics — routing and formatting only.
"""

from __future__ import annotations

import re
import string

from enum import Enum

from rag.narrative.models import NarrativeReconstruction
from rag.patterns.models import PatternReport


class Intent(Enum):
    ENTITY_SCOPED = "entity_scoped"
    ENTITY = "entity"
    THEME = "theme"
    CROSS_TOPIC = "cross_topic"
    TEMPORAL = "temporal"
    TIMELINE = "timeline"
    UNKNOWN = "unknown"


# Keyword sets per intent.  Scored by hit count, ties broken by declaration order.
_INTENT_KEYWORDS: list[tuple[Intent, frozenset[str]]] = [
    (Intent.ENTITY, frozenset({
        "who", "people", "person", "persons", "names", "entities", "entity",
        "recurring", "mentioned", "characters", "shows",
    })),
    (Intent.THEME, frozenset({
        "theme", "themes", "topic", "topics", "subject", "subjects",
        "cluster", "clusters", "pattern", "patterns", "coming",
        "main", "important",
    })),
    (Intent.CROSS_TOPIC, frozenset({
        "bridge", "bridges", "bridging", "connect", "connects", "connecting",
        "across", "cross", "multiple", "overlap", "overlapping", "span",
        "appears",
    })),
    (Intent.TEMPORAL, frozenset({
        "when", "intense", "intensity", "active", "activity", "burst",
        "bursts", "busy", "dense", "concentrated", "period", "periods",
    })),
    (Intent.TIMELINE, frozenset({
        "change", "changed", "changes", "evolve", "evolved", "evolution",
        "over", "time", "timeline", "shift", "shifted", "transition",
        "transitions", "progress", "progression",
    })),
]


# Question words that suggest the user is asking about a specific entity.
_ENTITY_SCOPED_QUESTION_WORDS: frozenset[str] = frozenset({
    "what", "how", "when", "timeline", "happened", "with",
    "about", "did", "history", "went", "go", "evolve", "evolved",
})


def _normalize_query(query: str) -> str:
    """Lowercase and strip punctuation for matching."""
    return query.lower().translate(str.maketrans("", "", string.punctuation))


def _detect_entity(query: str, report: PatternReport) -> str | None:
    """Find the best-matching entity from the report in the query.

    Uses whole-word case-insensitive matching.  If multiple entities match,
    returns the one with the highest occurrence_count.
    """
    normalized = _normalize_query(query)

    matches: list[tuple[str, int]] = []
    for entity in report.entities:
        # Whole-word match: entity name boundaries must be word edges.
        pattern = r"\b" + re.escape(entity.name.lower()) + r"\b"
        if re.search(pattern, normalized):
            matches.append((entity.name, entity.occurrence_count))

    if not matches:
        return None

    # Highest occurrence count wins; ties broken alphabetically for determinism.
    matches.sort(key=lambda m: (-m[1], m[0]))
    return matches[0][0]


def classify_intent(
    query: str,
    report: PatternReport | None = None,
) -> Intent:
    """Classify a user question into one of the supported intents.

    When *report* is provided, ENTITY_SCOPED is checked first: if the query
    contains both a known entity name and a question pattern, ENTITY_SCOPED
    wins before keyword scoring.
    """
    tokens = set(query.lower().split())

    # Check ENTITY_SCOPED before keyword scoring.
    if report is not None:
        entity = _detect_entity(query, report)
        if entity is not None:
            # Require at least one question-pattern word alongside the entity.
            stripped_tokens = set(_normalize_query(query).split())
            if stripped_tokens & _ENTITY_SCOPED_QUESTION_WORDS:
                return Intent.ENTITY_SCOPED

    best_intent = Intent.UNKNOWN
    best_score = 0

    for intent, keywords in _INTENT_KEYWORDS:
        score = len(tokens & keywords)
        if score > best_score:
            best_score = score
            best_intent = intent

    return best_intent


def route_answer(
    query: str,
    report: PatternReport,
    narratives: list[NarrativeReconstruction],
) -> str:
    """Route a user question to the appropriate data and produce an answer."""
    intent = classify_intent(query, report)

    if intent == Intent.ENTITY_SCOPED:
        entity = _detect_entity(query, report)
        assert entity is not None  # guaranteed by classify_intent
        return _answer_entity_scoped(entity, report)
    if intent == Intent.ENTITY:
        return _answer_entity(report)
    if intent == Intent.THEME:
        return _answer_theme(report)
    if intent == Intent.CROSS_TOPIC:
        return _answer_cross_topic(report)
    if intent == Intent.TEMPORAL:
        return _answer_temporal(report)
    if intent == Intent.TIMELINE:
        return _answer_timeline(report, narratives)
    return _answer_unknown(report)


def _answer_entity_scoped(entity_name: str, report: PatternReport) -> str:
    """Produce an answer scoped to a single entity using filtered data."""
    # Filter entities.
    entity_match = None
    for ent in report.entities:
        if ent.name.lower() == entity_name.lower():
            entity_match = ent
            break

    if entity_match is None:
        return f"No data found for entity '{entity_name}'."

    # Filter clusters: keep those where entity appears in key_entities.
    scoped_clusters = tuple(
        c for c in report.clusters
        if entity_name in c.key_entities
    )

    # Filter entity_cluster_links.
    scoped_links = tuple(
        l for l in report.entity_cluster_links
        if l.entity == entity_name
    )

    # Filter temporal_bursts.
    scoped_bursts = tuple(
        b for b in report.temporal_bursts
        if entity_name in b.entities
    )

    # Build answer.
    lines = [f"{entity_match.name} — {entity_match.occurrence_count} occurrences", ""]

    # Date span.
    dates = [o.created_at for o in entity_match.occurrences if o.created_at]
    if len(dates) >= 2:
        lines.append(f"  Active period: {dates[0]} to {dates[-1]}")
        lines.append("")

    # Clusters.
    if scoped_clusters:
        lines.append(f"  Appears in {len(scoped_clusters)} topic cluster{'s' if len(scoped_clusters) != 1 else ''}:")
        for c in scoped_clusters[:3]:
            date_part = f" ({c.date_range})" if c.date_range else ""
            lines.append(f"    - {c.label}{date_part}")
        if len(scoped_clusters) > 3:
            lines.append(f"    ... and {len(scoped_clusters) - 3} more")
        lines.append("")

    # Cross-cluster links.
    if scoped_links:
        link = scoped_links[0]
        lines.append(f"  Bridges {link.cluster_count} clusters across {link.total_phase_count} phases")
        lines.append("")

    # Temporal bursts.
    if scoped_bursts:
        lines.append(f"  Activity bursts:")
        for b in scoped_bursts[:3]:
            lines.append(f"    - {b.date_range}: {b.burst_size} phases")
        lines.append("")

    # Snippets: show up to 3 excerpts.
    snippets = entity_match.occurrences[:3]
    if snippets:
        lines.append("  Key mentions:")
        for occ in snippets:
            date_prefix = f"[{occ.created_at}] " if occ.created_at else ""
            lines.append(f"    - {date_prefix}{occ.excerpt}")

    return "\n".join(lines)


def _answer_entity(report: PatternReport) -> str:
    if not report.entities:
        return "No recurring entities found in the data."

    lines = [f"The most recurring people/entities ({len(report.entities)} total):", ""]
    for i, ent in enumerate(report.entities[:5], 1):
        dates = [o.created_at for o in ent.occurrences if o.created_at]
        span = ""
        if len(dates) >= 2:
            span = f", span {dates[0]} to {dates[-1]}"
        lines.append(f"  {i}. {ent.name} — {ent.occurrence_count} occurrences{span}")
    if len(report.entities) > 5:
        lines.append(f"  ... and {len(report.entities) - 5} more")
    return "\n".join(lines)


def _answer_theme(report: PatternReport) -> str:
    if not report.clusters:
        return "No topic clusters found in the data."

    lines = [f"The main recurring themes ({len(report.clusters)} clusters):", ""]
    for i, cluster in enumerate(report.clusters[:5], 1):
        date_part = f" ({cluster.date_range})" if cluster.date_range else ""
        lines.append(f"  {i}. {cluster.label}{date_part} — {cluster.phase_count} phases")
        if cluster.key_entities:
            lines.append(f"     Key people: {', '.join(cluster.key_entities)}")
    return "\n".join(lines)


def _answer_cross_topic(report: PatternReport) -> str:
    if not report.entity_cluster_links:
        return "No entities bridge multiple topic clusters."

    lines = ["Entities that connect multiple topics:", ""]
    for link in report.entity_cluster_links:
        lines.append(f"  {link.entity} — appears in {link.cluster_count} clusters ({link.total_phase_count} phases):")
        for cl in link.cluster_labels:
            lines.append(f"    - {cl}")
    return "\n".join(lines)


def _answer_temporal(report: PatternReport) -> str:
    if not report.temporal_bursts:
        return "No periods of concentrated activity found."

    lines = [f"Periods of intense activity ({len(report.temporal_bursts)} bursts):", ""]
    for burst in report.temporal_bursts[:5]:
        ent_part = f" — involving {', '.join(burst.entities)}" if burst.entities else ""
        lines.append(f"  {burst.date_range}: {burst.burst_size} phases{ent_part}")
    return "\n".join(lines)


def _answer_timeline(
    report: PatternReport,
    narratives: list[NarrativeReconstruction],
) -> str:
    all_transitions: list[str] = []
    all_gaps: list[str] = []

    for narr in narratives:
        for t in narr.transitions:
            all_transitions.append(f"{t.from_phase} → {t.to_phase}: {t.description}")
        for g in narr.gaps:
            all_gaps.append(g.description)

    if not all_transitions and not all_gaps and not report.temporal_bursts:
        return "Not enough data to describe how things changed over time."

    lines = ["How things evolved:", ""]

    if report.temporal_bursts:
        lines.append("  Activity concentration:")
        for burst in report.temporal_bursts[:3]:
            ent_part = f" ({', '.join(burst.entities)})" if burst.entities else ""
            lines.append(f"    {burst.date_range}: {burst.burst_size} phases{ent_part}")
        lines.append("")

    if all_transitions:
        lines.append(f"  Transitions detected: {len(all_transitions)}")
        for t in all_transitions[:5]:
            lines.append(f"    - {t}")
        lines.append("")

    if all_gaps:
        lines.append(f"  Gaps in activity: {len(all_gaps)}")
        for g in all_gaps[:5]:
            lines.append(f"    - {g}")

    return "\n".join(lines)


def _answer_unknown(report: PatternReport) -> str:
    lines = [
        "I can answer questions about:",
        "",
        "  - Who are the main people? (entity-focused)",
        "  - What happened with [name]? (entity-scoped)",
        "  - What are the main themes? (theme-focused)",
        "  - Who connects multiple topics? (cross-topic)",
        "  - When were things most active? (temporal intensity)",
        "  - What changed over time? (timeline/transitions)",
    ]
    return "\n".join(lines)
