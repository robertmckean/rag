"""Deterministic query routing over pattern extraction outputs.

Maps user questions to the appropriate subset of PatternReport and
NarrativeReconstruction data, then produces concise grounded answers.
No LLM, no new analytics — routing and formatting only.
"""

from __future__ import annotations

from enum import Enum

from rag.narrative.models import NarrativeReconstruction
from rag.patterns.models import PatternReport


class Intent(Enum):
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


def classify_intent(query: str) -> Intent:
    """Classify a user question into one of the supported intents."""
    tokens = set(query.lower().split())

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
    intent = classify_intent(query)

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
        "  - What are the main themes? (theme-focused)",
        "  - Who connects multiple topics? (cross-topic)",
        "  - When were things most active? (temporal intensity)",
        "  - What changed over time? (timeline/transitions)",
    ]
    return "\n".join(lines)
