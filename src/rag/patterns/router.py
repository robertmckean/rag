"""Deterministic query routing over pattern extraction outputs.

Maps user questions to the appropriate subset of PatternReport and
NarrativeReconstruction data, then produces concise grounded answers.
No LLM, no new analytics — routing and formatting only.
"""

from __future__ import annotations

import re
import string

from enum import Enum

from rag.narrative.models import NarrativePhase, NarrativeReconstruction, NarrativeTransition
from rag.narrative.positions import (
    Contradiction,
    ThinkingEvolution,
    build_thinking_evolution,
    collect_positions_for_entity,
    detect_contradictions,
    extract_positions,
)
from rag.patterns.models import PatternReport


class Intent(Enum):
    CONTRADICTION = "contradiction"
    EVOLUTION = "evolution"
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


# Evolution keywords — any of these alongside an entity triggers EVOLUTION.
_EVOLUTION_KEYWORDS: frozenset[str] = frozenset({
    "evolve", "evolved", "thinking", "change", "changed",
    "path", "journey", "develop", "developed", "progression",
})

# Strong evolution keywords — these trigger EVOLUTION even without an entity.
_EVOLUTION_STRONG_KEYWORDS: frozenset[str] = frozenset({
    "thinking", "journey", "path", "progression",
    "develop", "developed", "evolve", "evolved",
})


# Contradiction/change keywords — these trigger CONTRADICTION when present.
_CONTRADICTION_KEYWORDS: frozenset[str] = frozenset({
    "contradict", "contradiction", "contradicted",
    "reverse", "reversed", "reversal",
    "soften", "softened", "softening",
    "strengthen", "strengthened", "strengthening",
    "grew", "grow", "growth", "clearer",
})

# Phrase-level contradiction triggers checked against normalized query.
_CONTRADICTION_PHRASES: tuple[str, ...] = (
    "changed my mind",
    "change my mind",
    "become clearer",
)


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
    stripped_tokens = set(_normalize_query(query).split())
    normalized_query = _normalize_query(query)

    # Check for contradiction keywords/phrases.
    has_contradiction = bool(stripped_tokens & _CONTRADICTION_KEYWORDS) or any(
        p in normalized_query for p in _CONTRADICTION_PHRASES
    )

    # CONTRADICTION takes priority over EVOLUTION when explicit.
    if has_contradiction:
        return Intent.CONTRADICTION

    # Check EVOLUTION before ENTITY_SCOPED — evolution queries take priority.
    if report is not None:
        entity = _detect_entity(query, report)
        has_evolution = bool(stripped_tokens & _EVOLUTION_KEYWORDS)
        has_strong_evolution = bool(stripped_tokens & _EVOLUTION_STRONG_KEYWORDS)

        if entity is not None and has_evolution:
            return Intent.EVOLUTION
        if has_strong_evolution:
            return Intent.EVOLUTION

        # Check ENTITY_SCOPED: entity + question word, no evolution signal.
        if entity is not None:
            if stripped_tokens & _ENTITY_SCOPED_QUESTION_WORDS:
                return Intent.ENTITY_SCOPED
    else:
        # No report — still check for strong evolution keywords.
        has_strong_evolution = bool(stripped_tokens & _EVOLUTION_STRONG_KEYWORDS)
        if has_strong_evolution:
            return Intent.EVOLUTION

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

    if intent == Intent.CONTRADICTION:
        entity = _detect_entity(query, report)
        return _answer_contradiction(entity, query, narratives)
    if intent == Intent.EVOLUTION:
        entity = _detect_entity(query, report)
        return _answer_evolution(entity, query, narratives)
    if intent == Intent.ENTITY_SCOPED:
        entity = _detect_entity(query, report)
        assert entity is not None  # guaranteed by classify_intent
        return _answer_entity_scoped(entity, report, narratives)
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


def _phase_mentions_entity(phase: NarrativePhase, entity_name: str) -> bool:
    """Check whether *entity_name* appears as a whole word in the phase label or description."""
    pattern = r"\b" + re.escape(entity_name) + r"\b"
    return bool(
        re.search(pattern, phase.label, re.IGNORECASE)
        or re.search(pattern, phase.description, re.IGNORECASE)
    )


def _filter_narrative_for_entity(
    narratives: list[NarrativeReconstruction],
    entity_name: str,
) -> tuple[tuple[NarrativePhase, ...], tuple[NarrativeTransition, ...]]:
    """Filter narrative phases and transitions to those mentioning *entity_name*.

    Returns (filtered_phases, scoped_transitions) across all narratives.
    Transitions between retained phases are kept if they existed originally;
    for retained phases that were separated by removed phases, a minimal gap
    transition is created.
    """
    all_phases: list[NarrativePhase] = []
    # Map from (from_label, to_label) to original transition for quick lookup.
    original_transitions: dict[tuple[str, str], NarrativeTransition] = {}

    for narr in narratives:
        for phase in narr.timeline:
            if _phase_mentions_entity(phase, entity_name):
                all_phases.append(phase)
        for t in narr.transitions:
            original_transitions[(t.from_phase, t.to_phase)] = t

    # Build transitions between consecutive retained phases.
    scoped_transitions: list[NarrativeTransition] = []
    for i in range(len(all_phases) - 1):
        from_phase = all_phases[i]
        to_phase = all_phases[i + 1]
        key = (from_phase.label, to_phase.label)
        if key in original_transitions:
            scoped_transitions.append(original_transitions[key])
        else:
            # Phases were not originally adjacent — insert a gap transition.
            scoped_transitions.append(NarrativeTransition(
                from_phase=from_phase.label,
                to_phase=to_phase.label,
                description="No direct transition observed (intermediate phases filtered out)",
                evidence_ids=(),
                support="partially_supported",
            ))

    return tuple(all_phases), tuple(scoped_transitions)


def _answer_contradiction(
    entity_name: str | None,
    query: str,
    narratives: list[NarrativeReconstruction],
) -> str:
    """Produce a contradiction/change answer from extracted positions."""
    # Collect positions from all narrative phases.
    all_positions = []
    for narr in narratives:
        for phase in narr.timeline:
            all_positions.extend(extract_positions(phase))

    # Scope to entity when present.
    label = entity_name or query
    if entity_name:
        scoped = list(collect_positions_for_entity(tuple(all_positions), entity_name))
    else:
        scoped = all_positions

    lines = [f"Contradiction/change analysis: {label}", ""]

    # Case A: fewer than 2 positions.
    if len(scoped) < 2:
        lines.append("  Insufficient evidence to determine contradiction or change.")
        if scoped:
            lines.append("")
            lines.append("  Extracted position(s):")
            for p in scoped:
                date_part = f"[{p.date}] " if p.date else ""
                lines.append(f"    - {date_part}({p.stance_marker}) {p.text}")
        return "\n".join(lines)

    contradictions = detect_contradictions(label, scoped)

    # Case B: 2+ positions, no contradictions.
    if not contradictions:
        lines.append("  No contradiction or meaningful change signal detected.")
        lines.append("")
        lines.append(f"  Positions ({len(scoped)}):")
        from rag.narrative.positions import _sort_key
        for p in sorted(scoped, key=_sort_key):
            date_part = f"[{p.date}] " if p.date else ""
            lines.append(f"    - {date_part}({p.stance_marker}) {p.text}")
        return "\n".join(lines)

    # Case C: 1+ contradictions detected.
    lines.append("  Possible contradiction/change detected.")
    lines.append("")
    lines.append(f"  Detected change(s) ({len(contradictions)}):")
    for c in contradictions:
        lines.append(f"    [{c.date_range}] possible {c.change_type}")
        lines.append(f"      Earlier: ({c.earlier.stance_marker}) {c.earlier.text}")
        lines.append(f"      Later:   ({c.later.stance_marker}) {c.later.text}")
        lines.append(f"      Signal:  {c.signal}")

    return "\n".join(lines)


def _answer_evolution(
    entity_name: str | None,
    query: str,
    narratives: list[NarrativeReconstruction],
) -> str:
    """Produce an evolution answer from extracted positions."""
    # Collect positions from all narrative phases.
    all_positions = []
    for narr in narratives:
        for phase in narr.timeline:
            all_positions.extend(extract_positions(phase))

    # Scope to entity when present.
    label = entity_name or query
    if entity_name:
        scoped = collect_positions_for_entity(tuple(all_positions), entity_name)
    else:
        scoped = tuple(all_positions)

    evo = build_thinking_evolution(label, scoped)

    lines = [f"Thinking evolution: {label}", ""]

    # Case A: fewer than 2 positions.
    if len(evo.positions) < 2:
        lines.append("  Insufficient evidence to determine evolution.")
        if evo.positions:
            lines.append("")
            lines.append("  Extracted position(s):")
            for p in evo.positions:
                date_part = f"[{p.date}] " if p.date else ""
                lines.append(f"    - {date_part}({p.stance_marker}) {p.text}")
        return "\n".join(lines)

    # Case B: 2+ positions, no shifts.
    if not evo.shifts:
        lines.append("  Position appears stable over time.")
        lines.append("")
        lines.append(f"  Positions ({len(evo.positions)}):")
        for p in evo.positions:
            date_part = f"[{p.date}] " if p.date else ""
            lines.append(f"    - {date_part}({p.stance_marker}) {p.text}")
        return "\n".join(lines)

    # Case C: 2+ positions with shifts.
    lines.append("  Possible evolution detected.")
    lines.append("")
    lines.append(f"  Positions ({len(evo.positions)}):")
    for p in evo.positions:
        date_part = f"[{p.date}] " if p.date else ""
        lines.append(f"    - {date_part}({p.stance_marker}) {p.text}")
    lines.append("")
    lines.append(f"  Shifts ({len(evo.shifts)}):")
    for s in evo.shifts:
        lines.append(f"    - {s}")

    return "\n".join(lines)


def _answer_entity_scoped(
    entity_name: str,
    report: PatternReport,
    narratives: list[NarrativeReconstruction],
) -> str:
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
        lines.append("")

    # Scoped timeline from narrative data.
    scoped_phases, scoped_transitions = _filter_narrative_for_entity(
        narratives, entity_name,
    )

    if not scoped_phases:
        lines.append("  Timeline: no timeline data found for this entity.")
    elif len(scoped_phases) == 1:
        phase = scoped_phases[0]
        date_part = f" ({phase.date_range})" if phase.date_range else ""
        lines.append("  Timeline (1 phase):")
        lines.append(f"    1. {phase.label}{date_part}")
        lines.append(f"       {phase.description}")
    else:
        lines.append(f"  Timeline ({len(scoped_phases)} phases):")
        for i, phase in enumerate(scoped_phases, 1):
            date_part = f" ({phase.date_range})" if phase.date_range else ""
            lines.append(f"    {i}. {phase.label}{date_part}")
            lines.append(f"       {phase.description}")
        if scoped_transitions:
            lines.append("")
            lines.append(f"  Transitions ({len(scoped_transitions)}):")
            for t in scoped_transitions:
                lines.append(f"    - {t.from_phase} → {t.to_phase}: {t.description}")

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
        "  - How did my thinking about [name] change? (evolution)",
        "  - Where did I contradict myself? (contradiction/change)",
        "  - What are the main themes? (theme-focused)",
        "  - Who connects multiple topics? (cross-topic)",
        "  - When were things most active? (temporal intensity)",
        "  - What changed over time? (timeline/transitions)",
    ]
    return "\n".join(lines)
