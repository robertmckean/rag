"""Core narrative reconstruction logic.

Builds a ``NarrativeReconstruction`` directly from retrieved evidence items
without requiring a Phase 5 ``AnswerResult``.  The pipeline is:

    evidence → sort → group into phases → detect transitions → detect gaps
    → detect limitations → generate summary → NarrativeReconstruction
"""

from __future__ import annotations

import re
from pathlib import Path

from rag.answering.generator_llm import _NON_ENTITY_WORDS
from rag.answering.models import EvidenceItem
from rag.answering.qualify import focus_terms as extract_focus_terms, select_evidence
from rag.narrative.models import (
    NarrativeGap,
    NarrativeLimitation,
    NarrativePhase,
    NarrativeReconstruction,
    NarrativeTransition,
)
from rag.retrieval.lexical import RetrievalFilters, retrieve_message_windows


DEFAULT_PHASE_WINDOW_DAYS = 7
DEFAULT_GAP_THRESHOLD_DAYS = 30
DEFAULT_NARRATIVE_LIMIT = 10
DEFAULT_NARRATIVE_MAX_EVIDENCE = 10


# Build a narrative reconstruction from a run directory and query.
def build_narrative_from_run(
    run_dir: Path,
    query: str,
    *,
    retrieval_mode: str = "relevance",
    channel: str = "bm25",
    limit: int = DEFAULT_NARRATIVE_LIMIT,
    max_evidence: int = DEFAULT_NARRATIVE_MAX_EVIDENCE,
    filters: RetrievalFilters | None = None,
    phase_window_days: int = DEFAULT_PHASE_WINDOW_DAYS,
    gap_threshold_days: int = DEFAULT_GAP_THRESHOLD_DAYS,
) -> NarrativeReconstruction:
    retrieval_results = retrieve_message_windows(
        run_dir.resolve(),
        query,
        limit=limit,
        mode=retrieval_mode,
        filters=filters,
        channel=channel,
    )
    evidence = select_evidence(retrieval_results, query, max_evidence=max_evidence)
    return build_narrative(
        query,
        evidence,
        phase_window_days=phase_window_days,
        gap_threshold_days=gap_threshold_days,
    )


# Build a narrative reconstruction from pre-selected evidence items.
def build_narrative(
    query: str,
    evidence: tuple[EvidenceItem, ...],
    *,
    phase_window_days: int = DEFAULT_PHASE_WINDOW_DAYS,
    gap_threshold_days: int = DEFAULT_GAP_THRESHOLD_DAYS,
) -> NarrativeReconstruction:
    if not evidence:
        return NarrativeReconstruction(
            query=query,
            summary="No evidence was retrieved for this query.",
            timeline=(),
            transitions=(),
            gaps=(),
            limitations=(),
            evidence_count=0,
        )

    sorted_evidence = _sort_chronologically(evidence)
    limitations = _detect_limitations(sorted_evidence)
    query_terms = extract_focus_terms(query)
    phases = _group_into_phases(sorted_evidence, query_terms, phase_window_days)
    transitions = _detect_transitions(phases, sorted_evidence)
    gaps = _detect_gaps(phases, gap_threshold_days)
    summary = _generate_summary(query, phases, transitions, gaps)

    return NarrativeReconstruction(
        query=query,
        summary=summary,
        timeline=tuple(phases),
        transitions=tuple(transitions),
        gaps=tuple(gaps),
        limitations=tuple(limitations),
        evidence_count=len(evidence),
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _parse_date(timestamp: str | None) -> str | None:
    """Extract YYYY-MM-DD prefix from an ISO timestamp, or None."""
    if not timestamp or len(timestamp) < 10:
        return None
    match = re.match(r"\d{4}-\d{2}-\d{2}", timestamp)
    return match.group(0) if match else None


def _days_between(date_a: str, date_b: str) -> int:
    """Approximate day distance between two YYYY-MM-DD strings."""
    try:
        ya, ma, da = int(date_a[:4]), int(date_a[5:7]), int(date_a[8:10])
        yb, mb, db = int(date_b[:4]), int(date_b[5:7]), int(date_b[8:10])
        # Rough Julian-day calculation — sufficient for gap/window checks.
        jd_a = ya * 365 + ma * 30 + da
        jd_b = yb * 365 + mb * 30 + db
        return abs(jd_b - jd_a)
    except (ValueError, IndexError):
        return 999


def _evidence_id(item: EvidenceItem) -> str:
    return f"e{item.rank}"


_COHERENCE_STOPWORDS = frozenset({
    "the", "and", "was", "were", "are", "that", "this", "with", "for", "from",
    "but", "not", "had", "has", "have", "been", "being", "about", "just",
    "some", "also", "very", "would", "could", "should", "there", "their",
    "they", "them", "then", "than", "what", "when", "where", "which", "who",
    "how", "all", "can", "did", "does", "into", "its", "may", "more",
    "most", "her", "him", "his", "she", "out", "our", "own", "too", "you",
})


def content_terms_from_text(text: str) -> set[str]:
    """Extract lowercase content words (4+ chars, no stopwords) from arbitrary text.

    Shared helper used by both narrative grouping and topic clustering.
    """
    return {
        w.lower()
        for w in re.findall(r"[A-Za-z]{4,}", text)
        if w.lower() not in _COHERENCE_STOPWORDS
    }


def _excerpt_terms(item: EvidenceItem) -> set[str]:
    """Extract lowercase content words (4+ chars, no stopwords) from an evidence excerpt."""
    return content_terms_from_text(item.citation.excerpt)


def entity_terms_from_text(text: str) -> set[str]:
    """Extract capitalized proper-noun candidates from arbitrary text.

    Shared helper used by both narrative grouping and pattern extraction.
    """
    return {
        m.group(0)
        for m in re.finditer(r"\b[A-Z][a-z]{2,}\b", text)
        if m.group(0) not in _NON_ENTITY_WORDS
    }


def _entity_terms(item: EvidenceItem) -> set[str]:
    """Extract capitalized proper-noun candidates from an evidence excerpt."""
    return entity_terms_from_text(item.citation.excerpt)


def _topic_overlap(terms_a: set[str], terms_b: set[str]) -> float:
    """Jaccard similarity between two term sets."""
    if not terms_a or not terms_b:
        return 0.0
    intersection = terms_a & terms_b
    union = terms_a | terms_b
    return len(intersection) / len(union)


_TOPIC_COHERENCE_THRESHOLD = 0.08


def _sort_chronologically(evidence: tuple[EvidenceItem, ...]) -> list[EvidenceItem]:
    """Sort evidence by created_at.  Items without timestamps sort last."""

    def sort_key(item: EvidenceItem) -> tuple[int, str, int]:
        ts = item.citation.created_at or ""
        has_ts = 0 if ts else 1
        return (has_ts, ts, item.rank)

    return sorted(evidence, key=sort_key)


def _group_into_phases(
    sorted_evidence: list[EvidenceItem],
    query_terms: tuple[str, ...],
    phase_window_days: int,
) -> list[NarrativePhase]:
    """Group evidence into phases requiring both temporal proximity and topic coherence."""
    if not sorted_evidence:
        return []

    query_term_set = {t.lower() for t in query_terms}

    # Each group is a list of evidence items in the current phase.
    groups: list[list[EvidenceItem]] = [[sorted_evidence[0]]]

    for item in sorted_evidence[1:]:
        current_group = groups[-1]
        last_item = current_group[-1]

        # Check temporal proximity.
        date_cur = _parse_date(item.citation.created_at)
        date_last = _parse_date(last_item.citation.created_at)

        within_window = False
        if date_cur and date_last:
            within_window = _days_between(date_last, date_cur) <= phase_window_days
        elif not date_cur and not date_last:
            # Both lack timestamps — group by topic only.
            within_window = True

        # Check topic coherence (query terms excluded — they inflate overlap
        # because every evidence item is about the query by construction).
        item_terms = (_excerpt_terms(item) | _entity_terms(item)) - query_term_set
        group_terms: set[str] = set()
        for gi in current_group:
            group_terms |= _excerpt_terms(gi) | _entity_terms(gi)
        group_terms -= query_term_set

        coherent = _topic_overlap(item_terms, group_terms) >= _TOPIC_COHERENCE_THRESHOLD

        if within_window and coherent:
            current_group.append(item)
        else:
            groups.append([item])

    # Convert groups to NarrativePhase objects.
    phases: list[NarrativePhase] = []
    for group in groups:
        phases.append(_build_phase(group, query_terms))

    return phases


def _build_phase(items: list[EvidenceItem], query_terms: tuple[str, ...]) -> NarrativePhase:
    """Construct a NarrativePhase from a group of evidence items."""
    evidence_ids = tuple(_evidence_id(item) for item in items)

    # Date range.
    dates = [_parse_date(item.citation.created_at) for item in items]
    valid_dates = sorted(d for d in dates if d)
    if valid_dates:
        if valid_dates[0] == valid_dates[-1]:
            date_range = valid_dates[0]
        else:
            date_range = f"{valid_dates[0]} to {valid_dates[-1]}"
    else:
        date_range = None

    # Deterministic label: date range + dominant entities (user-sourced preferred).
    user_entities: dict[str, int] = {}
    all_entities: dict[str, int] = {}
    for item in items:
        for ent in _entity_terms(item):
            all_entities[ent] = all_entities.get(ent, 0) + 1
            if item.author_role == "user":
                user_entities[ent] = user_entities.get(ent, 0) + 1
    # Prefer entities from user items; fall back to all entities if no user items.
    entity_source = user_entities if user_entities else all_entities
    sorted_entities = sorted(entity_source.items(), key=lambda x: (-x[1], x[0]))
    dominant = [e[0] for e in sorted_entities[:2]]

    if date_range and dominant:
        label = f"{date_range}: {', '.join(dominant)}"
    elif date_range:
        label = date_range
    elif dominant:
        label = ", ".join(dominant)
    else:
        label = f"Phase (evidence {', '.join(evidence_ids)})"

    # Description: grounded in evidence excerpts, user-role first.
    user_items = [item for item in items if item.author_role == "user"]
    assistant_items = [item for item in items if item.author_role != "user"]
    ordered_items = user_items + assistant_items

    description_parts: list[str] = []
    for item in ordered_items:
        date = _parse_date(item.citation.created_at)
        prefix = f"({date}) " if date else ""
        excerpt = item.citation.excerpt
        if len(excerpt) > 150:
            excerpt = excerpt[:147] + "..."
        role = item.author_role or "unknown"
        description_parts.append(f"{prefix}[{role}] \"{excerpt}\"")

    description = " | ".join(description_parts)

    # Support: single-evidence = supported, multi-evidence = partially_supported.
    support = "supported" if len(items) == 1 else "partially_supported"

    return NarrativePhase(
        label=label,
        description=description,
        evidence_ids=evidence_ids,
        date_range=date_range,
        support=support,
    )


def _detect_transitions(
    phases: list[NarrativePhase],
    sorted_evidence: list[EvidenceItem],
) -> list[NarrativeTransition]:
    """Detect shifts between adjacent phases.

    Priority order:
    1. Explicit topic change (entities/terms differ)
    2. Explicit stated change (evidence text references change)
    3. Temporal discontinuity (large time gap)
    Sentiment shift is secondary — only noted when no stronger signal exists.
    """
    if len(phases) < 2:
        return []

    # Build a map from evidence_id to EvidenceItem for term extraction.
    evidence_by_id: dict[str, EvidenceItem] = {}
    for item in sorted_evidence:
        evidence_by_id[_evidence_id(item)] = item

    transitions: list[NarrativeTransition] = []

    for i in range(len(phases) - 1):
        phase_a = phases[i]
        phase_b = phases[i + 1]

        # Gather terms and entities for each phase.
        terms_a: set[str] = set()
        entities_a: set[str] = set()
        for eid in phase_a.evidence_ids:
            if eid in evidence_by_id:
                terms_a |= _excerpt_terms(evidence_by_id[eid])
                entities_a |= _entity_terms(evidence_by_id[eid])

        terms_b: set[str] = set()
        entities_b: set[str] = set()
        for eid in phase_b.evidence_ids:
            if eid in evidence_by_id:
                terms_b |= _excerpt_terms(evidence_by_id[eid])
                entities_b |= _entity_terms(evidence_by_id[eid])

        # Evidence IDs for this transition: last item(s) of phase_a + first item(s) of phase_b.
        transition_evidence = (phase_a.evidence_ids[-1:] + phase_b.evidence_ids[:1])

        # 1. Topic change: new entities appearing, old ones disappearing.
        new_entities = entities_b - entities_a
        lost_entities = entities_a - entities_b
        topic_shift_parts: list[str] = []
        if new_entities:
            topic_shift_parts.append(f"new topics: {', '.join(sorted(new_entities)[:3])}")
        if lost_entities:
            topic_shift_parts.append(f"prior topics absent: {', '.join(sorted(lost_entities)[:3])}")

        # 2. Temporal discontinuity.
        date_a_end = _parse_date_from_phase(phase_a, evidence_by_id, last=True)
        date_b_start = _parse_date_from_phase(phase_b, evidence_by_id, last=False)
        time_gap_days = None
        temporal_part = ""
        if date_a_end and date_b_start:
            time_gap_days = _days_between(date_a_end, date_b_start)
            if time_gap_days > 0:
                temporal_part = f"{time_gap_days}-day gap ({date_a_end} to {date_b_start})"

        # 3. Stated change: check if phase_b evidence references change language.
        stated_change = _detect_stated_change(phase_b, evidence_by_id)

        # Build description from strongest signals.
        desc_parts: list[str] = []
        if topic_shift_parts:
            desc_parts.append("Topic shift: " + "; ".join(topic_shift_parts))
        if stated_change:
            desc_parts.append(f"Stated change: {stated_change}")
        if temporal_part:
            desc_parts.append(f"Time: {temporal_part}")

        # Secondary: sentiment shift only if no other signal.
        if not desc_parts:
            sentiment = _detect_sentiment_shift(phase_a, phase_b, evidence_by_id)
            if sentiment:
                desc_parts.append(f"Tone shift: {sentiment}")

        if not desc_parts:
            desc_parts.append("Sequential continuation")

        description = ". ".join(desc_parts)

        transitions.append(NarrativeTransition(
            from_phase=phase_a.label,
            to_phase=phase_b.label,
            description=description,
            evidence_ids=transition_evidence,
            support="partially_supported",
        ))

    return transitions


def _parse_date_from_phase(
    phase: NarrativePhase,
    evidence_by_id: dict[str, EvidenceItem],
    *,
    last: bool,
) -> str | None:
    """Get the first or last date from a phase's evidence."""
    ids = phase.evidence_ids
    search_order = reversed(ids) if last else iter(ids)
    for eid in search_order:
        item = evidence_by_id.get(eid)
        if item:
            d = _parse_date(item.citation.created_at)
            if d:
                return d
    return None


_CHANGE_PATTERNS = re.compile(
    r"\b(started|stopped|changed|decided|realized|moved|left|returned|embraced|shifted|switched)\b",
    re.IGNORECASE,
)


def _detect_stated_change(
    phase: NarrativePhase,
    evidence_by_id: dict[str, EvidenceItem],
) -> str | None:
    """Check if phase evidence explicitly references a change event."""
    for eid in phase.evidence_ids:
        item = evidence_by_id.get(eid)
        if not item:
            continue
        match = _CHANGE_PATTERNS.search(item.citation.excerpt)
        if match:
            # Return a short context window around the change word.
            start = max(0, match.start() - 30)
            end = min(len(item.citation.excerpt), match.end() + 50)
            context = item.citation.excerpt[start:end].strip()
            if start > 0:
                context = "..." + context
            if end < len(item.citation.excerpt):
                context = context + "..."
            return f'"{context}"'
    return None


_POSITIVE_CUES = {"better", "amazing", "improved", "love", "enjoy", "happy", "great", "wonderful"}
_NEGATIVE_CUES = {"down", "unhappy", "frustrated", "angry", "draining", "exhausted", "sad", "worried"}


def _detect_sentiment_shift(
    phase_a: NarrativePhase,
    phase_b: NarrativePhase,
    evidence_by_id: dict[str, EvidenceItem],
) -> str | None:
    """Detect a sentiment shift between two phases (secondary signal only)."""

    def phase_sentiment(phase: NarrativePhase) -> tuple[int, int]:
        pos = neg = 0
        for eid in phase.evidence_ids:
            item = evidence_by_id.get(eid)
            if not item:
                continue
            words = set(item.citation.excerpt.lower().split())
            pos += len(words & _POSITIVE_CUES)
            neg += len(words & _NEGATIVE_CUES)
        return pos, neg

    pos_a, neg_a = phase_sentiment(phase_a)
    pos_b, neg_b = phase_sentiment(phase_b)

    if pos_a > neg_a and neg_b > pos_b:
        return "positive to negative"
    if neg_a > pos_a and pos_b > neg_b:
        return "negative to positive"
    return None


def _detect_gaps(
    phases: list[NarrativePhase],
    gap_threshold_days: int,
) -> list[NarrativeGap]:
    """Find temporal holes between phases exceeding the gap threshold."""
    gaps: list[NarrativeGap] = []

    for i in range(len(phases) - 1):
        date_a = _last_date_from_range(phases[i].date_range)
        date_b = _first_date_from_range(phases[i + 1].date_range)

        if not date_a or not date_b:
            continue

        days = _days_between(date_a, date_b)
        if days >= gap_threshold_days:
            gaps.append(NarrativeGap(
                description=f"No evidence between {date_a} and {date_b} ({days} days)",
                reason="Retrieval did not surface messages from this period",
            ))

    return gaps


def _first_date_from_range(date_range: str | None) -> str | None:
    if not date_range:
        return None
    return date_range[:10] if len(date_range) >= 10 else None


def _last_date_from_range(date_range: str | None) -> str | None:
    if not date_range:
        return None
    if " to " in date_range:
        return date_range.split(" to ")[-1][:10]
    return date_range[:10] if len(date_range) >= 10 else None


def _detect_limitations(sorted_evidence: list[EvidenceItem]) -> list[NarrativeLimitation]:
    """Detect data-quality limitations in the evidence set."""
    limitations: list[NarrativeLimitation] = []
    has_null_ts = False
    has_ambiguous_order = False

    prev_date: str | None = None
    for item in sorted_evidence:
        # Truncated excerpts.
        if len(item.citation.excerpt) >= 195:
            limitations.append(NarrativeLimitation(
                description=f"Evidence {_evidence_id(item)} excerpt is truncated at ~200 characters",
                kind="truncated_excerpt",
            ))

        # Null timestamps.
        ts = item.citation.created_at
        if not ts or len(ts) < 10:
            if not has_null_ts:
                limitations.append(NarrativeLimitation(
                    description="One or more evidence items lack timestamps; chronological ordering may be unreliable",
                    kind="null_timestamp",
                ))
                has_null_ts = True
            continue

        # Ambiguous ordering: same date as previous item.
        cur_date = _parse_date(ts)
        if cur_date and cur_date == prev_date and not has_ambiguous_order:
            limitations.append(NarrativeLimitation(
                description=f"Multiple evidence items share the date {cur_date}; intra-day ordering is based on timestamp precision",
                kind="ambiguous_ordering",
            ))
            has_ambiguous_order = True
        prev_date = cur_date

    return limitations


def _generate_summary(
    query: str,
    phases: list[NarrativePhase],
    transitions: list[NarrativeTransition],
    gaps: list[NarrativeGap],
) -> str:
    """Generate a deterministic summary from phases and transitions."""
    if not phases:
        return "No evidence was retrieved for this query."

    parts: list[str] = []

    # Opening: scope statement.
    date_range_parts = [p.date_range for p in phases if p.date_range]
    if date_range_parts:
        first_date = _first_date_from_range(date_range_parts[0])
        last_date = _last_date_from_range(date_range_parts[-1])
        if first_date and last_date and first_date != last_date:
            parts.append(f"The evidence spans from {first_date} to {last_date} across {len(phases)} phase(s).")
        elif first_date:
            parts.append(f"The evidence is concentrated around {first_date} across {len(phases)} phase(s).")
    else:
        parts.append(f"The evidence covers {len(phases)} phase(s) without clear date markers.")

    # Phase summaries (one sentence each, up to 4).
    for phase in phases[:4]:
        n_items = len(phase.evidence_ids)
        date_part = f" ({phase.date_range})" if phase.date_range else ""
        parts.append(f"{phase.label}{date_part}: {n_items} evidence item(s), {phase.support}.")

    if len(phases) > 4:
        parts.append(f"({len(phases) - 4} additional phase(s) not shown in summary.)")

    # Gaps note.
    if gaps:
        gap_descriptions = [g.description for g in gaps[:2]]
        parts.append("Notable gap(s): " + "; ".join(gap_descriptions) + ".")

    return " ".join(parts)
