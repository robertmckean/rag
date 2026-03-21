"""Phase 13A — Deterministic user-position extraction from narrative phases.

Scans narrative phase descriptions for explicit stance markers in user-role
segments only.  Produces ``Position`` frozen dataclasses that record what the
user stated, when, and which entity (if any) is mentioned.

No LLM, no embeddings, no semantic analysis — pure string/rule matching.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

from rag.narrative.builder import entity_terms_from_text
from rag.narrative.models import NarrativePhase


# ---------------------------------------------------------------------------
# Stance markers — explicit first-person position language
# ---------------------------------------------------------------------------

STANCE_MARKERS: tuple[str, ...] = (
    # Core position markers (from spec).
    "i think",
    "i believe",
    "i decided",
    "i realized",
    "i concluded",
    "my view is",
    "i was wrong about",
    "i changed my mind",
    "i no longer",
    "i used to think",
    # Negated stance.
    "i don't think",
    "i don't believe",
    # Interpretive/epistemic stance.
    "i feel like",
    "i noticed",
    "i learned",
    "i understand",
    "i'm sure",
    "i figured out",
)

# Unicode smart-quote variants that should be treated as ASCII apostrophes.
_SMART_APOSTROPHES = "\u2018\u2019\u02bc"  # left single, right single, modifier letter

def _normalize_apostrophes(text: str) -> str:
    """Replace Unicode smart quotes with ASCII apostrophes for matching."""
    for ch in _SMART_APOSTROPHES:
        text = text.replace(ch, "'")
    return text

# Pre-compiled pattern: match any stance marker as a whole phrase (case-insensitive).
# Sort longest-first so "i don't think" is tried before "i think".
_SORTED_MARKERS = sorted(STANCE_MARKERS, key=len, reverse=True)
_STANCE_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(m) for m in _SORTED_MARKERS) + r")\b",
    re.IGNORECASE,
)

# Pattern to split pipe-separated description into individual segments.
_SEGMENT_SPLIT = re.compile(r"\s*\|\s*")

# Pattern to extract role and quoted text from a description segment.
# Matches: (optional_date) [role] "excerpt"
_SEGMENT_PARSE = re.compile(
    r"(?:\(([^)]*)\)\s*)?\[(\w+)\]\s*\"(.*?)\"$",
    re.DOTALL,
)


# ---------------------------------------------------------------------------
# Position model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Position:
    """A user-authored position statement extracted from a narrative phase."""

    text: str               # extracted position statement
    date: str | None        # phase date if available (YYYY-MM-DD or range)
    entity: str | None      # associated entity if explicitly present, else None
    evidence_id: str        # source evidence item id
    stance_marker: str      # exact matched marker (lowercased)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def _parse_segments(description: str) -> list[tuple[str | None, str, str]]:
    """Parse a phase description into (date, role, text) triples."""
    raw_segments = _SEGMENT_SPLIT.split(description)
    result: list[tuple[str | None, str, str]] = []
    for seg in raw_segments:
        seg = seg.strip()
        m = _SEGMENT_PARSE.match(seg)
        if m:
            date = m.group(1) or None
            role = m.group(2)
            text = m.group(3)
            result.append((date, role, text))
    return result


def _find_sentence_containing(text: str, marker_start: int, marker_end: int) -> str:
    """Extract the sentence that contains the stance marker.

    Uses simple sentence boundary detection (period/exclamation/question mark
    followed by space or end of string).  Falls back to the full text if no
    boundaries are found.
    """
    # Find sentence boundaries.
    boundaries = [0]
    for m in re.finditer(r"[.!?]\s+", text):
        boundaries.append(m.end())
    boundaries.append(len(text))

    # Find the sentence that contains the marker.
    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end = boundaries[i + 1]
        if start <= marker_start < end:
            return text[start:end].strip()

    return text.strip()


def extract_positions(
    phase: NarrativePhase,
    role_filter: str = "user",
) -> tuple[Position, ...]:
    """Extract explicit stance-marker positions from user-role segments of a phase.

    Parameters
    ----------
    phase : NarrativePhase
        The narrative phase to scan.
    role_filter : str
        Only segments with this role are scanned (default ``"user"``).

    Returns
    -------
    tuple[Position, ...]
        Extracted positions, ordered by appearance in the description.
    """
    segments = _parse_segments(phase.description)

    # Build a mapping from user-segment index to evidence_id.
    # User segments appear first in description (Phase 12A ordering), so
    # map them to evidence_ids in order, cycling if necessary.
    evidence_ids = phase.evidence_ids
    if not evidence_ids:
        return ()

    positions: list[Position] = []
    user_segment_idx = 0

    for _seg_date, seg_role, seg_text in segments:
        if seg_role != role_filter:
            continue

        # Assign evidence_id: cycle through available IDs.
        eid = evidence_ids[user_segment_idx % len(evidence_ids)]
        user_segment_idx += 1

        # Normalize smart quotes for matching but preserve original for output.
        normalized_text = _normalize_apostrophes(seg_text)

        # Scan for stance markers in this segment.
        for match in _STANCE_PATTERN.finditer(normalized_text):
            matched_marker = match.group(0).lower()
            sentence = _find_sentence_containing(seg_text, match.start(), match.end())

            # Entity association: check if an entity is explicitly in the
            # extracted sentence.
            entities = entity_terms_from_text(sentence)
            entity = sorted(entities)[0] if entities else None

            positions.append(Position(
                text=sentence,
                date=phase.date_range,
                entity=entity,
                evidence_id=eid,
                stance_marker=matched_marker,
            ))

    return tuple(positions)


# ---------------------------------------------------------------------------
# Phase 13B — Temporal comparison
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ThinkingEvolution:
    """Temporal comparison of user-authored positions for one entity/topic."""

    entity: str                       # target entity or topic
    positions: tuple[Position, ...]   # chronologically ordered
    shifts: tuple[str, ...]           # shift signals between adjacent positions

    def to_dict(self) -> dict[str, object]:
        return {
            "entity": self.entity,
            "positions": [p.to_dict() for p in self.positions],
            "shifts": list(self.shifts),
        }


# Negation terms that signal a reversed stance.
_NEGATION_TERMS = frozenset({
    "don't", "dont", "not", "no longer", "never", "wrong", "changed my mind",
})

# Positive stance terms.
_POSITIVE_STANCE = frozenset({
    "trust", "love", "enjoy", "sure", "confident", "believe", "great",
    "good", "better", "understand", "appreciate", "comfortable", "happy",
})

# Negative stance terms.
_NEGATIVE_STANCE = frozenset({
    "distrust", "hate", "dislike", "doubt", "uncertain", "wrong",
    "bad", "worse", "frustrated", "angry", "uncomfortable", "worried",
    "draining", "exhausted", "disappointed",
})

# Explicit self-revision markers (subset of stance markers).
_SELF_REVISION_MARKERS = frozenset({
    "i changed my mind", "i was wrong about", "i no longer", "i used to think",
})


def _position_tokens(text: str) -> set[str]:
    """Lowercase tokens from position text for comparison."""
    normalized = _normalize_apostrophes(text.lower())
    return set(re.findall(r"[a-z']+", normalized))


def _has_negation(tokens: set[str]) -> bool:
    """Check whether any negation term appears in the token set."""
    for neg in _NEGATION_TERMS:
        neg_parts = neg.split()
        if len(neg_parts) == 1:
            if neg in tokens:
                return True
        else:
            # Multi-word: check if all parts present (rough heuristic).
            if all(p in tokens for p in neg_parts):
                return True
    return False


def _stance_valence(tokens: set[str]) -> int:
    """Return +1 for positive, -1 for negative, 0 for neutral."""
    pos = len(tokens & _POSITIVE_STANCE)
    neg = len(tokens & _NEGATIVE_STANCE)
    if pos > neg:
        return 1
    if neg > pos:
        return -1
    return 0


def _detect_shift(earlier: Position, later: Position) -> str | None:
    """Detect a possible shift between two adjacent positions.

    Returns a conservative description string, or None if no shift detected.
    """
    # Check for explicit self-revision marker in the later position.
    if later.stance_marker in _SELF_REVISION_MARKERS:
        return f"explicit self-revision detected: \"{later.stance_marker}\""

    earlier_tokens = _position_tokens(earlier.text)
    later_tokens = _position_tokens(later.text)

    # Negation change.
    earlier_neg = _has_negation(earlier_tokens)
    later_neg = _has_negation(later_tokens)
    if earlier_neg != later_neg:
        if later_neg:
            return "possible shift: negation introduced in later position"
        else:
            return "possible shift: negation removed in later position"

    # Sentiment-bearing term change.
    earlier_valence = _stance_valence(earlier_tokens)
    later_valence = _stance_valence(later_tokens)
    if earlier_valence != 0 and later_valence != 0 and earlier_valence != later_valence:
        direction = (
            "positive to negative" if earlier_valence > 0 else "negative to positive"
        )
        return f"possible shift from {direction} stance"

    return None


def _sort_key(pos: Position) -> tuple[int, str, str]:
    """Sort positions: dated first (ascending), undated last, then by text."""
    if pos.date is None:
        return (1, "", pos.text)
    # Use first 10 chars of date (YYYY-MM-DD) for sorting.
    return (0, pos.date[:10], pos.text)


def build_thinking_evolution(
    entity: str,
    positions: tuple[Position, ...],
) -> ThinkingEvolution:
    """Build a temporal comparison for one entity/topic.

    Parameters
    ----------
    entity : str
        The entity or topic being compared.
    positions : tuple[Position, ...]
        All extracted positions relevant to this entity.

    Returns
    -------
    ThinkingEvolution
        Chronologically ordered positions with detected shifts.
    """
    sorted_positions = tuple(sorted(positions, key=_sort_key))

    shifts: list[str] = []
    for i in range(len(sorted_positions) - 1):
        shift = _detect_shift(sorted_positions[i], sorted_positions[i + 1])
        if shift:
            shifts.append(
                f"Between \"{sorted_positions[i].date or 'undated'}\" and "
                f"\"{sorted_positions[i + 1].date or 'undated'}\": {shift}"
            )

    return ThinkingEvolution(
        entity=entity,
        positions=sorted_positions,
        shifts=tuple(shifts),
    )


def collect_positions_for_entity(
    all_positions: tuple[Position, ...],
    entity: str,
) -> tuple[Position, ...]:
    """Filter positions to those matching a target entity (case-insensitive).

    Falls back to all positions if no entity-specific matches exist.
    """
    entity_lower = entity.lower()
    entity_specific = tuple(
        p for p in all_positions
        if p.entity is not None and p.entity.lower() == entity_lower
    )
    if entity_specific:
        return entity_specific

    # Fallback: positions whose text mentions the entity as a whole word.
    pattern = re.compile(r"\b" + re.escape(entity) + r"\b", re.IGNORECASE)
    text_matches = tuple(
        p for p in all_positions
        if pattern.search(p.text)
    )
    return text_matches


# ---------------------------------------------------------------------------
# Phase 14A — Contradiction / change signal detection
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Contradiction:
    """A detected contradiction or change signal between two positions."""

    entity: str           # target entity or topic
    earlier: Position     # chronologically earlier position
    later: Position       # chronologically later position
    signal: str           # deterministic reason string
    date_range: str       # concise range, e.g. "2025-03-01 to 2025-06-01"
    change_type: str      # "reversal" | "softening" | "strengthening" | "evolution"

    def to_dict(self) -> dict[str, object]:
        return {
            "entity": self.entity,
            "earlier": self.earlier.to_dict(),
            "later": self.later.to_dict(),
            "signal": self.signal,
            "date_range": self.date_range,
            "change_type": self.change_type,
        }


# ---------------------------------------------------------------------------
# Phase 14B — Change type classification
# ---------------------------------------------------------------------------

# Marker strength ordering: higher = more certain/decisive.
_MARKER_STRENGTH: dict[str, int] = {
    "i feel like": 1,
    "i noticed": 1,
    "i think": 2,
    "i don't think": 2,
    "i believe": 3,
    "i don't believe": 3,
    "i understand": 3,
    "i learned": 3,
    "i figured out": 3,
    "i realized": 4,
    "i'm sure": 5,
    "i decided": 5,
    "i concluded": 5,
    # Self-revision markers — not used for strength comparison.
    "i was wrong about": 0,
    "i changed my mind": 0,
    "i no longer": 0,
    "i used to think": 0,
}


def _classify_change_type(earlier: Position, later: Position, signal: str) -> str:
    """Classify a detected change signal into a change type.

    Priority:
    1. reversal — explicit self-revision, strong negation change, or sentiment reversal
    2. softening — later marker weaker than earlier
    3. strengthening — later marker stronger than earlier
    4. evolution — fallback for shifts that don't fit the above
    """
    # Explicit self-revision markers are always reversals.
    if later.stance_marker in _SELF_REVISION_MARKERS:
        return "reversal"

    # Sentiment reversal (positive ↔ negative) is a reversal.
    if "positive to negative" in signal or "negative to positive" in signal:
        return "reversal"

    # Strong negation change is a reversal.
    if "negation introduced" in signal or "negation removed" in signal:
        earlier_strength = _MARKER_STRENGTH.get(earlier.stance_marker, 2)
        later_strength = _MARKER_STRENGTH.get(later.stance_marker, 2)
        # If strengths are comparable, negation flip = reversal.
        if abs(earlier_strength - later_strength) <= 1:
            return "reversal"
        # If later is weaker + negated, it's a softening of the opposite stance.
        if later_strength < earlier_strength:
            return "softening"
        # If later is stronger + negated, it's a strengthening of the opposite stance.
        if later_strength > earlier_strength:
            return "strengthening"

    # No negation/sentiment signal — check marker strength for softening/strengthening.
    earlier_strength = _MARKER_STRENGTH.get(earlier.stance_marker, 2)
    later_strength = _MARKER_STRENGTH.get(later.stance_marker, 2)
    if later_strength < earlier_strength:
        return "softening"
    if later_strength > earlier_strength:
        return "strengthening"

    return "evolution"


def _date_range_str(earlier: Position, later: Position) -> str:
    """Build a concise date range string from two positions."""
    e = earlier.date[:10] if earlier.date else "undated"
    l = later.date[:10] if later.date else "undated"
    if e == l:
        return e
    return f"{e} to {l}"


def detect_contradictions(
    entity: str,
    positions: list[Position] | tuple[Position, ...],
) -> list[Contradiction]:
    """Detect contradiction/change signals between adjacent chronological positions.

    Uses the same three heuristics as ``_detect_shift``:
    1. Explicit self-revision markers in the later position.
    2. Negation introduced or removed between positions.
    3. Sentiment-bearing term reversal (positive ↔ negative).

    Parameters
    ----------
    entity : str
        The entity or topic being compared.
    positions : list or tuple of Position
        Positions to compare (will be sorted chronologically).

    Returns
    -------
    list[Contradiction]
        Detected signals, ordered chronologically.  Empty if no signals found.
    """
    sorted_positions = sorted(positions, key=_sort_key)

    contradictions: list[Contradiction] = []
    for i in range(len(sorted_positions) - 1):
        earlier = sorted_positions[i]
        later = sorted_positions[i + 1]
        signal = _detect_shift(earlier, later)
        if signal:
            change_type = _classify_change_type(earlier, later, signal)
            contradictions.append(Contradiction(
                entity=entity,
                earlier=earlier,
                later=later,
                signal=signal,
                date_range=_date_range_str(earlier, later),
                change_type=change_type,
            ))

    return contradictions
