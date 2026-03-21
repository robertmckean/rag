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
