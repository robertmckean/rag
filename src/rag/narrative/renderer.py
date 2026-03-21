"""Output formatters for narrative reconstructions."""

from __future__ import annotations

import json

from rag.narrative.models import NarrativeReconstruction


def render_json(narrative: NarrativeReconstruction) -> str:
    return json.dumps(narrative.to_dict(), indent=2, ensure_ascii=False)


def render_text(narrative: NarrativeReconstruction) -> str:
    lines: list[str] = []

    lines.append(f"Narrative: {narrative.query}")
    lines.append("")
    lines.append("Summary:")
    lines.append(f"  {narrative.summary}")
    lines.append("")

    if narrative.timeline:
        lines.append("Timeline:")
        for i, phase in enumerate(narrative.timeline, 1):
            date_part = f" ({phase.date_range})" if phase.date_range else ""
            lines.append(f"  [{i}] {phase.label}{date_part} -- {phase.support}")
            # Wrap description lines.
            for desc_part in phase.description.split(" | "):
                lines.append(f"      {desc_part.strip()}")
            lines.append(f"      Evidence: {', '.join(phase.evidence_ids)}")
            lines.append("")

    if narrative.transitions:
        lines.append("Transitions:")
        for trans in narrative.transitions:
            lines.append(f"  {trans.from_phase}  -->  {trans.to_phase}")
            lines.append(f"    {trans.description}")
            lines.append(f"    Evidence: {', '.join(trans.evidence_ids)}")
            lines.append(f"    ({trans.support})")
            lines.append("")

    if narrative.gaps:
        lines.append("Gaps:")
        for gap in narrative.gaps:
            lines.append(f"  - {gap.description}")
            lines.append(f"    Reason: {gap.reason}")
        lines.append("")

    if narrative.limitations:
        lines.append("Limitations:")
        for lim in narrative.limitations:
            lines.append(f"  - [{lim.kind}] {lim.description}")
        lines.append("")

    lines.append(f"Evidence count: {narrative.evidence_count}")
    return "\n".join(lines)


def render_debug(narrative: NarrativeReconstruction) -> str:
    """Text format with additional structural detail for inspection."""
    text = render_text(narrative)
    lines = [text, "", "=== DEBUG ===", ""]

    # Phase detail.
    for i, phase in enumerate(narrative.timeline, 1):
        lines.append(f"Phase {i}: {phase.label}")
        lines.append(f"  evidence_ids: {phase.evidence_ids}")
        lines.append(f"  date_range: {phase.date_range}")
        lines.append(f"  support: {phase.support}")
        lines.append(f"  description_length: {len(phase.description)}")
        lines.append("")

    # Transition detail.
    for i, trans in enumerate(narrative.transitions, 1):
        lines.append(f"Transition {i}: {trans.from_phase} -> {trans.to_phase}")
        lines.append(f"  evidence_ids: {trans.evidence_ids}")
        lines.append(f"  description: {trans.description}")
        lines.append("")

    return "\n".join(lines)
