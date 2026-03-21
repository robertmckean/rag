"""Output formatters for pattern extraction reports."""

from __future__ import annotations

import json

from rag.patterns.models import PatternReport


def render_json(report: PatternReport) -> str:
    return json.dumps(report.to_dict(), indent=2, ensure_ascii=False)


def render_text(report: PatternReport) -> str:
    lines: list[str] = []

    lines.append(f"Pattern Report: {report.query}")
    lines.append("")

    if not report.entities:
        lines.append("No recurring entities found.")
        lines.append("")
        lines.append(f"Evidence count: {report.evidence_count}")
        return "\n".join(lines)

    lines.append(f"Recurring entities: {len(report.entities)}")
    lines.append("")

    for i, entity in enumerate(report.entities, 1):
        lines.append(f"  [{i}] {entity.name} ({entity.occurrence_count} occurrences)")
        for occ in entity.occurrences:
            date_part = f" ({occ.created_at})" if occ.created_at else ""
            lines.append(f"      - {occ.evidence_id}{date_part}: {occ.excerpt}")
        lines.append("")

    lines.append(f"Evidence count: {report.evidence_count}")
    return "\n".join(lines)
