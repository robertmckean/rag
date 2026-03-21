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

    if report.entities:
        lines.append(f"Recurring entities: {len(report.entities)}")
        lines.append("")

        for i, entity in enumerate(report.entities, 1):
            lines.append(f"  [{i}] {entity.name} ({entity.occurrence_count} occurrences)")
            for occ in entity.occurrences:
                date_part = f" ({occ.created_at})" if occ.created_at else ""
                lines.append(f"      - {occ.evidence_id}{date_part}: {occ.excerpt}")
            lines.append("")
    else:
        lines.append("No recurring entities found.")
        lines.append("")

    if report.clusters:
        lines.append(f"Topic clusters: {len(report.clusters)}")
        lines.append("")

        for i, cluster in enumerate(report.clusters, 1):
            date_part = f" ({cluster.date_range})" if cluster.date_range else ""
            lines.append(f"  [{i}] {cluster.label}{date_part} -- {cluster.phase_count} phases")
            if cluster.key_entities:
                lines.append(f"      Entities: {', '.join(cluster.key_entities)}")
            if cluster.key_terms:
                lines.append(f"      Terms: {', '.join(cluster.key_terms)}")
            lines.append(f"      Phases: {', '.join(cluster.phase_labels)}")
            lines.append(f"      Evidence: {', '.join(cluster.evidence_ids)}")
            lines.append("")

    lines.append(f"Evidence count: {report.evidence_count}")
    return "\n".join(lines)
