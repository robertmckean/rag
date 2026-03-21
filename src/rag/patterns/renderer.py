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

    if report.entity_cluster_links:
        lines.append(f"Cross-cluster entities: {len(report.entity_cluster_links)}")
        lines.append("")

        for link in report.entity_cluster_links:
            lines.append(f"  {link.entity} ({link.cluster_count} clusters, {link.total_phase_count} phases)")
            for cl in link.cluster_labels:
                lines.append(f"      - {cl}")
            lines.append("")

    if report.temporal_bursts:
        lines.append(f"Temporal bursts: {len(report.temporal_bursts)}")
        lines.append("")

        for burst in report.temporal_bursts:
            lines.append(f"  {burst.date_range} -- {burst.burst_size} phases")
            if burst.entities:
                lines.append(f"      Entities: {', '.join(burst.entities)}")
            for pl in burst.phase_labels:
                lines.append(f"      - {pl}")
            lines.append("")

    # Summary emphasis: top entities and clusters.
    if report.entities or report.clusters:
        lines.append("---")
        lines.append("Summary")
        lines.append("")
        if report.entities:
            top = report.entities[:3]
            names = ", ".join(f"{e.name} ({e.occurrence_count})" for e in top)
            lines.append(f"  Top entities: {names}")
        if report.clusters:
            top = report.clusters[:3]
            labels = ", ".join(f"{c.label} ({c.phase_count} phases)" for c in top)
            lines.append(f"  Top themes: {labels}")
        if report.entity_cluster_links:
            bridging = ", ".join(l.entity for l in report.entity_cluster_links[:3])
            lines.append(f"  Cross-topic: {bridging}")
        if report.temporal_bursts:
            burst_summary = ", ".join(f"{b.date_range} ({b.burst_size} phases)" for b in report.temporal_bursts[:3])
            lines.append(f"  Activity bursts: {burst_summary}")
        lines.append("")

    lines.append(f"Evidence count: {report.evidence_count}")
    return "\n".join(lines)
