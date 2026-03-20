"""Read-only message quality analysis for normalized message streams."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


def analyze_message_quality(messages_path: Path) -> dict[str, object]:
    """Analyze normalized messages.jsonl and return a deterministic summary."""
    messages = _load_messages(messages_path)
    total_messages = len(messages)

    messages_by_provider = Counter()
    messages_by_author_role = Counter()
    messages_with_attachments = 0
    messages_with_non_text_content_blocks = 0

    empty_messages: list[dict[str, object]] = []
    null_text_nonempty_blocks: list[dict[str, object]] = []
    system_metadata_only: list[dict[str, object]] = []
    content_blocks_empty: list[dict[str, object]] = []
    malformed_block_examples: list[dict[str, object]] = []
    blocks_missing_type = 0
    blocks_missing_text = 0

    empty_by_provider = Counter()
    empty_by_author_role = Counter()

    for message in messages:
        provider = _string_or_default(message.get("provider"), "unknown")
        author_role = _string_or_default(message.get("author_role"), "unknown")
        sender = _string_or_default(message.get("sender"), "unknown")
        text = _normalize_text(message.get("text"))
        attachments = message.get("attachments")
        content_blocks = message.get("content_blocks")

        if not isinstance(attachments, list):
            attachments = []
        if not isinstance(content_blocks, list):
            content_blocks = []

        messages_by_provider[provider] += 1
        messages_by_author_role[author_role] += 1

        if attachments:
            messages_with_attachments += 1

        if any(_string_or_default(block.get("type"), "") != "text" for block in content_blocks if isinstance(block, dict)):
            messages_with_non_text_content_blocks += 1

        if not content_blocks:
            content_blocks_empty.append(_sample_record(message))

        block_texts = []
        meaningful_block_texts = []
        for block in content_blocks:
            if not isinstance(block, dict):
                continue
            block_type = block.get("type")
            block_text = _normalize_text(block.get("text"))
            if block_type in (None, ""):
                blocks_missing_type += 1
                if len(malformed_block_examples) < 3:
                    malformed_block_examples.append(
                        {
                            "message_id": message.get("message_id"),
                            "issue": "missing_type",
                            "block": block,
                        }
                    )
            if block.get("text", None) is None or (isinstance(block.get("text"), str) and not block.get("text").strip()):
                blocks_missing_text += 1
                if len(malformed_block_examples) < 3:
                    malformed_block_examples.append(
                        {
                            "message_id": message.get("message_id"),
                            "issue": "missing_or_empty_text",
                            "block": block,
                        }
                    )
            block_texts.append(block_text)
            if block_text:
                meaningful_block_texts.append(block_text)

        has_meaningful_block_text = bool(meaningful_block_texts)
        no_attachment = not attachments
        text_missing = text is None

        if text_missing and (not content_blocks or not has_meaningful_block_text) and no_attachment:
            empty_messages.append(message)
            empty_by_provider[provider] += 1
            empty_by_author_role[author_role] += 1

        if text_missing and has_meaningful_block_text:
            null_text_nonempty_blocks.append(message)

        if _is_system_or_metadata_sender(author_role, sender) and not _has_meaningful_content(text, meaningful_block_texts):
            system_metadata_only.append(message)

    report = {
        "messages_path": str(messages_path),
        "total_messages": total_messages,
        "distribution_summary": {
            "total_messages": total_messages,
            "messages_by_provider": dict(sorted(messages_by_provider.items())),
            "messages_by_author_role": dict(sorted(messages_by_author_role.items())),
            "messages_with_attachments": messages_with_attachments,
            "messages_with_non_text_content_blocks": messages_with_non_text_content_blocks,
        },
        "empty_messages": {
            "count": len(empty_messages),
            "percentage": _percentage(len(empty_messages), total_messages),
            "by_provider": dict(sorted(empty_by_provider.items())),
            "by_author_role": dict(sorted(empty_by_author_role.items())),
            "samples": _sample_records(empty_messages, limit=5),
        },
        "null_text_but_nonempty_content_blocks": {
            "count": len(null_text_nonempty_blocks),
            "percentage": _percentage(len(null_text_nonempty_blocks), total_messages),
            "samples": _sample_records(null_text_nonempty_blocks, limit=3),
        },
        "system_metadata_only_messages": {
            "count": len(system_metadata_only),
            "percentage": _percentage(len(system_metadata_only), total_messages),
            "recommendation": (
                "candidate_for_filtering"
                if system_metadata_only
                else "keep"
            ),
            "samples": _sample_records(system_metadata_only, limit=5),
        },
        "content_blocks_integrity": {
            "messages_with_empty_content_blocks": len(content_blocks_empty),
            "blocks_missing_type": blocks_missing_type,
            "blocks_missing_text": blocks_missing_text,
            "messages_with_non_text_content_blocks": messages_with_non_text_content_blocks,
            "malformed_examples": malformed_block_examples,
        },
    }
    return report


def write_message_quality_report(report: dict[str, object], output_path: Path) -> None:
    """Write the JSON analysis report."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def render_message_quality_report(report: dict[str, object]) -> str:
    """Render a human-readable report for CLI output."""
    lines: list[str] = []
    summary = report["distribution_summary"]
    empty = report["empty_messages"]
    null_text = report["null_text_but_nonempty_content_blocks"]
    system_only = report["system_metadata_only_messages"]
    integrity = report["content_blocks_integrity"]

    lines.append("Message Quality Report")
    lines.append(f"messages_path: {report['messages_path']}")
    lines.append("")
    lines.append("Distribution Summary")
    lines.append(f"  total_messages: {summary['total_messages']}")
    lines.append("  messages_by_provider:")
    for key, value in summary["messages_by_provider"].items():
        lines.append(f"    - {key}: {value}")
    lines.append("  messages_by_author_role:")
    for key, value in summary["messages_by_author_role"].items():
        lines.append(f"    - {key}: {value}")
    lines.append(f"  messages_with_attachments: {summary['messages_with_attachments']}")
    lines.append(f"  messages_with_non_text_content_blocks: {summary['messages_with_non_text_content_blocks']}")
    lines.append("")
    lines.append("Empty Messages")
    lines.append(f"  count: {empty['count']}")
    lines.append(f"  percentage: {empty['percentage']:.2f}")
    lines.append("  by_provider:")
    for key, value in empty["by_provider"].items():
        lines.append(f"    - {key}: {value}")
    lines.append("  by_author_role:")
    for key, value in empty["by_author_role"].items():
        lines.append(f"    - {key}: {value}")
    lines.extend(_render_samples(empty["samples"]))
    lines.append("")
    lines.append("Null Text But Non-Empty Content Blocks")
    lines.append(f"  count: {null_text['count']}")
    lines.append(f"  percentage: {null_text['percentage']:.2f}")
    lines.extend(_render_samples(null_text["samples"]))
    lines.append("")
    lines.append("System Or Metadata-Only Messages")
    lines.append(f"  count: {system_only['count']}")
    lines.append(f"  percentage: {system_only['percentage']:.2f}")
    lines.append(f"  recommendation: {system_only['recommendation']}")
    lines.extend(_render_samples(system_only["samples"]))
    lines.append("")
    lines.append("Content Blocks Integrity")
    lines.append(f"  messages_with_empty_content_blocks: {integrity['messages_with_empty_content_blocks']}")
    lines.append(f"  blocks_missing_type: {integrity['blocks_missing_type']}")
    lines.append(f"  blocks_missing_text: {integrity['blocks_missing_text']}")
    lines.append(f"  messages_with_non_text_content_blocks: {integrity['messages_with_non_text_content_blocks']}")
    lines.append("  malformed_examples:")
    for example in integrity["malformed_examples"]:
        lines.append(
            f"    - message_id: {example.get('message_id')} issue: {example.get('issue')}"
        )
    return "\n".join(lines)


def _load_messages(messages_path: Path) -> list[dict[str, object]]:
    messages: list[dict[str, object]] = []
    with messages_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if isinstance(payload, dict):
                messages.append(payload)
    return messages


def _normalize_text(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def _is_system_or_metadata_sender(author_role: str, sender: str) -> bool:
    if author_role == "system":
        return True
    if sender not in {"unknown", "human", "assistant", "user", ""}:
        return True
    return False


def _has_meaningful_content(text: str | None, block_texts: list[str]) -> bool:
    return bool(text or block_texts)


def _sample_records(messages: list[dict[str, object]], limit: int = 3) -> list[dict[str, str | None]]:
    ordered = sorted(
        (_sample_record(message) for message in messages),
        key=lambda item: (
            _string_or_default(item.get("provider"), "unknown"),
            _string_or_default(item.get("message_id"), ""),
        ),
    )
    return ordered[:limit]


def _sample_record(message: dict[str, object]) -> dict[str, str | None]:
    preview = _normalize_text(message.get("text"))
    if preview is None:
        content_blocks = message.get("content_blocks")
        if isinstance(content_blocks, list):
            for block in content_blocks:
                if isinstance(block, dict):
                    preview = _normalize_text(block.get("text"))
                    if preview:
                        break
    if preview and len(preview) > 120:
        preview = preview[:117] + "..."
    return {
        "message_id": _string_or_default(message.get("message_id"), ""),
        "provider": _string_or_default(message.get("provider"), "unknown"),
        "snippet": preview,
    }


def _string_or_default(value: object, default: str) -> str:
    if isinstance(value, str):
        return value
    return default


def _percentage(count: int, total: int) -> float:
    if total == 0:
        return 0.0
    return (count / total) * 100.0


def _render_samples(samples: list[dict[str, str | None]]) -> list[str]:
    lines = ["  samples:"]
    if not samples:
        lines.append("    - none")
        return lines
    for sample in samples:
        lines.append(
            f"    - {sample.get('message_id')} | provider={sample.get('provider')} | snippet={sample.get('snippet')}"
        )
    return lines
