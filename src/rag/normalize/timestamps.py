"""Timestamp normalization helpers."""

from __future__ import annotations

from datetime import datetime, timezone


def normalize_timestamp(value: object) -> str | None:
    """Normalize source timestamps to UTC ISO-8601 with trailing Z."""
    if value is None:
        return None

    if isinstance(value, bool):
        return None

    if isinstance(value, (int, float)):
        return _format_datetime(datetime.fromtimestamp(value, tz=timezone.utc))

    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None

        try:
            numeric = float(text)
        except ValueError:
            numeric = None

        if numeric is not None:
            return _format_datetime(datetime.fromtimestamp(numeric, tz=timezone.utc))

        iso_text = text.replace("Z", "+00:00")
        parsed = datetime.fromisoformat(iso_text)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        else:
            parsed = parsed.astimezone(timezone.utc)
        return _format_datetime(parsed)

    return None


def _format_datetime(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
