"""Shared CLI output helpers used by all rag CLI commands."""

from __future__ import annotations

import sys


# Print terminal output with replacement fallback for consoles that cannot encode some message text.
def safe_print(value: str) -> None:
    encoding = sys.stdout.encoding or "utf-8"
    safe_value = value.encode(encoding, errors="replace").decode(encoding, errors="replace")
    print(safe_value)


# Print CLI errors to stderr with the same encoding fallback used for normal output.
def safe_print_error(value: str) -> None:
    encoding = sys.stderr.encoding or "utf-8"
    safe_value = value.encode(encoding, errors="replace").decode(encoding, errors="replace")
    print(safe_value, file=sys.stderr)
