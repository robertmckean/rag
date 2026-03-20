"""CLI for raw export readiness inspection."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.rag.config import DEFAULT_INPUT_PATH
from src.rag.inspection.inventory import ProviderInventory, inspect_raw_inputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect raw provider folders and report export-readiness status."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="Raw input root containing provider subfolders.",
    )
    return parser


def render_provider_report(inventory: ProviderInventory, raw_root: Path) -> list[str]:
    relative_root = inventory.root.relative_to(raw_root.parent)
    lines = [
        f"{inventory.provider}:",
        f"  folder: {relative_root}",
        f"  status: {inventory.status}",
        f"  non_hidden_files: {inventory.file_count}",
    ]
    if inventory.exists and inventory.top_level_entries:
        lines.append("  top_level_entries:")
        lines.extend(f"    - {entry}" for entry in inventory.top_level_entries)
    if inventory.exists and inventory.file_count:
        lines.append("  discovered_files:")
        lines.extend(f"    - {path}" for path in inventory.relative_files)
    return lines


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    raw_root = args.input.resolve()
    inventories = inspect_raw_inputs(raw_root)

    print(f"raw_input_root: {raw_root}")
    for inventory in inventories:
        for line in render_provider_report(inventory, raw_root):
            print(line)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
