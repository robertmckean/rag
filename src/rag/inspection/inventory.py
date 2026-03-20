"""Inspection-time readiness checks for raw provider input folders."""

from dataclasses import dataclass
from pathlib import Path


PROVIDERS = ("chatgpt", "claude")


@dataclass(frozen=True)
class ProviderInventory:
    """Discovery summary for a single provider input folder."""

    provider: str
    root: Path
    exists: bool
    file_count: int
    relative_files: tuple[str, ...]

    @property
    def status(self) -> str:
        if not self.exists:
            return "missing"
        if self.file_count == 0:
            return "empty"
        return "ready"

    @property
    def top_level_entries(self) -> tuple[str, ...]:
        entries = {path.split("\\", 1)[0].split("/", 1)[0] for path in self.relative_files}
        return tuple(sorted(entries))


def iter_visible_files(root: Path) -> tuple[Path, ...]:
    """Return non-hidden files rooted under the provided directory."""
    if not root.exists():
        return ()

    files: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if any(part.startswith(".") for part in path.relative_to(root).parts):
            continue
        files.append(path)
    return tuple(sorted(files))


def inspect_provider_root(root: Path, provider: str) -> ProviderInventory:
    """Inspect one provider folder for non-hidden export artifacts."""
    provider_root = root / provider
    visible_files = iter_visible_files(provider_root)
    return ProviderInventory(
        provider=provider,
        root=provider_root,
        exists=provider_root.exists(),
        file_count=len(visible_files),
        relative_files=tuple(str(path.relative_to(provider_root)) for path in visible_files),
    )


def inspect_raw_inputs(raw_root: Path) -> tuple[ProviderInventory, ...]:
    """Inspect all supported provider folders beneath the raw input root."""
    return tuple(inspect_provider_root(raw_root, provider) for provider in PROVIDERS)
