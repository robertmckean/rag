from pathlib import Path

from src.rag.inspection.inventory import inspect_raw_inputs


# Inventory tests focus on readiness states rather than provider-specific parsing.
# Hidden placeholder files should not make a provider look ready.
# Missing and populated directories must remain easy to distinguish in CLI output.

# Verify that provider roots are classified correctly as empty or ready.
def test_inventory_distinguishes_missing_empty_and_ready(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    (raw_root / "chatgpt").mkdir(parents=True)
    (raw_root / "chatgpt" / ".gitkeep").write_text("", encoding="utf-8")
    (raw_root / "claude").mkdir(parents=True)
    (raw_root / "claude" / "History_Claude").mkdir()
    (raw_root / "claude" / "History_Claude" / "conversations.json").write_text(
        "[]", encoding="utf-8"
    )

    inventories = {item.provider: item for item in inspect_raw_inputs(raw_root)}

    assert inventories["chatgpt"].status == "empty"
    assert inventories["claude"].status == "ready"
    assert inventories["claude"].relative_files == ("History_Claude/conversations.json",)


# Verify that absent provider directories report as missing.
def test_inventory_marks_missing_provider_folder(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    raw_root.mkdir()

    inventories = {item.provider: item for item in inspect_raw_inputs(raw_root)}

    assert inventories["chatgpt"].status == "missing"
    assert inventories["claude"].status == "missing"
