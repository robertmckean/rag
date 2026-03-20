from pathlib import Path

from src.rag.cli.inspect_exports import main


# This integration test covers the CLI wrapper rather than the raw inventory helper alone.
# The goal is to confirm that readiness information is actually printed for users.
# A small synthetic tree is enough because the CLI only formats existing inventory data.

# Verify that the inspection CLI reports provider readiness and discovered bundle names.
def test_inspect_exports_cli_reports_provider_statuses(tmp_path: Path, capsys) -> None:
    raw_root = tmp_path / "raw"
    (raw_root / "chatgpt" / "History_ChatGPT").mkdir(parents=True)
    (raw_root / "chatgpt" / "History_ChatGPT" / "conversations-000.json").write_text(
        "[]", encoding="utf-8"
    )
    (raw_root / "claude").mkdir(parents=True)

    exit_code = main(["--input", str(raw_root)])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "chatgpt:" in captured.out
    assert "status: ready" in captured.out
    assert "status: empty" in captured.out
    assert "History_ChatGPT" in captured.out
