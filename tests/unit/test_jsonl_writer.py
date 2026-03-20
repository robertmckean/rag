import json
from pathlib import Path

from src.rag.storage.jsonl_writer import write_jsonl


# The JSONL writer is the last step before canonical records hit disk.
# Stable key ordering is important because deterministic output underpins run comparisons.
# This test keeps the behavior minimal and focused on exact line output.

# Verify that the writer emits one sorted JSON object per line.
def test_write_jsonl_writes_one_record_per_line(tmp_path: Path) -> None:
    output = tmp_path / "sample.jsonl"
    records = [{"b": 2, "a": 1}, {"a": 3}]

    write_jsonl(output, records)

    lines = output.read_text(encoding="utf-8").splitlines()
    assert lines == [
        json.dumps({"a": 1, "b": 2}, ensure_ascii=True, sort_keys=True),
        json.dumps({"a": 3}, ensure_ascii=True, sort_keys=True),
    ]
