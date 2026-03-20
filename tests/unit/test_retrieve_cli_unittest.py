import io
import json
import shutil
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from rag.cli.retrieve import main


# These tests cover the retrieval CLI boundary rather than the lexical scoring internals.
# The failure paths stay small so error handling remains readable and deterministic.
# CLI behavior is validated through stdout/stderr because that is the user-facing contract here.
class RetrieveCliTests(unittest.TestCase):
    # Create a small isolated run directory for the CLI failure-path checks.
    def setUp(self) -> None:
        self.tmp_root = Path("tests/_tmp_retrieve_cli")
        if self.tmp_root.exists():
            shutil.rmtree(self.tmp_root, ignore_errors=True)
        self.tmp_root.mkdir(parents=True)

    # Remove the temporary run directory after each CLI test.
    def tearDown(self) -> None:
        if self.tmp_root.exists():
            shutil.rmtree(self.tmp_root, ignore_errors=True)

    # Capture the retrieval CLI output streams and exit code for one invocation.
    def _run_cli(self, *args: str) -> tuple[int, str, str]:
        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()
        with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
            exit_code = main(list(args))
        return exit_code, stdout_buffer.getvalue(), stderr_buffer.getvalue()

    # Write one minimal valid normalized run directory for CLI checks that should succeed.
    def _write_minimal_run(self, run_dir: Path) -> None:
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "manifest.json").write_text(
            json.dumps({"run_id": "cli-fixture"}, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
        (run_dir / "conversations.jsonl").write_text(
            json.dumps(
                {
                    "conversation_id": "chatgpt:conversation:test",
                    "provider": "chatgpt",
                    "source_conversation_id": "test",
                    "title": "CLI Fixture",
                    "summary": None,
                    "created_at": "2025-03-01T00:00:00Z",
                    "updated_at": "2025-03-01T00:05:00Z",
                    "message_count": 1,
                    "participant_summary": {"roles": ["user"], "authors": []},
                    "source_artifact": {"root": "History_ChatGPT", "conversation_file": "conversations-000.json", "sidecar_files": []},
                    "source_metadata": {},
                },
                ensure_ascii=True,
            )
            + "\n",
            encoding="utf-8",
        )
        (run_dir / "messages.jsonl").write_text(
            json.dumps(
                {
                    "message_id": "chatgpt:message:test",
                    "conversation_id": "chatgpt:conversation:test",
                    "provider": "chatgpt",
                    "source_message_id": "test",
                    "parent_message_id": None,
                    "sequence_index": 0,
                    "author_role": "user",
                    "author_name": None,
                    "sender": None,
                    "created_at": "2025-03-01T00:00:00Z",
                    "updated_at": None,
                    "text": "resume planning note",
                    "content_blocks": [{"type": "text", "text": "resume planning note", "start_timestamp": None, "stop_timestamp": None, "citations": None}],
                    "attachments": [],
                    "source_artifact": {"conversation_file": "conversations-000.json", "raw_message_path": "mapping.test.message"},
                    "source_metadata": {},
                },
                ensure_ascii=True,
            )
            + "\n",
            encoding="utf-8",
        )

    # Verify that the CLI reports a missing run directory cleanly.
    def test_reports_missing_run_directory(self) -> None:
        exit_code, stdout_value, stderr_value = self._run_cli(
            "--run-dir",
            str(self.tmp_root / "missing-run"),
            "--query",
            "resume",
        )

        self.assertEqual(exit_code, 2)
        self.assertEqual(stdout_value, "")
        self.assertIn("Run directory not found", stderr_value)

    # Verify that the CLI reports a missing required run file cleanly.
    def test_reports_missing_required_run_file(self) -> None:
        run_dir = self.tmp_root / "missing-messages"
        run_dir.mkdir(parents=True)
        (run_dir / "manifest.json").write_text('{"run_id":"bad"}\n', encoding="utf-8")
        (run_dir / "conversations.jsonl").write_text("{}\n", encoding="utf-8")

        exit_code, stdout_value, stderr_value = self._run_cli(
            "--run-dir",
            str(run_dir),
            "--query",
            "resume",
        )

        self.assertEqual(exit_code, 2)
        self.assertEqual(stdout_value, "")
        self.assertIn("Required run file not found", stderr_value)

    # Verify that malformed JSONL is surfaced as a readable CLI error.
    def test_reports_malformed_jsonl(self) -> None:
        run_dir = self.tmp_root / "bad-jsonl"
        self._write_minimal_run(run_dir)
        (run_dir / "messages.jsonl").write_text("{bad json}\n", encoding="utf-8")

        exit_code, stdout_value, stderr_value = self._run_cli(
            "--run-dir",
            str(run_dir),
            "--query",
            "resume",
        )

        self.assertEqual(exit_code, 2)
        self.assertEqual(stdout_value, "")
        self.assertIn("Malformed JSONL", stderr_value)

    # Verify that an empty result set is rendered explicitly rather than silently looking broken.
    def test_reports_empty_result_set(self) -> None:
        run_dir = self.tmp_root / "empty-results"
        self._write_minimal_run(run_dir)

        exit_code, stdout_value, stderr_value = self._run_cli(
            "--run-dir",
            str(run_dir),
            "--query",
            "burnout",
        )

        self.assertEqual(exit_code, 0)
        self.assertEqual(stderr_value, "")
        self.assertIn("result_count: 0", stdout_value)
        self.assertIn("no matching results", stdout_value)

    # Verify that the CLI writes structured JSON output for a successful retrieval run.
    def test_writes_json_output_on_success(self) -> None:
        run_dir = self.tmp_root / "json-out"
        json_out = self.tmp_root / "artifacts" / "results.json"
        self._write_minimal_run(run_dir)

        exit_code, stdout_value, stderr_value = self._run_cli(
            "--run-dir",
            str(run_dir),
            "--query",
            "resume",
            "--json-out",
            str(json_out),
        )

        self.assertEqual(exit_code, 0)
        self.assertEqual(stderr_value, "")
        self.assertTrue(json_out.exists())
        payload = json.loads(json_out.read_text(encoding="utf-8"))
        self.assertEqual(payload["result_count"], 1)
        self.assertEqual(payload["results"][0]["focal_message_id"], "chatgpt:message:test")
        self.assertIn("json_out:", stdout_value)


if __name__ == "__main__":
    unittest.main()
