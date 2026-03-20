import io
import json
import shutil
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from rag.cli.answer import main


# These tests cover the grounded-answer CLI contract rather than answer classification internals.
# The checks stay narrow: CLI argument behavior, JSON output, and readable failure handling.
# The fixture run is reused so the answer CLI stays aligned with the answer module behavior.
class AnswerCliTests(unittest.TestCase):
    # Prepare a temp area for answer CLI JSON-output checks.
    def setUp(self) -> None:
        self.run_dir = Path("tests/fixtures/answering/sample_run")
        self.tmp_root = Path("tests/_tmp_answer_cli")
        if self.tmp_root.exists():
            shutil.rmtree(self.tmp_root, ignore_errors=True)

    # Remove any temp artifacts created by the answer CLI tests.
    def tearDown(self) -> None:
        if self.tmp_root.exists():
            shutil.rmtree(self.tmp_root, ignore_errors=True)

    # Capture stdout, stderr, and exit code for one answer CLI invocation.
    def _run_cli(self, *args: str) -> tuple[int, str, str]:
        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()
        with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
            try:
                exit_code = main(list(args))
            except SystemExit as exc:
                exit_code = int(exc.code)
        return exit_code, stdout_buffer.getvalue(), stderr_buffer.getvalue()

    # Verify that the answer CLI rejects timeline mode cleanly at the parser boundary.
    def test_answer_cli_rejects_timeline_mode(self) -> None:
        exit_code, stdout_value, stderr_value = self._run_cli(
            "--run-dir",
            str(self.run_dir),
            "--query",
            "burnout",
            "--retrieval-mode",
            "timeline",
        )

        self.assertNotEqual(exit_code, 0)
        self.assertEqual(stdout_value, "")
        self.assertIn("invalid choice", stderr_value)

    # Verify that the answer CLI can print structured JSON and write it to disk.
    def test_answer_cli_json_output_is_stable(self) -> None:
        json_out = self.tmp_root / "results" / "answer.json"
        exit_code, stdout_value, stderr_value = self._run_cli(
            "--run-dir",
            str(self.run_dir),
            "--query",
            "What have I said about burnout?",
            "--json",
            "--json-out",
            str(json_out),
        )

        self.assertEqual(exit_code, 0)
        self.assertEqual(stderr_value, "")
        self.assertTrue(json_out.exists())
        printed_payload = json.loads(stdout_value)
        file_payload = json.loads(json_out.read_text(encoding="utf-8"))
        self.assertEqual(printed_payload["answer_status"], "supported")
        self.assertEqual(file_payload, printed_payload)

    # Verify that normal terminal rendering remains readable on the happy path.
    def test_answer_cli_renders_terminal_summary(self) -> None:
        exit_code, stdout_value, stderr_value = self._run_cli(
            "--run-dir",
            str(self.run_dir),
            "--query",
            "What have I said about workload?",
        )

        self.assertEqual(exit_code, 0)
        self.assertEqual(stderr_value, "")
        self.assertIn("Grounded Answer", stdout_value)
        self.assertIn("grounding_mode: strict", stdout_value)
        self.assertIn("answer_status: ambiguous", stdout_value)
        self.assertIn("citations:", stdout_value)

    # Verify that the CLI wires --llm, --llm-model, and --grounding-mode through to the answer pipeline.
    def test_answer_cli_wires_llm_and_grounding_flags(self) -> None:
        with patch("rag.cli.answer.answer_query") as mocked_answer_query:
            mocked_answer_query.return_value = type(
                "StubAnswerResult",
                (),
                {
                    "to_dict": lambda self: {
                        "request": {"grounding_mode": "conversational_memory"},
                        "query": "burnout",
                        "answer_status": "supported",
                        "answer": "stub",
                        "evidence_used": [],
                        "gaps": [],
                        "conflicts": [],
                        "citations": [],
                        "retrieval_summary": {
                            "run_id": "stub-run",
                            "retrieval_mode": "relevance",
                            "retrieval_limit": 8,
                            "retrieved_result_count": 1,
                            "evidence_used_count": 1,
                        },
                        "diagnostics": {
                            "focus_terms": [],
                            "selected_evidence_count": 0,
                            "qualified_evidence_count": 0,
                            "support_basis": None,
                            "composition_used": False,
                            "supporting_excerpt_count": 0,
                            "user_excerpt_count": 0,
                            "assistant_excerpt_count": 0,
                            "coverage_terms": [],
                            "coverage_ratio": 0.0,
                            "window_id": None,
                            "status_reason": None,
                            "rejected_evidence": [],
                        },
                    }
                },
            )()
            exit_code, stdout_value, stderr_value = self._run_cli(
                "--run-dir",
                str(self.run_dir),
                "--query",
                "burnout",
                "--grounding-mode",
                "conversational_memory",
                "--llm",
                "--llm-model",
                "gpt-5-mini",
                "--json",
            )

        self.assertEqual(exit_code, 0)
        self.assertEqual(stderr_value, "")
        self.assertTrue(stdout_value)
        self.assertTrue(mocked_answer_query.call_args.kwargs["llm"])
        self.assertEqual(mocked_answer_query.call_args.kwargs["llm_model"], "gpt-5-mini")
        self.assertEqual(mocked_answer_query.call_args.kwargs["grounding_mode"], "conversational_memory")


if __name__ == "__main__":
    unittest.main()
