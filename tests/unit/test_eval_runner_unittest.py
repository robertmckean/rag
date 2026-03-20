import io
import json
import shutil
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from rag.cli.eval import main
from rag.eval.runner import eval_report_json, load_query_bank, render_eval_report, run_benchmark


# These tests cover the eval runner and CLI surface rather than answer classification details.
# The benchmark fixture stays small so results are stable and easy to inspect.
# The runner must exercise the real answer pipeline, not a separate mocked path.
class EvalRunnerTests(unittest.TestCase):
    # Point the eval harness at the dedicated answer fixture run and compact query bank.
    def setUp(self) -> None:
        self.run_dir = Path("tests/fixtures/answering/sample_run")
        self.bench_path = Path("tests/fixtures/eval/query_bank.json")
        self.tmp_root = Path("tests/_tmp_eval_cli")
        if self.tmp_root.exists():
            shutil.rmtree(self.tmp_root, ignore_errors=True)

    # Clean up any temp JSON output from CLI checks.
    def tearDown(self) -> None:
        if self.tmp_root.exists():
            shutil.rmtree(self.tmp_root, ignore_errors=True)

    # Verify that the benchmark file loads into explicit case models.
    def test_loads_query_bank(self) -> None:
        cases = load_query_bank(self.bench_path)

        self.assertGreaterEqual(len(cases), 8)
        self.assertEqual(cases[0].id, "supported_burnout_late_2025")

    # Verify that the runner aggregates pass/fail totals and status accuracy correctly.
    def test_runner_produces_aggregate_summary(self) -> None:
        summary, case_results = run_benchmark(self.run_dir, self.bench_path)

        self.assertEqual(summary.bench_cases, len(case_results))
        self.assertGreaterEqual(summary.passed, 1)
        self.assertGreaterEqual(summary.status_accuracy, 0.0)
        self.assertLessEqual(summary.status_accuracy, 1.0)

    # Verify that the JSON report stays structured and serialization-friendly.
    def test_runner_json_report_is_stable(self) -> None:
        summary, case_results = run_benchmark(self.run_dir, self.bench_path)
        payload = json.loads(eval_report_json(summary, case_results))

        self.assertIn("summary", payload)
        self.assertIn("results", payload)
        self.assertEqual(payload["summary"]["bench_cases"], len(payload["results"]))

    # Verify that fail-only rendering hides passing cases while keeping the failed ones visible.
    def test_fail_only_rendering(self) -> None:
        summary, case_results = run_benchmark(self.run_dir, self.bench_path)
        rendered = render_eval_report(summary, case_results, fail_only=True)

        self.assertIn("Eval Summary", rendered)
        self.assertNotIn("passed=True", rendered)

    # Verify that the eval CLI can write JSON output on the happy path.
    def test_eval_cli_json_output(self) -> None:
        json_out = self.tmp_root / "reports" / "eval.json"
        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()
        with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
            exit_code = main(
                [
                    "--run-dir",
                    str(self.run_dir),
                    "--bench",
                    str(self.bench_path),
                    "--json",
                    "--json-out",
                    str(json_out),
                ]
            )

        self.assertEqual(exit_code, 0)
        self.assertEqual(stderr_buffer.getvalue(), "")
        self.assertTrue(json_out.exists())
        payload = json.loads(json_out.read_text(encoding="utf-8"))
        self.assertIn("summary", payload)
        self.assertIn("results", payload)
        self.assertEqual(json.loads(stdout_buffer.getvalue()), payload)


if __name__ == "__main__":
    unittest.main()
