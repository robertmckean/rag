"""Unit tests for the pattern extraction CLI (Phase 7)."""

from __future__ import annotations

import io
import json
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from rag.cli.patterns import main


SAMPLE_RUN = Path("tests/fixtures/answering/sample_run")


class PatternCliTests(unittest.TestCase):
    """Tests for the pattern extraction CLI entry point."""

    def _run_cli(self, *args: str) -> tuple[int, str, str]:
        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()
        with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
            try:
                exit_code = main(list(args))
            except SystemExit as exc:
                exit_code = int(exc.code)
        return exit_code, stdout_buf.getvalue(), stderr_buf.getvalue()

    def test_json_output_single_query(self) -> None:
        exit_code, stdout, stderr = self._run_cli(
            "--run-dir", str(SAMPLE_RUN),
            "--queries", "What have I said about burnout?",
            "--format", "json",
        )
        self.assertEqual(exit_code, 0)
        self.assertEqual(stderr, "")
        d = json.loads(stdout)
        self.assertIn("query", d)
        self.assertIn("entities", d)
        self.assertIn("evidence_count", d)
        self.assertIsInstance(d["entities"], list)

    def test_json_output_multiple_queries(self) -> None:
        exit_code, stdout, stderr = self._run_cli(
            "--run-dir", str(SAMPLE_RUN),
            "--queries", "burnout", "sleep",
            "--format", "json",
        )
        self.assertEqual(exit_code, 0)
        d = json.loads(stdout)
        self.assertIn(";", d["query"])

    def test_text_output_single_query(self) -> None:
        exit_code, stdout, stderr = self._run_cli(
            "--run-dir", str(SAMPLE_RUN),
            "--queries", "What have I said about burnout?",
            "--format", "text",
        )
        self.assertEqual(exit_code, 0)
        self.assertEqual(stderr, "")
        self.assertIn("Pattern Report:", stdout)
        self.assertIn("Evidence count:", stdout)

    def test_text_output_multiple_queries(self) -> None:
        exit_code, stdout, stderr = self._run_cli(
            "--run-dir", str(SAMPLE_RUN),
            "--queries", "burnout", "sleep",
            "--format", "text",
        )
        self.assertEqual(exit_code, 0)
        self.assertIn("Pattern Report:", stdout)

    def test_empty_result_path(self) -> None:
        """Query with no recurring entities produces clean output."""
        exit_code, stdout, stderr = self._run_cli(
            "--run-dir", str(SAMPLE_RUN),
            "--queries", "xyzzy nonexistent topic",
            "--format", "text",
        )
        self.assertEqual(exit_code, 0)
        self.assertIn("No recurring entities found.", stdout)

    def test_missing_run_dir(self) -> None:
        exit_code, stdout, stderr = self._run_cli(
            "--run-dir", "tests/fixtures/nonexistent",
            "--queries", "burnout",
            "--format", "json",
        )
        self.assertEqual(exit_code, 2)
        self.assertIn("error", stderr)

    def test_missing_required_args(self) -> None:
        """Omitting --queries should fail."""
        exit_code, stdout, stderr = self._run_cli(
            "--run-dir", str(SAMPLE_RUN),
            "--format", "json",
        )
        self.assertNotEqual(exit_code, 0)
        self.assertIn("--queries", stderr)

    def test_invalid_format_rejected(self) -> None:
        exit_code, stdout, stderr = self._run_cli(
            "--run-dir", str(SAMPLE_RUN),
            "--queries", "burnout",
            "--format", "debug",
        )
        self.assertNotEqual(exit_code, 0)
        self.assertIn("invalid choice", stderr)

    def test_end_to_end_json_structure(self) -> None:
        """Full pipeline: narrative build -> extraction -> JSON render."""
        exit_code, stdout, stderr = self._run_cli(
            "--run-dir", str(SAMPLE_RUN),
            "--queries", "burnout", "sleep",
            "--format", "json",
        )
        self.assertEqual(exit_code, 0)
        d = json.loads(stdout)
        # Every entity must have valid structure.
        for entity in d["entities"]:
            self.assertIn("name", entity)
            self.assertIn("occurrences", entity)
            self.assertIn("occurrence_count", entity)
            self.assertEqual(entity["occurrence_count"], len(entity["occurrences"]))
            for occ in entity["occurrences"]:
                self.assertIn("evidence_id", occ)
                self.assertIn("created_at", occ)
                self.assertIn("excerpt", occ)

    def test_stable_output_across_runs(self) -> None:
        """Same input produces identical output (deterministic)."""
        args = [
            "--run-dir", str(SAMPLE_RUN),
            "--queries", "burnout",
            "--format", "json",
        ]
        _, stdout_1, _ = self._run_cli(*args)
        _, stdout_2, _ = self._run_cli(*args)
        self.assertEqual(stdout_1, stdout_2)


if __name__ == "__main__":
    unittest.main()
