import os
import subprocess
import sys
import unittest
from pathlib import Path


# This smoke test locks in standard unittest discovery so future test files stay discoverable.
# The child-process guard prevents recursive discovery spawning from re-invoking the same check forever.
# The assertion only checks that discovery finds real tests rather than silently reporting zero.
class UnittestDiscoverySmokeTests(unittest.TestCase):
    # Run the standard discovery command in a child process and verify that it finds the suite.
    def test_standard_unittest_discover_finds_tests(self) -> None:
        if os.environ.get("RAG_SKIP_DISCOVERY_SMOKE") == "1":
            return

        env = os.environ.copy()
        env["PYTHONPATH"] = "src"
        env["RAG_SKIP_DISCOVERY_SMOKE"] = "1"
        repo_root = Path(__file__).resolve().parents[2]
        result = subprocess.run(
            [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
            cwd=repo_root,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        combined_output = result.stdout + "\n" + result.stderr

        self.assertEqual(result.returncode, 0, msg=combined_output)
        self.assertNotIn("Ran 0 tests", combined_output)
        self.assertIn("Ran ", combined_output)


if __name__ == "__main__":
    unittest.main()
