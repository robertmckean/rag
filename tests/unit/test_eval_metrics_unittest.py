import unittest
from pathlib import Path

from rag.answering.answer import answer_query
from rag.eval.metrics import (
    assistant_support_leakage,
    citation_count_ok,
    citation_trace_valid,
    classify_failures,
    empty_support_correctness,
    evaluate_case_metrics,
    evidence_term_coverage_all,
    evidence_term_coverage_any,
    forbidden_term_violation,
    status_match,
)
from rag.eval.models import EvalCase, FailureType


# These tests keep the deterministic eval metrics narrow and explicit.
# They exercise the current answer pipeline outputs rather than inventing a separate eval-only representation.
# Failure classification checks stay targeted so regressions are easy to interpret.
class EvalMetricsTests(unittest.TestCase):
    # Reuse the grounded-answer fixture run so eval metrics stay aligned with real answer behavior.
    def setUp(self) -> None:
        self.run_dir = Path("tests/fixtures/answering/sample_run")

    # Verify that status_match succeeds for a clean supported case.
    def test_status_match(self) -> None:
        case = self._case("supported_burnout_late_2025", "What have I said about burnout late 2025?", "supported")
        result = answer_query(self.run_dir, case.query, retrieval_mode=case.retrieval_mode)

        self.assertTrue(status_match(case, result))

    # Verify that citation-count checks honor configured bounds.
    def test_citation_count_ok(self) -> None:
        case = EvalCase(
            id="count",
            query="What have I said about burnout?",
            retrieval_mode="relevance",
            expected_status="supported",
            expected_terms_any=(),
            expected_terms_all=(),
            forbidden_terms=(),
            expected_min_citations=1,
            expected_max_citations=2,
            notes=None,
        )
        result = answer_query(self.run_dir, case.query, retrieval_mode=case.retrieval_mode)

        self.assertTrue(citation_count_ok(case, result))

    # Verify that returned citations always trace back to selected evidence items.
    def test_citation_trace_valid(self) -> None:
        result = answer_query(self.run_dir, "What have I said about burnout late 2025?")

        self.assertTrue(citation_trace_valid(result))

    # Verify that expected any-term coverage succeeds and fails deterministically.
    def test_evidence_term_coverage_any(self) -> None:
        positive_case = EvalCase("pos", "burnout", "relevance", "supported", ("burnout",), (), (), None, None, None)
        negative_case = EvalCase("neg", "burnout", "relevance", "supported", ("daughter",), (), (), None, None, None)
        result = answer_query(self.run_dir, "What have I said about burnout late 2025?")

        self.assertTrue(evidence_term_coverage_any(positive_case, result))
        self.assertFalse(evidence_term_coverage_any(negative_case, result))

    # Verify that expected all-term coverage succeeds and fails deterministically.
    def test_evidence_term_coverage_all(self) -> None:
        positive_case = EvalCase("pos", "burnout", "relevance", "supported", (), ("burnout", "late", "2025"), (), None, None, None)
        negative_case = EvalCase("neg", "burnout", "relevance", "supported", (), ("burnout", "daughter"), (), None, None, None)
        result = answer_query(self.run_dir, "What have I said about burnout late 2025?")

        self.assertTrue(evidence_term_coverage_all(positive_case, result))
        self.assertFalse(evidence_term_coverage_all(negative_case, result))

    # Verify that forbidden-term checks catch leaked terms in answers or evidence.
    def test_forbidden_term_violation(self) -> None:
        violating_case = EvalCase("forbid", "burnout", "relevance", "supported", (), (), ("burnout",), None, None, None)
        clean_case = EvalCase("clean", "burnout", "relevance", "supported", (), (), ("daughter",), None, None, None)
        result = answer_query(self.run_dir, "What have I said about burnout late 2025?")

        self.assertTrue(forbidden_term_violation(violating_case, result))
        self.assertFalse(forbidden_term_violation(clean_case, result))

    # Verify that empty evidence must map to insufficient_evidence.
    def test_empty_support_correctness(self) -> None:
        result = answer_query(self.run_dir, "What have I said about Mecky's daughter?")

        self.assertTrue(empty_support_correctness(result))

    # Verify that failure classification stays stable for representative eval failures.
    def test_failure_classification(self) -> None:
        case = EvalCase(
            id="failure",
            query="What have I said about marathon training?",
            retrieval_mode="relevance",
            expected_status="supported",
            expected_terms_any=("marathon",),
            expected_terms_all=(),
            forbidden_terms=(),
            expected_min_citations=1,
            expected_max_citations=2,
            notes=None,
        )
        result = answer_query(self.run_dir, case.query, retrieval_mode=case.retrieval_mode)
        metrics = evaluate_case_metrics(case, result)
        failures = classify_failures(case, result, metrics)

        self.assertIn(FailureType.MISSED_SUPPORT, failures)
        self.assertIn(FailureType.TOPICAL_DRIFT, failures)
        self.assertIn(FailureType.CITATION_COUNT_FAILURE, failures)

    # Verify the assistant-leakage heuristic stays quiet for a normal supported case.
    def test_assistant_support_leakage(self) -> None:
        case = EvalCase(
            id="mecky",
            query="What have I said about Mecky's daughter?",
            retrieval_mode="relevance",
            expected_status="insufficient_evidence",
            expected_terms_any=(),
            expected_terms_all=(),
            forbidden_terms=(),
            expected_min_citations=0,
            expected_max_citations=0,
            notes=None,
        )
        result = answer_query(self.run_dir, case.query, retrieval_mode=case.retrieval_mode)

        self.assertFalse(assistant_support_leakage(case, result))

    # Build a compact benchmark case for metric checks.
    def _case(self, case_id: str, query: str, expected_status: str) -> EvalCase:
        return EvalCase(
            id=case_id,
            query=query,
            retrieval_mode="relevance",
            expected_status=expected_status,
            expected_terms_any=(),
            expected_terms_all=(),
            forbidden_terms=(),
            expected_min_citations=None,
            expected_max_citations=None,
            notes=None,
        )


if __name__ == "__main__":
    unittest.main()
