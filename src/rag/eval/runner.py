"""Eval runner for deterministic grounded-answer benchmarks."""

from __future__ import annotations

import json
from pathlib import Path

from rag.answering.answer import answer_query
from rag.eval.metrics import classify_failures, evaluate_case_metrics, failure_type_counts
from rag.eval.models import EvalCase, EvalCaseResult, EvalSummary


# The runner executes the real answer pipeline against a small benchmark bank.
# Reports stay concise because the goal is fast diagnosis, not a generic eval framework.
# Each case result keeps enough detail to explain what failed without dumping the full answer object.


# Load a JSON benchmark file into explicit eval-case models.
def load_query_bank(path: Path) -> tuple[EvalCase, ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Benchmark file must contain a JSON list: {path}")

    cases: list[EvalCase] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        cases.append(
            EvalCase(
                id=str(item["id"]),
                query=str(item["query"]),
                retrieval_mode=str(item.get("retrieval_mode", "relevance")),
                expected_status=str(item["expected_status"]),
                expected_terms_any=_string_tuple(item.get("expected_terms_any")),
                expected_terms_all=_string_tuple(item.get("expected_terms_all")),
                forbidden_terms=_string_tuple(item.get("forbidden_terms")),
                expected_min_citations=_optional_int(item.get("expected_min_citations")),
                expected_max_citations=_optional_int(item.get("expected_max_citations")),
                notes=_optional_string(item.get("notes")),
            )
        )
    return tuple(cases)


# Run the full benchmark bank against one normalized run and aggregate results.
def run_benchmark(run_dir: Path, bench_path: Path) -> tuple[EvalSummary, tuple[EvalCaseResult, ...]]:
    cases = load_query_bank(bench_path)
    case_results: list[EvalCaseResult] = []

    for case in cases:
        answer_result = answer_query(
            run_dir.resolve(),
            case.query,
            retrieval_mode=case.retrieval_mode,
            limit=8,
            max_evidence=5,
        )
        metrics = evaluate_case_metrics(case, answer_result)
        failures = classify_failures(case, answer_result, metrics)
        passed = all(metrics.values()) and not failures
        case_results.append(
            EvalCaseResult(
                case_id=case.id,
                query=case.query,
                expected_status=case.expected_status,
                actual_status=answer_result.answer_status.value,
                passed=passed,
                failure_types=failures,
                metrics=metrics,
                citation_count=len(answer_result.citations),
                evidence_used_count=len(answer_result.evidence_used),
                answer=answer_result.answer,
                notes=case.notes,
            )
        )

    results_tuple = tuple(case_results)
    passed_count = sum(1 for result in results_tuple if result.passed)
    summary = EvalSummary(
        run_id=str(run_dir.resolve().name),
        bench_path=str(bench_path.resolve()),
        bench_cases=len(results_tuple),
        passed=passed_count,
        failed=len(results_tuple) - passed_count,
        status_accuracy=_status_accuracy(results_tuple),
        failure_type_counts=failure_type_counts(results_tuple),
    )
    return summary, results_tuple


# Render a concise human-readable report for terminal review.
def render_eval_report(summary: EvalSummary, case_results: tuple[EvalCaseResult, ...], *, fail_only: bool = False) -> str:
    lines = [
        "Eval Summary",
        f"run_id: {summary.run_id}",
        f"bench_cases: {summary.bench_cases}",
        f"passed: {summary.passed}",
        f"failed: {summary.failed}",
        f"status_accuracy: {summary.status_accuracy:.2f}",
        "failure_types:",
    ]
    if summary.failure_type_counts:
        for failure_type, count in sorted(summary.failure_type_counts.items()):
            lines.append(f"  {failure_type}: {count}")
    else:
        lines.append("  (none)")

    visible_results = tuple(result for result in case_results if (not fail_only or not result.passed))
    if visible_results:
        lines.append("")
        lines.append("Cases:")
        for result in visible_results:
            lines.append(
                f"  - id={result.case_id} passed={result.passed} "
                f"expected_status={result.expected_status} actual_status={result.actual_status}"
            )
            if result.failure_types:
                lines.append(
                    "    failure_types=" + ", ".join(failure_type.value for failure_type in result.failure_types)
                )
            if result.notes:
                lines.append(f"    notes={result.notes}")
    return "\n".join(lines)


# Serialize the eval summary and per-case results into a stable JSON payload.
def eval_report_json(summary: EvalSummary, case_results: tuple[EvalCaseResult, ...]) -> str:
    payload = {
        "summary": summary.to_dict(),
        "results": [case_result.to_dict() for case_result in case_results],
    }
    return json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n"


# Compute simple status accuracy over the benchmark bank.
def _status_accuracy(case_results: tuple[EvalCaseResult, ...]) -> float:
    if not case_results:
        return 0.0
    matched = sum(1 for result in case_results if result.expected_status == result.actual_status)
    return matched / len(case_results)


# Normalize one optional list field from the benchmark JSON into a tuple of strings.
def _string_tuple(value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(str(item) for item in value if isinstance(item, str) and item.strip())


# Read one optional integer field from the benchmark JSON.
def _optional_int(value: object) -> int | None:
    return value if isinstance(value, int) else None


# Read one optional string field from the benchmark JSON.
def _optional_string(value: object) -> str | None:
    return value if isinstance(value, str) and value.strip() else None
