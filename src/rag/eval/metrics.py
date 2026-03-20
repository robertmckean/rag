"""Deterministic metrics for benchmarking grounded-answer behavior."""

from __future__ import annotations

from collections import Counter

from rag.answering.models import AnswerResult
from rag.eval.models import EvalCase, FailureType
from rag.retrieval.read_model import tokenize_query


# The eval metrics reuse the same lexical tokenization style as retrieval and answering.
# Each metric is intentionally small and inspectable so failures are easy to reason about.
# Failure classification stays rule-based because this harness is for deterministic regression checks.


ENTITY_STOPWORDS = frozenset({"about", "did", "have", "i", "my", "said", "what"})


# Normalize one term into the lowercase lexical token space used across the project.
def normalize_eval_term(term: str) -> str:
    return " ".join(tokenize_query(term))


# Compare actual and expected answer status directly.
def status_match(case: EvalCase, result: AnswerResult) -> bool:
    return result.answer_status.value == case.expected_status


# Check citation count bounds when the benchmark case specifies them.
def citation_count_ok(case: EvalCase, result: AnswerResult) -> bool:
    citation_count = len(result.citations)
    if case.expected_min_citations is not None and citation_count < case.expected_min_citations:
        return False
    if case.expected_max_citations is not None and citation_count > case.expected_max_citations:
        return False
    return True


# Verify that each returned citation corresponds to one of the selected evidence items.
def citation_trace_valid(result: AnswerResult) -> bool:
    evidence_keys = {
        (
            item.citation.provider,
            item.citation.conversation_id,
            item.citation.message_id,
            item.citation.excerpt,
        )
        for item in result.evidence_used
    }
    for citation in result.citations:
        citation_key = (
            citation.provider,
            citation.conversation_id,
            citation.message_id,
            citation.excerpt,
        )
        if citation_key not in evidence_keys:
            return False
    return True


# Check whether any expected term appears in qualified evidence excerpts.
def evidence_term_coverage_any(case: EvalCase, result: AnswerResult) -> bool:
    if not case.expected_terms_any:
        return True
    evidence_tokens = _evidence_token_set(result)
    return any(normalize_eval_term(term) in evidence_tokens for term in case.expected_terms_any)


# Check whether all expected terms appear in qualified evidence excerpts.
def evidence_term_coverage_all(case: EvalCase, result: AnswerResult) -> bool:
    if not case.expected_terms_all:
        return True
    evidence_tokens = _evidence_token_set(result)
    return all(normalize_eval_term(term) in evidence_tokens for term in case.expected_terms_all)


# Detect forbidden-term leakage in the answer text or qualified evidence excerpts.
def forbidden_term_violation(case: EvalCase, result: AnswerResult) -> bool:
    if not case.forbidden_terms:
        return False
    searchable_text = " ".join(
        [result.answer] + [item.citation.excerpt for item in result.evidence_used]
    )
    normalized_text = " ".join(tokenize_query(searchable_text))
    return any(normalize_eval_term(term) and normalize_eval_term(term) in normalized_text for term in case.forbidden_terms)


# Flag assistant-dominant factual support for entity-specific questions using deterministic heuristics.
def assistant_support_leakage(case: EvalCase, result: AnswerResult) -> bool:
    if not result.evidence_used:
        return False
    focus_terms = _query_focus_terms(case.query)
    if len(focus_terms) < 2:
        return False
    if not _looks_entity_specific(case.query):
        return False
    if result.evidence_used[0].author_role != "assistant":
        return False
    return all(item.author_role == "assistant" for item in result.evidence_used)


# Empty evidence is only correct when the answer status is insufficient_evidence.
def empty_support_correctness(result: AnswerResult) -> bool:
    if result.retrieval_summary.evidence_used_count == 0:
        return result.answer_status.value == "insufficient_evidence"
    return True


# Evaluate the metric bundle for one benchmark case.
def evaluate_case_metrics(case: EvalCase, result: AnswerResult) -> dict[str, bool]:
    return {
        "status_match": status_match(case, result),
        "citation_count_ok": citation_count_ok(case, result),
        "citation_trace_valid": citation_trace_valid(result),
        "evidence_term_coverage_any": evidence_term_coverage_any(case, result),
        "evidence_term_coverage_all": evidence_term_coverage_all(case, result),
        "forbidden_term_ok": not forbidden_term_violation(case, result),
        "assistant_support_leakage": not assistant_support_leakage(case, result),
        "empty_support_correctness": empty_support_correctness(result),
    }


# Map metric failures into stable failure types that are useful for targeted fixes.
def classify_failures(case: EvalCase, result: AnswerResult, metrics: dict[str, bool]) -> tuple[FailureType, ...]:
    failures: list[FailureType] = []
    actual_status = result.answer_status.value

    if not metrics["status_match"]:
        if actual_status == "supported":
            failures.append(FailureType.FALSE_SUPPORTED)
        elif actual_status == "partially_supported":
            failures.append(FailureType.FALSE_PARTIAL)
        elif actual_status == "ambiguous":
            failures.append(FailureType.FALSE_AMBIGUOUS)

        if case.expected_status == "supported":
            failures.append(FailureType.MISSED_SUPPORT)
        elif case.expected_status == "partially_supported":
            failures.append(FailureType.MISSED_PARTIAL)
        elif case.expected_status == "ambiguous":
            failures.append(FailureType.MISSED_AMBIGUITY)

    if not metrics["evidence_term_coverage_any"]:
        failures.append(FailureType.TOPICAL_DRIFT)
    if not metrics["evidence_term_coverage_all"]:
        failures.append(FailureType.MULTIPART_COVERAGE_FAILURE)
    if not metrics["citation_trace_valid"]:
        failures.append(FailureType.CITATION_INTEGRITY_FAILURE)
    if not metrics["citation_count_ok"]:
        failures.append(FailureType.CITATION_COUNT_FAILURE)
    if not metrics["forbidden_term_ok"]:
        failures.append(FailureType.FORBIDDEN_TERM_VIOLATION)
    if not metrics["assistant_support_leakage"]:
        failures.append(FailureType.ASSISTANT_LEAKAGE)

    unique_failures: list[FailureType] = []
    seen_values: set[str] = set()
    for failure in failures:
        if failure.value in seen_values:
            continue
        seen_values.add(failure.value)
        unique_failures.append(failure)
    return tuple(unique_failures)


# Count failure types across all evaluated cases.
def failure_type_counts(case_results: tuple[object, ...]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for case_result in case_results:
        for failure_type in case_result.failure_types:
            counter[failure_type.value] += 1
    return dict(counter)


# Build a set of normalized evidence excerpts for term-coverage checks.
def _evidence_token_set(result: AnswerResult) -> set[str]:
    tokens: set[str] = set()
    for item in result.evidence_used:
        normalized_excerpt = " ".join(tokenize_query(item.citation.excerpt))
        if normalized_excerpt:
            tokens.add(normalized_excerpt)
            tokens.update(normalized_excerpt.split())
    return tokens


# Extract meaningful query terms for entity-style heuristics.
def _query_focus_terms(query: str) -> tuple[str, ...]:
    tokens = []
    seen: set[str] = set()
    for token in tokenize_query(query):
        if token in ENTITY_STOPWORDS or len(token) < 2:
            continue
        if token in seen:
            continue
        seen.add(token)
        tokens.append(token)
    return tuple(tokens)


# Treat possessive entity queries as the current high-risk class for assistant leakage.
def _looks_entity_specific(query: str) -> bool:
    return "'s " in query.lower()
