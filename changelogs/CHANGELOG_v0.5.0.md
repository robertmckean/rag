# v0.5.0 - Phase 5 synthesis foundation: multi-evidence answers and benchmark fixtures

Date: 2026-03-21

## Implements

- expands QUESTION_STOPWORDS in the answering qualification layer to filter
  query-intent verbs (learn, conclude, happened, thinking, path) and leaked
  prepositions/determiners so focus_terms targets entities and topics only
- replaces single-excerpt answer generation with multi-evidence synthesis:
  _select_synthesis_excerpts() picks up to 4 non-redundant excerpts with
  user-authored evidence preferred, then _compose_supported_answer() or
  _compose_partial_answer() formats the output
- separates evidence selection from answer writing: qualification pipeline
  produces evidence items, synthesis picks what to show, composition formats it
- adds near-duplicate suppression in synthesis excerpt selection (60-char
  normalized prefix comparison)
- creates 5 frozen Phase 5 benchmark fixtures in tests/fixtures/phase5_benchmarks/
  with v0.4.6 retrieval outputs and answer baselines for regression comparison

## Benchmark results (5 queries, deterministic path)

- shadow work: partially_supported → supported (2 user excerpts, coverage 1.0)
- Benz: partially_supported → supported (4 user excerpts across conversations)
- Marc: partially_supported → supported (4 user excerpts showing confrontation arc)
- villa group: partially_supported → partially_supported (4 user excerpts, 1.0 coverage, conservative classifier)
- Butters: partially_supported → supported (4 user excerpts showing distancing arc)

## Release meaning

This is a major version bump because it introduces the synthesis layer as a real
capability. The system can now answer "what did I conclude across multiple
conversations" using grounded multi-evidence composition rather than quoting one
excerpt. The frozen benchmark fixtures establish the evaluation baseline for
iterative synthesis refinement.

## Validation

- PYTHONPATH=src python -m unittest discover -s tests -p "test_*.py"
- 116 tests pass, zero failures
- Live benchmark across 5 queries confirms multi-evidence synthesis and improved
  answer status classification
