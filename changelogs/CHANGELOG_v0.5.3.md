# v0.5.3 - Entity validation fix and expanded benchmark

Date: 2026-03-21

## Fixes

- fixes possessive-stripping gap in `_entity_surface_forms_permissive()`:
  evidence containing "Reap's" now also allows "Reap" in the answer, matching
  the behavior strict extraction already had
- this was the only structural hybrid fallback — meditation query now passes
  all 3 benchmark runs

## Implements

- expands benchmark from 5 to 15 frozen queries covering weak evidence,
  conflicting evidence, long-range chronology, generic-topic ambiguity, and
  technical/project-history
- updates eval script (`hybrid_eval.py`) to handle variable evidence counts
  (up to 5) and use fixture-recorded `answer_status` instead of hardcoded
  `SUPPORTED`
- adds `benchmark_expand.py` script for freezing new queries from real pipeline
- adds `mode_comparison.py` script for side-by-side deterministic vs hybrid
  evaluation through the real pipeline
- rewrites `llm_comparison_report.md` as full default-mode evaluation report
  with 6-dimension comparison, 3-run benchmark results, and operating decision

## Benchmark results (15 queries, 3 runs, post-fix)

| Run | Valid | Degraded | Invalid | Fallback | Rate |
|-----|-------|----------|---------|----------|------|
| 1   | 12    | 0        | 0       | 3        | 80%  |
| 2   | 15    | 0        | 0       | 0        | 100% |
| 3   | 13    | 0        | 0       | 2        | 87%  |

- Zero invalid or degraded across all 90 evaluations (pre+post fix)
- Run 2 achieved 100% — zero fallbacks across all 15 queries
- Remaining fallbacks are rotating LLM non-determinism, no structural issues

## Operating decision

- deterministic = default (safest, most inspectable)
- hybrid = recommended for richer answers (may fall back safely)
- both modes clearly labeled and always available via `--hybrid` flag

## Validation

- PYTHONPATH=src python -m unittest discover -s tests -p "test_*.py"
- 135 tests pass, zero failures (1 new: permissive possessive stripping)
