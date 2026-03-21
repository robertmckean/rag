# v0.5.2 - Hybrid validation framework and synthesis stabilization

Date: 2026-03-21

## Implements

- adds property-based validation framework for hybrid synthesis outputs
  (hybrid_validation.py) with five structural checks: no new entities, dates
  preserved, evidence bounded, status not inflated, evidence grounding
- classifies hybrid outputs as valid, degraded, invalid, or fallback based
  on property check results
- tightens hybrid prompt constraints with three surgical refinements: entity
  anchoring (verbatim spelling), date anchoring (created_at per excerpt), and
  evidence anchoring (cite every reference, flag truncation)
- adds synthesis_mode field to AnswerResult tracking which path produced the
  answer (deterministic, llm, hybrid, llm_fallback, hybrid_fallback)
- surfaces fallback in CLI output: render_answer_result prints clear notice
  when hybrid synthesis fails validation and falls back to deterministic
- updates evaluation script to classify and count valid/degraded/invalid/
  fallback outputs across the benchmark set

## Benchmark results (5 queries, hybrid path with property validation)

- Butters: valid — user voice quoted, timestamp preserved, truncation noted
- Marc: valid — 3 dated excerpts in chronological order
- Benz: valid — 7-month timeline preserved, truncation acknowledged
- shadow work: valid — 3 excerpts with thematic and chronological structure
- villa group: valid — multi-day span, user voice preserved

Pass rate: 5/5 valid (100%), up from 4/5 (80%) in v0.5.1

## Release meaning

This release establishes measurable reliability for hybrid synthesis. The
property-based validation framework means hybrid output quality is checked
structurally rather than by exact string matching, accommodating LLM
non-determinism while enforcing grounding, chronology, and voice preservation.
The synthesis_mode field and CLI fallback notice make the system transparent
about which path produced each answer.

## Validation

- PYTHONPATH=src python -m unittest discover -s tests -p "test_*.py"
- 134 tests pass, zero failures (12 new: 9 property-based + 3 synthesis mode)
- Live hybrid benchmark: 5/5 valid under property-based classification
