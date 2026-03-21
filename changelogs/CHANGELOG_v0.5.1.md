# v0.5.1 - Fix LLM validator, add hybrid synthesis mode, synthesis refinement

Date: 2026-03-21

## Implements

- fixes entity surface form validator that caused 100% silent fallback on all
  LLM synthesis queries: splits extraction into permissive (for allowed set)
  and strict mid-sentence-only (for answer check), adds _NON_ENTITY_WORDS set
  of 120 common English words, normalizes possessives, and includes evidence
  created_at timestamps in the allowed date source
- adds --hybrid CLI flag for constrained hybrid synthesis: deterministic
  evidence selection with LLM-compressed narrative that preserves user voice
  (quoting key phrases), chronological dates, and evidence boundaries
- adds explicit WARNING-level logging when LLM synthesis falls back to
  deterministic, replacing the previous silent try/except
- adds hybrid prompt (_prompt_instructions_hybrid) that instructs the LLM to
  quote user-authored excerpts, preserve dates chronologically, and not infer
  beyond evidence
- includes author_role in the evidence payload sent to the LLM in hybrid mode
  so the model can distinguish user vs assistant excerpts
- completes synthesis refinement from v0.5.0: multi-evidence excerpt
  compression (120 char limit), chronological ordering when timestamps span
  multiple days, _SynthesisEntry dataclass with timestamp and author_role,
  date labels on multi-evidence answers

## Benchmark results (5 queries, hybrid path)

- Butters: VALIDATED — quotes user phrases, preserves date, notes truncation
- Marc: VALIDATED — chronological annotated excerpts showing preparation arc
- Benz: VALIDATED — timeline with explicit gaps noted
- shadow work: VALIDATED — thematic grouping with chronological structure
- villa group: FALLBACK (non-deterministic LLM output, ~80% pass rate)

## Release meaning

This release makes the LLM synthesis path functional for the first time and
introduces the hybrid synthesis mode as the recommended optional LLM-augmented
path. Deterministic synthesis remains the default. The entity validator
redesign ensures the safety checks catch genuinely new entities without
false-positive rejection of normal English capitalization.

## Validation

- PYTHONPATH=src python -m unittest discover -s tests -p "test_*.py"
- 122 tests pass, zero failures (6 new tests for entity extraction, hybrid
  flag, and fallback logging)
- Live hybrid benchmark across 5 queries: 4/5 validated, 1/5 falls back
  to deterministic due to LLM non-determinism
