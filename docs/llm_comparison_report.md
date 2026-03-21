# Hybrid Synthesis Evaluation Report

Date: 2026-03-21
Benchmark version: v0.5.2 (frozen Phase 5 fixtures, v0.4.6 retrieval)
LLM model: gpt-5-mini (default)

## Executive Summary

After prompt constraint refinement and property-based validation, the hybrid
synthesis path achieves **5/5 valid** on the frozen benchmark set. Zero
fallbacks, zero invalid outputs, zero degraded outputs.

| Metric | v0.5.1 | v0.5.2 (this) |
|--------|--------|---------------|
| Valid | 4/5 (80%) | 5/5 (100%) |
| Fallback | 1/5 (20%) | 0/5 (0%) |
| Invalid | 0/5 | 0/5 |
| Degraded | 0/5 | 0/5 |

## Property-Based Validation

Each hybrid output is validated against five structural properties:

1. **No new entities** — every entity in the answer appears in the evidence or query
2. **Dates preserved** — when evidence spans multiple dates, at least one timestamp
   appears in the answer
3. **Evidence bounded** — citation count does not exceed evidence count, all IDs valid
4. **Status not inflated** — partially_supported answers are not presented as definitive
5. **Evidence grounding** — at least one 4-word n-gram from evidence appears in the answer

Outputs are classified as:
- **valid** — all properties pass
- **degraded** — dates missing or evidence grounding absent
- **invalid** — new entities or citation overflow
- **fallback** — LLM synthesis failed, fell back to deterministic

## Per-Query Results

### 1. "what did I conclude about Butters" — VALID

> You wrote (created_at: 2026-01-20): "OK… Next! I could just start ignoring
> butters, but maybe I should send him something similar. I could connect him to
> the villa and my need for distance." The excerpt is truncated.

Properties: entities OK, dates OK, bounded OK, grounding OK.

### 2. "what did I learn about Marc" — VALID

> 2025-02-07 (e1): "How should I open the meeting with Marc and Craig?"
> 2025-02-12 (e3): "Give me additional suggestions to make marc less
> intimidating to talk to, based on his profile."
> 2025-02-18 (e4): "I think you're being too vague about his responses."

Properties: all pass. Chronological ordering, user voice quoted, 3 citations.

### 3. "what happened with Benz" — VALID

> 2025-07-13: "I was thinking of sending Benz a message, asking if she wants to
> play pool. She's a lovely girl..."
> 2026-01-23: "I stopped at the bar to play pool with Benz. She was already
> playing with some old English guy." (excerpt truncated)

Properties: all pass. 7-month timeline preserved, voice quoted.

### 4. "what was my path to shadow work" — VALID

> 2025-11-24: "We're having an argument, I felt like she wasn't hearing me."
> 2025-12-08: "I believe the universe is essentially pixelated."
> 2025-12-10: "I embraced stoicism to help me understand what was happening to me."

Properties: all pass. 3 excerpts chronologically ordered with timestamps.

### 5. "what was my thinking about the villa group" — VALID

> 2026-01-20: "Aaron I like, even though he's fully a part of the villa..."
> 2026-01-20: "I could just start ignoring butters..."
> 2026-01-25: "Just the idea of being on the same list as Marc is enough to
> make me leave."

Properties: all pass. Multi-day span preserved, user voice quoted throughout.

## Prompt Constraint Changes (v0.5.1 -> v0.5.2)

Three targeted refinements in `_prompt_instructions_hybrid()`:

1. **Entity anchoring** — "Only use names, places, and entities that appear
   verbatim in the evidence or query. Spell them exactly as they appear."
   (was: "Do not introduce any names, places, or entities...")

2. **Date anchoring** — "When evidence spans multiple dates, include the
   created_at date for each excerpt as a chronological marker."
   (was: "Preserve dates from the evidence. Include them in chronological order
   when available.")

3. **Voice anchoring** — "Cite every evidence item you reference. Do not
   reference evidence you do not cite." + "When evidence is truncated or
   incomplete, say so — do not fill in the gap."
   (new constraints, not present in v0.5.1)

## Architecture Changes

- `AnswerResult.synthesis_mode` — tracks which path produced the answer:
  `deterministic`, `llm`, `hybrid`, `llm_fallback`, `hybrid_fallback`
- `render_answer_result()` — shows fallback notice in CLI when synthesis fails
- `hybrid_validation.py` — shared property-based validation used by tests and eval
- `HybridValidationResult` — structured output with per-property pass/fail and
  classification
- Evaluation script classifies outputs as valid/degraded/invalid/fallback

## Reliability Assessment

The hybrid path went from 80% (4/5) validation rate in v0.5.1 to 100% (5/5)
in v0.5.2. The improvement came from tighter prompt constraints that anchor
the LLM to entity names, dates, and evidence boundaries.

The remaining risk is LLM non-determinism — future runs may occasionally
produce outputs that fail validation. The fallback path is now visible in
both CLI output and the `synthesis_mode` field, so failures are never silent.

## Recommendation

Hybrid synthesis is now reliable enough for optional use. The deterministic
path remains the default. Next evaluation step: run the benchmark multiple
times to measure the statistical validation rate across LLM non-determinism.
