# Default Answer Mode Evaluation Report

Date: 2026-03-21
Benchmark version: v0.5.2 (frozen Phase 5 fixtures, v0.4.6 retrieval)
LLM model: gpt-5-mini (default)
Pipeline: real pipeline (not test harness)

## Executive Summary

Hybrid synthesis produces meaningfully better answers than deterministic on 4 of
5 benchmark queries. The 5th query (Butters) fell back to deterministic due to
LLM non-determinism, making the outputs identical.

**Recommendation: configurable default, deterministic ships first, hybrid
promoted after expanded benchmark validation.**

## Side-by-Side Comparison

| Query | Det chars | Hyb chars | Hyb mode | Winner |
|-------|-----------|-----------|----------|--------|
| Butters | 614 | 614 | hybrid_fallback | tie |
| Marc | 530 | 1125 | hybrid | hybrid |
| Benz | 523 | 1612 | hybrid | hybrid |
| shadow work | 338 | 811 | hybrid | hybrid |
| villa group | 643 | 1296 | hybrid | hybrid |

## Dimension-by-Dimension Analysis

### 1. Readability

**Deterministic**: bullet list of truncated excerpts prefixed with dates. Uniform
structure regardless of query. No connective tissue between excerpts.

**Hybrid**: narrative form with full ISO timestamps, connective phrases ("Together
these excerpts show"), and explicit boundary markers ("No additional inference has
been added"). Varies structure based on content — chronological listing for Marc,
synthesis paragraph for Benz.

**Winner: hybrid.** The narrative form reads as an answer rather than a search
result dump. Deterministic is scannable but doesn't answer the question — it
presents evidence and leaves synthesis to the reader.

### 2. Grounding / Faithfulness

**Deterministic**: excerpts are verbatim from evidence, truncated at 200 chars.
Cannot hallucinate because it only copies. No interpretation.

**Hybrid**: quotes user verbatim, labels truncation explicitly, includes "No
additional facts about Marc are stated in the excerpts." Occasionally includes
assistant responses for context (Benz query: assistant's "Benz was the latter"
comment). All claims traceable to evidence.

**Winner: tie with edge to deterministic.** Deterministic is unfailingly faithful
by construction. Hybrid is faithful under validation constraints but carries
inherent LLM risk. The 1/5 fallback rate on this run demonstrates the risk is
real, not theoretical.

### 3. Chronology Clarity

**Deterministic**: date prefix in `(YYYY-MM-DD)` format. Excerpts ordered
chronologically by the pipeline. No explicit timeline narrative.

**Hybrid**: full ISO timestamps with evidence IDs. Explicitly calls out time
spans ("25 years ago," "amazingly better than I did a year ago"). Villa group
hybrid pulled in an earlier 2025-03-06 excerpt that deterministic omitted,
providing broader timeline context.

**Winner: hybrid.** The full timestamps and narrative framing make temporal
relationships explicit. Deterministic shows dates but doesn't connect them.

### 4. User Voice Preservation

**Deterministic**: quotes the user's words verbatim (truncated at 200 chars).
No paraphrasing. No voice alteration.

**Hybrid**: also quotes verbatim, but labels quotes explicitly as "user" or
"assistant" with role attribution. Benz hybrid includes a synthesis paragraph
that paraphrases evidence — but clearly signals which parts are quoted vs
summarized.

**Winner: tie.** Both preserve user voice. Hybrid adds role attribution which
is useful context but occasionally paraphrases in synthesis paragraphs.

### 5. Inspectability

**Deterministic**: what you see is what the pipeline found. Evidence IDs map
directly to citations. No transformation layer to question.

**Hybrid**: `synthesis_mode` field explicitly records which path produced the
answer. Fallback notice visible in CLI. But the LLM transformation is opaque —
you can verify the output against evidence, but you can't see the LLM's
reasoning.

**Winner: deterministic.** Deterministic is fully transparent. Hybrid requires
post-hoc validation to trust, which is why we built the validation framework.

### 6. Trustworthiness

**Deterministic**: always produces the same output for the same evidence.
Never hallucinates. Never fails.

**Hybrid**: 4/5 produced valid output on this run. 1/5 fell back to
deterministic (Butters). The fallback is visible and safe — but it means the
user sees different quality levels across queries in the same session.

**Winner: deterministic.** Consistency matters for trust. A system that
sometimes gives great answers and sometimes falls back is harder to trust than
one that always gives adequate answers.

## Scoring Summary

| Dimension | Deterministic | Hybrid |
|-----------|---------------|--------|
| Readability | adequate | strong |
| Grounding | strong | strong (with risk) |
| Chronology | adequate | strong |
| Voice preservation | strong | strong |
| Inspectability | strong | adequate |
| Trustworthiness | strong | adequate |

Deterministic: 4 strong, 2 adequate
Hybrid: 3 strong, 1 strong-with-risk, 2 adequate

## Recommendation: Configurable Default

Neither mode dominates. Deterministic wins on trust and inspectability. Hybrid
wins on readability and chronology. The right answer is not to pick one — it's
to let the user choose, with deterministic as the shipping default.

### Why not hybrid default now

1. **Fallback inconsistency** — 1/5 queries fell back on this run. Users would
   see mixed quality across queries in the same session.
2. **Latency and cost** — hybrid requires an LLM API call per query. Deterministic
   is instant.
3. **Validation coverage** — 5 queries is too small to characterize the failure
   distribution. Need the expanded benchmark (10 more queries) to measure the
   real fallback rate.

### When to promote hybrid to default

Hybrid should become the default when:
- Expanded benchmark (15 queries) shows ≥90% valid rate across 3 consecutive runs
- Fallback rendering is polished enough that mixed-mode sessions feel coherent
- Latency is acceptable for the user's workflow

### Implementation

No code changes needed — `--hybrid` flag already exists. The recommendation is
a decision about defaults and documentation, not about architecture.

## Expanded Benchmark Results (15 queries, 3 runs)

10 new benchmark queries were frozen from real pipeline output and evaluated
alongside the original 5 queries. Each run calls the LLM independently.

### Pre-fix Results (v0.5.2)

| Run | Valid | Degraded | Invalid | Fallback | Rate |
|-----|-------|----------|---------|----------|------|
| 1 | 12 | 0 | 0 | 3 | 80% |
| 2 | 12 | 0 | 0 | 3 | 80% |
| 3 | 13 | 0 | 0 | 2 | 87% |

Meditation failed all 3 runs. Root cause: evidence e3 contains "Siem Reap's
temple environments" — permissive extractor captured `Reap's` but not the
base form `Reap`, so when the LLM wrote "Reap" the entity validator rejected it.

### Fix Applied

`_entity_surface_forms_permissive()` now strips possessives: when it finds
`Reap's`, it also adds `Reap` to the allowed set. Same logic strict extraction
already had. One-line change in `generator_llm.py`.

### Post-fix Results (v0.5.3)

| Run | Valid | Degraded | Invalid | Fallback | Rate |
|-----|-------|----------|---------|----------|------|
| 1 | 12 | 0 | 0 | 3 | 80% |
| 2 | 15 | 0 | 0 | 0 | 100% |
| 3 | 13 | 0 | 0 | 2 | 87% |

**Meditation: valid on all 3 runs.** Fix confirmed.

**Run 2 achieved 100%** — zero fallbacks across all 15 queries.

**Zero invalid or degraded across all 90 evaluations (pre+post).** Every
failure is a clean fallback to deterministic.

### Per-Query Stability (post-fix)

| Query | Category | Run 1 | Run 2 | Run 3 |
|-------|----------|-------|-------|-------|
| Butters | original | valid | valid | valid |
| Marc | original | valid | valid | valid |
| Benz | original | valid | valid | valid |
| shadow work | original | valid | valid | valid |
| villa group | original | valid | valid | valid |
| meditation | weak evidence | valid | valid | valid |
| cryptocurrency | weak evidence | valid | valid | valid |
| Thailand | conflicting | valid | valid | fallback |
| stay or leave | conflicting | valid | valid | valid |
| social life | long-range | fallback | valid | valid |
| stoicism | long-range | valid | valid | valid |
| learn about myself | ambiguous | valid | valid | valid |
| biggest decisions | ambiguous | valid | valid | valid |
| projects | technical | fallback | valid | valid |
| AI | technical | valid | valid | fallback |
| shadow work path | original | fallback | valid | valid |

### Fallback Analysis (post-fix)

**No consistent fallbacks.** Every query passed at least 2 of 3 runs.

**Rotating fallbacks (1/3 runs each):**
- social life, projects, shadow work path (Run 1)
- Thailand, AI (Run 3)

All failures are entity hallucination caught by the validator. No pattern —
different queries fail on different runs. This is irreducible LLM
non-determinism.

### Assessment Against Threshold

Average valid rate: **89%** (40/45). Best run: **100%**.

The ≥90% across 3 consecutive runs threshold is narrowly missed (80%, 100%, 87%).
However:
- Zero invalid or degraded in 90 total evaluations
- One run achieved 100%
- All fallbacks are safe and visible
- No structural failures remain

### Operating Decision

Deterministic remains the default. Hybrid is recommended for richer answers
with the understanding that ~10% of queries may fall back to deterministic.
The fallback is safe, visible, and never produces worse output than the default.

### New Benchmark Queries (frozen fixtures)

**Weak evidence (2 queries)**:
1. "what did I think about meditation" — 3 evidence items (supported).
   Stress test: thin evidence, LLM tempted to fill gaps with context.
2. "what was my opinion on cryptocurrency" — 3 evidence items
   (partially_supported). Stress test: evidence doesn't actually discuss
   cryptocurrency, hybrid must acknowledge this.

**Conflicting evidence (2 queries)**:
3. "how did I feel about living in Thailand" — 4 evidence items (supported).
   Stress test: sentiment shifts across months.
4. "did I want to stay or leave" — 2 evidence items (supported).
   Stress test: ambivalent topic, thin evidence.

**Long-range chronology (2 queries)**:
5. "how did my social life change over the past year" — 3 evidence items
   (partially_supported). Stress test: wide time range.
6. "what was my journey with stoicism" — 4 evidence items
   (partially_supported). Stress test: thematic evolution.

**Generic-topic ambiguity (2 queries)**:
7. "what did I learn about myself" — 3 evidence items (supported).
   Stress test: maximally broad query.
8. "what were my biggest decisions" — 4 evidence items
   (partially_supported). Stress test: subjective, multi-topic.

**Technical / project-history (2 queries)**:
9. "what projects was I working on" — 3 evidence items (supported).
   Stress test: factual/technical content.
10. "what did I discuss about AI" — 4 evidence items (supported).
    Stress test: high-volume evidence.
