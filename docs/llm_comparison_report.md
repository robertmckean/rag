# LLM Synthesis Comparison Report

Date: 2026-03-21
Benchmark version: v0.5.1 (frozen Phase 5 fixtures, v0.4.6 retrieval)
LLM model: gpt-5-mini (default)

## Executive Summary

Three synthesis paths were evaluated against 5 frozen benchmark queries:

| Path | Description | Validation rate | Voice | Dates | Grounding |
|------|-------------|-----------------|-------|-------|-----------|
| **Deterministic** | Quote-based, chronological bullets | 100% | User's own words | Preserved | Direct quotes |
| **--llm (rewrite)** | LLM paraphrase of evidence | 0% pre-fix, ~60% post-fix | Voice erased | Dropped | Paraphrase |
| **--hybrid (new)** | LLM-compressed narrative preserving voice/dates | 80% (4/5) | User voice quoted | Preserved | Quoted + summarized |

**Validator bug (fixed):** `_entity_surface_forms()` used a naive capitalized-word
regex that flagged sentence-initial words ("Because", "From", "You") as unseen
entities. Fixed by splitting into permissive extraction (for the allowed set) and
strict mid-sentence-only extraction (for the answer check), plus a non-entity
word list and possessive normalization.

**Fallback logging (added):** `_apply_optional_llm_synthesis` now logs a WARNING
when LLM synthesis fails, replacing the previous silent fallback.

**Hybrid mode (new):** `--hybrid` CLI flag sends a constrained prompt that
instructs the LLM to preserve user voice, include dates chronologically, and
quote rather than paraphrase.

## Hybrid vs Deterministic: Per-Query Results

### 1. "what did I conclude about Butters" -- VALIDATED

**Hybrid:**
> On 2026-01-20 you wrote: "I could just start ignoring butters, but maybe I
> should send him something similar. I could connect him to the villa and my need
> for distance. I was thinking about what makes butters unlikab" -- so your
> conclusion was that you might "start ignoring butters", considered sending him
> "something similar", and connected him to "the villa" and your "need for
> distance".

| Dimension | Deterministic | Hybrid |
|-----------|--------------|--------|
| Grounding | 4 direct quotes | Quotes key phrases from evidence |
| Chronology | 4 dated bullets | Date preserved, single-excerpt focus |
| Voice | Verbatim | Quotes user phrases within narrative |
| Compression | 4 lines | 1 paragraph, tighter |
| Narrative | List of excerpts | Coherent conclusion statement |

### 2. "what did I learn about Marc" -- VALIDATED

**Hybrid:**
> 2025-02-07: You asked, "How should I open the meeting with Marc and Craig?"
> 2025-02-12: You asked, "Give me additional suggestions to make marc less
> intimidating to talk to, based on his profile." -- indicates you perceive Marc
> as "intimidating". 2025-02-18: You wrote, "I think you're being too vague
> about his responses. I need the specifics to support every claim."

| Dimension | Deterministic | Hybrid |
|-----------|--------------|--------|
| Grounding | 4 direct quotes | Quotes with inline interpretation |
| Chronology | 4 dated bullets | 3 dated entries, chronological |
| Voice | Verbatim | Quotes user's exact words |
| Compression | 4 lines | 3 entries with annotations |
| Narrative | List | Annotated chronology showing preparation arc |

### 3. "what happened with Benz" -- VALIDATED

**Hybrid:**
> 2025-07-13: "I was thinking of sending Benz a message, asking if she wants to
> play pool. She's a lovely girl..." 2026-01-23: "I stopped at the bar to play
> pool with Benz. She was already playing with some old English guy..."
> Summary: You considered messaging Benz (2025-07-13) and later went to play
> pool with her (2026-01-23). The record stops after "put the coin."

| Dimension | Deterministic | Hybrid |
|-----------|--------------|--------|
| Grounding | 4 direct quotes | Quotes + summary anchored to evidence |
| Chronology | 4 dated bullets | Dated entries with explicit timeline summary |
| Voice | Verbatim | Quotes user's words, summarizes in second person |
| Compression | 4 lines | Compact timeline + summary |
| Narrative | List | Story arc with explicit gaps noted |

### 4. "what was my path to shadow work" -- VALIDATED

**Hybrid:**
> Chronological excerpts: 2025-11-24 -- "We're having an argument, I felt like
> she wasn't hearing me." 2025-12-08 -- "I believe the universe is essentially
> pixelated..." 2025-12-10 -- "I embraced stoicism to help me understand what
> was happening to me. That's where I learned that I could control my own mind."
> Together: (1) interpersonal conflict, (2) philosophical questioning, (3) travel
> and practice. The excerpts are truncated and do not fully connect.

| Dimension | Deterministic | Hybrid |
|-----------|--------------|--------|
| Grounding | 4 direct quotes | Quotes + thematic grouping |
| Chronology | 4 dated bullets | 3 dated entries + thematic summary |
| Voice | Verbatim | Quotes key user phrases |
| Compression | 4 lines | Compressed with thematic structure |
| Narrative | List | Thematic arc with explicit incompleteness |

### 5. "what was my thinking about the villa group" -- FALLBACK

The LLM output for this query non-deterministically introduces surface forms
that fail validation (varies by run). When it passes, the hybrid produces a
strong mixed-feelings narrative. The 80% validation rate reflects inherent
LLM non-determinism, not a systematic validator issue.

## Aggregate Assessment

### Hybrid advantages over deterministic

1. **Narrative coherence** -- Hybrid weaves excerpts into stories rather than
   listing bullets. The Benz answer shows a clear timeline arc.
2. **Thematic organization** -- Shadow work answer groups by theme (conflict,
   philosophy, travel) while preserving chronology.
3. **Explicit gap acknowledgment** -- Hybrid notes when excerpts are truncated
   or incomplete, rather than letting bullet points imply completeness.
4. **Inline annotation** -- Marc answer annotates what each excerpt reveals
   ("indicates you perceive Marc as intimidating") alongside the quote.

### Deterministic advantages over hybrid

1. **100% reliability** -- No API dependency, no validation failures, no
   non-determinism.
2. **Voice purity** -- User's exact words with zero paraphrase.
3. **Coverage guarantee** -- All 4 evidence items always appear.
4. **Predictability** -- Same input always produces same output.

### Hybrid advantages over old --llm (rewrite) path

1. **Voice preservation** -- Hybrid quotes user phrases; rewrite paraphrased everything.
2. **Chronological structure** -- Hybrid includes dates; rewrite dropped all dates.
3. **Second-person address** -- Hybrid says "you wrote"; rewrite said "the speaker."
4. **Status alignment** -- Hybrid doesn't under-claim; rewrite consistently hedged.

## Recommendation

**Deterministic remains the default. Hybrid is the recommended optional path.**

- `--hybrid` should replace `--llm` as the recommended LLM-augmented mode.
- `--llm` (rewrite mode) is preserved for backward compatibility but is not
  recommended -- it erases voice and drops chronology.
- The 80% validation rate means hybrid will fall back to deterministic ~1 in 5
  queries due to LLM non-determinism. This is acceptable because the fallback
  is the deterministic answer, which is always correct.
- Do not make hybrid the default until the validation rate is consistently >95%.

## Changes made

1. **Entity validator redesigned** -- Split into `_entity_surface_forms_permissive()`
   (for allowed set) and `_entity_surface_forms_strict()` (for answer check,
   mid-sentence only). Added `_NON_ENTITY_WORDS` set and possessive normalization.
2. **Fallback logging** -- `_apply_optional_llm_synthesis` logs WARNING on fallback.
3. **Hybrid prompt** -- `_prompt_instructions_hybrid()` instructs voice/date/boundary
   preservation. Evidence payload includes `author_role` in hybrid mode.
4. **`--hybrid` CLI flag** -- Wired through `answer_query()` to `LLMSynthesisRequest.hybrid`.
5. **6 new tests** -- Entity extraction, possessive handling, hybrid flag passthrough,
   fallback logging.
