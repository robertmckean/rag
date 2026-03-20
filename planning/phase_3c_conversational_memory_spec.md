# Phase 3C — Evidence Scoring
## Conversational-Memory Mode Design Spec (v2)

---

## 1. Definition

Conversational-memory mode is an optional answering mode for memory-style queries where the relevant concept is distributed across multiple nearby messages within a single retrieved conversation window.

It is:
- A bounded qualification-layer enhancement
- Grounded in existing retrieval windows
- Limited to one conversation window at a time
- Conservative in status assignment
- Designed for narrative recall and conversational memory queries

It is not:
- A retrieval redesign
- Cross-window or cross-conversation composition
- Semantic inference or entity resolution
- A replacement for Phase 3A strict grounding

Core principle:
- The retrieval window becomes the atomic unit of compositional support

---

## 2. Activation Model

- CLI: `--grounding-mode conversational_memory`
- API: `grounding_mode="conversational_memory"`

Default:
- Phase 3A strict mode remains default

Behavior:
- Explicit opt-in only
- No automatic switching

---

## 3. Evidence Model

- Retrieval windows are treated as atomic support candidates
- Individual excerpts remain visible but qualification occurs at the window level

Definitions:
- meaningful query terms = `scoring_terms` from ParsedQuery (or stopword-filtered equivalent)
- window_support_spans = excerpts contributing to support
- window_coverage_terms = union of scoring_terms across contributing excerpts

Atomic support unit:
- “This window jointly supports part of the query”

---

## 4. Composition Rules

### Allowed
- Only within a single retrieved window
- Only within one conversation
- Only using excerpts with direct lexical matches to scoring_terms
- Only when excerpts jointly cover query concepts
- Must include at least one user-authored excerpt

### Forbidden
- Cross-window composition
- Cross-conversation composition
- Use of assistant text as sole bridging evidence
- Composition without direct lexical grounding
- Inferring relationships not explicitly grounded

### Locality Constraints
- Same conversation: required
- Same window: required
- Sequence distance: bounded by retrieval window

---

## 5. Window Qualification Algorithm

For each retrieved window:

1. Identify all excerpts with direct matches to scoring_terms
2. Compute union of scoring_terms across those excerpts
3. Compute coverage ratio:
   coverage = matched_terms / total_scoring_terms

4. Window qualifies if:
   - coverage ≥ 0.75 (configurable 0.7–0.8)
   - at least one excerpt is user-authored
   - at least two excerpts contribute when no single excerpt satisfies strict support

---

## 6. Status Mapping

### supported
- One excerpt independently satisfies strict Phase 3A rules
- Other excerpts may add context only

### partially_supported
- Support requires composition across multiple excerpts in one window
- Default outcome for conversational-memory mode

### ambiguous
- Conflicting interpretations within or across windows

### insufficient_evidence
- No qualifying window

Rule:
- Distributed composition alone can NEVER produce supported

---

## 7. Interaction with Pipeline

### Selection
- No change

### Qualification
- Adds window-level qualification logic

### Answer Generation
- Must explicitly state when support is composed
- Must cite contributing excerpts
- Must expose status reasoning

---

## 8. Diagnostics / Scoring Output

Each result must include:

- support_basis: single_excerpt | window_composed
- composition_used: true/false
- supporting_excerpt_count
- user_excerpt_count
- assistant_excerpt_count
- coverage_terms
- coverage_ratio
- window_id
- status_reason

---

## 9. Relationship to Query Shaping

- Query shaping is complementary
- Used to refine retrieval or inspect components
- Must NOT be used to artificially construct support

---

## 10. Risks & Mitigations

### False Positives
Mitigation:
- Same-window restriction
- Lexical grounding requirement
- Coverage threshold
- No supported from composition

### Assistant Leakage
Mitigation:
- Require user-authored excerpt
- Assistant cannot bridge gaps alone

### Over-composition
Mitigation:
- No cross-window composition
- Strict term coverage rules

### User Over-trust
Mitigation:
- Explicit disclosure of composed support
- Full citation transparency

---

## 11. Non-Goals

- No change to default Phase 3A behavior
- No semantic inference or LLM-based grounding
- No cross-conversation reasoning
- No entity resolution
- No automatic mode switching

---

## 12. Minimal Implementation Policy

1. Opt-in only
2. Same-window composition only
3. Coverage-based qualification (≥75%)
4. At least one user-authored excerpt required
5. Composition maps to partially_supported
6. supported requires single-excerpt grounding
7. Full diagnostics exposed

---

## Summary

This design introduces a controlled, auditable mechanism for handling distributed conversational evidence while preserving the strict guarantees of Phase 3A.

It expands capability without weakening trust.
