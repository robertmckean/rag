# Roadmap: Current State → Manifesto

This is the authoritative checklist. Every item is grounded in the current
codebase and traced to a specific manifesto requirement.

---

## Current state (verified 2026-03-21)

What works:

- [x] Normalization with `author_role` on every message (Phase 1)
- [x] BM25 + semantic + hybrid retrieval (Phase 2-4)
- [x] User voice boost (1.25x) and assistant downweight (0.8x) in ranking
- [x] Assistant meta-commentary suppression (0.48x) in ranking
- [x] Speculative assistant content filtered from evidence
- [x] Deterministic grounded answers with support levels (Phase 3A)
- [x] Constrained LLM synthesis (Phase 3B)
- [x] Narrative reconstruction: phases, transitions, gaps (Phase 6)
- [x] Recurring entity extraction with alias normalization (Phase 7A)
- [x] Content snippets centered on entity mentions (Phase 7B)
- [x] Deterministic topic clustering via term-overlap (Phase 7C)
- [x] Entity noise suppression via expanded denylist (Phase 7D)
- [x] Cross-cluster entity links and temporal burst detection (Phase 8)
- [x] Deterministic query routing with 5 intent types (Phase 9)
- [x] Entity-scoped query routing via ENTITY_SCOPED intent (Phase 10A)
- [x] 366 tests passing

What does not work:

- [x] ~~Cannot scope queries to a specific entity~~ → entity-scoped routing works (Phase 10A)
- [ ] Cannot track how thinking about a topic evolved across time
- [ ] Cannot detect contradictions or reversals in past positions
- [ ] Cannot chain narratives across multiple topics/queries
- [ ] Cannot surface forgotten conclusions
- [ ] Cannot distinguish user experience from user-relayed explanation
- [ ] No dedup of assistant restatements of user content
- [ ] Narrative construction does not prefer user content over assistant content

---

## Phase 10 — Entity-scoped queries

Manifesto question: "What did I conclude about a person, relationship, or project?"

Prerequisite: none. Highest leverage immediate improvement.

### 10A: Entity scope in router

- [x] Add ENTITY_SCOPED intent to router (keyword: entity name + "what/how/when")
- [x] Detect the target entity name from the query string by matching against
      PatternReport.entities names (case-insensitive)
- [x] Filter PatternReport.entities to the target entity
- [x] Filter PatternReport.clusters to those containing the target in key_entities
- [x] Filter entity_cluster_links to the target entity
- [x] Filter temporal_bursts to those whose entities include the target
- [x] Return a scoped answer grounded only in the filtered data
- [x] Add `--answer "what happened with Marc"` test on real corpus
- [x] Add unit tests: entity detected, correct filtering, unknown entity fallback

### 10B: Entity scope in narrative

- [ ] Add entity filtering to narrative phases (include phase only if entity
      appears in its description via entity_terms_from_text)
- [ ] Produce a scoped timeline: only phases mentioning the target entity
- [ ] Wire scoped narrative into the router ENTITY_SCOPED answer formatter
- [ ] Add test: scoped timeline excludes unrelated phases
- [ ] Add test: scoped timeline preserves chronological order
- [ ] Validate on real corpus: `--answer "what happened with Marc"` shows
      Marc-only phases, not the full timeline

---

## Phase 11 — Assistant restatement dedup

Manifesto principle: "Duplicate evidence is not stronger evidence. Repeated
assistant restatements should not overwhelm distinct, primary support."

Prerequisite: none. Can run in parallel with Phase 10.

- [ ] Audit real corpus for assistant messages that closely restate a user
      message from the same conversation window
- [ ] Define restatement detection: assistant message whose content overlaps
      with a user message in the same window above a threshold (e.g., 60%
      token overlap after lowercasing and stopword removal)
- [ ] Add `is_assistant_restatement(assistant_text, user_texts)` helper in
      `src/rag/retrieval/types.py`
- [ ] Apply in evidence ranking: multiply restatement score by 0.5 so they
      rank below the original user message
- [ ] Add unit tests: restatement detected, non-restatement preserved,
      threshold boundary cases
- [ ] Validate on real corpus: check that assistant restatements drop below
      the user original in rankings

---

## Phase 12 — Narrative voice preference

Manifesto principle: "Prioritizes my lived experience, decisions, and thought
process over assistant commentary."

Prerequisite: Phase 11 (restatement dedup prevents double-counting).

### 12A: Role-aware phase descriptions

- [ ] In `builder.py` phase description construction, when building the
      pipe-separated excerpt list, place user-role excerpts before
      assistant-role excerpts within each phase
- [ ] When selecting the phase label, prefer user-authored excerpt text
      over assistant-authored excerpt text
- [ ] Add test: phase with mixed user/assistant evidence shows user content
      first in description
- [ ] Add test: phase with only assistant evidence still produces valid output

### 12B: Role-aware narrative summary

- [ ] In `_generate_summary`, count user-sourced vs assistant-sourced phases
- [ ] Include "X of Y phases grounded in user content" in the summary string
- [ ] Add test: summary reflects correct user/assistant ratio

---

## Phase 13 — Longitudinal thinking evolution

Manifesto question: "How did my thinking evolve over time?"

Prerequisite: Phase 10 (entity-scoped queries to focus on a topic).

### 13A: Position extraction (deterministic v1)

- [ ] Define stance markers: "I think", "I believe", "I decided", "I realized",
      "I concluded", "my view is", "I was wrong about", "I changed my mind",
      "I no longer", "I used to think"
- [ ] Build `extract_positions(phase_description, role_filter="user")` that
      finds sentences containing stance markers in user-role segments only
- [ ] Position model (frozen dataclass): `Position(text: str, date: str | None,
      entity: str | None, evidence_id: str, stance_marker: str)`
- [ ] Add test: stance marker detected in user text
- [ ] Add test: assistant text with same markers is excluded
- [ ] Add test: no false positives on non-stance sentences
- [ ] Add test: multiple positions extracted from one phase

### 13B: Temporal comparison

- [ ] For a given entity/topic, collect positions across all phases sorted by date
- [ ] Detect shifts: when the same entity has positions with different
      sentiment-bearing terms or negation of prior terms across time periods
- [ ] Build `ThinkingEvolution(entity: str, positions: tuple[Position, ...],
      shifts: tuple[str, ...])` frozen dataclass
- [ ] Add test: stable positions across time show no shifts
- [ ] Add test: contradictory positions across time show a shift
- [ ] Add test: positions without dates handled gracefully (sorted last)

### 13C: Router integration

- [ ] Add EVOLUTION intent to router (keywords: "evolve", "evolved",
      "thinking", "path", "journey", "develop", "developed", "grew")
- [ ] Wire evolution extraction into answer formatting
- [ ] Validate on real corpus: `--answer "how did my thinking about Marc change"`
- [ ] Assess: does the output show real progression or just restated data?

---

## Phase 14 — Contradiction and change detection

Manifesto question: "Where did I contradict myself, change, grow, or become clearer?"

Prerequisite: Phase 13A (position extraction).

### 14A: Contradiction signals (deterministic v1)

- [ ] Define contradiction heuristics:
      - Explicit self-correction: "I was wrong", "I changed my mind",
        "I no longer think"
      - Opposing stance markers on same entity across time:
        "I trust X" → "I don't trust X"
      - Sentiment reversal: positive stance terms → negative stance terms
        for same entity
- [ ] Build `detect_contradictions(positions: list[Position]) -> list[Contradiction]`
- [ ] Contradiction model (frozen dataclass): `Contradiction(entity: str,
      earlier: Position, later: Position, signal: str, date_range: str)`
- [ ] Add test: explicit self-correction detected
- [ ] Add test: stable repetition not flagged as contradiction
- [ ] Add test: different topics for same entity not conflated

### 14B: Change classification

- [ ] Classify each detected change as one of:
      - `reversal` — direct contradiction of prior position
      - `softening` — weaker version of prior position
      - `strengthening` — stronger version of prior position
      - `evolution` — shift to different but not contradictory position
- [ ] Add `change_type: str` field to Contradiction model
- [ ] Add test: reversal vs softening vs strengthening correctly classified
- [ ] Add test: classification is deterministic

### 14C: Router integration

- [ ] Add CONTRADICTION intent to router (keywords: "contradict",
      "contradiction", "reversed", "grew", "growth", "clearer", "changed mind")
- [ ] Wire contradiction detection into answer formatting
- [ ] Validate on real corpus: do detected contradictions feel real and grounded?

---

## Phase 15 — Multi-hop narrative chaining

Manifesto question: "What was my path to shadow work?"

Prerequisite: Phase 10 (entity scope), Phase 13 (longitudinal tracking).

### 15A: Cross-narrative phase linking

- [ ] Given a target topic, retrieve narratives from multiple related queries
- [ ] Determine related queries by finding entities that co-occur with the
      target topic in existing clusters (deterministic, no LLM)
- [ ] Limit expansion to max 5 related queries to avoid scope explosion
- [ ] Collect all phases across narratives that mention the target topic
      or related entities
- [ ] Sort by date to produce a unified cross-narrative timeline
- [ ] Deduplicate phases with identical evidence_ids

### 15B: Narrative chain model

- [ ] `NarrativeChain(topic: str, phases: tuple[ChainedPhase, ...],
      source_queries: tuple[str, ...], date_range: str | None)` frozen dataclass
- [ ] `ChainedPhase(phase_label: str, source_query: str, date: str | None,
      entities: tuple[str, ...], evidence_ids: tuple[str, ...])` frozen dataclass
- [ ] Serialization via `to_dict()` following repo convention
- [ ] Add test: phases from different queries merged into one chain
- [ ] Add test: duplicate phases deduplicated by evidence_ids
- [ ] Add test: chronological ordering preserved

### 15C: CLI and router integration

- [ ] Add `--chain-topic "shadow work"` flag to patterns CLI
- [ ] Add NARRATIVE_CHAIN intent to router (keywords: "path", "journey",
      "how did I get to", "what led to", "progression")
- [ ] Validate on real corpus: does the chain for "shadow work" produce a
      coherent path across conversations?
- [ ] Assess: does chaining reveal connections that single-query narrative misses?

---

## Phase 16 — Forgotten knowledge recovery

Manifesto question: "What did I already figure out that I've now forgotten?"

Prerequisite: Phase 13A (position extraction).

### 16A: Conclusion strength scoring

- [ ] Score positions by strength:
      - Explicit decision markers ("I decided", "I realized") score higher
        than observation markers ("I think", "I notice")
      - Positions with 2+ supporting evidence items score higher
      - User-role positions score higher than any assistant-role positions
- [ ] Build `score_conclusion_strength(position: Position, evidence_count: int) -> float`
- [ ] Define "strong conclusion" threshold
- [ ] Add test: strong conclusion scores higher than weak observation

### 16B: Recency gap detection

- [ ] For each strong conclusion, check whether its entity+topic combination
      appears in later phases (same entity + similar content terms)
- [ ] Flag conclusions that were stated once strongly but never revisited
      as "potentially forgotten"
- [ ] Build `ForgottenInsight(position: Position, last_seen: str,
      days_since: int, reinforcement_count: int)` frozen dataclass
- [ ] Add test: reinforced conclusion not flagged as forgotten
- [ ] Add test: unreinforced old conclusion flagged as forgotten
- [ ] Add test: recent conclusion not flagged regardless of reinforcement

### 16C: Router integration

- [ ] Add FORGOTTEN intent to router (keywords: "forgot", "forgotten",
      "already knew", "figured out", "used to know", "rediscover")
- [ ] Format answer showing forgotten conclusions with dates and evidence
- [ ] Validate on real corpus: are the surfaced conclusions genuinely
      worth rediscovering, or is it noise?

---

## Phase 17 — Experience vs explanation separation

Manifesto distinction: "experience from explanation, my voice from assistant voice"

Prerequisite: Phase 12 (narrative voice preference), Phase 13A (position extraction).

### 17A: Content type classification (deterministic v1)

- [ ] Define heuristics for user-role messages:
      - `lived_experience`: first-person past tense narrative
        ("I went", "I felt", "I told him", "that happened when")
      - `reflection`: first-person present tense analysis
        ("I think", "I realize", "I believe", "it seems to me")
      - `instruction`: directive to assistant
        ("tell me", "explain", "help me", "can you")
      - `relayed_explanation`: quoting external advice
        ("they said", "the advice was", "according to")
- [ ] For assistant-role messages: always classify as `explanation`
- [ ] Build `classify_content_type(text: str, role: str) -> str`
- [ ] Add test: each content type correctly classified
- [ ] Add test: assistant text always classified as explanation
- [ ] Add test: ambiguous text defaults to reflection (conservative)

### 17B: Content type in evidence and rendering

- [ ] Add content_type field to EvidenceItem (or a parallel annotation)
- [ ] In narrative phase descriptions, annotate content type alongside role
- [ ] In pattern report rendering, show content type distribution per entity
      (e.g., "Marc: 4 experiences, 2 reflections, 1 explanation")
- [ ] Add test: rendering shows content types correctly
- [ ] Add test: empty content type handled gracefully

### 17C: Content type preference in synthesis

- [ ] In grounded answer synthesis, prefer `lived_experience` and `reflection`
      evidence over `instruction` and `relayed_explanation`
- [ ] Apply as a scoring multiplier (similar to voice boost) not a hard filter
- [ ] Add test: answer built from experience+reflection when available
- [ ] Add test: falls back to other types when no experience exists
- [ ] Validate on real corpus: do answers now feel more grounded in personal
      experience?

---

## Phase 18 — Regression protection and grounding enforcement

Manifesto principle: "If the system sounds good but is weakly grounded, it has failed."

Prerequisite: all prior phases complete.

### 18A: Benchmark query set

- [ ] Curate 10-15 real queries spanning all intent types:
      entity, theme, cross-topic, temporal, timeline, entity-scoped,
      evolution, contradiction, chain, forgotten
- [ ] For each query, record expected: intent, top entities, top clusters,
      answer structure, key assertions
- [ ] Store in `tests/fixtures/benchmark_queries.json`

### 18B: End-to-end integration tests

- [ ] Build `tests/integration/test_benchmark_queries.py`
- [ ] Run the full pipeline (retrieval → narrative → patterns → routing)
      for each benchmark query
- [ ] Assert: correct intent classification
- [ ] Assert: no hallucinated entities (every entity in answer exists in data)
- [ ] Assert: every claim traces to an evidence_id

### 18C: Output stability locks

- [ ] For 5 key benchmark queries, snapshot PatternReport.to_dict() as golden
      files in `tests/fixtures/golden/`
- [ ] Add regression test: current output matches golden file
- [ ] Document update procedure: regenerate golden files only with explicit
      justification in commit message

### 18D: Grounding audit mode

- [ ] Add `--audit` flag to patterns CLI
- [ ] Audit output: for each claim in the answer, list the supporting
      evidence chain (evidence_id → excerpt → source message)
- [ ] Verify no answer section lacks evidence references
- [ ] Add test: audit mode produces valid output for all intent types

---

## Phase ordering and dependencies

```
Phase 10 (entity scope)  ──────────────────┐
Phase 11 (restatement dedup)  ─────────┐   │
                                       v   v
Phase 12 (narrative voice)  ───> Phase 13 (longitudinal)
                                       │
                                       v
                                 Phase 14 (contradictions)
                                       │
Phase 10 + Phase 13 ──────────> Phase 15 (multi-hop chaining)
Phase 13A ─────────────────────> Phase 16 (forgotten knowledge)
Phase 12 + Phase 13A ──────────> Phase 17 (experience vs explanation)
Phases 10-17 ──────────────────> Phase 18 (regression protection)
```

Phases 10 and 11 can run in parallel. Everything else is sequential.

---

## What is explicitly out of scope

- Vector databases or external search infrastructure
- UI or visualization layer
- Cross-run retrieval or analysis
- Chatbot or conversational follow-up behavior
- LLM-based reranking or semantic rewriting
- Automatic alias discovery (explicit static map only)
- Performance optimization (not a current bottleneck)
- Documentation-only work (docs update alongside code changes only)
- Public onboarding or packaging (not the current goal)
- Multi-user support

---

## Manifesto traceability

| Manifesto question / principle | Phase |
|---|---|
| "What did I conclude about a person?" | 10 |
| "Duplicate evidence is not stronger evidence" | 11 |
| "Prioritizes my lived experience over assistant commentary" | 12 |
| "How did my thinking evolve over time?" | 13 |
| "Where did I contradict myself, change, grow?" | 14 |
| "What was my path to shadow work?" | 15 |
| "What did I already figure out that I've now forgotten?" | 16 |
| "What was advice, and what was lived experience?" | 17 |
| "If the system sounds good but is weakly grounded, it has failed" | 18 |
| "The system must remain inspectable" | 18D |

---

## How to use this document

1. Work top to bottom. Do not skip phases.
2. Each checkbox is one unit of work. Check it when done and tested.
3. Run the full test suite before starting each new phase.
4. Validate on real corpus at each phase boundary.
5. If a step proves wrong or unnecessary, strike it out with a note explaining
   why — do not silently delete it.
6. This document is the single source of truth for remaining work.
