# v0.7.2 — Entity noise suppression, derived signals, query routing, roadmap

## Changes

### Entity extraction quality (Phase 7D)
- Expanded `_NON_ENTITY_WORDS` with ~200 common English words that appeared as
  false-positive entities at sentence/excerpt starts (verbs, adjectives, adverbs,
  prepositions, generic nouns, interjections).
- Merged `_LABEL_NOISE_WORDS` into the unified `_NON_ENTITY_WORDS` set.
- Simplified `entity_terms_from_text` to check a single filter set.
- Before/after validation on real corpus: 25 false positives eliminated,
  zero legitimate entities lost.

### Derived signals (Phase 8)
- Added `EntityClusterLink` model: entities bridging 2+ topic clusters.
- Added `TemporalBurst` model: periods with 3+ phases within a 7-day window.
- Extended `PatternReport` with `entity_cluster_links` and `temporal_bursts`.
- Extended text/JSON renderer with cross-cluster entity section, temporal burst
  section, and summary emphasis section.

### Query routing (Phase 9)
- Added `src/rag/patterns/router.py`: deterministic keyword-based intent
  classification with 5 intents (entity, theme, cross-topic, temporal, timeline)
  plus unknown fallback.
- Added `--answer` flag to `rag.cli.patterns` CLI for routed question answering.
- Answers are concise, grounded in PatternReport/Narrative data, no speculation.

### Project roadmap
- Added `TO_DO.md`: manifesto-aligned roadmap with 9 remaining phases,
  ~90 checkboxes, dependency graph, and manifesto traceability table.
- Added `docs/PROJECT_MANIFESTO.md`: project vision document.

## Test coverage
- 342 tests passing (67 new: 8 entity quality, 13 cross-cluster/burst extraction,
  19 derived signal rendering, 27 router intent/answer tests).
