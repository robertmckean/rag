# v0.7.1 — Content snippets and topic clustering

## Changes

### Content snippets (Phase 7B)
- Entity occurrence excerpts now show ~120-char content snippets centered on the
  entity mention instead of bare phase labels.
- Snippet extraction searches pipe-separated phase description segments, strips
  date/role prefixes, and falls back to the phase label when no mention is found.
- Alias variants are included in the snippet search so that Mark/Marc both match.

### Topic clustering (Phase 7C)
- Added `TopicCluster` model with label, phase_labels, evidence_ids, date_range,
  key_entities, key_terms, and phase_count fields.
- Deterministic agglomerative clustering via Jaccard term-overlap similarity
  (threshold 0.10) across narrative phases — no LLM or embedding dependencies.
- Query terms excluded from clustering similarity to prevent all-query-about-X
  phases from collapsing into one cluster.
- Description noise tokens (`user`, `assistant`, `unknown`) filtered from term
  sets and cluster labels.
- Entity names excluded from `key_terms` to avoid redundancy with `key_entities`.
- Clusters with 2+ phases emitted; ordered by phase_count desc then label asc.
- JSON and text renderers extended with cluster output section.

### Shared helpers
- Extracted `content_terms_from_text()` as a public helper in
  `rag.narrative.builder` for reuse by both narrative grouping and topic
  clustering (matches the earlier `entity_terms_from_text()` extraction).

## Test coverage
- 275 tests passing (30 new: 10 clustering, 7 snippet, 8 TopicCluster model,
  6 cluster rendering, minus 1 assertion correction).
