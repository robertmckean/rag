# v0.6.0 - Phase 6: Narrative Reconstruction

Date: 2026-03-21

## Implements

- adds grounded narrative reconstruction layer (`src/rag/narrative/`) that
  builds structured timelines from retrieved evidence without requiring
  Phase 5 AnswerResult
- groups evidence into phases requiring both temporal proximity and topic
  coherence — same-date evidence with different topics gets separate phases
- detects transitions between phases prioritizing explicit topic change,
  stated change, and temporal discontinuity over sentiment shift
- separates gaps (missing temporal/coverage continuity) from limitations
  (truncated excerpts, null timestamps, ambiguous ordering)
- deterministic phase labeling using date range + dominant entities
- configurable `phase_window_days` (default 7) and `gap_threshold_days`
  (default 30) as builder parameters and CLI flags
- all transitions carry `evidence_ids` for auditability
- all transitions are `partially_supported` by rule; single-evidence phases
  are `supported`, multi-evidence phases are `partially_supported`
- adds narrative CLI (`python -m rag.cli.narrative`) with `--format json`,
  `--format text`, and `--format debug` output modes
- adds `NarrativeReconstruction` schema with `NarrativePhase`,
  `NarrativeTransition`, `NarrativeGap`, and `NarrativeLimitation`

## Schema

```
NarrativeReconstruction:
  query: str
  summary: str
  timeline: [NarrativePhase]
  transitions: [NarrativeTransition]
  gaps: [NarrativeGap]
  limitations: [NarrativeLimitation]
  evidence_count: int
```

## Benchmark review (5 queries, manual inspection)

- Butters: 10 phases, 73-day gap flagged, Steve/Butters identity grouped
- Marc: 8 phases, Craig meeting grouped across 3 days, Benz denial separate
- Benz: 8 phases, 101-day and 171-day gaps flagged, Instagram phase distinct
- shadow work: 10 phases, Review → Shadow → Core Wound → Zeno progression
- villa group: 11 phases, 4 temporal gaps flagged across 10-month span

All 5 passed: no wrong groupings, no fabricated transitions, no misleading
summaries, gaps and limitations surfaced where expected.

## Validation

- PYTHONPATH=src python -m unittest discover -s tests -p "test_*.py"
- 174 tests pass, zero failures (39 new narrative tests)
- No existing files modified — clean new layer
