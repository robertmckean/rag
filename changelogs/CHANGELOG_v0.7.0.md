# v0.7.0 - Phase 7: Recurring-Entity Pattern Extraction

Date: 2026-03-21

## Implements

- adds recurring-entity pattern extraction layer (`src/rag/patterns/`) that
  identifies named entities appearing across 2+ distinct narrative phases
- extracts entities using the same `entity_terms_from_text()` helper shared
  with Phase 6 narrative grouping — no duplication of extraction logic
- applies explicit alias normalization via static `ENTITY_ALIASES` map to
  merge known spelling variants (e.g. Mark → Marc) unconditionally
- alias map is authoritative: co-occurrence of variants in the same phase
  does not block the merge
- entities sorted deterministically by occurrence count desc, then name asc
- occurrences sorted by date asc, nulls last
- each occurrence grounded to phase evidence_id, date, and phase label
- adds pattern CLI (`python -m rag.cli.patterns`) accepting multiple
  `--queries` with `--format json` and `--format text` output modes
- builds narratives per query using existing Phase 6 pipeline, then runs
  cross-narrative entity extraction
- adds `PatternReport`, `RecurringEntity`, `EntityOccurrence` schema as
  frozen dataclasses with `to_dict()` serialization

## Schema

```
PatternReport:
  query: str
  entities: [RecurringEntity]
  evidence_count: int

RecurringEntity:
  name: str
  occurrences: [EntityOccurrence]
  occurrence_count: int

EntityOccurrence:
  evidence_id: str
  created_at: str | None
  excerpt: str
```

## Alias Map (seeded)

```
Mark → Marc
```

## Real corpus validation (3 queries)

```
python -m rag.cli.patterns \
  --run-dir data/normalized/runs/combined-live-check \
  --queries "What happened with Marc?" "What happened with Benz?" \
            "What was my experience with burnout?" \
  --format text
```

- Marc: 7 occurrences spanning 2025-02-04 to 2026-02-03
- Benz: 6 occurrences spanning 2025-03-06 to 2026-01-23
- Craig: 5 occurrences spanning 2025-02-04 to 2025-10-31
- "Mark" fully collapsed into "Marc" via alias normalization
- No false positives, no noise entities, deterministic output

## Validation

- PYTHONPATH=src python -m unittest discover -s tests -p "test_*.py"
- 245 tests pass, zero failures (71 new pattern tests)
- No existing files modified beyond builder.py shared helper extraction
