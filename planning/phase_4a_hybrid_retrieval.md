# Phase 4A — Hybrid Retrieval

## Objective

Introduce semantic retrieval (embeddings) alongside existing BM25 retrieval to resolve vocabulary mismatch failures and improve recall of relevant evidence.

This phase establishes a **dual-channel retrieval system**:
* Lexical retrieval (BM25)
* Semantic retrieval (embeddings)

The goal is not perfect ranking or precision, but to prove that semantic retrieval can recover relevant results that BM25 cannot.

---

## Problem Statement

The current system relies entirely on BM25, which is based on lexical overlap.

This creates a hard limitation:

* Queries fail when the vocabulary used in the query does not match the vocabulary in the source text.

### Example Failure

* Query: "Larry guitar playing"
* Evidence: "I was playing the bass"

BM25 result:
* No match (no shared keywords)

Conclusion:
* This class of failure cannot be solved with ranking, thresholds, or query tuning.
* A semantic retrieval layer is required.

---

## Approach

### Dual Retrieval Strategy

For each query:

1. Run BM25 retrieval (existing system)
2. Run embedding-based semantic retrieval (new)
3. Merge results
4. Deduplicate
5. Return combined result set

---

## Retrieval Channels

### 1. BM25 (Existing)

* Input: raw query string
* Output: ranked list of results based on lexical similarity
* Strength: precision when keywords match
* Weakness: fails on vocabulary mismatch

---

### 2. Embeddings (New)

* Input: raw query string → embedding vector
* Corpus: precomputed embeddings for all messages (or chunks)
* Output: top-K results by vector similarity (cosine similarity)

* Strength: captures meaning and paraphrase
* Weakness: may return loosely related or noisy results

---

## Result Merging

Initial implementation should be simple and deterministic:

1. Combine BM25 and embedding result sets
2. Deduplicate by message_id
3. Preserve both scores:
   * bm25_score
   * embedding_score (cosine similarity)
4. Apply a simple combined ranking:
   * e.g. prioritize:
     * results appearing in both sets
     * then by highest individual score

No complex weighting or tuning in this phase.

---

## Expected Behavior Changes

### Before (BM25 only)

System behavior:
> "Find results that use the same words as the query"

### After (Hybrid)

System behavior:
> "Find results that mean the same thing as the query, even if phrased differently"

---

## Human-Testable Success Criteria

Evaluation is based on known failure cases, not metrics.

---

### Test 1 — Vocabulary Mismatch

Query:
> "Larry guitar playing"

Evidence:
> "I was playing the bass"

Expected:
* BM25: does not retrieve
* Hybrid: retrieves relevant result

---

### Test 2 — Paraphrase

Query:
> "discussion about burnout"

Evidence:
> "I’ve been feeling exhausted and unmotivated"

Expected:
* Hybrid retrieves relevant result

---

### Test 3 — Conceptual Similarity

Query:
> "issues with Marc and women"

Evidence:
* "Marc was being weird about girls"
* "he got possessive"

Expected:
* Hybrid surfaces relevant messages across phrasing differences

---

### Test 4 — False Positive Risk (Guardrail)

Query:
> "Marc behavior toward women"

Evidence (separate):
* "Marc showed up late"
* "we talked about women"

Expected:
* Hybrid may retrieve both
* Phase 3 grounding must NOT combine into a false claim

---

## Critical Principle

Hybrid retrieval increases recall, not correctness.

Correctness remains enforced by Phase 3 grounding.

> Retrieval can be noisy.
> Grounding must remain strict.

---

## Non-Goals (Phase 4A)

* No ranking optimization
* No score calibration
* No threshold tuning
* No reranking models
* No chunking strategy changes
* No database/vector index optimization

This phase is strictly about capability introduction.

---

## Data Requirements

### Embedding Corpus

Each message (or chunk) must have:

* message_id
* text content
* embedding vector

Embeddings must be generated once and stored for reuse.

---

## Implementation Outline

### Step 1 — Embedding Generation

* Select embedding model
* Generate embeddings for all messages
* Store results locally (file-based, consistent with Phase 2 design)

---

### Step 2 — Query Embedding

* Convert incoming query into embedding vector

---

### Step 3 — Similarity Search

* Compute cosine similarity between query vector and corpus
* Return top-K results

---

### Step 4 — Merge with BM25

* Combine BM25 and embedding results
* Deduplicate
* Attach both scores

---

### Step 5 — CLI Integration

Extend retrieval CLI:



Output should clearly indicate:
* source: bm25 / embedding / both
* scores for each

---

## Output Expectations

Each result should include:

* message_id
* text snippet
* bm25_score (if present)
* embedding_score (if present)
* source:
  * bm25
  * embedding
  * hybrid (if in both)

---

## Definition of Done

Phase 4A is complete when:

* Hybrid retrieval is implemented and callable via CLI
* Embeddings are generated and stored for the corpus
* Known failure cases (e.g. Larry/guitar) are resolved
* Results include both lexical and semantic matches
* Phase 3 grounding continues to behave correctly with expanded retrieval

---

## Strategic Outcome

Completion of Phase 4A transitions the system from:

* Keyword search

to:

* Semantic memory retrieval

This is a foundational capability required for all future improvements.