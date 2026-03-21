# Known Limitations

## Current Retrieval Boundary

Current retrieval in this repository now has three channels over a single
normalized run:

- BM25 lexical retrieval
- semantic retrieval over stored message embeddings
- hybrid retrieval that unions both

BM25 remains the lexical baseline. Semantic retrieval broadens recall, but the
answer layer still requires strict grounding.

That means retrieval depends on direct vocabulary overlap between the query and
the normalized message text.

## Lexical Mismatch Failure Mode

BM25 fails when the query vocabulary differs from the source wording, even when
the underlying topic is relevant.

Example:

- Query: `What have I said about Larry's guitar playing?`
- Retrieved evidence in the real run includes messages about Larry and
  "playing the bass"
- The query term `guitar` does not appear in those candidate excerpts
- Result: no strict or conversational-memory qualification succeeds

This is a true lexical-retrieval limitation, not a grounding bug.

## Why This Matters

The current answer pipeline is behaving correctly when it refuses to claim
support for a concept that is not lexically grounded in the retrieved evidence.

The system can now explain this failure mode explicitly with
`rag.cli.answer --debug-qualification`, but it cannot solve the mismatch with
BM25 tuning alone.

## Conclusion

This limitation is not expected to be solved by more BM25 tweaking, ranking
adjustments, or answer-layer prompt changes alone.

Phase 4A addresses it by adding embeddings as a second retrieval channel.
Phase 4B adds retrieval quality filters: a minimum token count for semantic
candidates (excluding ultra-short fragments) and focal-visible window dedup
(preventing near-duplicate overlapping windows from dense threads).

Hybrid retrieval quality still depends on model choice, artifact coverage, and
future evaluation work. See [retrieval_guidance.md](retrieval_guidance.md) for
channel recommendations, query benchmarking guidance, and remaining quality gaps.
