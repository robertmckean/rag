# Known Limitations

## Current Retrieval Boundary

Current retrieval in this repository is BM25-based lexical retrieval over a
single normalized run.

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
adjustments, or answer-layer prompt changes.

The next retrieval milestone is hybrid retrieval:

- keep BM25 as the lexical channel
- add embeddings as a second semantic retrieval channel
- compare hybrid retrieval against this pure-BM25 baseline
