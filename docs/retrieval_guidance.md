# Retrieval Guidance

## Recommended Defaults

Hybrid is the recommended default retrieval channel for real queries.

As of v0.4.6, hybrid retrieval consistently outperforms BM25-only and
semantic-only on synthesis-oriented and autobiographical questions. It keeps the
lexical anchor on specific terms while adding semantic recall for related
concepts that share no vocabulary with the query.

Use `--channel hybrid` unless you have a specific reason to isolate one signal.

## Channel Characteristics

**BM25** — best when the query contains exact phrases that appear in the source
text. Strong for regression testing and targeted lookups. Brittle when query
vocabulary diverges from source wording.

**Semantic** — best for conceptual recall where the topic is present but the
exact words are not. Weaker at precision: high-similarity matches on short or
generic text can appear without strong grounding. The minimum token filter
(SEMANTIC_MIN_TOKEN_COUNT = 4) mitigates the worst of this.

**Hybrid** — unions BM25 and semantic candidates. Messages found by both
channels receive a ranking bonus. This is the strongest channel for questions
that span multiple conversations or require thematic synthesis.

## Query Benchmarking

Conceptual and developmental queries are more informative benchmarks than simple
named-entity lookups.

Entity queries like "What did Larry say about guitar?" are useful for regression
testing but do not stress the retrieval system meaningfully. They succeed or fail
based on exact vocabulary overlap and tell you little about ranking quality.

Queries like "what was my path to shadow work" or "how did I develop my self
reflection and Jungian shadow work" exercise all three retrieval dimensions:
vocabulary matching, semantic similarity, and result diversity across
conversations and providers.

Use conceptual queries as the primary quality benchmark going forward.

## Remaining Quality Gaps

- Trivial entity-match queries (e.g., "Should I avoid ice in Cambodia?") rank on
  keyword overlap without semantic depth. This is a lexical relevance boundary,
  not a retrieval bug — synthesis-layer evaluation is the right next step.
- Window context messages with null text appear as `text=None` in CLI output.
  These are system/tool messages that were normalized without displayable text.

## Resolved Quality Gaps (v0.4.6)

- Cross-provider dedup: near-duplicate messages sent to both ChatGPT and Claude
  are collapsed by comparing normalized focal text prefixes (100 chars).
- User-voice preference: user messages score 1.25x, assistant messages 0.8x,
  so the user's own words rank above assistant reactions.
- Assistant meta-commentary suppression: messages opening with process/reaction
  filler phrases receive an additional 0.6x penalty without being hard-filtered.
