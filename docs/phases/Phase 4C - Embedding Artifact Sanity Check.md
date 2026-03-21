### Embedding Artifact Sanity Check
Run: `data/normalized/runs/combined-live-check/message_embeddings.jsonl`

Status: usable for retrieval

Validated:
* JSONL readable end-to-end
* 60,677 embedded records
* 60,677 unique message_ids
* eligible-message count matches embedded-record count exactly
* embedding dimension consistent at 1536
* model consistent: `text-embedding-3-small`
* no nulls, NaN, inf, all-zero vectors, or malformed lines
* random source-alignment checks passed
* filtering behavior matched current rules

Non-critical issues:
* `original_token_count` / `stored_token_count` appear to be character-count based, not tokenizer-based
* some repeated short texts remain embedded, producing repeated identical vectors as expected

Conclusion:
* safe to use for semantic/hybrid retrieval
* token metadata should be corrected in a future refactor
* filter tuning for low-information short prompts can be revisited later if retrieval noise becomes noticeable