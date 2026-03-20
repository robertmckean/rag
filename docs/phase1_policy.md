# Phase 1 Policy

## Included Data

Included in phase 1 canonical streams:
- ChatGPT conversations from `conversations-*.json`
- Claude conversations from `conversations.json`
- canonical conversation records
- canonical message records
- normalized timestamps
- provider-qualified canonical IDs
- message `content_blocks`
- derived plain-text `text`
- attachment and file references only
- minimal allowlisted `source_metadata`

## Excluded Data

Excluded from phase 1 canonical streams:
- embeddings
- retrieval indexes
- vector stores
- search layers
- copied attachment blobs
- ChatGPT sidecar streams as conversation/message data
- Claude `memories.json`
- Claude `projects.json`
- Claude `users.json` as conversation/message data
- full provider settings/profile payloads
- broad provider metadata dumps outside the allowlist

## ChatGPT Visible-Chain Policy

ChatGPT message order is reconstructed from the visible transcript only:
- walk each conversation's `current_node` ancestor chain
- reverse that chain to produce forward transcript order
- include only nodes on that visible chain
- exclude branch nodes not on the visible chain
- exclude nodes where `message = null` from canonical messages

This is the phase-1 deterministic linearization policy.

## Claude Sidecar Exclusion Policy

Claude canonical conversation/message streams are built from:
- `data/raw/claude/History_Claude/conversations.json`

These files are excluded from canonical conversation/message streams:
- `memories.json`
- `projects.json`
- `users.json`

They are not merged into `conversations.jsonl` or `messages.jsonl`.

## Attachment Handling Policy

Phase 1 preserves references only:
- preserve attachment identifiers and lightweight path hints when available
- do not copy blobs into normalized outputs
- do not inline binary content into canonical records

## Low-Signal Structural Records

Phase 1 canonical output may preserve low-signal structural records, including:
- empty ChatGPT system messages
- empty ChatGPT tool messages
- empty shell messages where the raw provider export contains no meaningful text payload

This is expected phase-1 behavior.

These records are preserved because phase 1 prioritizes:
- source fidelity
- deterministic normalization
- provider provenance

Retrieval-oriented filtering is intentionally deferred to a later phase.

## PII Minimization Policy

Phase 1 retains only the minimal provenance needed for normalization correctness:
- provider-qualified IDs
- source conversation/message IDs
- limited allowlisted metadata
- Claude `account_uuid` only where already included in the conversation metadata allowlist

Phase 1 excludes:
- full user profiles
- emails
- phone numbers
- settings payloads
- unrelated sidecar account data

## Source Metadata Allowlist

Conversation-level allowlist:
- ChatGPT:
  - `conversation_origin`
  - `is_archived`
  - `is_starred`
  - `is_read_only`
  - `is_do_not_remember`
  - `is_study_mode`
  - `default_model_slug`
- Claude:
  - `account_uuid`

Message-level allowlist:
- ChatGPT:
  - `content_type`
  - `channel`
  - `status`
- Claude:
  - `flags`
