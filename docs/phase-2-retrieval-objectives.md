# Phase 2 — Retrieval Objectives

## Questions to Support
* Find conversations relevant to a user query, using the normalized export outputs as the source of truth.
* Find individual messages relevant to a user query when the useful content is narrower than the full conversation.
* Retrieve prior context for a topic, project, person, decision, problem, date/time, or recurring line of thought.
* Support queries that ask for direct recall, such as “find where I talked about X” or “show prior discussion of Y.”
* Support queries that combine topic and time, such as “show discussions about burnout from late 2025” or “find resume-related conversations from March.”
* Support queries that combine topic and source, such as “find Claude conversations about RAG architecture.”
* Support queries that seek actionable prior content, such as decisions made, plans proposed, next steps, constraints, or conclusions.
* Support chronological exploration of a topic across multiple conversations.
* Support retrieval for downstream synthesis, where a later layer may summarize or analyze returned results.
* Return enough metadata with each result to let a human or later pipeline understand why it was retrieved and where it came from.
* Preserve traceability from every retrieved result back to the canonical normalized records.
* Provide chronologically ordered retrieval of all discussions within a specific date or date range, either generalized or topic-filtered.

## Retrieval Consumers
* Human browsing is a first-class consumer of retrieval results.
* LLM augmentation is a first-class consumer of retrieval results.
* Human browsing needs results that are interpretable, inspectable, and easy to trace back to original conversations and messages.
* LLM augmentation needs results that are compact, relevant, and structurally consistent enough to be injected as context.
* Phase 2 should support both consumers without creating separate retrieval systems.
* The retrieval layer should return stable, explicit records that either a UI or an LLM-oriented formatter can consume later.

## Success Criteria
* A relevant query should return genuinely useful results near the top of the result set.
* Top results should usually include the right conversation, the right message, or both, depending on the query.
* Retrieval should favor usefulness over raw volume; a small number of highly relevant results is better than a long noisy list.
* Results must preserve enough surrounding context to be understandable, especially when a single message is returned.
* Retrieved records must include stable identifiers and source metadata so results are debuggable and reproducible.
* The retrieval contract must work directly against the normalized Phase 1 outputs without requiring Phase 1 changes.
* Retrieval quality should support both exact/obvious matches and semantically related matches later, but Phase 2 may begin with lexical and metadata-driven relevance.
* The design should allow ranking to improve incrementally without changing the retrieval contract.
* The system should support metadata filtering cleanly, including source platform, conversation, message role, and timestamp-derived constraints.
* Retrieval latency should be reasonable for local interactive use; exact targets can be defined later, but the design should assume responsive human-facing workflows rather than offline batch-only execution.
* The system should make it obvious why a result was returned, at least through transparent fields and inspectable ranking inputs.
* Good retrieval means results are useful for the next step: reading, reviewing, summarizing, or injecting into an LLM prompt.

## Out of Scope (Phase 2)
* No changes to the Phase 1 normalization contract unless a hard blocker is discovered.
* No UI implementation.
* No clustering, topic modeling, or thought-trajectory visualization.
* No summarization or insight-generation layer.
* No sentiment, psychology, journaling, or behavioral-analysis features.
* No final product packaging decisions.
* No advanced personalization logic.
* No agentic workflows on top of retrieval.
* No embedding pipeline implementation unless explicitly added in a later slice.
* No vector database dependency in the initial retrieval contract.
* No attempt to solve all ranking quality problems in the first implementation slice.
* No irreversible storage or indexing decisions that prevent later hybrid retrieval.
* No commercial product features beyond what is required to support the core retrieval architecture.