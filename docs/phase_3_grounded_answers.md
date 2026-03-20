# Phase 3 — Grounded Answers

## Objective

Build the first grounded answer pipeline on top of Phase 2 retrieval.

The system must allow a user to ask a natural-language question against a normalized run and receive:

- a direct answer  
- an explicit answer status  
- supporting citations  
- visible gaps or conflicts  
- structured JSON output  
- readable terminal output  

This phase is **not** a chatbot. It is a deterministic, evidence-bounded answering layer.

---

## Core Design Rule

**Retrieval determines available evidence.  
Answering determines how that evidence is summarized.**

The answer layer must not invent new retrieval semantics.

---

## Scope

### In Scope

- Answer CLI  
- Answer data models  
- Evidence selection from retrieval results  
- Deterministic answer-status classification  
- Deterministic/template-based answer generation  
- Citation assembly  
- Terminal rendering  
- JSON output and `--json-out`  
- Focused unit and integration tests  

### Out of Scope

- LLM integration (Phase 3B)  
- Database or vector store  
- UI  
- Cross-run federation  
- Timeline answer mode  
- Autonomous follow-up questions  
- Retrieval redesign  
- Answer verification passes  
- Token budgeting beyond simple caps  

---

## CLI

### Command

```bash
python -m rag.cli.answer \
  --run-dir data/normalized/runs/<run_id> \
  --query "What have I said about burnout?" \
  --retrieval-mode relevance \
  --grounding-mode strict \
  --limit 8 \
  --max-evidence 5
```

### Arguments

Required:
- `--run-dir`
- `--query`

Optional:
- `--retrieval-mode relevance|newest|oldest` (default: relevance)
- `--grounding-mode strict|conversational_memory` (default: strict)
- `--limit`
- `--max-evidence` (default: 5)
- `--json`
- `--json-out`

---

## Architecture

```
src/rag/answering/
  models.py
  answer.py

src/rag/cli/
  answer.py
```

---

## Phase 3A Success Criteria

- grounded answers produced  
- explicit status classification  
- citations are correct and traceable  
- conversational-memory composition is explicit opt-in and stays within one retrieved window  
- no unsupported claims  
- ambiguity and gaps surfaced  
- tests pass  

---

## Future (Phase 3B)

- LLM-backed generation  
- token budgeting  
- grounding prompt contracts  
- optional verification  
