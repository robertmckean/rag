"""Run 10 new benchmark queries through the real pipeline and freeze as fixtures.

Each query is run through answer_query() to capture real retrieval + evidence,
then the evidence payload is serialized as a frozen benchmark fixture.
"""

from __future__ import annotations

import io
import json
import os
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from pathlib import Path

from rag.answering.answer import answer_query


RUN_DIR = Path("data/normalized/runs/combined-live-check").resolve()
FIXTURES_DIR = Path("tests/fixtures/phase5_benchmarks")

QUERIES = [
    # Weak evidence (2)
    "what did I think about meditation",
    "what was my opinion on cryptocurrency",
    # Conflicting evidence (2)
    "how did I feel about living in Thailand",
    "did I want to stay or leave",
    # Long-range chronology (2)
    "how did my social life change over the past year",
    "what was my journey with stoicism",
    # Generic-topic ambiguity (2)
    "what did I learn about myself",
    "what were my biggest decisions",
    # Technical / project-history (2)
    "what projects was I working on",
    "what did I discuss about AI",
]


def query_to_filename(query: str) -> str:
    return query.lower().replace(" ", "_").replace("?", "").replace("/", "_") + ".json"


def main() -> None:
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)

    for query in QUERIES:
        print(f"\n{'='*80}")
        print(f"QUERY: {query}")

        result = answer_query(
            RUN_DIR,
            query,
            retrieval_mode="relevance",
            grounding_mode="strict",
            limit=8,
            max_evidence=5,
        )

        print(f"  status: {result.answer_status.value}")
        print(f"  evidence: {len(result.evidence_used)}")
        print(f"  citations: {len(result.citations)}")

        # Build fixture payload matching existing format
        fixture_results = []
        for item in result.evidence_used:
            c = item.citation
            fixture_results.append({
                "provider": c.provider,
                "conversation_id": c.conversation_id,
                "conversation_title": c.title,
                "focal_message_id": c.message_id,
                "messages": [
                    {
                        "author_role": item.author_role,
                        "created_at": c.created_at,
                        "content_blocks": [
                            {"type": "text", "text": c.excerpt}
                        ],
                    }
                ],
            })

        fixture = {
            "query": query,
            "answer_status": result.answer_status.value,
            "evidence_count": len(result.evidence_used),
            "results": fixture_results,
        }

        fname = query_to_filename(query)
        out_path = FIXTURES_DIR / fname
        out_path.write_text(
            json.dumps(fixture, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"  fixture: {out_path}")

        # Print first 120 chars of answer for quick inspection
        preview = result.answer[:120].replace("\n", " ")
        print(f"  answer preview: {preview}...")

    print(f"\n{'='*80}")
    print(f"Done. {len(QUERIES)} fixtures written to {FIXTURES_DIR}")


if __name__ == "__main__":
    main()
