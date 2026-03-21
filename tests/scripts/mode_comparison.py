"""Compare deterministic vs hybrid answer modes on the 5 frozen benchmark queries.

Runs both modes through the real pipeline (not the test harness) and captures
structured output for side-by-side comparison across 6 dimensions.
"""

from __future__ import annotations

import io
import json
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from pathlib import Path

from rag.answering.answer import answer_query


RUN_DIR = Path("data/normalized/runs/combined-live-check").resolve()

QUERIES = [
    "what did I conclude about Butters",
    "what did I learn about Marc",
    "what happened with Benz",
    "what was my path to shadow work",
    "what was my thinking about the villa group",
]


def run_query(query: str, hybrid: bool) -> dict:
    result = answer_query(
        RUN_DIR,
        query,
        retrieval_mode="relevance",
        grounding_mode="strict",
        limit=8,
        max_evidence=5,
        llm=hybrid,
        hybrid=hybrid,
    )
    return {
        "query": query,
        "mode": "hybrid" if hybrid else "deterministic",
        "answer_status": result.answer_status.value,
        "synthesis_mode": result.synthesis_mode,
        "answer": result.answer,
        "evidence_count": len(result.evidence_used),
        "citation_count": len(result.citations),
    }


def main() -> None:
    results = []

    for query in QUERIES:
        print(f"\n{'='*80}")
        print(f"QUERY: {query}")

        # Deterministic
        print("  Running deterministic...")
        det = run_query(query, hybrid=False)
        results.append(det)

        # Hybrid
        print("  Running hybrid...")
        hyb = run_query(query, hybrid=True)
        results.append(hyb)

        print(f"\n--- DETERMINISTIC ({det['answer_status']}) ---")
        print(det["answer"])
        print(f"\n--- HYBRID ({hyb['answer_status']}, synthesis_mode={hyb['synthesis_mode']}) ---")
        print(hyb["answer"])

    # Summary
    print(f"\n{'='*80}")
    print("COMPARISON SUMMARY")
    print(f"{'='*80}")
    for query in QUERIES:
        det = next(r for r in results if r["query"] == query and r["mode"] == "deterministic")
        hyb = next(r for r in results if r["query"] == query and r["mode"] == "hybrid")
        print(f"\n{query}:")
        print(f"  det: status={det['answer_status']}, evidence={det['evidence_count']}, citations={det['citation_count']}, chars={len(det['answer'])}")
        print(f"  hyb: status={hyb['answer_status']}, evidence={hyb['evidence_count']}, citations={hyb['citation_count']}, chars={len(hyb['answer'])}, synthesis={hyb['synthesis_mode']}")

    # Write JSON for detailed inspection
    out_path = Path("tests/scripts/mode_comparison_output.json")
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nFull output written to {out_path}")


if __name__ == "__main__":
    main()
