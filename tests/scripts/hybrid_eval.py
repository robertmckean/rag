"""Classified evaluation of deterministic vs hybrid synthesis on Phase 5 benchmarks.

Runs both paths on each frozen benchmark fixture and classifies the hybrid output as:
  valid      — meets all property constraints
  degraded   — worse than deterministic (less grounded, missing dates, timeline loss)
  invalid    — violates constraints (new entities, citation overflow)
  equivalent — valid hybrid that restates deterministic content differently
  fallback   — LLM synthesis failed, fell back to deterministic
"""

from __future__ import annotations

import io
import json
import os
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from rag.answering.generator_llm import LLMSynthesisRequest, synthesize_answer_with_llm
from rag.answering.hybrid_validation import validate_hybrid_output
from rag.answering.models import AnswerStatus, Citation, EvidenceItem
from rag.answering.status import classify_answer_status, generate_answer_text


def _load_evidence(data: dict) -> tuple[EvidenceItem, ...]:
    items: list[EvidenceItem] = []
    for i, r in enumerate(data["results"][:5]):
        messages = r.get("messages", [])
        focal_msg = None
        for msg in messages:
            if msg.get("author_role") == "user":
                focal_msg = msg
                break
        if not focal_msg:
            focal_msg = messages[0] if messages else None
        if not focal_msg:
            continue
        excerpt = ""
        for block in focal_msg.get("content_blocks", []):
            if block.get("type") == "text" and block.get("text"):
                excerpt = block["text"][:200]
                break
        if not excerpt:
            continue
        citation = Citation(
            provider=r.get("provider", ""),
            conversation_id=r.get("conversation_id", ""),
            title=r.get("conversation_title", ""),
            message_id=r.get("focal_message_id", ""),
            created_at=focal_msg.get("created_at", ""),
            excerpt=excerpt,
        )
        items.append(
            EvidenceItem(
                rank=i + 1,
                source_result_rank=i + 1,
                window_id="",
                score=0.0,
                retrieval_mode="bm25",
                author_role="user",
                matched_terms=(),
                citation=citation,
            )
        )
    return tuple(items)


class _FakeQual:
    def __init__(self, evidence: tuple[EvidenceItem, ...]) -> None:
        self.evidence_items = evidence
        self.composition_used = False


class _FakeDiag:
    def __init__(self, count: int) -> None:
        self.selected_evidence_count = count
        self.qualified_evidence_count = count
        self.composition_used = False
        self.rejected_evidence = ()


def main() -> None:
    fixtures_dir = "tests/fixtures/phase5_benchmarks"
    fixtures = sorted(os.listdir(fixtures_dir))

    counts = {"valid": 0, "degraded": 0, "invalid": 0, "equivalent": 0, "fallback": 0}
    total = 0

    print("# Hybrid Synthesis Evaluation - Phase 5 Benchmarks")
    print()

    for fname in fixtures:
        path = os.path.join(fixtures_dir, fname)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        query = data["query"]
        evidence = _load_evidence(data)
        fixture_status_str = data.get("answer_status", "supported")
        fixture_status = AnswerStatus(fixture_status_str) if fixture_status_str in {s.value for s in AnswerStatus} else AnswerStatus.SUPPORTED
        total += 1

        # Deterministic path
        decision = classify_answer_status(query, _FakeQual(evidence))
        det_answer = generate_answer_text(decision, evidence, _FakeDiag(len(evidence)))

        # Hybrid path
        request = LLMSynthesisRequest(
            query=query,
            answer_status=fixture_status,
            evidence_items=evidence,
            gaps=(),
            conflicts=(),
            hybrid=True,
        )

        hybrid_answer = ""
        hybrid_citations: tuple[str, ...] = ()
        classification = "fallback"

        try:
            result = synthesize_answer_with_llm(request)
            hybrid_answer = result.answer_text
            hybrid_citations = result.citation_ids

            # Validate with property checks
            validation = validate_hybrid_output(
                answer_text=hybrid_answer,
                citation_ids=hybrid_citations,
                query=query,
                evidence_items=evidence,
                answer_status=fixture_status,
            )
            classification = validation.classification
        except Exception:
            classification = "fallback"

        counts[classification] += 1

        print("=" * 80)
        print(f"QUERY: {query}")
        print(f"CLASSIFICATION: {classification}")
        print()
        print("DETERMINISTIC:")
        print(det_answer)
        print()
        if classification == "fallback":
            print("HYBRID: (fell back to deterministic)")
        else:
            print(f"HYBRID ({classification}, citations={hybrid_citations}):")
            print(hybrid_answer)
            if classification != "valid":
                print(f"  Failures: {validation.failures}")
        print()

    print("=" * 80)
    print("SUMMARY")
    print(f"  Total: {total}")
    for cls in ("valid", "degraded", "invalid", "equivalent", "fallback"):
        print(f"  {cls}: {counts[cls]}")
    print()


if __name__ == "__main__":
    main()
