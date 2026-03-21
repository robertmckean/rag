"""Side-by-side evaluation of deterministic vs hybrid synthesis on Phase 5 benchmarks."""

from __future__ import annotations

import io
import json
import os
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from rag.answering.generator_llm import LLMSynthesisRequest, synthesize_answer_with_llm
from rag.answering.models import AnswerStatus, Citation, EvidenceItem
from rag.answering.status import classify_answer_status, generate_answer_text


def _load_evidence(data: dict) -> tuple[EvidenceItem, ...]:
    items: list[EvidenceItem] = []
    for i, r in enumerate(data["results"][:4]):
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

    print("# Hybrid vs Deterministic Evaluation - Phase 5 Benchmarks")
    print()

    for fname in fixtures:
        path = os.path.join(fixtures_dir, fname)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        query = data["query"]
        evidence = _load_evidence(data)

        # Deterministic path
        decision = classify_answer_status(query, _FakeQual(evidence))
        det_answer = generate_answer_text(decision, evidence, _FakeDiag(len(evidence)))

        # Hybrid path
        request = LLMSynthesisRequest(
            query=query,
            answer_status=AnswerStatus.SUPPORTED,
            evidence_items=evidence,
            gaps=(),
            conflicts=(),
            hybrid=True,
        )
        try:
            result = synthesize_answer_with_llm(request)
            hybrid_answer = result.answer_text
            hybrid_citations = result.citation_ids
            hybrid_status = "VALIDATED"
        except Exception as exc:
            hybrid_answer = f"(fallback: {exc})"
            hybrid_citations = ()
            hybrid_status = "FALLBACK"

        print("=" * 80)
        print(f"QUERY: {query}")
        print(f"Det status: {decision.status.value}")
        print()
        print("DETERMINISTIC:")
        print(det_answer)
        print()
        print(f"HYBRID ({hybrid_status}, citations={hybrid_citations}):")
        print(hybrid_answer)
        print()


if __name__ == "__main__":
    main()
