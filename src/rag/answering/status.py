"""Support status classification, gap wording, conflict detection, and answer text generation."""

from __future__ import annotations

from dataclasses import dataclass

from rag.answering.models import (
    AnswerDiagnostics,
    AnswerStatus,
    EvidenceItem,
)
from rag.answering.qualify import (
    _QualificationOutcome,
    focus_terms,
    has_combined_focus_support,
    is_strong_evidence_item,
    requires_combined_focus_support,
)
from rag.retrieval.read_model import tokenize_query


# Import cue sets from qualify to keep them in one place.
from rag.answering.qualify import POSITIVE_CUES, NEGATIVE_CUES


@dataclass(frozen=True)
class _StatusDecision:
    status: AnswerStatus
    gaps: tuple[str, ...]
    conflicts: tuple[str, ...]
    focus_terms: tuple[str, ...]


# Classify the answer status deterministically before any answer prose is generated.
def classify_answer_status(query: str, qualification: _QualificationOutcome) -> _StatusDecision:
    focus_terms_value = focus_terms(query)
    evidence_items = qualification.evidence_items
    if not evidence_items:
        return _StatusDecision(
            status=AnswerStatus.INSUFFICIENT_EVIDENCE,
            gaps=("No relevant retrieval evidence was available for this question.",),
            conflicts=(),
            focus_terms=focus_terms_value,
        )

    strong_items = tuple(item for item in evidence_items if is_strong_evidence_item(item, focus_terms_value))
    if not strong_items:
        return _StatusDecision(
            status=AnswerStatus.INSUFFICIENT_EVIDENCE,
            gaps=("The retrieved evidence was too weak or too indirect to support an answer.",),
            conflicts=(),
            focus_terms=focus_terms_value,
        )

    conflicts = detect_conflicts(strong_items, focus_terms_value)
    if conflicts:
        return _StatusDecision(
            status=AnswerStatus.AMBIGUOUS,
            gaps=(),
            conflicts=conflicts,
            focus_terms=focus_terms_value,
        )

    if qualification.composition_used:
        gaps = ("Support is distributed across multiple nearby excerpts within one retrieved conversation window.",)
        return _StatusDecision(
            status=AnswerStatus.PARTIALLY_SUPPORTED,
            gaps=gaps,
            conflicts=(),
            focus_terms=focus_terms_value,
        )

    covered_terms = {term for item in strong_items for term in item.matched_terms}
    uncovered_terms = tuple(term for term in focus_terms_value if term not in covered_terms)
    combined_focus_supported = has_combined_focus_support(strong_items, focus_terms_value)
    requires_combined = requires_combined_focus_support(query, focus_terms_value)

    narrow_single_item_case = len(focus_terms_value) <= 1 and len(strong_items) == 1
    if len(focus_terms_value) >= 2:
        if combined_focus_supported and not uncovered_terms:
            return _StatusDecision(
                status=AnswerStatus.SUPPORTED,
                gaps=(),
                conflicts=(),
                focus_terms=focus_terms_value,
            )
        if requires_combined and not combined_focus_supported:
            return _StatusDecision(
                status=AnswerStatus.INSUFFICIENT_EVIDENCE,
                gaps=("The retrieved evidence does not support the combined query concept directly.",),
                conflicts=(),
                focus_terms=focus_terms_value,
            )
    elif not uncovered_terms and (len(strong_items) >= 2 or narrow_single_item_case):
        return _StatusDecision(
            status=AnswerStatus.SUPPORTED,
            gaps=(),
            conflicts=(),
            focus_terms=focus_terms_value,
        )

    gaps: list[str] = []
    if uncovered_terms:
        gaps.append(f"The retrieved evidence does not directly cover: {', '.join(uncovered_terms)}.")
    if len(strong_items) == 1 and not narrow_single_item_case:
        gaps.append("Only one strong evidence item was available for this question.")
    if not gaps:
        gaps.append("The retrieved evidence is relevant but too thin for a fully supported answer.")
    return _StatusDecision(
        status=AnswerStatus.PARTIALLY_SUPPORTED,
        gaps=tuple(gaps),
        conflicts=(),
        focus_terms=focus_terms_value,
    )


# Distinguish between retrieval miss and qualification failure so insufficiency gaps stay accurate.
def refine_insufficient_evidence_wording(
    decision: _StatusDecision,
    diagnostics: AnswerDiagnostics,
) -> _StatusDecision:
    if decision.status is not AnswerStatus.INSUFFICIENT_EVIDENCE:
        return decision
    if diagnostics.selected_evidence_count <= 0 or diagnostics.qualified_evidence_count > 0:
        return decision
    if diagnostics.composition_used:
        gap = "Relevant retrieval candidates were found, but the composed support stayed too weak to justify a grounded answer."
    elif any("combined_concept_not_supported" in item.rejection_reasons for item in diagnostics.rejected_evidence):
        gap = "Relevant retrieval candidates were found, but none directly supported the full combined query concept after evidence qualification."
    else:
        gap = "Relevant retrieval candidates were found, but none met the evidence qualification rules for this question."
    return _StatusDecision(
        status=decision.status,
        gaps=(gap,),
        conflicts=decision.conflicts,
        focus_terms=decision.focus_terms,
    )


# Detect direct positive/negative tension across selected excerpts so status does not hide conflicts.
def detect_conflicts(evidence_items: tuple[EvidenceItem, ...], focus_terms_value: tuple[str, ...]) -> tuple[str, ...]:
    positive_excerpts: list[str] = []
    negative_excerpts: list[str] = []

    for item in evidence_items:
        excerpt_tokens = set(tokenize_query(item.citation.excerpt))
        if focus_terms_value and not any(term in excerpt_tokens for term in focus_terms_value):
            continue
        if excerpt_tokens & POSITIVE_CUES:
            positive_excerpts.append(item.citation.excerpt)
        if excerpt_tokens & NEGATIVE_CUES:
            negative_excerpts.append(item.citation.excerpt)

    if positive_excerpts and negative_excerpts:
        return (
            f'Positive and negative evidence both appear, including "{positive_excerpts[0]}" and "{negative_excerpts[0]}".',
        )
    return ()


# Maximum excerpt length for multi-evidence answers.  Shorter than the single-evidence
# path because several excerpts compose the answer and verbosity dilutes the signal.
MULTI_EVIDENCE_EXCERPT_LENGTH = 120


@dataclass(frozen=True)
class _SynthesisEntry:
    excerpt: str
    created_at: str | None
    author_role: str | None


# Generate deterministic grounded answer prose from the selected evidence and precomputed status.
# Multi-evidence synthesis: compose across deduplicated user-preferred windows rather than
# quoting the single strongest excerpt.
def generate_answer_text(
    decision: _StatusDecision,
    evidence_items: tuple[EvidenceItem, ...],
    diagnostics: AnswerDiagnostics,
) -> str:
    if decision.status is AnswerStatus.INSUFFICIENT_EVIDENCE:
        return "I did not find enough relevant evidence in this run to answer that question confidently."

    selected = _select_synthesis_entries(evidence_items)
    if not selected:
        return "Based on the retrieved conversation evidence, the available support is limited but relevant."

    if decision.status is AnswerStatus.AMBIGUOUS:
        quoted = [f'"{e.excerpt}"' for e in selected[:2]]
        return "The retrieved evidence is mixed. Conflicting relevant excerpts include " + " and ".join(quoted) + "."

    if decision.status is AnswerStatus.PARTIALLY_SUPPORTED:
        return _compose_partial_answer(selected)

    return _compose_supported_answer(selected)


# Pick the best non-redundant evidence entries for answer synthesis.
# User-authored first, then by evidence rank, with near-duplicate suppression.
# Returns structured entries with timestamps for chronological ordering.
def _select_synthesis_entries(
    evidence_items: tuple[EvidenceItem, ...],
    *,
    max_entries: int = 4,
) -> list[_SynthesisEntry]:
    user_items = [item for item in evidence_items if item.author_role == "user"]
    assistant_items = [item for item in evidence_items if item.author_role != "user"]
    ordered = user_items + assistant_items

    accepted_excerpts: list[str] = []
    entries: list[_SynthesisEntry] = []
    for item in ordered:
        excerpt = item.citation.excerpt
        if not excerpt:
            continue
        if _is_near_duplicate_excerpt(excerpt, accepted_excerpts):
            continue
        accepted_excerpts.append(excerpt)
        entries.append(_SynthesisEntry(
            excerpt=excerpt,
            created_at=item.citation.created_at,
            author_role=item.author_role,
        ))
        if len(entries) >= max_entries:
            break
    return entries


# Detect near-duplicate excerpts by comparing normalized 60-char prefixes.
def _is_near_duplicate_excerpt(excerpt: str, accepted: list[str]) -> bool:
    prefix = excerpt[:60].lower().strip()
    if len(prefix) < 30:
        return False
    for existing in accepted:
        if existing[:60].lower().strip() == prefix:
            return True
    return False


# Order entries chronologically when timestamps span more than one day.
# Falls back to the original selection order when timestamps are missing or same-day.
def _chronological_entries(entries: list[_SynthesisEntry]) -> list[_SynthesisEntry]:
    dated = [(e, e.created_at or "") for e in entries]
    dates = sorted({ts[:10] for _, ts in dated if len(ts) >= 10})
    if len(dates) < 2:
        return entries
    return sorted(entries, key=lambda e: e.created_at or "")


# Format a timestamp as a short human-readable date label.
def _date_label(created_at: str | None) -> str:
    if not created_at or len(created_at) < 10:
        return ""
    return created_at[:10]


# Compress an excerpt to the multi-evidence length limit.
def _compress_excerpt(excerpt: str, *, limit: int = MULTI_EVIDENCE_EXCERPT_LENGTH) -> str:
    if len(excerpt) <= limit:
        return excerpt
    return excerpt[: limit - 3] + "..."


# Compose a supported answer from multiple grounded excerpts with chronological structure.
def _compose_supported_answer(entries: list[_SynthesisEntry]) -> str:
    if len(entries) == 1:
        return f'Based on the retrieved evidence: "{entries[0].excerpt}".'
    ordered = _chronological_entries(entries)
    lines = ["Based on the retrieved evidence across multiple conversations:"]
    for entry in ordered:
        date = _date_label(entry.created_at)
        compressed = _compress_excerpt(entry.excerpt)
        if date:
            lines.append(f'- ({date}) "{compressed}"')
        else:
            lines.append(f'- "{compressed}"')
    return "\n".join(lines)


# Compose a partially-supported answer acknowledging gaps while presenting available evidence.
def _compose_partial_answer(entries: list[_SynthesisEntry]) -> str:
    if len(entries) == 1:
        return (
            "The retrieved evidence only partially answers the question. "
            f'The strongest relevant excerpt says "{entries[0].excerpt}".'
        )
    ordered = _chronological_entries(entries)
    lines = ["The retrieved evidence only partially answers the question. The most relevant excerpts are:"]
    for entry in ordered:
        date = _date_label(entry.created_at)
        compressed = _compress_excerpt(entry.excerpt)
        if date:
            lines.append(f'- ({date}) "{compressed}"')
        else:
            lines.append(f'- "{compressed}"')
    return "\n".join(lines)
