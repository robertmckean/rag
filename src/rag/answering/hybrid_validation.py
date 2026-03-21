"""Property-based validation for hybrid synthesis outputs.

These checks validate structural properties of hybrid answers without requiring
exact string equality.  They are used by both unit tests and the evaluation
script to classify hybrid outputs as valid, degraded, invalid, or equivalent.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from rag.answering.generator_llm import (
    _entity_surface_forms_permissive,
    _entity_surface_forms_strict,
    MONTH_NAMES,
)
from rag.answering.models import AnswerStatus, EvidenceItem


@dataclass(frozen=True)
class HybridValidationResult:
    no_new_entities: bool
    dates_preserved: bool
    evidence_bounded: bool
    status_not_inflated: bool
    has_evidence_grounding: bool
    failures: tuple[str, ...]

    @property
    def is_valid(self) -> bool:
        return not self.failures

    @property
    def classification(self) -> str:
        if self.is_valid:
            return "valid"
        if not self.no_new_entities or not self.evidence_bounded:
            return "invalid"
        if not self.dates_preserved or not self.has_evidence_grounding:
            return "degraded"
        return "invalid"


# Validate a hybrid synthesis answer against property-based constraints.
def validate_hybrid_output(
    answer_text: str,
    citation_ids: tuple[str, ...],
    query: str,
    evidence_items: tuple[EvidenceItem, ...],
    answer_status: AnswerStatus,
) -> HybridValidationResult:
    failures: list[str] = []

    # 1. No new entities: every entity in the answer must appear in the evidence or query.
    no_new_entities = _check_no_new_entities(answer_text, query, evidence_items, failures)

    # 2. Dates preserved: when evidence spans multiple dates, at least one timestamp or
    #    chronological marker must appear in the answer.
    dates_preserved = _check_dates_preserved(answer_text, evidence_items, failures)

    # 3. Evidence bounded: citation count must not exceed evidence count.
    evidence_bounded = _check_evidence_bounded(citation_ids, evidence_items, failures)

    # 4. Status not inflated: hybrid must not upgrade partially_supported to supported.
    status_not_inflated = _check_status_not_inflated(answer_text, answer_status, failures)

    # 5. Has evidence grounding: every major claim should reference evidence content.
    has_evidence_grounding = _check_evidence_grounding(answer_text, evidence_items, failures)

    return HybridValidationResult(
        no_new_entities=no_new_entities,
        dates_preserved=dates_preserved,
        evidence_bounded=evidence_bounded,
        status_not_inflated=status_not_inflated,
        has_evidence_grounding=has_evidence_grounding,
        failures=tuple(failures),
    )


def _check_no_new_entities(
    answer_text: str,
    query: str,
    evidence_items: tuple[EvidenceItem, ...],
    failures: list[str],
) -> bool:
    allowed_source = " ".join(
        [
            query,
            *(item.citation.excerpt for item in evidence_items),
            *(item.citation.created_at or "" for item in evidence_items),
        ]
    )
    allowed = _entity_surface_forms_permissive(allowed_source)
    strict = _entity_surface_forms_strict(answer_text)
    unseen = strict - allowed
    if unseen:
        failures.append(f"new_entities: {sorted(unseen)}")
        return False
    # Also check years and months.
    allowed_years = set(re.findall(r"\b(?:19|20)\d{2}\b", allowed_source))
    answer_years = set(re.findall(r"\b(?:19|20)\d{2}\b", answer_text))
    unseen_years = answer_years - allowed_years
    if unseen_years:
        failures.append(f"new_years: {sorted(unseen_years)}")
        return False
    allowed_months = {m for m in MONTH_NAMES if m in allowed_source}
    answer_months = {m for m in MONTH_NAMES if m in answer_text}
    unseen_months = answer_months - allowed_months
    if unseen_months:
        failures.append(f"new_months: {sorted(unseen_months)}")
        return False
    return True


def _check_dates_preserved(
    answer_text: str,
    evidence_items: tuple[EvidenceItem, ...],
    failures: list[str],
) -> bool:
    # Extract distinct date prefixes (YYYY-MM-DD) from evidence timestamps.
    evidence_dates = set()
    for item in evidence_items:
        ts = item.citation.created_at or ""
        if len(ts) >= 10:
            evidence_dates.add(ts[:10])
    if len(evidence_dates) < 2:
        # Single date or no dates — nothing to preserve.
        return True
    # The answer should contain at least one date or year from the evidence.
    answer_years = set(re.findall(r"\b(?:19|20)\d{2}\b", answer_text))
    answer_date_fragments = set(re.findall(r"\d{4}-\d{2}-\d{2}", answer_text))
    if answer_years or answer_date_fragments:
        return True
    failures.append("dates_missing: evidence spans multiple dates but answer contains no date markers")
    return False


def _check_evidence_bounded(
    citation_ids: tuple[str, ...],
    evidence_items: tuple[EvidenceItem, ...],
    failures: list[str],
) -> bool:
    valid_ids = {f"e{item.rank}" for item in evidence_items}
    for cid in citation_ids:
        if cid not in valid_ids:
            failures.append(f"invalid_citation: {cid} not in {sorted(valid_ids)}")
            return False
    if len(citation_ids) > len(evidence_items):
        failures.append(f"citation_overflow: {len(citation_ids)} citations for {len(evidence_items)} evidence items")
        return False
    return True


def _check_status_not_inflated(
    answer_text: str,
    answer_status: AnswerStatus,
    failures: list[str],
) -> bool:
    if answer_status is not AnswerStatus.PARTIALLY_SUPPORTED:
        return True
    # A partially_supported answer should not read as confidently definitive.
    # Check for strong definitive phrasing that implies full support.
    definitive_phrases = [
        "you definitively concluded",
        "the evidence clearly shows",
        "the answer is clear",
        "fully supported by",
    ]
    lower = answer_text.lower()
    for phrase in definitive_phrases:
        if phrase in lower:
            failures.append(f"status_inflation: '{phrase}' in partially_supported answer")
            return False
    return True


def _check_evidence_grounding(
    answer_text: str,
    evidence_items: tuple[EvidenceItem, ...],
    failures: list[str],
) -> bool:
    if not evidence_items:
        return True
    # Check that at least one word sequence from the evidence appears in the answer.
    # Use 4-word n-grams from evidence excerpts as grounding anchors.
    answer_lower = answer_text.lower()
    grounded = False
    for item in evidence_items:
        excerpt_words = item.citation.excerpt.lower().split()
        for i in range(len(excerpt_words) - 3):
            ngram = " ".join(excerpt_words[i : i + 4])
            if ngram in answer_lower:
                grounded = True
                break
        if grounded:
            break
    if not grounded:
        failures.append("no_evidence_grounding: answer does not contain any 4-word sequence from evidence excerpts")
        return False
    return True
