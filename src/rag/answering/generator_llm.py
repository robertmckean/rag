"""Optional constrained LLM synthesis over pre-qualified Phase 3A evidence."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from rag.answering.models import AnswerStatus, EvidenceItem
from rag.retrieval.read_model import tokenize_query


# Phase 3B is strictly a wording layer on top of deterministic Phase 3A outputs.
# The generator only sees already-qualified evidence and the precomputed answer status.
# Any parsing or validation failure must fall back to the deterministic Phase 3A answer.

DEFAULT_LLM_MODEL = "gpt-5-mini"
GENERIC_ENTITY_WORDS = frozenset(
    {
        "Based",
        "Earlier",
        "Evidence",
        "I",
        "Later",
        "The",
        "This",
    }
)
MONTH_NAMES = frozenset(
    {
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    }
)


@dataclass(frozen=True)
class LLMSynthesisRequest:
    query: str
    answer_status: AnswerStatus
    evidence_items: tuple[EvidenceItem, ...]
    gaps: tuple[str, ...]
    conflicts: tuple[str, ...]
    model: str | None = None


@dataclass(frozen=True)
class LLMSynthesisResult:
    answer_text: str
    citation_ids: tuple[str, ...]


# Build the bounded evidence payload passed to the LLM synthesis step.
def _build_evidence_payload(evidence_items: tuple[EvidenceItem, ...]) -> tuple[dict[str, object], ...]:
    payload: list[dict[str, object]] = []
    for item in evidence_items:
        payload.append(
            {
                "evidence_id": f"e{item.rank}",
                "provider": item.citation.provider,
                "conversation_id": item.citation.conversation_id,
                "created_at": item.citation.created_at,
                "excerpt": item.citation.excerpt,
            }
        )
    return tuple(payload)


# Build the tightly constrained prompt payload for the synthesis layer.
def _build_prompt(request: LLMSynthesisRequest) -> str:
    payload = {
        "query": request.query,
        "answer_status": request.answer_status.value,
        "gaps": list(request.gaps),
        "conflicts": list(request.conflicts),
        "evidence": list(_build_evidence_payload(request.evidence_items)),
        "required_output": {
            "answer_text": "string",
            "citation_ids": ["evidence_id"],
        },
    }
    return json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True)


# Return the fixed synthesis instructions used for every LLM-backed answer rewrite.
def _prompt_instructions() -> str:
    return (
        "You are rewriting a grounded answer using only the supplied evidence.\n"
        "Rules:\n"
        "- Use only the provided evidence excerpts.\n"
        "- Do not add outside knowledge or unsupported facts.\n"
        "- Do not change the supplied answer_status.\n"
        "- Reflect incompleteness or conflict when the evidence is incomplete or mixed.\n"
        "- Output valid JSON with keys answer_text and citation_ids only.\n"
        "- citation_ids must reference only the supplied evidence_id values.\n"
        "- Keep the answer concise and factual.\n"
    )


# Create the OpenAI client lazily so deterministic tests can run without the package installed.
def _create_openai_client() -> object:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("The OpenAI Python SDK is not installed in the current environment.") from exc
    return OpenAI()


# Call the OpenAI Responses API and return the raw text output.
def _call_llm(request: LLMSynthesisRequest) -> str:
    client = _create_openai_client()
    response = client.responses.create(
        model=request.model or DEFAULT_LLM_MODEL,
        instructions=_prompt_instructions(),
        input=_build_prompt(request),
    )
    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str) and output_text.strip():
        return output_text
    raise ValueError("The LLM synthesis response did not include any output text.")


# Parse the JSON response emitted by the constrained synthesis prompt.
def _parse_llm_output(value: str) -> LLMSynthesisResult:
    payload = json.loads(value)
    if not isinstance(payload, dict):
        raise ValueError("The LLM synthesis response must be a JSON object.")
    answer_text = payload.get("answer_text")
    citation_ids = payload.get("citation_ids")
    if not isinstance(answer_text, str) or not answer_text.strip():
        raise ValueError("The LLM synthesis response must include a non-empty answer_text.")
    if not isinstance(citation_ids, list) or not all(isinstance(item, str) for item in citation_ids):
        raise ValueError("The LLM synthesis response must include citation_ids as a list of strings.")
    return LLMSynthesisResult(
        answer_text=" ".join(answer_text.split()),
        citation_ids=tuple(citation_ids),
    )


# Reject synthesis output that cites unavailable evidence or introduces obvious new entities or dates.
def _validate_synthesis_result(
    request: LLMSynthesisRequest,
    result: LLMSynthesisResult,
) -> LLMSynthesisResult:
    if not result.answer_text:
        raise ValueError("The LLM synthesis response was empty.")

    valid_citation_ids = {f"e{item.rank}" for item in request.evidence_items}
    if any(citation_id not in valid_citation_ids for citation_id in result.citation_ids):
        raise ValueError("The LLM synthesis response referenced unknown evidence IDs.")
    if request.answer_status is not AnswerStatus.INSUFFICIENT_EVIDENCE and request.evidence_items and not result.citation_ids:
        raise ValueError("The LLM synthesis response must cite at least one evidence item.")
    if _contains_unseen_surface_forms(result.answer_text, request):
        raise ValueError("The LLM synthesis response introduced entity or date surface forms not present in the evidence.")
    return result


# Detect obviously new entity/date surface forms that are not present in the query or evidence package.
def _contains_unseen_surface_forms(answer_text: str, request: LLMSynthesisRequest) -> bool:
    allowed_source = " ".join(
        [request.query, *(item.citation.excerpt for item in request.evidence_items)]
    )
    allowed_entities = _entity_surface_forms(allowed_source)
    answer_entities = _entity_surface_forms(answer_text)
    if not answer_entities.issubset(allowed_entities):
        return True

    allowed_years = set(re.findall(r"\b(?:19|20)\d{2}\b", allowed_source))
    answer_years = set(re.findall(r"\b(?:19|20)\d{2}\b", answer_text))
    if not answer_years.issubset(allowed_years):
        return True

    allowed_months = {month for month in MONTH_NAMES if month in allowed_source}
    answer_months = {month for month in MONTH_NAMES if month in answer_text}
    if not answer_months.issubset(allowed_months):
        return True

    return False


# Extract light-weight entity surface forms from text without introducing semantic interpretation.
def _entity_surface_forms(value: str) -> set[str]:
    surfaces: set[str] = set()
    for match in re.finditer(r"\b[A-Z][A-Za-z']+\b", value):
        token = match.group(0)
        if token in GENERIC_ENTITY_WORDS:
            continue
        surfaces.add(token)
    return surfaces


# Run the constrained synthesis request end to end and return a validated structured result.
def synthesize_answer_with_llm(request: LLMSynthesisRequest) -> LLMSynthesisResult:
    raw_output = _call_llm(request)
    parsed_output = _parse_llm_output(raw_output)
    return _validate_synthesis_result(request, parsed_output)
