"""Semantic and hybrid candidate ranking for the retrieval layer."""

from __future__ import annotations

from rag.embeddings.client import DEFAULT_EMBEDDING_MODEL, OpenAIEmbeddingClient
from rag.embeddings.similarity import cosine_similarity
from rag.embeddings.store import load_embedding_records
from rag.retrieval.read_model import LoadedRun, normalize_lexical_text, tokenize_query
from rag.retrieval.scoring import (
    candidate_sort_key,
    hybrid_candidate_sort_key,
    with_mode_details,
)
from rag.retrieval.types import (
    BM25_K1,
    BM25_B,
    USER_VOICE_BOOST,
    ASSISTANT_VOICE_FACTOR,
    ASSISTANT_META_COMMENTARY_FACTOR,
    ASSISTANT_RESTATEMENT_FACTOR,
    is_assistant_meta_commentary,
    is_assistant_restatement,
    get_nearby_user_texts,
    ParsedQuery,
    RetrievalFilters,
    STOPWORD_FILTER_ENABLED,
    _Candidate,
)

# Semantic candidates with fewer tokens than this threshold are excluded from ranking.
# Short messages like "Larry." produce deceptively high cosine similarity without useful context.
SEMANTIC_MIN_TOKEN_COUNT = 4


# Rank message-level candidates with stored embeddings and query cosine similarity.
def rank_semantic_candidates(
    loaded_run: LoadedRun,
    parsed_query: ParsedQuery,
    filters: RetrievalFilters,
    *,
    limit: int,
    embedding_model: str,
    embedding_client: object | None,
    message_matches_filters_fn: object,
) -> list[_Candidate]:
    normalized_query = normalize_lexical_text(parsed_query.raw_query)
    if not normalized_query:
        return []

    records = load_embedding_records(loaded_run.run_dir)
    matching_records = [record for record in records if record.embedding_model == embedding_model]
    if not matching_records:
        available_models = sorted({record.embedding_model for record in records})
        if embedding_model not in available_models:
            raise ValueError(
                f"Embeddings for model '{embedding_model}' not found in {loaded_run.run_dir}. "
                f"Available models: {available_models or '(none)'}"
            )

    client = embedding_client or OpenAIEmbeddingClient()
    query_vectors = client.embed_texts([normalized_query], model=embedding_model)
    if not query_vectors:
        return []
    query_vector = tuple(float(value) for value in query_vectors[0])

    # Lazy per-conversation index for restatement detection.
    _conv_index_cache: dict[str, dict[str, int]] = {}

    def _get_message_index(conv_id: str, msg_id: str) -> int | None:
        if conv_id not in _conv_index_cache:
            msgs = loaded_run.messages_by_conversation_id.get(conv_id, ())
            _conv_index_cache[conv_id] = {
                str(m.get("message_id", "")): idx for idx, m in enumerate(msgs)
            }
        return _conv_index_cache[conv_id].get(msg_id)

    candidates: list[_Candidate] = []
    for record in matching_records:
        message = loaded_run.message_by_id.get(record.message_id)
        if message is None or not message_matches_filters_fn(message, filters):
            continue
        record_tokens = tokenize_query(record.text)
        if len(record_tokens) < SEMANTIC_MIN_TOKEN_COUNT:
            continue
        similarity = cosine_similarity(query_vector, record.embedding)
        if similarity <= 0.0:
            continue
        voice_factor = USER_VOICE_BOOST if record.author_role == "user" else ASSISTANT_VOICE_FACTOR
        if record.author_role == "assistant" and is_assistant_meta_commentary(record.text):
            voice_factor *= ASSISTANT_META_COMMENTARY_FACTOR
        if record.author_role == "assistant":
            focal_idx = _get_message_index(record.conversation_id, record.message_id)
            if focal_idx is not None:
                conv_msgs = loaded_run.messages_by_conversation_id.get(record.conversation_id, ())
                user_texts = get_nearby_user_texts(conv_msgs, focal_idx)
                if user_texts and is_assistant_restatement(record.text, user_texts):
                    voice_factor *= ASSISTANT_RESTATEMENT_FACTOR
        voiced_score = similarity * voice_factor
        candidates.append(
            _Candidate(
                score=voiced_score,
                base_score=voiced_score,
                recency_boost=0.0,
                bm25_score=None,
                semantic_similarity=round(similarity, 6),
                retrieval_sources=("semantic",),
                provider=record.provider,
                conversation_id=record.conversation_id,
                message_id=record.message_id,
                created_at=record.created_at,
                created_at_value=_parse_iso_timestamp(record.created_at) if record.created_at else None,
                match_basis={
                    "raw_query": parsed_query.raw_query,
                    "normalized_query_terms": list(parsed_query.normalized_query_terms),
                    "scoring_terms": list(parsed_query.scoring_terms),
                    "quoted_phrases": list(parsed_query.quoted_phrases),
                    "normalized_phrase_targets": list(parsed_query.normalized_phrase_targets),
                    "stopword_filter_enabled": STOPWORD_FILTER_ENABLED,
                    "stopword_filtered_terms": list(parsed_query.stopword_filtered_terms),
                    "matched_text_terms": sorted(set(parsed_query.scoring_terms) & set(tokenize_query(record.text))),
                    "matched_phrase_targets": [],
                    "matched_metadata_filters": _matched_filters_dict(filters),
                    "scoring_features": {
                        "query_terms": list(parsed_query.scoring_terms),
                        "bm25_k1": BM25_K1,
                        "bm25_b": BM25_B,
                        "document_length": len(tokenize_query(record.text)),
                        "average_document_length": None,
                        "bm25_term_contributions": [],
                        "bm25_score": None,
                        "exact_phrase_match": False,
                        "matched_phrase_count": 0,
                        "exact_phrase_boost": 0.0,
                        "title_overlap": 0,
                        "title_boost": 0.0,
                        "retrieval_mode": "relevance",
                        "recency_boost": 0.0,
                        "chronological_rank_basis": "semantic_similarity_desc",
                        "relevance_score": round(similarity, 6),
                        "ranking_score": round(similarity, 6),
                        "final_score": round(similarity, 6),
                        "semantic_similarity": round(similarity, 6),
                    },
                    "focal_match_excerpt": _build_excerpt({"text": record.text}),
                    "retrieval_sources": ["semantic"],
                },
            )
        )

    return sorted(candidates, key=candidate_sort_key)[:limit]


# Merge BM25 and semantic candidates on message id while preserving both provenance and scores.
def merge_hybrid_candidates(
    bm25_candidates: list[_Candidate],
    semantic_candidates: list[_Candidate],
) -> list[_Candidate]:
    candidate_by_message_id: dict[str, _Candidate] = {}

    for candidate in bm25_candidates + semantic_candidates:
        existing = candidate_by_message_id.get(candidate.message_id)
        if existing is None:
            candidate_by_message_id[candidate.message_id] = candidate
            continue
        candidate_by_message_id[candidate.message_id] = _combine_candidate(existing, candidate)

    merged = list(candidate_by_message_id.values())
    for index, candidate in enumerate(sorted(merged, key=hybrid_candidate_sort_key), start=1):
        merged[index - 1] = with_mode_details(
            candidate,
            mode="relevance",
            ranking_score=candidate.base_score,
            recency_boost=0.0,
        )
    return sorted(merged, key=hybrid_candidate_sort_key)


# Combine one BM25 candidate and one semantic candidate into a single hybrid result row.
def _combine_candidate(left: _Candidate, right: _Candidate) -> _Candidate:
    sources = tuple(sorted(set(left.retrieval_sources + right.retrieval_sources)))
    bm25_score = left.bm25_score if left.bm25_score is not None else right.bm25_score
    semantic_similarity = (
        left.semantic_similarity if left.semantic_similarity is not None else right.semantic_similarity
    )
    bm25_signal = bm25_score or 0.0
    semantic_signal = semantic_similarity or 0.0
    both_bonus = 10.0 if len(sources) == 2 else 0.0
    hybrid_score = both_bonus + max(bm25_signal, semantic_signal)

    match_basis = dict(left.match_basis)
    scoring_features = dict(match_basis.get("scoring_features", {}))
    scoring_features["bm25_score"] = bm25_score
    scoring_features["semantic_similarity"] = semantic_similarity
    scoring_features["hybrid_ranking_score"] = round(hybrid_score, 6)
    scoring_features["relevance_score"] = round(hybrid_score, 6)
    scoring_features["ranking_score"] = round(hybrid_score, 6)
    scoring_features["final_score"] = round(hybrid_score, 6)
    scoring_features["chronological_rank_basis"] = "hybrid_rank_desc"
    match_basis["scoring_features"] = scoring_features
    match_basis["retrieval_sources"] = list(sources)

    merged_terms = sorted(set(match_basis.get("matched_text_terms", [])) | set(right.match_basis.get("matched_text_terms", [])))
    match_basis["matched_text_terms"] = merged_terms

    return _Candidate(
        score=hybrid_score,
        base_score=hybrid_score,
        recency_boost=0.0,
        bm25_score=bm25_score,
        semantic_similarity=semantic_similarity,
        retrieval_sources=sources,
        provider=left.provider,
        conversation_id=left.conversation_id,
        message_id=left.message_id,
        created_at=left.created_at or right.created_at,
        created_at_value=left.created_at_value or right.created_at_value,
        match_basis=match_basis,
    )


# --- Small helpers duplicated from lexical to avoid circular imports ---

from datetime import datetime, timezone


def _parse_iso_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def _matched_filters_dict(filters: RetrievalFilters) -> dict[str, object]:
    matched: dict[str, object] = {}
    if filters.provider:
        matched["provider"] = filters.provider
    if filters.conversation_id:
        matched["conversation_id"] = filters.conversation_id
    if filters.date_from:
        matched["date_from"] = filters.date_from
    if filters.date_to:
        matched["date_to"] = filters.date_to
    if filters.author_role:
        matched["author_role"] = filters.author_role
    return matched


def _build_excerpt(message: dict[str, object]) -> str | None:
    from rag.utils import string_or_none
    text = string_or_none(message.get("text")) or ""
    if not text:
        return None
    excerpt = text.strip()
    if len(excerpt) > 160:
        excerpt = excerpt[:157] + "..."
    return excerpt
