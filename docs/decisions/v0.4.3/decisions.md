Decision: shared embedding client/model contract moved from builder.py to embeddings/client.py
Reason: query-time retrieval must not import build-time orchestration code
