"""Embedding client protocol and default model constant shared by build and retrieval paths."""

from __future__ import annotations

from typing import Protocol


DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"


class EmbeddingClient(Protocol):
    # Embed a batch of texts with one configured model.
    def embed_texts(self, texts: list[str], *, model: str) -> list[list[float]]:
        ...


class OpenAIEmbeddingClient:
    # Call the modern OpenAI embeddings API through the current Python SDK.
    def embed_texts(self, texts: list[str], *, model: str) -> list[list[float]]:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("OpenAI Python SDK is required for embedding generation.") from exc
        client = OpenAI()
        response = client.embeddings.create(model=model, input=texts)
        return [list(item.embedding) for item in response.data]
