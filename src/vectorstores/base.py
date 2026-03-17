"""Base interfaces for vector database integrations."""

from __future__ import annotations

from abc import ABC, abstractmethod

from src.domain.models import DocumentChunk


class VectorStore(ABC):
    """Abstract interface for storing and querying chunk embeddings."""

    @abstractmethod
    def upsert(self, chunks: list[DocumentChunk], embeddings: list[list[float]]) -> None:
        """Persist chunk embeddings and metadata."""

    @abstractmethod
    def similarity_search(self, query_embedding: list[float], top_k: int = 5) -> list[DocumentChunk]:
        """Retrieve the most similar chunks for a query embedding."""
