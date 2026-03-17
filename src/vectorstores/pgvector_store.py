"""Placeholder `pgvector` implementation for future indexing and retrieval."""

from __future__ import annotations

from src.domain.models import DocumentChunk
from src.vectorstores.base import VectorStore


class PGVectorStore(VectorStore):
    """`pgvector` integration stub kept ready for the next iteration."""

    def __init__(self, database_url: str) -> None:
        self._database_url = database_url

    def upsert(self, chunks: list[DocumentChunk], embeddings: list[list[float]]) -> None:
        """Persist chunk embeddings to PostgreSQL with `pgvector`.

        The schema and write path are intentionally deferred for now.
        """
        raise NotImplementedError("pgvector persistence will be implemented next.")

    def similarity_search(self, query_embedding: list[float], top_k: int = 5) -> list[DocumentChunk]:
        """Search for similar chunks in `pgvector`.

        Retrieval is intentionally deferred for now.
        """
        raise NotImplementedError("pgvector retrieval will be implemented next.")
