"""Service layer for future embedding and vector indexing workflows."""

from __future__ import annotations

from src.domain.models import IngestionReport
from src.embeddings.openai_embedder import OpenAIEmbedder
from src.vectorstores.base import VectorStore


class GuidelineIndexService:
    """Coordinate future indexing of prepared chunks into a vector store."""

    def __init__(self, embedder: OpenAIEmbedder, vector_store: VectorStore) -> None:
        self._embedder = embedder
        self._vector_store = vector_store

    def index(self, report: IngestionReport) -> None:
        """Embed and store chunks for retrieval.

        This is intentionally deferred until the next implementation phase.
        """
        raise NotImplementedError("Vector indexing will be implemented in the next iteration.")
