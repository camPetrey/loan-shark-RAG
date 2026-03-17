"""OpenAI embedding client wrapper for future vector indexing."""

from __future__ import annotations

from collections.abc import Sequence

from openai import OpenAI

from src.core.config import Settings


class OpenAIEmbedder:
    """Thin wrapper around the OpenAI embeddings API."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client = OpenAI(api_key=settings.openai_api_key) if settings.openai_api_key else None

    @property
    def is_configured(self) -> bool:
        """Return whether an OpenAI API key is available."""
        return self._client is not None

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed a list of texts with the configured OpenAI model."""
        if not self._client:
            raise RuntimeError("OPENAI_API_KEY is not configured.")

        response = self._client.embeddings.create(
            model=self._settings.openai_embedding_model,
            input=list(texts),
        )
        return [item.embedding for item in response.data]
