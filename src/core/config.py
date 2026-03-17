"""Centralized application settings loaded from environment variables."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings for API, ingestion, and external services."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "Loan Shark API"
    app_env: str = "development"
    log_level: str = "INFO"

    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    openai_embedding_model: str = "text-embedding-3-small"
    openai_chat_model: str = "gpt-4.1-mini"

    data_dir: Path = Path("data")
    artifacts_dir: Path = Path("artifacts")

    pgvector_database_url: str = ""
    vector_store_backend: str = "pgvector"

    chunk_size_tokens: int = 800
    chunk_overlap_tokens: int = 120

    @property
    def default_ingestion_output_path(self) -> Path:
        """Return the default path for ingestion preview artifacts."""
        return self.artifacts_dir / "ingestion_preview.json"


@lru_cache
def get_settings() -> Settings:
    """Return a cached settings instance."""
    return Settings()
