"""Core domain models for guideline documents and chunk metadata."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field


class SourcePage(BaseModel):
    """Represents raw text extracted from a single PDF page."""

    page_number: int
    text: str


class DocumentMetadata(BaseModel):
    """Describes a source guideline document."""

    document_id: str
    title: str
    source_path: str
    source_type: str = "pdf"


class DocumentChunk(BaseModel):
    """A citation-friendly chunk produced from a guideline document."""

    chunk_id: str
    document_id: str
    text: str
    page_start: int
    page_end: int
    token_count: int
    citation: str
    metadata: dict[str, str | int] = Field(default_factory=dict)


class ParsedDocument(BaseModel):
    """A parsed document and its extracted pages."""

    metadata: DocumentMetadata
    pages: list[SourcePage]

    @property
    def page_count(self) -> int:
        """Return the number of extracted pages."""
        return len(self.pages)


class IngestionReport(BaseModel):
    """Structured output describing an ingestion preview run."""

    documents: list[ParsedDocument]
    chunks: list[DocumentChunk]

    @property
    def document_count(self) -> int:
        """Return the number of parsed documents."""
        return len(self.documents)

    @property
    def chunk_count(self) -> int:
        """Return the number of generated chunks."""
        return len(self.chunks)


def path_to_document_id(path: Path) -> str:
    """Create a stable document id from a file path."""
    return path.stem.lower().replace(" ", "_")
