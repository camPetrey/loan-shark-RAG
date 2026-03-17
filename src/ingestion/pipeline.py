"""Composable ingestion pipeline that parses and chunks guideline PDFs."""

from __future__ import annotations

from pathlib import Path

from src.core.config import Settings
from src.domain.models import IngestionReport, ParsedDocument
from src.ingestion.chunker import TextChunker
from src.ingestion.pdf_parser import PDFParser


class IngestionPipeline:
    """Orchestrate document parsing and chunk generation."""

    def __init__(self, parser: PDFParser, chunker: TextChunker) -> None:
        self._parser = parser
        self._chunker = chunker

    @classmethod
    def from_settings(cls, settings: Settings) -> "IngestionPipeline":
        """Create a pipeline using the current application settings."""
        return cls(
            parser=PDFParser(),
            chunker=TextChunker(
                chunk_size_tokens=settings.chunk_size_tokens,
                chunk_overlap_tokens=settings.chunk_overlap_tokens,
            ),
        )

    def run(self, pdf_paths: list[Path]) -> IngestionReport:
        """Parse and chunk one or more PDF documents."""
        documents: list[ParsedDocument] = []
        chunks = []

        for pdf_path in pdf_paths:
            parsed_document = self._parser.parse(pdf_path)
            documents.append(parsed_document)
            chunks.extend(self._chunker.chunk_document(parsed_document))

        return IngestionReport(documents=documents, chunks=chunks)
