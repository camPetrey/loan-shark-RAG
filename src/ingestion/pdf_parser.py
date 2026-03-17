"""PDF parsing utilities for extracting mortgage guideline text page by page."""

from __future__ import annotations

from pathlib import Path

import pymupdf

from src.domain.models import DocumentMetadata, ParsedDocument, SourcePage, path_to_document_id


class PDFParser:
    """Parse PDF documents into structured page content."""

    def parse(self, pdf_path: Path) -> ParsedDocument:
        """Extract text from a PDF and return a parsed document model."""
        document = pymupdf.open(pdf_path)
        pages: list[SourcePage] = []

        try:
            for index, page in enumerate(document, start=1):
                text = page.get_text("text")
                normalized_text = text.strip()
                if normalized_text:
                    pages.append(SourcePage(page_number=index, text=normalized_text))
        finally:
            document.close()

        return ParsedDocument(
            metadata=DocumentMetadata(
                document_id=path_to_document_id(pdf_path),
                title=pdf_path.stem.replace("_", " "),
                source_path=str(pdf_path),
            ),
            pages=pages,
        )
