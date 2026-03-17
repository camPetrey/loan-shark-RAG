"""PDF loader for page-by-page text extraction from guideline documents."""

from dataclasses import dataclass
from pathlib import Path

import pymupdf

DEFAULT_PDF_PATH = Path("data/02042026_Single_Family_Selling_Guide_highlighting.pdf")


@dataclass(frozen=True)
class LoadedPage:
    """Represents text extracted from a single PDF page."""

    page_number: int
    text: str


class PDFLoader:
    """Load a guideline PDF and extract text page by page using PyMuPDF."""

    def load(self, pdf_path: Path = DEFAULT_PDF_PATH) -> list[LoadedPage]:
        """Return ordered pages with extracted text from the provided PDF file."""
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        pages: list[LoadedPage] = []
        document = pymupdf.open(pdf_path)

        try:
            for index in range(document.page_count):
                page = document.load_page(index)
                page_text = page.get_text("text").strip() #type: ignore
                pages.append(LoadedPage(page_number=index + 1, text=page_text))
        finally:
            document.close()

        return pages
