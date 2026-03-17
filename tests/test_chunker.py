"""Tests for section-based chunking and heading detection."""

from __future__ import annotations

from pathlib import Path

from src.ingestion.chunker import FallbackTokenEncoder, TextChunker
from src.ingestion.loader import LoadedPage, PDFLoader


def build_chunker(chunk_size_tokens: int = 40, chunk_overlap_tokens: int = 6) -> TextChunker:
    """Create a chunker with an offline-safe tokenizer for deterministic tests."""
    chunker = TextChunker(
        chunk_size_tokens=chunk_size_tokens,
        chunk_overlap_tokens=chunk_overlap_tokens,
    )
    chunker._encoding = FallbackTokenEncoder()
    return chunker


def test_heading_detection_accepts_expected_guideline_headings() -> None:
    """Heading heuristics should recognize common guideline section labels."""
    chunker = build_chunker()

    assert chunker._looks_like_heading("General Requirements")
    assert chunker._looks_like_heading("Part A, Doing Business with Fannie Mae")
    assert chunker._looks_like_heading("A2-1-03, Indemnification for Losses (10/06/2021)")


def test_heading_detection_rejects_body_text_and_noise() -> None:
    """Heading heuristics should ignore sentences and repeated page noise."""
    chunker = build_chunker()

    assert not chunker._looks_like_heading(
        "The lender must maintain the recording for the minimum period required by applicable law."
    )
    assert chunker._is_noise_line("Published February 4, 2026")
    assert chunker._is_noise_line("17")
    assert chunker._is_noise_line("Fannie Mae Copyright Notice")


def test_chunk_pages_preserves_sections_and_subheadings() -> None:
    """Chunking should start a new section when a new heading appears."""
    chunker = build_chunker(chunk_size_tokens=120, chunk_overlap_tokens=10)
    pages = [
        LoadedPage(
            page_number=1,
            text=(
                "General Requirements\n"
                "Lenders must verify the borrower's income.\n"
                "Documentation must be retained.\n"
                "Application After Enforcement Relief\n"
                "A repurchase request may still be issued in limited circumstances.\n"
                "Indemnification Process\n"
                "The seller must respond within the required timeline."
            ),
        )
    ]

    chunks = chunker.chunk_pages(pages)

    assert [chunk.heading for chunk in chunks] == [
        "General Requirements",
        "Application After Enforcement Relief",
        "Indemnification Process",
    ]
    assert "Lenders must verify the borrower's income." in chunks[0].text
    assert "A repurchase request may still be issued" in chunks[1].text
    assert "The seller must respond within the required timeline." in chunks[2].text


def test_chunk_pages_adds_overlap_when_section_is_split() -> None:
    """Oversized sections should preserve trailing context in the next chunk."""
    chunker = build_chunker(chunk_size_tokens=18, chunk_overlap_tokens=5)
    pages = [
        LoadedPage(
            page_number=7,
            text=(
                "General Requirements\n"
                "alpha beta gamma delta epsilon\n"
                "zeta eta theta iota kappa\n"
                "lambda mu nu xi omicron\n"
                "pi rho sigma tau upsilon"
            ),
        )
    ]

    chunks = chunker.chunk_pages(pages)

    assert len(chunks) > 1
    assert chunks[0].heading == "General Requirements"
    assert chunks[1].heading == "General Requirements"
    assert "lambda mu nu xi omicron" in chunks[0].text
    assert "lambda mu nu xi omicron" in chunks[1].text


def test_chunk_pages_skips_table_of_contents_pages_from_real_pdf() -> None:
    """TOC pages in the source PDF should not become chunks."""
    pdf_path = Path("data/02042026_Single_Family_Selling_Guide_highlighting.pdf")
    loader = PDFLoader()
    chunker = build_chunker(chunk_size_tokens=200, chunk_overlap_tokens=20)

    pages = loader.load(pdf_path)[2:4]
    chunks = chunker.chunk_pages(pages)

    assert chunks == []


def test_chunk_pages_finds_real_headings_in_body_pages() -> None:
    """Real content pages should still produce known section headings."""
    pdf_path = Path("data/02042026_Single_Family_Selling_Guide_highlighting.pdf")
    loader = PDFLoader()
    chunker = build_chunker(chunk_size_tokens=200, chunk_overlap_tokens=20)

    pages = loader.load(pdf_path)[23:25]
    chunks = chunker.chunk_pages(pages)
    headings = [chunk.heading for chunk in chunks]

    assert "Delivery Methods" in headings
    assert "True Sale" in headings
    assert "Recent Related Announcements" in headings
    assert "Introduction" in headings
