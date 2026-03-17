"""Section-based chunking utilities for guideline text extracted by the loader."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

import tiktoken

from src.domain.models import DocumentChunk, ParsedDocument, SourcePage
from src.ingestion.loader import LoadedPage

HEADING_PATTERN = re.compile(r"^[A-Z0-9][A-Za-z0-9/,&()\-.' ]{0,120}:?$")
SECTION_CODE_PATTERN = re.compile(r"^[A-Z]\d[\w\-]*, .+\(\d{2}/\d{2}/\d{4}\)$")
ROMAN_NUMERAL_PATTERN = re.compile(r"^[IVXLCDM]+$")
NOISE_LINES = {"Selling Guide", "Fannie Mae Single Family"}


class TokenEncoder(Protocol):
    """Protocol for objects that can encode text into token ids."""

    def encode(self, text: str) -> list[int]: # type: ignore
        """Encode text into a list of token ids."""


class FallbackTokenEncoder:
    """Offline-safe approximation used when `tiktoken` assets are unavailable."""

    def encode(self, text: str) -> list[int]:
        """Approximate tokenization with a whitespace split."""
        return text.split() # type: ignore


@dataclass(frozen=True)
class SectionChunk:
    """A section-oriented chunk ready for future embedding."""

    heading: str
    text: str
    page_start: int
    page_end: int
    token_count: int


@dataclass(frozen=True)
class SectionBuffer:
    """In-progress section assembled from heading and body lines."""

    heading: str
    lines: list[tuple[int, str]]


class TextChunker:
    """Chunk guideline pages by detected section headings with token overlap."""

    def __init__(self, chunk_size_tokens: int, chunk_overlap_tokens: int) -> None:
        self._chunk_size_tokens = chunk_size_tokens
        self._chunk_overlap_tokens = chunk_overlap_tokens
        self._encoding = self._build_encoder()

    def chunk_pages(self, pages: list[LoadedPage]) -> list[SectionChunk]:
        """Create section-based chunks directly from loader output."""
        sections = self._build_sections(pages)
        chunks: list[SectionChunk] = []

        for section in sections:
            chunks.extend(self._split_section(section))

        return chunks

    def chunk_document(self, document: ParsedDocument) -> list[DocumentChunk]:
        """Create domain chunks from a parsed document for compatibility."""
        section_chunks = self.chunk_pages(
            [LoadedPage(page_number=page.page_number, text=page.text) for page in document.pages]
        )

        chunks: list[DocumentChunk] = []
        for index, section_chunk in enumerate(section_chunks):
            chunks.append(
                DocumentChunk(
                    chunk_id=f"{document.metadata.document_id}-chunk-{index:04d}",
                    document_id=document.metadata.document_id,
                    text=section_chunk.text,
                    page_start=section_chunk.page_start,
                    page_end=section_chunk.page_end,
                    token_count=section_chunk.token_count,
                    citation=(
                        f"{document.metadata.title}, {section_chunk.heading}, "
                        f"pp. {section_chunk.page_start}-{section_chunk.page_end}"
                    ),
                    metadata={
                        "title": document.metadata.title,
                        "source_path": document.metadata.source_path,
                        "heading": section_chunk.heading,
                    },
                )
            )

        return chunks

    def _build_sections(self, pages: list[LoadedPage]) -> list[SectionBuffer]:
        """Group page text into sections based on detected headings."""
        sections: list[SectionBuffer] = []
        current_heading = "Opening Content"
        current_lines: list[tuple[int, str]] = []

        for page in pages:
            if self._is_table_of_contents_page(page):
                continue

            for raw_line in page.text.splitlines():
                line = self._normalize_line(raw_line)
                if not line or self._is_noise_line(line):
                    continue

                if self._looks_like_heading(line):
                    if self._has_body_content(current_lines, current_heading):
                        sections.append(SectionBuffer(heading=current_heading, lines=current_lines))
                    current_heading = line.rstrip(":")
                    current_lines = [(page.page_number, line.rstrip(":"))]
                    continue

                current_lines.append((page.page_number, line))

        if self._has_body_content(current_lines, current_heading):
            sections.append(SectionBuffer(heading=current_heading, lines=current_lines))

        return sections

    def _split_section(self, section: SectionBuffer) -> list[SectionChunk]:
        """Split a section into token-sized chunks with overlap."""
        if not section.lines:
            return []

        heading_line = section.heading
        heading_tokens = self._count_tokens(heading_line)
        body_lines = section.lines[1:] if section.lines and section.lines[0][1] == heading_line else section.lines

        if not body_lines:
            page_number = section.lines[0][0]
            text = heading_line
            return [
                SectionChunk(
                    heading=heading_line,
                    text=text,
                    page_start=page_number,
                    page_end=page_number,
                    token_count=self._count_tokens(text),
                )
            ]

        chunks: list[SectionChunk] = []
        current_lines: list[tuple[int, str]] = []
        current_tokens = heading_tokens

        for page_number, line in body_lines:
            line_tokens = self._count_tokens(line)
            if current_lines and current_tokens + line_tokens > self._chunk_size_tokens:
                chunks.append(self._build_section_chunk(heading_line, current_lines))
                current_lines = self._overlap_lines(current_lines)
                current_tokens = heading_tokens + self._count_tokens(
                    "\n".join(text for _, text in current_lines)
                )

            current_lines.append((page_number, line))
            current_tokens += line_tokens

        if current_lines:
            chunks.append(self._build_section_chunk(heading_line, current_lines))

        return chunks

    def _build_section_chunk(
        self,
        heading: str,
        lines: list[tuple[int, str]],
    ) -> SectionChunk:
        """Create a chunk model from a heading and a set of body lines."""
        page_numbers = [page_number for page_number, _ in lines]
        body_text = "\n".join(text for _, text in lines)
        text = f"{heading}\n{body_text}".strip()
        return SectionChunk(
            heading=heading,
            text=text,
            page_start=min(page_numbers),
            page_end=max(page_numbers),
            token_count=self._count_tokens(text),
        )

    def _overlap_lines(self, lines: list[tuple[int, str]]) -> list[tuple[int, str]]:
        """Retain trailing lines so the next chunk overlaps with the previous one."""
        retained: list[tuple[int, str]] = []
        token_total = 0

        for page_number, line in reversed(lines):
            line_tokens = self._count_tokens(line)
            if retained and token_total + line_tokens > self._chunk_overlap_tokens:
                break
            retained.insert(0, (page_number, line))
            token_total += line_tokens

        return retained

    def _looks_like_heading(self, line: str) -> bool:
        """Heuristically detect heading-like lines in guideline text."""
        if SECTION_CODE_PATTERN.match(line):
            return True

        if not HEADING_PATTERN.match(line):
            return False

        if line.endswith("."):
            return False

        words = line.split()
        if len(words) > 14:
            return False

        alpha_words = [word for word in words if any(char.isalpha() for char in word)]
        title_case_words = [word for word in alpha_words if word[:1].isupper()]
        return bool(alpha_words) and len(title_case_words) >= max(1, len(alpha_words) - 1)

    def _is_noise_line(self, line: str) -> bool:
        """Filter out repeated headers, footers, and page numbers."""
        if line in NOISE_LINES:
            return True

        if line == "Fannie Mae Copyright Notice":
            return True

        if line.startswith("Published "):
            return True

        if ROMAN_NUMERAL_PATTERN.fullmatch(line):
            return True

        if "..." in line:
            return True

        return line.isdigit()

    def _has_body_content(self, lines: list[tuple[int, str]], heading: str) -> bool:
        """Return whether a section contains content beyond its heading line."""
        if not lines:
            return False

        if heading == "Opening Content":
            return True

        return any(text != heading for _, text in lines)

    def _is_table_of_contents_page(self, page: LoadedPage) -> bool:
        """Detect table-of-contents pages to avoid chunking index entries as content."""
        lines = [self._normalize_line(line) for line in page.text.splitlines() if self._normalize_line(line)]
        dot_leader_lines = sum(1 for line in lines if "..." in line)
        dated_heading_lines = sum(1 for line in lines if SECTION_CODE_PATTERN.match(line))
        return dot_leader_lines >= 3 or dated_heading_lines >= 6

    def _normalize_line(self, line: str) -> str:
        """Normalize PDF text lines for heading detection and chunk assembly."""
        return " ".join(line.split())

    def _count_tokens(self, text: str) -> int:
        """Count tokens using the default OpenAI-compatible encoding."""
        return len(self._encoding.encode(text))

    def _build_encoder(self) -> TokenEncoder:
        """Create a tokenizer, falling back to an offline-safe approximation."""
        try:
            return tiktoken.get_encoding("cl100k_base")
        except Exception:
            return FallbackTokenEncoder()
