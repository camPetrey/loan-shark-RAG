"""CLI command for parsing mortgage guideline PDFs into chunk artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.core.config import get_settings
from src.ingestion.pipeline import IngestionPipeline


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""
    parser = argparse.ArgumentParser(description="Parse PDF guidelines into chunk artifacts.")
    parser.add_argument(
        "--input",
        required=True,
        help="A PDF file path or a directory containing PDF files.",
    )
    parser.add_argument(
        "--output",
        help="Optional JSON output path. If omitted, a default artifact path is used.",
    )
    return parser


def resolve_pdf_paths(input_path: Path) -> list[Path]:
    """Resolve a single PDF or all PDFs within a directory."""
    if input_path.is_file():
        return [input_path]

    if input_path.is_dir():
        return sorted(path for path in input_path.glob("*.pdf") if path.is_file())

    raise FileNotFoundError(f"Input path was not found: {input_path}")


def main() -> None:
    """Run the ingestion preview workflow from the command line."""
    args = build_parser().parse_args()
    settings = get_settings()
    pipeline = IngestionPipeline.from_settings(settings)

    input_path = Path(args.input)
    pdf_paths = resolve_pdf_paths(input_path)
    report = pipeline.run(pdf_paths)

    output_path = Path(args.output) if args.output else settings.default_ingestion_output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "documents": report.document_count,
                "chunks": report.chunk_count,
                "output": str(output_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
