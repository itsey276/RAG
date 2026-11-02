#!/usr/bin/env python3
"""Convert PDFs in ./sources into embedded chunks in ./chunks.db."""

import argparse
import os
import sqlite3
import sys
import tempfile
from pathlib import Path

from embed_chunks import DEFAULT_BATCH_SIZE, embed_database
from md_to_chunks import store_chunks
from pdf_to_md import convert_pdf_to_md


def build_vector_store(
    sources_dir: Path,
    database_path: Path,
    api_key: str,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> tuple[int, int]:
    """Convert every PDF in sources_dir, store its chunks, then embed them."""
    pdf_files = sorted(
        (path for path in sources_dir.iterdir() if path.is_file() and path.suffix.lower() == ".pdf"),
        key=lambda path: path.name.lower(),
    )
    if not pdf_files:
        raise ValueError(f"No PDF files found in {sources_dir}")

    chunk_count = 0
    with tempfile.TemporaryDirectory(prefix="rag-pipeline-") as temp_dir:
        temp_path = Path(temp_dir)
        for pdf_path in pdf_files:
            markdown_path = temp_path / f"{pdf_path.stem}.md"
            convert_pdf_to_md(pdf_path, markdown_path)
            count = store_chunks(markdown_path, database_path, source_file_path=pdf_path)
            chunk_count += count
            print(f"Processed {pdf_path}: {count} chunks")

    embedded_count = embed_database(database_path, api_key, batch_size=batch_size)
    return chunk_count, embedded_count


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Convert PDFs from a source directory and embed their chunks into SQLite."
    )
    parser.add_argument(
        "--sources",
        type=Path,
        default=script_dir / "sources",
        help="directory containing PDFs (default: ./sources beside this script)",
    )
    parser.add_argument(
        "--database",
        type=Path,
        default=script_dir / "chunks.db",
        help="output SQLite vector store (default: ./chunks.db beside this script)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"embedding API batch size (default: {DEFAULT_BATCH_SIZE})",
    )
    args = parser.parse_args()

    api_key = os.environ.get("UF_API_KEY")
    if not api_key:
        parser.error("UF_API_KEY environment variable is not set")
    if not args.sources.is_dir():
        parser.error(f"Sources directory does not exist: {args.sources}")
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")

    try:
        chunk_count, embedded_count = build_vector_store(
            args.sources, args.database, api_key, args.batch_size
        )
    except (OSError, UnicodeError, sqlite3.Error, ValueError, RuntimeError) as exc:
        print(f"Pipeline failed: {exc}", file=sys.stderr)
        return 1

    print(
        f"Done: processed {chunk_count} chunks; generated {embedded_count} embeddings "
        f"in {args.database}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
