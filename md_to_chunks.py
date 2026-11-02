#!/usr/bin/env python3
"""Split a Markdown file into overlapping character chunks stored in SQLite."""

import argparse
import sqlite3
from pathlib import Path

CHUNK_SIZE = 2000
OVERLAP = 50


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = OVERLAP):
    """Yield (start, end, text) windows with overlap between adjacent chunks."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size")

    step = chunk_size - overlap
    for start in range(0, len(text), step):
        end = min(start + chunk_size, len(text))
        yield start, end, text[start:end]


def store_chunks(
    markdown_path: Path,
    database_path: Path,
    source_file_path: Path | None = None,
) -> int:
    """Store Markdown chunks, optionally recording a separate source-file path."""
    text = markdown_path.read_text(encoding="utf-8")
    source_file = str((source_file_path or markdown_path).resolve())
    chunks = list(chunk_text(text))

    database_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            """CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY,
                source_file TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                start_char INTEGER NOT NULL,
                end_char INTEGER NOT NULL,
                content TEXT NOT NULL,
                UNIQUE (source_file, chunk_index)
            )"""
        )
        connection.execute("DELETE FROM chunks WHERE source_file = ?", (source_file,))
        connection.executemany(
            """INSERT INTO chunks
                (source_file, chunk_index, start_char, end_char, content)
                VALUES (?, ?, ?, ?, ?)""",
            [
                (source_file, index, start, end, content)
                for index, (start, end, content) in enumerate(chunks)
            ],
        )

    return len(chunks)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Store overlapping 2,000-character Markdown chunks in SQLite."
    )
    parser.add_argument("markdown", type=Path, help="Markdown file to chunk")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="SQLite database path (default: same path as the Markdown file with a .db extension)",
    )
    args = parser.parse_args()

    if not args.markdown.is_file():
        parser.error(f"Markdown file does not exist: {args.markdown}")

    database_path = args.output or args.markdown.with_suffix(".db")
    try:
        count = store_chunks(args.markdown, database_path)
    except (OSError, UnicodeError, sqlite3.Error, ValueError) as exc:
        parser.error(f"Could not create chunks: {exc}")

    print(f"Stored {count} chunks in {database_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
