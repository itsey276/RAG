#!/usr/bin/env python3
"""Generate OpenAI-compatible embeddings for chunks in a SQLite database."""

import argparse
import os
import sqlite3
import struct
import sys
from pathlib import Path
from typing import Any

MODEL = "nomic-embed-text-v1.5"
BASE_URL = "https://api.ai.it.ufl.edu/v1"
DEFAULT_BATCH_SIZE = 64


def vector_to_blob(vector: list[float]) -> bytes:
    """Serialize an embedding as little-endian IEEE-754 32-bit floats."""
    return struct.pack(f"<{len(vector)}f", *vector)


def ensure_embedding_columns(connection: sqlite3.Connection) -> None:
    """Add embedding storage columns to an existing chunks table if needed."""
    columns = {
        row[1] for row in connection.execute("PRAGMA table_info(chunks)")
    }
    if not columns:
        raise ValueError("Database does not contain a 'chunks' table")
    if "embedding" not in columns:
        connection.execute("ALTER TABLE chunks ADD COLUMN embedding BLOB")
    if "embedding_model" not in columns:
        connection.execute("ALTER TABLE chunks ADD COLUMN embedding_model TEXT")


def embed_database(
    database_path: Path,
    api_key: str,
    batch_size: int = DEFAULT_BATCH_SIZE,
    force: bool = False,
    client: Any = None,
) -> int:
    """Embed chunks that do not yet have an embedding; return number processed."""
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if client is None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "The openai package is required. Install it with: pip install openai"
            ) from exc
        client = OpenAI(api_key=api_key, base_url=BASE_URL)

    with sqlite3.connect(database_path) as connection:
        ensure_embedding_columns(connection)
        condition = "" if force else "WHERE embedding IS NULL"
        rows = connection.execute(
            f"SELECT id, content FROM chunks {condition} ORDER BY id"
        ).fetchall()

        processed = 0
        for offset in range(0, len(rows), batch_size):
            batch = rows[offset : offset + batch_size]
            response = client.embeddings.create(
                model=MODEL,
                input=[content for _, content in batch],
            )
            vectors = sorted(response.data, key=lambda item: item.index)
            if len(vectors) != len(batch):
                raise RuntimeError(
                    f"Embedding API returned {len(vectors)} vectors for {len(batch)} chunks"
                )

            connection.executemany(
                "UPDATE chunks SET embedding = ?, embedding_model = ? WHERE id = ?",
                [
                    (vector_to_blob(item.embedding), MODEL, row[0])
                    for row, item in zip(batch, vectors)
                ],
            )
            processed += len(batch)

    return processed


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Add nomic-embed-text-v1.5 embeddings to chunks in SQLite."
    )
    parser.add_argument("database", type=Path, help="SQLite database created by md_to_chunks.py")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"number of chunks per API request (default: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="re-embed chunks that already have embeddings",
    )
    args = parser.parse_args()

    api_key = os.environ.get("UF_API_KEY")
    if not api_key:
        parser.error("UF_API_KEY environment variable is not set")
    if not args.database.is_file():
        parser.error(f"Database file does not exist: {args.database}")

    try:
        count = embed_database(args.database, api_key, args.batch_size, args.force)
    except (OSError, sqlite3.Error, ValueError, RuntimeError) as exc:
        print(f"Error embedding chunks: {exc}", file=sys.stderr)
        return 1

    print(f"Embedded {count} chunks in {args.database}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
