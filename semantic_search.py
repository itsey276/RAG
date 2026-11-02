#!/usr/bin/env python3
"""Search embedded SQLite chunks using cosine similarity."""

import argparse
import math
import os
import sqlite3
import struct
import sys
from pathlib import Path
from typing import Any

from embed_chunks import BASE_URL, MODEL


def blob_to_vector(blob: bytes) -> tuple[float, ...]:
    """Deserialize little-endian IEEE-754 32-bit floats from a SQLite BLOB."""
    if len(blob) % 4:
        raise ValueError("Stored embedding has an invalid byte length")
    if not blob:
        raise ValueError("Stored embedding is empty")
    return struct.unpack(f"<{len(blob) // 4}f", blob)


def cosine_similarity(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    """Return cosine similarity for two vectors of equal length."""
    if len(left) != len(right):
        raise ValueError("Embedding dimensions do not match")
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return sum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)


def search_database(
    query: str,
    database_path: Path,
    api_key: str,
    top_k: int = 5,
    client: Any = None,
) -> list[dict[str, Any]]:
    """Embed query and return the most similar chunks, ranked by cosine score."""
    if not query.strip():
        raise ValueError("Query must not be empty")
    if top_k < 1:
        raise ValueError("top_k must be at least 1")

    if client is None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "The openai package is required. Install it with: pip install openai"
            ) from exc
        client = OpenAI(api_key=api_key, base_url=BASE_URL)

    response = client.embeddings.create(model=MODEL, input=[query])
    if not response.data:
        raise RuntimeError("Embedding API returned no vectors for the query")
    query_vector = tuple(response.data[0].embedding)

    with sqlite3.connect(database_path) as connection:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(chunks)")}
        if not columns:
            raise ValueError("Database does not contain a 'chunks' table")
        required = {"id", "source_file", "chunk_index", "content", "embedding"}
        missing = required - columns
        if missing:
            raise ValueError(
                "Chunks table is missing required columns: " + ", ".join(sorted(missing))
            )
        rows = connection.execute(
            """SELECT id, source_file, chunk_index, content, embedding, embedding_model
               FROM chunks WHERE embedding IS NOT NULL ORDER BY id"""
            if "embedding_model" in columns
            else """SELECT id, source_file, chunk_index, content, embedding, NULL
               FROM chunks WHERE embedding IS NOT NULL ORDER BY id"""
        ).fetchall()

    if not rows:
        raise ValueError("No embedded chunks found; run embed_chunks.py first")

    results = []
    for chunk_id, source_file, chunk_index, content, blob, embedding_model in rows:
        if embedding_model and embedding_model != MODEL:
            raise ValueError(
                f"Chunk {chunk_id} uses model {embedding_model!r}, expected {MODEL!r}"
            )
        vector = blob_to_vector(blob)
        score = cosine_similarity(query_vector, vector)
        results.append(
            {
                "id": chunk_id,
                "source_file": source_file,
                "chunk_index": chunk_index,
                "content": content,
                "score": score,
            }
        )

    results.sort(key=lambda result: result["score"], reverse=True)
    return results[:top_k]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=f"Search chunks.db using {MODEL} embeddings."
    )
    parser.add_argument("query", help="text to search for")
    parser.add_argument(
        "--database",
        type=Path,
        default=Path("chunks.db"),
        help="SQLite database path (default: chunks.db)",
    )
    parser.add_argument(
        "--top-k", type=int, default=5, help="number of results to show (default: 5)"
    )
    args = parser.parse_args()

    api_key = os.environ.get("UF_API_KEY")
    if not api_key:
        parser.error("UF_API_KEY environment variable is not set")
    if not args.database.is_file():
        parser.error(f"Database file does not exist: {args.database}")
    if args.top_k < 1:
        parser.error("--top-k must be at least 1")

    try:
        results = search_database(args.query, args.database, api_key, args.top_k)
    except (OSError, sqlite3.Error, ValueError, RuntimeError) as exc:
        print(f"Error searching chunks: {exc}", file=sys.stderr)
        return 1

    for rank, result in enumerate(results, start=1):
        print(
            f"{rank}. score={result['score']:.4f} "
            f"source={result['source_file']} chunk={result['chunk_index']}"
        )
        print(result["content"])
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
