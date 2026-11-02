#!/usr/bin/env python3
"""Extract text from a PDF and save it as a Markdown file."""

import argparse
from pathlib import Path
import sys

from PyPDF2 import PdfReader


def convert_pdf_to_md(pdf_path: Path, md_path: Path) -> None:
    """Extract each PDF page's text and write it to a Markdown file."""
    reader = PdfReader(str(pdf_path))
    sections = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        sections.append(f"## Page {page_number}\n\n{text}")

    md_path.write_text("\n\n---\n\n".join(sections) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract text from a PDF and save it as Markdown."
    )
    parser.add_argument("pdf", type=Path, help="PDF file to convert")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="output Markdown path (default: same path as the PDF with a .md extension)",
    )
    args = parser.parse_args()

    if not args.pdf.is_file():
        parser.error(f"PDF file does not exist: {args.pdf}")

    output_path = args.output or args.pdf.with_suffix(".md")
    try:
        convert_pdf_to_md(args.pdf, output_path)
    except Exception as exc:
        print(f"Error converting {args.pdf}: {exc}", file=sys.stderr)
        return 1

    print(f"Created {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
