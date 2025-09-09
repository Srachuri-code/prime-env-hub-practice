"""
Simple PDF to Markdown converter.

Usage:
  python context/pdf_to_md.py INPUT_PDF [--output OUTPUT_MD] [--force]

Notes:
  - Requires the 'pdfminer.six' package.
  - Produces a lightweight Markdown approximation by normalizing bullets and paragraphs.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from typing import Optional


def convert_pdf_to_markdown(pdf_path: str, output_path: str, overwrite: bool = False) -> None:
    try:
        from pdfminer.high_level import extract_text
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Missing dependency: pdfminer.six. Install with 'pip install pdfminer.six'"
        ) from exc

    if not os.path.isfile(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    if os.path.exists(output_path) and not overwrite:
        raise FileExistsError(
            f"Output file already exists: {output_path}. Use --force to overwrite."
        )

    # Extract raw text
    raw_text: str = extract_text(pdf_path) or ""

    # Normalize newlines
    text = raw_text.replace("\r\n", "\n").replace("\r", "\n")

    # Fix common hyphenation across line breaks (e.g., "exam-\nple" -> "example")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

    # Convert bullet symbols at line-start to Markdown '- '
    text = re.sub(r"(?m)^\s*[•●◦▪‣·]\s*", "- ", text)

    # Normalize numbered list styles like "1)" or "(1)" to Markdown "1."
    text = re.sub(r"(?m)^\s*\(?(\d+)\)\s+", r"\1. ", text)

    # Collapse excessive blank lines to at most two
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Trim trailing spaces on each line
    text = re.sub(r"(?m)[ \t]+$", "", text)

    # Prepend a small header comment
    header = f"<!-- Generated from '{os.path.basename(pdf_path)}' by pdf_to_md.py -->\n\n"
    markdown_output = header + text

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(markdown_output)


def guess_output_path(pdf_path: str) -> str:
    base, _ = os.path.splitext(pdf_path)
    return base + ".md"


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert a PDF to Markdown (simple heuristics)")
    parser.add_argument("input_pdf", help="Path to the input PDF file")
    parser.add_argument("--output", "-o", dest="output_md", help="Output Markdown path (default: same name with .md)")
    parser.add_argument("--force", "-f", action="store_true", help="Overwrite output if it exists")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    input_pdf = os.path.abspath(args.input_pdf)
    output_md = os.path.abspath(args.output_md) if args.output_md else guess_output_path(input_pdf)

    try:
        convert_pdf_to_markdown(input_pdf, output_md, overwrite=args.force)
    except Exception as exc:
        sys.stderr.write(f"Error: {exc}\n")
        return 1

    print(f"Wrote Markdown: {output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


