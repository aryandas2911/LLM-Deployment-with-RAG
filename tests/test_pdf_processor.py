r"""Manually test PDF extraction and chunking with a sample PDF.

Usage:
    python tests/test_pdf_processor.py path\to\sample.pdf
"""

import argparse
import sys
from pathlib import Path

# When this file is run as a script, Python looks in tests/ first. Add the
# project root so the backend package can be imported without installing it.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.pdf_processor import extract_pdf_text, split_documents


def make_preview(text, limit=200):
    """Return a single-line preview so terminal output stays readable."""
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[:limit].rstrip() + "..."


def main():
    parser = argparse.ArgumentParser(
        description="Extract text from a PDF and split it into chunks."
    )
    parser.add_argument("pdf_path", help="Path to a small sample PDF file")
    args = parser.parse_args()

    try:
        documents = extract_pdf_text(args.pdf_path)
        chunks = split_documents(documents)
    except (FileNotFoundError, ValueError, OSError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    print(f"Pages with extractable text: {len(documents)}", flush=True)
    print(f"Generated chunks: {len(chunks)}", flush=True)

    if not documents:
        print(
            "No extractable text was found. This PDF may be scanned and need "
            "OCR, which is outside this mini-project's scope.",
            file=sys.stderr,
        )
        return 1

    missing_metadata = []
    for index, chunk in enumerate(chunks, start=1):
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        if not source or not isinstance(page, int) or page < 1:
            missing_metadata.append(index)

    if missing_metadata:
        print(
            "Error: these chunks lost their source or page metadata: "
            f"{missing_metadata}",
            file=sys.stderr,
        )
        return 1

    print("\nFirst chunks:")
    for index, chunk in enumerate(chunks[:3], start=1):
        print(f"\nChunk {index}")
        print(f"  Source: {chunk.metadata['source']}")
        print(f"  Page: {chunk.metadata['page']}")
        print(f"  Preview: {make_preview(chunk.page_content)}")

    print("\nAll chunks retain source and page metadata.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
