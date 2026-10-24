r"""Test the Pinecone vector-store integration with a sample PDF.

Usage:
    python tests\test_vector_store.py path\to\sample.pdf

This script uploads vectors to Pinecone and queries them. It does not contact
the Kaggle LLM server, and it does not delete any indexes or vectors.
"""

import argparse
import sys
import time
from pathlib import Path

# Add the project root so backend imports work when this file is run directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.embeddings import generate_embeddings
from backend.pdf_processor import extract_pdf_text, split_documents
from backend.vector_store import get_pinecone_index, search_documents, store_documents


def make_preview(text, limit=200):
    """Return a compact single-line preview for terminal output."""
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[:limit].rstrip() + "..."


def main():
    parser = argparse.ArgumentParser(
        description="Upload PDF chunks to Pinecone and run a semantic query."
    )
    parser.add_argument("pdf_path", help="Path to a small sample PDF file")
    args = parser.parse_args()

    try:
        documents = extract_pdf_text(args.pdf_path)
        chunks = split_documents(documents)
        if not chunks:
            print(
                "No text chunks were produced. The PDF may be empty or scanned.",
                file=sys.stderr,
            )
            return 1

        embeddings = generate_embeddings(chunks)
        index = get_pinecone_index()
        store_documents(chunks, embeddings, index)

        # Pinecone writes are eventually consistent. A short pause gives a new
        # upload a chance to become visible, though results may still take longer
        # to appear for a large index.
        print("Waiting briefly for Pinecone to make the vectors searchable...")
        time.sleep(2)

        question = "What is machine learning?"
        results = search_documents(question, index, top_k=3)
    except (FileNotFoundError, ValueError, OSError, RuntimeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    except Exception as error:
        print(
            f"Pinecone or network error: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return 1

    if not results:
        print(
            "No matches were returned. Pinecone may still be indexing the "
            "vectors; wait a few seconds and run the query again.",
            file=sys.stderr,
        )
        return 1

    print(f"\nQuestion: {question}")
    print("Higher similarity scores mean the chunk is more semantically related.")

    for rank, result in enumerate(results, start=1):
        if (
            not result.get("text")
            or not result.get("source")
            or result.get("page") is None
            or result.get("score") is None
        ):
            print(
                f"Error: result {rank} is missing required metadata.",
                file=sys.stderr,
            )
            return 1

        print(f"\n{rank}. Similarity: {result['score']:.4f}")
        print(f"   Source: {result['source']}")
        print(f"   Page: {result['page']}")
        print(f"   Preview: {make_preview(result['text'])}")

    print("\nAll returned results contain text, source, page, and score metadata.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
