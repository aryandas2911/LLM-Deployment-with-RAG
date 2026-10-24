r"""Test local retrieval and prompt construction without calling an LLM.

Usage:
    python tests\test_rag_pipeline.py "What is machine learning?"

This script uses Pinecone and the local embedding model. It does not contact
Kaggle, does not require a vLLM server, and never prints API keys.
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.rag_pipeline import NOT_FOUND_ANSWER, build_rag_prompt, retrieve_context

PROMPT_PREVIEW_LIMIT = 3000


def make_preview(text, limit=PROMPT_PREVIEW_LIMIT):
    """Keep terminal output readable while showing the important prompt parts."""
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "\n...[prompt preview truncated]"


def main():
    parser = argparse.ArgumentParser(
        description="Retrieve chunks and preview the RAG prompt without an LLM."
    )
    parser.add_argument("question", help="Question to embed and search for")
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of Pinecone chunks to retrieve (default: 3)",
    )
    args = parser.parse_args()

    try:
        chunks = retrieve_context(args.question, top_k=args.top_k)
        messages = build_rag_prompt(args.question, chunks)
    except (FileNotFoundError, ValueError, OSError, RuntimeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    except Exception as error:
        print(
            f"Unexpected error: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return 1

    print(f"Question: {args.question.strip()}")
    print(f"Retrieved chunks: {len(chunks)}")

    if not chunks:
        print(f"No chunks were retrieved. The LLM would return: {NOT_FOUND_ANSWER}")
    else:
        print("\nRetrieved sources:")
        for rank, chunk in enumerate(chunks, start=1):
            score = chunk.get("score")
            score_text = f"{score:.4f}" if isinstance(score, (int, float)) else "unknown"
            print(
                f"{rank}. {chunk.get('source', 'unknown')} "
                f"(page {chunk.get('page', 'unknown')}, score {score_text})"
            )

    print("\nPrompt preview:")
    for message in messages:
        role = getattr(message, "type", "message")
        content = getattr(message, "content", "")
        print(f"\n[{role}]")
        print(make_preview(content if isinstance(content, str) else str(content)))

    print("\nNo LLM call was made. This test works without Kaggle or vLLM.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
