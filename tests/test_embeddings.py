r"""Test local embeddings and a small semantic-retrieval demo.

Usage:
    python tests\test_embeddings.py path\to\sample.pdf

The script reuses the PDF extraction and chunking functions from Task 6, then
computes embeddings locally on the CPU. It does not contact the Kaggle LLM.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from langchain_core.documents import Document

# Running a file inside tests/ puts tests/ first on sys.path. Add the project
# root so the backend package can be imported without installing the project.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.embeddings import (
    EMBEDDING_DIMENSION,
    EMBEDDING_MODEL_NAME,
    generate_embeddings,
    load_embedding_model,
)
from backend.pdf_processor import extract_pdf_text, split_documents


def make_preview(text, limit=200):
    """Return a compact single-line preview for terminal output."""
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[:limit].rstrip() + "..."


def cosine_similarities(query_vector, document_vectors):
    """Return one cosine similarity score per document vector.

    Cosine similarity compares vector direction rather than length. A score
    near 1 means the texts point in a similar direction; near 0 means they are
    mostly unrelated. Using the same embedding model for the question and the
    chunks is required for these directions to be comparable.
    """
    query = np.asarray(query_vector, dtype=np.float32)
    documents = np.asarray(document_vectors, dtype=np.float32)

    numerator = documents @ query
    denominator = np.linalg.norm(documents, axis=1) * np.linalg.norm(query)

    # A zero vector has no direction, so its similarity is defined as 0 here.
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator != 0,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Embed PDF chunks and show a small semantic-similarity demo."
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
    except (FileNotFoundError, ValueError, OSError, RuntimeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    print(f"Embedding model: {EMBEDDING_MODEL_NAME} (CPU)")
    print(f"Number of chunks: {len(chunks)}")
    print(f"Number of embeddings: {len(embeddings)}")
    print(f"Dimension of first embedding: {len(embeddings[0])}")
    print(
        "First 10 values of first embedding: "
        f"{[round(value, 6) for value in embeddings[0][:10]]}"
    )

    if len(embeddings) != len(chunks):
        print(
            "Error: the number of embeddings does not match the number of chunks.",
            file=sys.stderr,
        )
        return 1

    bad_dimensions = [
        index
        for index, vector in enumerate(embeddings, start=1)
        if len(vector) != EMBEDDING_DIMENSION
    ]
    if bad_dimensions:
        print(
            f"Error: these embeddings do not have {EMBEDDING_DIMENSION} dimensions: "
            f"{bad_dimensions}",
            file=sys.stderr,
        )
        return 1

    # The first small demo shows that the model is loaded once and reused.
    model = load_embedding_model()
    if model is not load_embedding_model():
        print("Error: the embedding model was loaded more than once.", file=sys.stderr)
        return 1

    # A question must use the same model as the chunks so their vectors can be
    # compared in the same semantic space.
    question = "What is machine learning?"
    question_vector = generate_embeddings([Document(page_content=question)])[0]
    similarities = cosine_similarities(question_vector, embeddings)

    top_count = min(3, len(chunks))
    top_indices = np.argsort(similarities)[::-1][:top_count]

    print("\nSemantic similarity demo")
    print(f"Question: {question}")
    print("Higher cosine similarity means the chunk is more related to the question.")

    for rank, index in enumerate(top_indices, start=1):
        chunk = chunks[index]
        print(f"\n{rank}. Similarity: {similarities[index]:.4f}")
        print(f"   Source: {chunk.metadata.get('source', 'unknown')}")
        print(f"   Page: {chunk.metadata.get('page', 'unknown')}")
        print(f"   Preview: {make_preview(chunk.page_content)}")

    print("\nEmbedding checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
