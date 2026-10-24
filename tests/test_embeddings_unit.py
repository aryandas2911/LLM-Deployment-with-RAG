"""Embedding tests using the real local Sentence Transformers model."""

import numpy as np
import pytest
from langchain_core.documents import Document

from backend.embeddings import (
    EMBEDDING_DIMENSION,
    generate_embeddings,
    load_embedding_model,
)


def test_embedding_dimension_matches_pinecone_index():
    assert EMBEDDING_DIMENSION == 384


def test_model_loads_with_expected_dimension():
    model = load_embedding_model()
    dimension = (
        model.get_embedding_dimension()
        if hasattr(model, "get_embedding_dimension")
        else model.get_sentence_embedding_dimension()
    )

    assert dimension == EMBEDDING_DIMENSION


def test_embeddings_have_expected_dimension_order_and_values():
    documents = [
        Document(page_content="Machine learning learns patterns from data."),
        Document(page_content="Pinecone stores vectors for semantic search."),
    ]

    vectors = generate_embeddings(documents)

    assert len(vectors) == len(documents)
    assert all(len(vector) == EMBEDDING_DIMENSION for vector in vectors)

    expected = load_embedding_model().encode(
        [document.page_content for document in documents],
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    assert np.allclose(np.asarray(vectors), expected, atol=1e-6)


def test_generate_embeddings_rejects_empty_input():
    with pytest.raises(ValueError, match="at least one"):
        generate_embeddings([])


def test_generate_embeddings_rejects_empty_text():
    with pytest.raises(ValueError, match="nonempty page_content"):
        generate_embeddings([Document(page_content="   ")])
