"""Pinecone logic tests using mocks instead of a live Pinecone account."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from langchain_core.documents import Document

import backend.vector_store as vector_store


def _document(text="Machine learning learns patterns.", page=1):
    return Document(
        page_content=text,
        metadata={"source": "sample.pdf", "page": page},
    )


def test_store_documents_creates_stable_ids_and_metadata():
    index = Mock()
    index.upsert.return_value = SimpleNamespace(upserted_count=1)
    document = _document()

    first_count = vector_store.store_documents(
        [document], [[0.0] * 384], index
    )
    first_vectors = index.upsert.call_args.kwargs["vectors"]

    second_count = vector_store.store_documents(
        [document], [[0.0] * 384], index
    )
    second_vectors = index.upsert.call_args.kwargs["vectors"]

    assert first_count == 1
    assert second_count == 1
    assert first_vectors[0]["id"] == second_vectors[0]["id"]
    assert first_vectors[0]["metadata"] == {
        "source": "sample.pdf",
        "page": 1,
        "text": "Machine learning learns patterns.",
    }
    assert index.upsert.call_args.kwargs["namespace"] == "rag-mini-project"


def test_store_documents_rejects_mismatched_input_lengths():
    with pytest.raises(ValueError, match="number of documents"):
        vector_store.store_documents(
            [_document()],
            [[0.0] * 384, [0.0] * 384],
            Mock(),
        )


def test_store_documents_rejects_wrong_embedding_dimension():
    with pytest.raises(ValueError, match="384"):
        vector_store.store_documents(
            [_document()],
            [[0.0] * 10],
            Mock(),
        )


def test_search_documents_preserves_results_and_query_settings(monkeypatch):
    monkeypatch.setattr(
        vector_store,
        "generate_embeddings",
        lambda documents: [[0.1] * 384],
    )
    index = Mock()
    index.query.return_value = SimpleNamespace(
        matches=[
            SimpleNamespace(
                metadata={
                    "text": "Machine learning learns patterns.",
                    "source": "sample.pdf",
                    "page": 2,
                },
                score=0.82,
            )
        ]
    )

    results = vector_store.search_documents("What is ML?", index, top_k=4)

    assert results == [
        {
            "text": "Machine learning learns patterns.",
            "source": "sample.pdf",
            "page": 2,
            "score": 0.82,
        }
    ]
    assert index.query.call_args.kwargs["top_k"] == 4
    assert index.query.call_args.kwargs["include_metadata"] is True
    assert index.query.call_args.kwargs["namespace"] == "rag-mini-project"


@pytest.mark.parametrize("matches", [None, []])
def test_search_documents_handles_empty_results(monkeypatch, matches):
    monkeypatch.setattr(
        vector_store,
        "generate_embeddings",
        lambda documents: [[0.1] * 384],
    )
    index = Mock()
    index.query.return_value = SimpleNamespace(matches=matches)

    assert vector_store.search_documents("What is ML?", index) == []


def test_get_pinecone_index_validates_existing_index(monkeypatch):
    description = SimpleNamespace(
        dimension=384,
        metric="cosine",
        status=SimpleNamespace(ready=True),
    )

    class FakePinecone:
        def has_index(self, name):
            assert name == "test-index"
            return True

        def describe_index(self, name):
            assert name == "test-index"
            return description

        def Index(self, name):
            assert name == "test-index"
            return "index-client"

    monkeypatch.setattr(vector_store, "load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.setattr(vector_store, "Pinecone", lambda api_key: FakePinecone())
    monkeypatch.setenv("PINECONE_API_KEY", "test-key")
    monkeypatch.setenv("PINECONE_INDEX_NAME", "test-index")

    assert vector_store.get_pinecone_index() == "index-client"


def test_get_pinecone_index_rejects_incompatible_dimension(monkeypatch):
    description = SimpleNamespace(
        dimension=1536,
        metric="cosine",
        status=SimpleNamespace(ready=True),
    )

    class FakePinecone:
        def has_index(self, name):
            return True

        def describe_index(self, name):
            return description

    monkeypatch.setattr(vector_store, "load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.setattr(vector_store, "Pinecone", lambda api_key: FakePinecone())
    monkeypatch.setenv("PINECONE_API_KEY", "test-key")
    monkeypatch.setenv("PINECONE_INDEX_NAME", "test-index")

    with pytest.raises(RuntimeError, match="dimension"):
        vector_store.get_pinecone_index()


def test_get_pinecone_index_rejects_incompatible_metric(monkeypatch):
    description = SimpleNamespace(
        dimension=384,
        metric="euclidean",
        status=SimpleNamespace(ready=True),
    )

    class FakePinecone:
        def has_index(self, name):
            return True

        def describe_index(self, name):
            return description

    monkeypatch.setattr(vector_store, "load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.setattr(vector_store, "Pinecone", lambda api_key: FakePinecone())
    monkeypatch.setenv("PINECONE_API_KEY", "test-key")
    monkeypatch.setenv("PINECONE_INDEX_NAME", "test-index")

    with pytest.raises(RuntimeError, match="cosine"):
        vector_store.get_pinecone_index()


def test_get_pinecone_index_creates_missing_index(monkeypatch):
    captured = {}
    description = SimpleNamespace(
        dimension=384,
        metric="cosine",
        status=SimpleNamespace(ready=True),
    )

    class FakePinecone:
        def has_index(self, name):
            return False

        def create_index(self, **kwargs):
            captured["create"] = kwargs

        def describe_index(self, name):
            return description

        def Index(self, name):
            return "index-client"

    monkeypatch.setattr(vector_store, "load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.setattr(vector_store, "Pinecone", lambda api_key: FakePinecone())
    monkeypatch.setenv("PINECONE_API_KEY", "test-key")
    monkeypatch.setenv("PINECONE_INDEX_NAME", "test-index")

    assert vector_store.get_pinecone_index() == "index-client"
    assert captured["create"]["name"] == "test-index"
    assert captured["create"]["dimension"] == 384
    assert captured["create"]["metric"] == "cosine"
    assert captured["create"]["vector_type"] == "dense"


def test_get_pinecone_index_requires_api_key(monkeypatch):
    monkeypatch.setattr(vector_store, "load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.delenv("PINECONE_API_KEY", raising=False)
    monkeypatch.setenv("PINECONE_INDEX_NAME", "test-index")

    with pytest.raises(ValueError, match="PINECONE_API_KEY"):
        vector_store.get_pinecone_index()
