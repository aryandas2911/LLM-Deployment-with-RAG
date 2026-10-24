"""FastAPI endpoint tests that do not require Pinecone or a running LLM.

Run from the project root:

    python -m pytest tests\test_api.py

External services are mocked so the tests are fast, deterministic, and safe to
run without API keys. The mocks replace only the network-facing pieces; request
validation and response handling still use the real FastAPI application.
"""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document

import backend.main as main

client = TestClient(main.app)


def test_health_returns_ok():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_upload_rejects_non_pdf():
    response = client.post(
        "/upload",
        files={"file": ("notes.txt", b"not a pdf", "text/plain")},
    )

    assert response.status_code == 400
    assert "pdf" in response.json()["detail"].lower()


def test_upload_rejects_empty_file():
    response = client.post(
        "/upload",
        files={"file": ("empty.pdf", b"", "application/pdf")},
    )

    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


def test_upload_rejects_invalid_pdf_and_deletes_temp_file(monkeypatch):
    captured = {}
    real_extract = main.extract_pdf_text

    def capture_path(pdf_path):
        captured["temp_path"] = Path(pdf_path)
        return real_extract(pdf_path)

    monkeypatch.setattr(main, "extract_pdf_text", capture_path)

    response = client.post(
        "/upload",
        files={"file": ("broken.pdf", b"not really a pdf", "application/pdf")},
    )

    assert response.status_code == 400
    assert "not a readable pdf" in response.json()["detail"].lower()
    assert not captured["temp_path"].exists()


def test_upload_rejects_oversized_file(monkeypatch):
    # Use a tiny limit so the test does not need to allocate 10 MB.
    monkeypatch.setattr(main, "MAX_UPLOAD_BYTES", 10)

    response = client.post(
        "/upload",
        files={"file": ("large.pdf", b"x" * 11, "application/pdf")},
    )

    assert response.status_code == 413
    assert "too large" in response.json()["detail"].lower()


@pytest.mark.parametrize(
    "payload",
    [
        {"question": "   "},
        {"question": "What is machine learning?", "top_k": 0},
        {"question": "What is machine learning?", "top_k": 11},
    ],
)
def test_chat_rejects_invalid_requests(payload):
    response = client.post("/chat", json=payload)

    assert response.status_code == 400
    assert response.json()["detail"]


def test_chat_retrieval_does_not_call_llm(monkeypatch):
    retrieved = [
        {
            "text": "Machine learning is a field of artificial intelligence.",
            "source": "sample.pdf",
            "page": 2,
            "score": 0.74,
        }
    ]
    calls = {}

    def fake_retrieve(question, top_k=3):
        calls["question"] = question
        calls["top_k"] = top_k
        return retrieved

    def fail_generation(*args, **kwargs):
        raise AssertionError("generate_answer must not be called in retrieval mode.")

    monkeypatch.setattr(main, "retrieve_context", fake_retrieve)
    monkeypatch.setattr(main, "generate_answer", fail_generation)

    response = client.post(
        "/chat",
        json={
            "question": "What is machine learning?",
            "top_k": 2,
            "generate": False,
        },
    )

    assert response.status_code == 200
    assert calls == {"question": "What is machine learning?", "top_k": 2}
    assert response.json() == {
        "question": "What is machine learning?",
        "generate": False,
        "answer": None,
        "sources": retrieved,
    }


def test_generation_returns_503_without_configured_endpoint(monkeypatch):
    # Prevent a real .env file from making this test environment-dependent.
    monkeypatch.setattr(main, "load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.delenv("VLLM_API_BASE_URL", raising=False)
    monkeypatch.delenv("VLLM_API_KEY", raising=False)

    response = client.post(
        "/chat",
        json={"question": "What is machine learning?", "generate": True},
    )

    assert response.status_code == 503
    detail = response.json()["detail"]
    assert "VLLM_API_BASE_URL" in detail
    assert "Kaggle" in detail


def test_generation_uses_existing_rag_pipeline(monkeypatch):
    calls = {}

    monkeypatch.setattr(
        main,
        "_get_llm_settings",
        lambda: ("https://llm.example/v1", "test-key"),
    )

    def fake_generate(question, api_base_url, api_key, top_k=3):
        calls["question"] = question
        calls["api_base_url"] = api_base_url
        calls["api_key"] = api_key
        calls["top_k"] = top_k
        return {
            "answer": "Machine learning learns patterns from data.",
            "sources": [
                {"source": "sample.pdf", "page": 2, "score": 0.74}
            ],
        }

    monkeypatch.setattr(main, "generate_answer", fake_generate)

    response = client.post(
        "/chat",
        json={
            "question": "What is machine learning?",
            "top_k": 3,
            "generate": True,
        },
    )

    assert response.status_code == 200
    assert calls == {
        "question": "What is machine learning?",
        "api_base_url": "https://llm.example/v1",
        "api_key": "test-key",
        "top_k": 3,
    }
    assert response.json()["answer"].startswith("Machine learning")
    assert response.json()["sources"][0]["source"] == "sample.pdf"


def test_generation_maps_timeout_to_504(monkeypatch):
    monkeypatch.setattr(
        main,
        "_get_llm_settings",
        lambda: ("https://llm.example/v1", "test-key"),
    )

    def fail_generation(*args, **kwargs):
        raise RuntimeError("The vLLM request timed out. The server may be loading.")

    monkeypatch.setattr(main, "generate_answer", fail_generation)

    response = client.post(
        "/chat",
        json={"question": "What is machine learning?", "generate": True},
    )

    assert response.status_code == 504
    assert "timed out" in response.json()["detail"].lower()


def test_generation_maps_pinecone_configuration_error_to_503(monkeypatch):
    monkeypatch.setattr(
        main,
        "_get_llm_settings",
        lambda: ("https://llm.example/v1", "test-key"),
    )

    def fail_generation(*args, **kwargs):
        raise ValueError("PINECONE_API_KEY is missing.")

    monkeypatch.setattr(main, "generate_answer", fail_generation)

    response = client.post(
        "/chat",
        json={"question": "What is machine learning?", "generate": True},
    )

    assert response.status_code == 503
    assert "PINECONE_API_KEY" in response.json()["detail"]


def test_upload_uses_existing_pipeline_and_deletes_temp_file(monkeypatch):
    captured = {}
    source_document = Document(
        page_content="Machine learning learns patterns from data.",
        metadata={"source": "temporary-name.pdf", "page": 1},
    )
    chunk = Document(
        page_content="Machine learning learns patterns from data.",
        metadata={"source": "temporary-name.pdf", "page": 1},
    )

    def fake_extract(pdf_path):
        captured["temp_path"] = Path(pdf_path)
        return [source_document]

    def fake_split(documents):
        captured["documents"] = documents
        return [chunk]

    def fake_embeddings(documents):
        captured["embedding_documents"] = documents
        return [[0.0] * 384]

    def fake_index():
        return object()

    def fake_store(documents, embeddings, index):
        captured["stored_documents"] = documents
        captured["stored_embeddings"] = embeddings
        return len(documents)

    monkeypatch.setattr(main, "extract_pdf_text", fake_extract)
    monkeypatch.setattr(main, "split_documents", fake_split)
    monkeypatch.setattr(main, "generate_embeddings", fake_embeddings)
    monkeypatch.setattr(main, "get_pinecone_index", fake_index)
    monkeypatch.setattr(main, "store_documents", fake_store)

    response = client.post(
        "/upload",
        files={
            "file": (
                "sample.pdf",
                b"%PDF-1.4 fake test bytes",
                "application/pdf",
            )
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "filename": "sample.pdf",
        "pages_processed": 1,
        "chunks_created": 1,
        "vectors_uploaded": 1,
        "message": "PDF processed and stored in Pinecone successfully.",
    }
    assert captured["documents"][0].metadata["source"] == "sample.pdf"
    assert captured["embedding_documents"] == [chunk]
    assert captured["stored_documents"] == [chunk]
    assert not captured["temp_path"].exists()
