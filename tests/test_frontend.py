from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import requests
from streamlit.testing.v1 import AppTest

APP_PATH = Path(__file__).resolve().parents[1] / "frontend" / "app.py"

UPLOAD_RESPONSE = {
    "filename": "sample.pdf",
    "pages_processed": 3,
    "chunks_created": 8,
    "vectors_uploaded": 8,
    "message": "PDF processed and stored in Pinecone successfully.",
}


class FakeResponse:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self.payload = payload if payload is not None else {"status": "ok"}

    def json(self):
        return self.payload


@contextmanager
def app_with_mocks(get_response=None, post_response=None):
    if get_response is None:
        get_response = FakeResponse(payload={"status": "ok"})
    if post_response is None:
        post_response = FakeResponse(payload={})

    with patch("requests.get", return_value=get_response), patch(
        "requests.post", return_value=post_response
    ) as post:
        app = AppTest.from_file(str(APP_PATH), default_timeout=10).run()
        yield app, post


def all_rendered_text(app):
    values = []
    for element_type in (
        "title",
        "header",
        "subheader",
        "markdown",
        "caption",
        "text",
        "success",
        "error",
        "info",
        "warning",
    ):
        for element in getattr(app, element_type):
            values.append(str(element.value))
    for metric in app.metric:
        values.append(str(metric.label))
        values.append(str(metric.value))
    return "\n".join(values)


def test_health_status_does_not_expose_secret_values(monkeypatch):
    monkeypatch.setenv("PINECONE_API_KEY", "pinecone-secret-value")
    monkeypatch.setenv("VLLM_API_KEY", "vllm-secret-value")

    with app_with_mocks() as (app, _):
        assert not app.exception
        assert [item.value for item in app.sidebar.success] == [
            "Backend connected."
        ]
        rendered = all_rendered_text(app)
        assert "pinecone-secret-value" not in rendered
        assert "vllm-secret-value" not in rendered
        assert "does not prove that an LLM endpoint is available" in rendered


def test_backend_unavailable_is_explained():
    with patch(
        "requests.get",
        side_effect=requests.ConnectionError("backend is down"),
    ):
        app = AppTest.from_file(str(APP_PATH), default_timeout=10).run()

    assert not app.exception
    assert [item.value for item in app.sidebar.error] == [
        "Backend unavailable."
    ]
    assert "Start the backend and refresh" in all_rendered_text(app)


def test_upload_displays_stats_and_does_not_repeat_same_file():
    with app_with_mocks(
        post_response=FakeResponse(payload=UPLOAD_RESPONSE)
    ) as (app, post):
        pdf_bytes = b"%PDF-1.4 fake test document"
        app.file_uploader(key="pdf_uploader").set_value(
            ("sample.pdf", pdf_bytes, "application/pdf")
        )
        app.button(key="upload_pdf_button").click()
        app.run()

        assert not app.exception
        assert post.call_count == 1
        assert post.call_args.kwargs["files"]["file"] == (
            "sample.pdf",
            pdf_bytes,
            "application/pdf",
        )
        assert (
            "PDF processed and stored in Pinecone successfully."
            in all_rendered_text(app)
        )

        metrics = {item.label: item.value for item in app.metric}
        assert metrics == {
            "Pages processed": "3",
            "Chunks created": "8",
            "Vectors uploaded": "8",
        }

        app.button(key="upload_pdf_button").click()
        app.run()

        assert post.call_count == 1
        assert "already uploaded in this session" in all_rendered_text(app)


def test_upload_rejects_missing_file_without_calling_backend():
    with app_with_mocks() as (app, post):
        app.button(key="upload_pdf_button").click()
        app.run()

        assert post.call_count == 0
        assert "Choose a PDF before uploading." in all_rendered_text(app)


def test_upload_invalid_response_is_handled():
    with app_with_mocks(
        post_response=FakeResponse(payload={"unexpected": "response"})
    ) as (app, _):
        app.file_uploader(key="pdf_uploader").set_value(
            (
                "sample.pdf",
                b"%PDF-1.4 fake test document",
                "application/pdf",
            )
        )
        app.button(key="upload_pdf_button").click()
        app.run()

        assert not app.exception
        assert "missing expected fields" in all_rendered_text(app)
        assert not app.metric


def test_empty_question_is_not_sent():
    with app_with_mocks() as (app, post):
        app.chat_input(key="chat_input").set_value("   ")
        app.run()

        assert post.call_count == 0
        assert "Enter a non-empty question" in all_rendered_text(app)
        assert not app.session_state["messages"]


def test_document_search_renders_sources_and_keeps_history():
    chat_response = {
        "question": "What is machine learning?",
        "generate": False,
        "answer": None,
        "sources": [
            {
                "text": "Machine learning is a field of artificial intelligence.",
                "source": "sample.pdf",
                "page": 2,
                "score": 0.7432,
            }
        ],
    }
    with app_with_mocks(
        post_response=FakeResponse(payload=chat_response)
    ) as (app, post):
        app.chat_input(key="chat_input").set_value(
            "What is machine learning?"
        )
        app.run()

        assert not app.exception
        assert post.call_count == 1
        assert post.call_args.kwargs["json"] == {
            "question": "What is machine learning?",
            "top_k": 3,
            "generate": False,
        }

        rendered = all_rendered_text(app)
        assert "Retrieved document content" in rendered
        assert "does not generate an AI answer" in rendered
        assert "sample.pdf | page 2 | similarity 0.7432" in rendered
        assert (
            "Machine learning is a field of artificial intelligence."
            in rendered
        )
        assert len(app.chat_message) == 2
        assert app.session_state["messages"][0]["content"] == (
            "What is machine learning?"
        )

        app.run()

        assert post.call_count == 1
        assert len(app.chat_message) == 2
        assert len(app.session_state["messages"]) == 2


def test_ai_answer_renders_generated_answer_and_sources():
    chat_response = {
        "question": "What is machine learning?",
        "generate": True,
        "answer": "Machine learning learns patterns from data.",
        "sources": [
            {
                "source": "sample.pdf",
                "page": 2,
                "score": 0.7432,
            }
        ],
    }
    with app_with_mocks(
        post_response=FakeResponse(payload=chat_response)
    ) as (app, post):
        app.radio(key="response_mode").set_value("AI Answer")
        app.run()
        app.chat_input(key="chat_input").set_value(
            "What is machine learning?"
        )
        app.run()

        assert not app.exception
        assert post.call_count == 1
        assert post.call_args.kwargs["json"]["generate"] is True
        rendered = all_rendered_text(app)
        assert "Machine learning learns patterns from data." in rendered
        assert "sample.pdf | page 2 | similarity 0.7432" in rendered


def test_ai_answer_503_is_explained_without_fallback():
    detail = (
        "Generation mode requires a permitted, reachable vLLM endpoint. "
        "Set VLLM_API_BASE_URL and VLLM_API_KEY in .env."
    )
    with app_with_mocks(
        post_response=FakeResponse(
            status_code=503,
            payload={"detail": detail},
        )
    ) as (app, post):
        app.radio(key="response_mode").set_value("AI Answer")
        app.run()
        app.chat_input(key="chat_input").set_value(
            "What is machine learning?"
        )
        app.run()

        assert not app.exception
        assert post.call_count == 1
        assert post.call_args.kwargs["json"]["generate"] is True
        assert detail in all_rendered_text(app)
        assert app.session_state["messages"][-1]["error"] == detail
