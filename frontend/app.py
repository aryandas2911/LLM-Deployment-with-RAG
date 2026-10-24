"""Streamlit frontend for the local FastAPI RAG backend.

The page uploads a PDF to FastAPI, shows indexing statistics, and keeps a
session-only chat. Document Search retrieves Pinecone chunks; AI Answer asks
the backend to generate an answer when a permitted LLM endpoint is configured.
"""

import hashlib
import os
from pathlib import Path

import requests
import streamlit as st
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

DEFAULT_API_URL = "http://127.0.0.1:8001"
_configured_api_url = os.getenv("FASTAPI_BASE_URL", "").strip()
FASTAPI_BASE_URL = (_configured_api_url or DEFAULT_API_URL).rstrip("/")

HEALTH_TIMEOUT_SECONDS = 2
REQUEST_TIMEOUT_SECONDS = (5, 180)
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
TOP_K = 3

st.set_page_config(page_title="LLM Deployment with RAG", layout="wide")


def check_backend_health():
    """Return whether FastAPI is reachable, plus a readable failure message."""
    try:
        response = requests.get(
            f"{FASTAPI_BASE_URL}/health",
            timeout=HEALTH_TIMEOUT_SECONDS,
        )
    except requests.ConnectionError:
        return False, "Could not connect to FastAPI. Start the backend and refresh."
    except requests.Timeout:
        return False, "The FastAPI health check timed out. The server may be starting."
    except requests.RequestException:
        return False, "The FastAPI health check failed. Check the backend terminal."

    if response.status_code != 200:
        return False, f"The backend health check returned HTTP {response.status_code}."

    try:
        payload = response.json()
    except ValueError:
        return False, "The backend returned an invalid health response."

    if isinstance(payload, dict) and payload.get("status") == "ok":
        return True, ""

    return False, "The backend health response was not recognized."


def _backend_error(response):
    """Extract a safe error message from a FastAPI JSON response."""
    try:
        payload = response.json()
    except ValueError:
        return f"The backend returned HTTP {response.status_code}."

    if isinstance(payload, dict):
        detail = payload.get("detail")
        if isinstance(detail, str) and detail.strip():
            return detail.strip()

    return f"The backend returned HTTP {response.status_code}."


def upload_document(filename, content):
    """Send one PDF to FastAPI and return the response or a readable error."""
    try:
        response = requests.post(
            f"{FASTAPI_BASE_URL}/upload",
            files={"file": (filename, content, "application/pdf")},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except requests.ConnectionError:
        return None, "Could not connect to FastAPI. Start the backend and try again."
    except requests.Timeout:
        return None, "The upload timed out. Check the backend and try again."
    except requests.RequestException:
        return None, "The upload request failed. Check the backend terminal."

    if not 200 <= response.status_code < 300:
        return None, _backend_error(response)

    try:
        payload = response.json()
    except ValueError:
        return None, "The backend returned an invalid upload response."

    required_fields = (
        "filename",
        "pages_processed",
        "chunks_created",
        "vectors_uploaded",
    )
    if not isinstance(payload, dict) or any(
        field not in payload for field in required_fields
    ):
        return None, "The backend upload response was missing expected fields."

    return payload, None


def send_chat(question, generate):
    """Send a question to FastAPI in retrieval or generation mode."""
    request_body = {
        "question": question,
        "top_k": TOP_K,
        "generate": generate,
    }

    try:
        response = requests.post(
            f"{FASTAPI_BASE_URL}/chat",
            json=request_body,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except requests.ConnectionError:
        return None, "Could not connect to FastAPI. Start the backend and try again."
    except requests.Timeout:
        return None, "The question timed out. Check the backend and try again."
    except requests.RequestException:
        return None, "The chat request failed. Check the backend terminal."

    if not 200 <= response.status_code < 300:
        return None, _backend_error(response)

    try:
        payload = response.json()
    except ValueError:
        return None, "The backend returned an invalid chat response."

    if not isinstance(payload, dict):
        return None, "The backend chat response was not a JSON object."

    sources = payload.get("sources")
    if not isinstance(sources, list) or any(
        not isinstance(source, dict) for source in sources
    ):
        return None, "The backend chat response contained invalid sources."

    if generate:
        answer = payload.get("answer")
        if not isinstance(answer, str) or not answer.strip():
            return None, "The backend did not return a usable generated answer."

    return payload, None


def _format_score(score):
    """Format a similarity score safely; older results may not contain one."""
    if isinstance(score, (int, float)) and not isinstance(score, bool):
        return f"{score:.4f}"
    return "n/a"


def _source_label(source):
    """Build a short citation label for one retrieved chunk."""
    filename = source.get("source") or "unknown source"
    page = source.get("page", "unknown")
    score = _format_score(source.get("score"))
    return f"{filename} | page {page} | similarity {score}"


def render_assistant_message(message):
    """Render a stored assistant response, including its mode and errors."""
    if message.get("error"):
        st.error(message["error"])
        return

    mode = message.get("mode", "Document Search")
    sources = message.get("sources") or []

    if mode == "Document Search":
        st.markdown("**Retrieved document content**")
        st.caption(
            "These passages come from the indexed PDF. This mode does not "
            "generate an AI answer."
        )

        if not sources:
            st.info("No document chunks were returned for this question.")
            return

        for index, source in enumerate(sources, start=1):
            with st.container(border=True):
                st.caption(f"Source {index}: {_source_label(source)}")
                text = source.get("text")
                if isinstance(text, str) and text.strip():
                    st.write(text)
                else:
                    st.caption("No text was returned for this chunk.")
        return

    st.markdown("**Answer**")
    answer = message.get("answer")
    if isinstance(answer, str) and answer.strip():
        st.write(answer)
    else:
        st.warning("The backend returned an empty answer.")

    st.markdown("**Sources**")
    if not sources:
        st.caption("No sources were returned for this answer.")
        return

    for source in sources:
        st.markdown(f"- {_source_label(source)}")


# Session state keeps data across Streamlit reruns but not across browser restarts.
st.session_state.setdefault("messages", [])
st.session_state.setdefault("upload_result", None)
st.session_state.setdefault("uploaded_signature", None)

st.title("LLM Deployment with RAG")
st.write(
    "Upload a PDF, search its indexed chunks, or ask the configured AI model "
    "to answer from the retrieved evidence."
)
st.caption(
    "Document Search works with the local FastAPI backend and Pinecone. "
    "AI Answer also requires a permitted, reachable vLLM endpoint."
)

with st.sidebar:
    st.header("Document")

    backend_available, health_message = check_backend_health()
    if backend_available:
        st.success("Backend connected.")
    else:
        st.error("Backend unavailable.")
        st.caption(health_message)

    st.caption(f"FastAPI: {FASTAPI_BASE_URL}")
    st.caption(
        "This status checks FastAPI only. It does not prove that an LLM "
        "endpoint is available."
    )

    st.divider()
    st.subheader("Upload a PDF")
    st.write(
        "Choose a text-based PDF up to 10 MB. FastAPI extracts the text, "
        "creates embeddings, and stores the chunks in Pinecone."
    )

    uploaded_file = st.file_uploader(
        "PDF file",
        type=["pdf"],
        key="pdf_uploader",
        max_upload_size=10,
        help="Scanned PDFs without a text layer need OCR and are not supported.",
    )

    upload_clicked = st.button(
        "Upload PDF",
        key="upload_pdf_button",
        width="stretch",
    )

    if upload_clicked:
        if uploaded_file is None:
            st.warning("Choose a PDF before uploading.")
        else:
            filename = Path(uploaded_file.name).name.strip()
            content = uploaded_file.getvalue()

            if not filename.lower().endswith(".pdf"):
                st.error("Choose a file with a .pdf extension.")
            elif not content:
                st.error("The selected PDF is empty.")
            elif len(content) > MAX_UPLOAD_BYTES:
                st.error("The PDF is larger than 10 MB.")
            else:
                file_digest = hashlib.sha256(content).hexdigest()
                signature = f"{filename}:{file_digest}"

                if signature == st.session_state.uploaded_signature:
                    st.info(
                        "This PDF is already uploaded in this session. "
                        "Choose another file to upload a different document."
                    )
                else:
                    with st.spinner("Uploading and indexing the PDF..."):
                        result, error = upload_document(filename, content)

                    if error:
                        st.error(error)
                    else:
                        st.session_state.upload_result = result
                        st.session_state.uploaded_signature = signature

    upload_result = st.session_state.upload_result
    if upload_result:
        st.divider()
        st.subheader("Indexed document")
        st.success(
            upload_result.get(
                "message",
                "PDF processed and stored in Pinecone successfully.",
            )
        )
        st.write(f"**File:** {upload_result.get('filename', 'unknown')}")

        left_column, right_column = st.columns(2)
        left_column.metric("Pages processed", upload_result["pages_processed"])
        right_column.metric("Chunks created", upload_result["chunks_created"])
        left_column.metric("Vectors uploaded", upload_result["vectors_uploaded"])

st.subheader("Ask the document")
mode = st.radio(
    "Response mode",
    ["Document Search", "AI Answer"],
    horizontal=True,
    key="response_mode",
)

if mode == "Document Search":
    st.caption(
        "Retrieval only: returns the most relevant document chunks with "
        "filename, page, and similarity score. The LLM is not called."
    )
else:
    st.caption(
        "Retrieval plus generation: the backend sends the retrieved evidence "
        "to a configured vLLM endpoint. If none is available, the request "
        "returns HTTP 503 instead of switching providers."
    )

prompt = st.chat_input(
    "Ask a question about the uploaded PDF",
    key="chat_input",
)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "user":
            st.write(message["content"])
        else:
            render_assistant_message(message)

if not st.session_state.messages and prompt is None:
    st.info("Upload a PDF, then ask a question about it.")

st.caption("Chat history is kept only for this browser session.")

if prompt is not None:
    question = prompt.strip() if isinstance(prompt, str) else ""

    if not question:
        st.warning("Enter a non-empty question before sending.")
    else:
        generate = mode == "AI Answer"
        user_message = {"role": "user", "content": question}
        st.session_state.messages.append(user_message)

        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            spinner_text = (
                "Retrieving evidence and generating an answer..."
                if generate
                else "Searching the indexed document..."
            )
            with st.spinner(spinner_text):
                response, error = send_chat(question, generate=generate)

            if error:
                assistant_message = {
                    "role": "assistant",
                    "mode": mode,
                    "answer": None,
                    "sources": [],
                    "error": error,
                }
                st.error(error)
            else:
                assistant_message = {
                    "role": "assistant",
                    "mode": mode,
                    "answer": response.get("answer"),
                    "sources": response.get("sources", []),
                    "error": None,
                }
                render_assistant_message(assistant_message)

        st.session_state.messages.append(assistant_message)
