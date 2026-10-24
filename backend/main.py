"""Local FastAPI backend for PDF upload and document-based question answering.

The API has three endpoints:

- ``GET /health`` reports whether the server is running.
- ``POST /upload`` extracts a PDF, creates embeddings, and stores them in Pinecone.
- ``POST /chat`` retrieves document chunks and optionally asks the configured LLM
  to generate an answer.

Retrieval works locally with Pinecone. Generation works only when a permitted,
reachable vLLM endpoint is configured in ``.env``. The Kaggle vLLM server uses
loopback inside Kaggle and cannot be reached from this Windows machine. This
module does not create a tunnel or expose the Kaggle server publicly.
"""

import logging
import os
import tempfile
import time
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from backend.embeddings import generate_embeddings
from backend.pdf_processor import extract_pdf_text, split_documents
from backend.rag_pipeline import generate_answer, retrieve_context
from backend.vector_store import get_pinecone_index, store_documents

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Read the upload in small chunks. This lets the API stop at 10 MB instead of
# loading a potentially unlimited file into memory.
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
UPLOAD_CHUNK_BYTES = 1024 * 1024

logger = logging.getLogger(__name__)

app = FastAPI(
    title="LLM Deployment with RAG",
    description="Local API for PDF upload and document-based question answering.",
    version="0.1.0",
)


class ChatRequest(BaseModel):
    """JSON body accepted by the chat endpoint."""

    question: str = Field(description="Question to ask about the uploaded document.")
    top_k: int = Field(
        default=3,
        description="Number of Pinecone chunks to retrieve (1 to 10).",
    )
    generate: bool = Field(
        default=False,
        description="Use the configured LLM when true; retrieval only when false.",
    )


def _get_llm_settings():
    """Return configured LLM settings, or ``None`` when generation is unavailable.

    Credentials come from the environment or ``.env``. They are never hardcoded
    and never printed. Placeholder values from ``.env.example`` are treated as
    unconfigured so the API does not try to call an example URL.
    """
    load_dotenv(PROJECT_ROOT / ".env")

    api_base_url = os.getenv("VLLM_API_BASE_URL", "").strip()
    api_key = os.getenv("VLLM_API_KEY", "").strip()

    if not api_base_url or not api_key:
        return None

    if api_base_url.startswith("https://your-") or api_key.startswith("your_"):
        return None

    return api_base_url, api_key


def _clear_exception(error):
    """Drop traceback references that can keep a Windows file locked.

    Python tracebacks retain frame locals. PyMuPDF's exception chain can retain
    the open document, so the temporary PDF stays locked until the traceback is
    released. Clearing the chain before responding lets the finally block delete
    the file while the request is still running.
    """
    cause = error.__cause__
    context = error.__context__

    for item in (error, cause, context):
        if item is not None:
            item.__traceback__ = None
            item.__cause__ = None
            item.__context__ = None


def _delete_temp_file(temp_path):
    """Delete a temporary upload, retrying briefly for Windows file handles."""
    for attempt in range(5):
        try:
            temp_path.unlink(missing_ok=True)
            return
        except OSError:
            if attempt == 4:
                logger.warning(
                    "Could not delete temporary upload file %s", temp_path
                )
                return
            time.sleep(0.1)


@app.get("/health", summary="Check whether the API is running")
def health():
    """Small health check used by browsers, scripts, and deployment tools."""
    return {"status": "ok"}


@app.post("/upload", summary="Upload and index a PDF")
async def upload_pdf(
    file: UploadFile = File(
        description="PDF file to upload. Maximum size: 10 MB."
    ),
):
    """Extract, chunk, embed, and store a PDF in Pinecone.

    The upload is streamed to a generated temporary filename. The original
    filename is restored in document metadata so Pinecone search results show a
    useful citation instead of the random temporary name.
    """
    original_filename = Path(file.filename or "").name.strip()

    # Check the extension before writing anything to disk. Content validation is
    # handled by the existing PDF processor after the bytes are saved.
    if (
        not original_filename
        or Path(original_filename).suffix.lower() != ".pdf"
    ):
        raise HTTPException(
            status_code=400,
            detail="Upload a PDF file with a .pdf extension.",
        )

    temp_path = None

    try:
        # NamedTemporaryFile creates a safe random filename and closes the file
        # before the PDF processor opens it by path.
        with tempfile.NamedTemporaryFile(
            mode="wb", suffix=".pdf", prefix="rag-upload-", delete=False
        ) as temp_file:
            temp_path = Path(temp_file.name)
            total_bytes = 0

            while True:
                chunk = await file.read(UPLOAD_CHUNK_BYTES)
                if not chunk:
                    break

                total_bytes += len(chunk)
                if total_bytes > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=(
                            "PDF is too large. The maximum upload size is "
                            "10 MB."
                        ),
                    )

                temp_file.write(chunk)

        if total_bytes == 0:
            raise HTTPException(
                status_code=400,
                detail="The uploaded PDF is empty.",
            )

        try:
            documents = extract_pdf_text(temp_path)
        except ValueError as error:
            logger.info("Rejected upload %s: %s", original_filename, error)
            _clear_exception(error)
            raise HTTPException(
                status_code=400,
                detail=(
                    "The uploaded file is not a readable PDF. Check that it is "
                    "not corrupted or password-protected."
                ),
            ) from None
        except OSError as error:
            logger.exception("Could not open uploaded PDF %s", original_filename)
            _clear_exception(error)
            raise HTTPException(
                status_code=500,
                detail="The uploaded PDF could not be opened by the server.",
            ) from None

        if not documents:
            raise HTTPException(
                status_code=400,
                detail=(
                    "No extractable text was found in the PDF. The file may be "
                    "scanned and require OCR."
                ),
            )

        # The temporary filename must not become the citation shown to users.
        for document in documents:
            document.metadata["source"] = original_filename

        chunks = split_documents(documents)
        if not chunks:
            raise HTTPException(
                status_code=400,
                detail="The PDF text could not be split into chunks.",
            )

        try:
            embeddings = generate_embeddings(chunks)
        except (ValueError, RuntimeError) as error:
            logger.exception("Embedding failed for %s", original_filename)
            raise HTTPException(
                status_code=503,
                detail=(
                    "The embedding model could not process this PDF. The model "
                    "may need to be downloaded, or the machine may be out of "
                    "memory."
                ),
            ) from error

        try:
            index = get_pinecone_index()
            vectors_uploaded = store_documents(chunks, embeddings, index)
        except (ValueError, TimeoutError) as error:
            logger.warning(
                "Pinecone was not ready for %s: %s", original_filename, error
            )
            raise HTTPException(
                status_code=503,
                detail=(
                    "Pinecone is not configured or is unavailable. Check "
                    "PINECONE_API_KEY and PINECONE_INDEX_NAME, then try again."
                ),
            ) from error
        except Exception as error:
            logger.exception("Pinecone upload failed for %s", original_filename)
            raise HTTPException(
                status_code=503,
                detail=(
                    "Pinecone upload failed. Check your connection, index, and "
                    "Pinecone service status, then try again."
                ),
            ) from error

        return {
            "filename": original_filename,
            "pages_processed": len(documents),
            "chunks_created": len(chunks),
            "vectors_uploaded": vectors_uploaded,
            "message": "PDF processed and stored in Pinecone successfully.",
        }

    finally:
        # Always close the upload and delete the generated temporary file, even
        # when validation, extraction, embedding, or Pinecone storage fails.
        try:
            await file.close()
        finally:
            if temp_path is not None:
                _delete_temp_file(temp_path)


@app.post("/chat", summary="Retrieve document chunks or generate an answer")
def chat(request: ChatRequest):
    """Answer from the indexed document with retrieval or retrieval + generation.

    Retrieval mode (``generate=false``) never calls the LLM. Generation mode
    (``generate=true``) calls the existing ``generate_answer()`` function only
    when a permitted endpoint and API key are configured.
    """
    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question must not be empty.",
        )

    if not 1 <= request.top_k <= 10:
        raise HTTPException(
            status_code=400,
            detail="top_k must be between 1 and 10.",
        )

    if not request.generate:
        try:
            sources = retrieve_context(question, top_k=request.top_k)
        except ValueError as error:
            logger.warning("Pinecone retrieval is not configured: %s", error)
            raise HTTPException(
                status_code=503,
                detail=(
                    "Pinecone retrieval is not configured correctly. Check "
                    "PINECONE_API_KEY and PINECONE_INDEX_NAME."
                ),
            ) from error
        except TimeoutError as error:
            logger.warning("Pinecone retrieval timed out: %s", error)
            raise HTTPException(
                status_code=504,
                detail="Pinecone retrieval timed out. Try again shortly.",
            ) from error
        except Exception as error:
            logger.exception("Pinecone retrieval failed")
            raise HTTPException(
                status_code=503,
                detail=(
                    "Pinecone retrieval failed. Check your connection and try "
                    "again."
                ),
            ) from error

        return {
            "question": question,
            "generate": False,
            "answer": None,
            "sources": sources,
        }

    settings = _get_llm_settings()
    if settings is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Generation mode requires a permitted, reachable vLLM endpoint. "
                "Set VLLM_API_BASE_URL and VLLM_API_KEY in .env. The Kaggle "
                "loopback server cannot be reached from this Windows machine."
            ),
        )

    api_base_url, api_key = settings

    try:
        result = generate_answer(
            question,
            api_base_url,
            api_key,
            top_k=request.top_k,
        )
    except ValueError as error:
        error_text = str(error).lower()

        if "pinecone" in error_text:
            logger.warning("Pinecone retrieval is not configured: %s", error)
            raise HTTPException(
                status_code=503,
                detail=(
                    "Pinecone retrieval is not configured correctly. Check "
                    "PINECONE_API_KEY and PINECONE_INDEX_NAME."
                ),
            ) from error

        logger.warning("LLM configuration is invalid: %s", error)
        raise HTTPException(
            status_code=503,
            detail=(
                "The LLM endpoint configuration is invalid. Check "
                "VLLM_API_BASE_URL and VLLM_API_KEY."
            ),
        ) from error
    except TimeoutError as error:
        logger.warning("RAG request timed out: %s", error)
        raise HTTPException(
            status_code=504,
            detail="The RAG request timed out. Try again when the services are ready.",
        ) from error
    except RuntimeError as error:
        error_text = str(error).lower()

        if "timed out" in error_text:
            logger.warning("LLM request timed out")
            raise HTTPException(
                status_code=504,
                detail="The LLM request timed out. Try again when the endpoint is ready.",
            ) from error

        if "could not connect" in error_text:
            logger.warning("Could not connect to the configured LLM endpoint")
            raise HTTPException(
                status_code=503,
                detail=(
                    "Could not reach the configured LLM endpoint. Check that "
                    "it is running and permitted from this machine."
                ),
            ) from error

        if "pinecone" in error_text:
            logger.warning("Pinecone retrieval failed during generation: %s", error)
            raise HTTPException(
                status_code=503,
                detail=(
                    "Pinecone retrieval failed. Check the index, connection, "
                    "and Pinecone service status, then try again."
                ),
            ) from error

        logger.warning("LLM generation failed: %s", error)
        raise HTTPException(
            status_code=502,
            detail=(
                "The LLM endpoint returned an error. Check its logs and "
                "configuration, then try again."
            ),
        ) from error
    except Exception as error:
        logger.exception("Unexpected generation failure")
        raise HTTPException(
            status_code=502,
            detail="The RAG request failed unexpectedly. Check the server logs.",
        ) from error

    return {
        "question": question,
        "generate": True,
        "answer": result.get("answer"),
        "sources": result.get("sources", []),
    }
