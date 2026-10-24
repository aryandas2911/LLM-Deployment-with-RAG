# Testing Report

**Project:** LLM Deployment with RAG  
**Date:** 2026-10-24  
**Environment:** Windows (win32), Python 3.12.10, local virtual environment  
**Code tested:** commit `fca69a7` plus the Task 12 working-tree changes  
**External services available:** none for live Pinecone or live vLLM testing

## Overall result

```text
59 passed in 50.66s
```

The automated suite passed completely. Tests that need Pinecone or a running vLLM server use mocks; the PDF extraction and embedding tests use the real local libraries and the cached `all-MiniLM-L6-v2` model.

Live Pinecone, live vLLM, and the Kaggle GPU notebooks were **not tested** in this environment. Those results must not be described as passing until they are actually executed with the required accounts, keys, and GPU session.

## Test environment

| Item | Value |
| --- | --- |
| Python | 3.12.10 |
| pytest | 9.1.1 |
| Streamlit | 1.66.0 |
| FastAPI | 0.158.0 |
| Pinecone SDK | 6.0.2 |
| PyMuPDF | 1.28.2 |
| sentence-transformers | 5.7.0 |
| openai | 2.54.0 |
| langchain-core | 0.3.86 |
| langchain-openai | 0.3.35 |

Additional checks:

```text
python -m pip check
No broken requirements found.

python -m compileall -q backend frontend tests
PASS
```

## Automated tests

Run command:

```powershell
python -m pytest -q
```

| Component | Test performed | Expected result | Actual result | Status |
| --- | --- | --- | --- | --- |
| API | `GET /health` | HTTP 200 with `{"status": "ok"}` | HTTP 200 and exact body | PASS |
| API | Upload a non-PDF filename | HTTP 400 with a PDF-extension message | HTTP 400, message present | PASS |
| API | Upload an empty `.pdf` | HTTP 400 with an empty-file message | HTTP 400, message present | PASS |
| API | Upload an invalid PDF and inspect temp file | HTTP 400 and temp file deleted | HTTP 400; captured temp path did not exist | PASS |
| API | Upload a body larger than the real 10 MB limit | HTTP 413, no indexing attempt | HTTP 413; limit constant is 10 MB | PASS |
| API | Upload a generated valid PDF with real PyMuPDF extraction | HTTP 200, page/chunk/vector counts, temp file deleted | HTTP 200; source/page metadata correct; temp deleted | PASS |
| API | Invalid chat payloads: blank question, `top_k=0`, `top_k=11` | HTTP 400 | All three cases returned HTTP 400 | PASS |
| API | Retrieval mode | Does not call the LLM; returns source dictionaries | `generate_answer` was not called; response matched | PASS |
| API | Generation with no configured endpoint | HTTP 503 explaining `VLLM_API_BASE_URL` and Kaggle limitations | HTTP 503 with expected detail | PASS |
| API | Generation calls the existing RAG pipeline | Answer and sources returned without exposing key handling | HTTP 200; pipeline arguments and response correct | PASS |
| API | Generation error containing a fake API key | No key value in the response or captured warning logs | HTTP 502; fake key absent from both response and `caplog` | PASS |
| API | LLM timeout | HTTP 504 | HTTP 504 with timeout message | PASS |
| API | Pinecone configuration failure during generation | HTTP 503 | HTTP 503 with Pinecone guidance | PASS |
| API | Mocked upload pipeline and temp cleanup | Pipeline called with metadata; temp file deleted | Response and captured arguments correct; temp deleted | PASS |
| Frontend | Fake secret values in environment | Values are never displayed | Health status rendered; no fake secret text found | PASS |
| Frontend | Backend unavailable | Clear error message; no crash | Error state rendered | PASS |
| Frontend | Upload statistics and duplicate prevention | Stats rendered; same upload is not repeated | Stats and duplicate warning rendered | PASS |
| Frontend | Upload with no file | No backend call | Warning shown; backend not called | PASS |
| Frontend | Invalid upload response | Readable error; no traceback | Error message rendered | PASS |
| Frontend | Empty question | No request sent | Warning shown; no request | PASS |
| Frontend | Document Search with sources | Sources rendered; chat history retained | Sources and history rendered | PASS |
| Frontend | AI Answer success | Answer and sources rendered | Answer and sources rendered | PASS |
| Frontend | AI Answer HTTP 503 | Explanation shown; no fake fallback answer | Error explanation rendered | PASS |
| PDF | Real two-page PDF | Two Documents with correct text and page metadata | Two pages; source/page metadata correct | PASS |
| PDF | Chunking | Multiple chunks; source/page preserved | Multiple chunks; metadata preserved | PASS |
| PDF | Invalid PDF bytes | `ValueError` | `ValueError` with readable-PDF message | PASS |
| PDF | Empty PDF file | `ValueError` | `ValueError` with readable-PDF message | PASS |
| PDF | PDF page with no text layer | No Documents | Empty list returned | PASS |
| PDF | Invalid chunk settings | `ValueError` for all invalid combinations | Four invalid combinations rejected | PASS |
| Embeddings | Dimension constant | 384 | 384 | PASS |
| Embeddings | Model loading | Real model loads with expected dimension | Model loaded; dimension 384 | PASS |
| Embeddings | Embedding order and values | Output order matches input; values match model encoding | Shape, order, and values matched | PASS |
| Embeddings | Empty document list | Clear `ValueError` | Rejected with clear message | PASS |
| Embeddings | Empty document text | Clear `ValueError` | Rejected with clear message | PASS |
| RAG prompt | Evidence labels and role separation | System contains rules and source/page labels; human contains the question | All checks passed | PASS |
| RAG prompt | Retrieved text tries to close the context tag | Closing tag is escaped; only the template's closing tag remains | Escape present; one real closing tag | PASS |
| RAG prompt | Empty retrieval | Explicit no-evidence context | Expected text present | PASS |
| RAG prompt | Invalid retrieval values | `ValueError` | All invalid values rejected | PASS |
| RAG pipeline | No retrieved chunks | Return not-found answer without calling the LLM | `ChatOpenAI` was not called | PASS |
| RAG pipeline | Answerable retrieval | Answer trimmed; sources returned; client settings correct | Answer and sources matched; settings captured | PASS |
| RAG pipeline | Missing API key | `ValueError` before retrieval | Rejected with key message | PASS |
| Vector store | Stable vector IDs and metadata | Repeated upload produces the same ID and metadata | IDs stable; metadata correct; namespace correct | PASS |
| Vector store | Mismatched document/embedding lengths | `ValueError` | Rejected | PASS |
| Vector store | Wrong embedding dimension | `ValueError` naming 384 dimensions | Rejected | PASS |
| Vector store | Search query settings and result mapping | Uses `top_k`, metadata, namespace; maps text/source/page/score | All query and result checks passed | PASS |
| Vector store | Empty search results (`None` and `[]`) | Empty list | Empty list for both cases | PASS |
| Vector store | Existing compatible index | Validated and returned without recreation | Validation passed; client returned | PASS |
| Vector store | Existing index with wrong dimension | `RuntimeError` | Rejected | PASS |
| Vector store | Existing index with wrong metric | `RuntimeError` | Rejected | PASS |
| Vector store | Missing index | Created with dimension 384, cosine metric, dense vectors | Creation arguments captured and correct | PASS |
| Vector store | Missing Pinecone API key | `ValueError` naming `PINECONE_API_KEY` | Rejected | PASS |

## Headless server checks

| Component | Test performed | Expected result | Actual result | Status |
| --- | --- | --- | --- | --- |
| FastAPI process | Start Uvicorn on `127.0.0.1:8002` and request `/health` | HTTP 200 with status ok | HTTP 200, `{"status":"ok"}` | PASS |
| Streamlit process | Start headless on `127.0.0.1:8508` and request `/_stcore/health` | HTTP 200 with `ok` | HTTP 200, `ok` | PASS |

These checks confirm that both server processes start, but they do not replace a full browser test.

## Security and repository checks

| Check | Expected result | Actual result | Status |
| --- | --- | --- | --- |
| Tracked `.env` or `kaggle.json` | None | None found with `git ls-files` | PASS |
| Tracked PDFs or private keys | None | None found | PASS |
| `.gitignore` coverage | Ignore secrets, environments, caches, uploads, PDFs, model weights | `git check-ignore` matched `.env`, `.venv/`, `__pycache__/`, `.pytest_cache/`, `sample.pdf`, `models/`, `huggingface/`, `uploads/`, and notebook checkpoints | PASS |
| Notebook outputs | No credentials in committed outputs | Both Kaggle notebooks have zero output entries; no key patterns found | PASS |
| Git history scan | No real Pinecone/OpenAI-style keys | No `pcsk_` or long `sk-` pattern found; deliberate test/placeholder strings only | PASS |
| Tunnel code | No ngrok, cloudflared, localtunnel, serveo, or SSH reverse-tunnel code | No matches in `backend/` or `frontend/` | PASS |
| Local-only services | Documentation and commands use loopback addresses | Commands use `127.0.0.1`; README warns against `0.0.0.0` | PASS |
| Error responses and logs | No API key values returned in user-facing errors or warning logs | API messages reference variable names, not values; fake-secret backend and frontend tests passed | PASS |
| Prompt injection | Retrieved text cannot close the context tag early | Escaping test passed; this is a structural safeguard, not a complete defense | PASS |

The scan is a targeted credential and repository review, not a full security audit.

## External tests not executed

| Component | Test performed | Expected result | Actual result | Status |
| --- | --- | --- | --- | --- |
| Pinecone live index | Create/describe an index with dimension 384 and cosine metric | Compatible index ready | No `.env` / `PINECONE_API_KEY` available | NOT TESTED |
| Pinecone live upload | Upload real PDF chunks and inspect vectors in the Pinecone console | Vectors and metadata appear in the namespace | Not executed; mocked logic only | NOT TESTED |
| Pinecone live retrieval | Query the real index with an answerable question | Relevant chunks with source/page/score | Not executed; mocked logic only | NOT TESTED |
| vLLM local generation | Send a prompt to a permitted reachable OpenAI-compatible endpoint | Grounded answer and source list | No permitted reachable endpoint; Kaggle loopback inaccessible | NOT TESTED |
| Kaggle vLLM notebook | Run `kaggle/llm_server.ipynb` on a GPU session | Server starts and API checks pass | Not executed locally; requires Kaggle GPU | NOT TESTED |
| Kaggle RAG integration | Run `kaggle/test_rag_integration.ipynb` with Kaggle Secrets | Retrieval and generation work in one session | Not executed locally; requires Kaggle GPU and secrets | NOT TESTED |
| Remote HTTPS client | Run `tests/test_remote_llm.py` against a permitted endpoint | Authenticated response returned | No remote endpoint available | NOT TESTED |
| Manual live retrieval helper | Run `tests/test_rag_pipeline.py` against Pinecone | Retrieved chunks and prompt preview | Live Pinecone unavailable | NOT TESTED |
| Manual live vector helper | Run `tests/test_vector_store.py` against Pinecone | Uploaded vectors and query results | Live Pinecone unavailable | NOT TESTED |
| Browser interaction | Upload, click, and inspect the app in a real browser | No visual/runtime errors | Only Streamlit AppTest and headless health check used | NOT TESTED |
| OCR / scanned PDFs | Upload an image-only PDF | Extracted text | OCR is not implemented | NOT TESTED / OUT OF SCOPE |
| Load and concurrency | Multiple simultaneous uploads and queries | Stable performance and isolation | Not measured; single-user local design | NOT TESTED |
| Live end-to-end chain | Streamlit to FastAPI to Pinecone to vLLM | Complete answer with citations | External components unavailable; individual layers tested separately | NOT TESTED |

## Bugs and cleanup found during final review

No reproducible functional defect was found in the backend, frontend, PDF, embedding, prompt, or mocked vector-store paths. The following cleanup items were fixed:

| Item | Change | Reason |
| --- | --- | --- |
| `requirements.txt` | Removed unused `langchain` and `langchain-community`; added direct `openai` and `pydantic` declarations | Match actual imports and avoid unnecessary packages |
| `.env.example` | Removed unused `VLLM_MODEL_NAME` and `EMBEDDING_MODEL_NAME` assignments; documented that model names are pinned in code | Those variables had no effect on the backend |
| `.gitignore` | Added `*.pdf`, model weights, `.cache/`, and `huggingface/` | Keep local uploads and downloaded model data out of Git |
| `README.md` | Rewrote as a GitHub-ready overview | Old version was task-history-heavy and less useful to a new reader |
| `backend/main.py` | Generic LLM failure logging no longer includes lower-layer exception text | Defense in depth: avoid logging a value if a future client exception contains sensitive request details |
| Tests | Added real PDF, embedding, prompt, mocked vector-store, and secret-redaction coverage | Final validation previously relied on manual scripts for these areas |
| `TESTING.md` | Added this dated test record | Separate tested facts from setup and feature documentation |

## How to reproduce

From the project root with the virtual environment activated:

```powershell
python -m pip check
python -m compileall -q backend frontend tests
python -m pytest -q
```

Expected result:

```text
59 passed
```

The embedding tests require the `all-MiniLM-L6-v2` model to be cached locally or downloadable from Hugging Face. The first run may take longer and needs internet access.

To reproduce the live checks that were not possible here, provide:

1. a real `PINECONE_API_KEY` and a compatible `llm-rag-index`;
2. a permitted, reachable vLLM endpoint and key for local AI Answer mode; or
3. a Kaggle GPU session for the two Kaggle notebooks.

Do not use a Kaggle loopback URL from the local machine. This project intentionally does not create a tunnel.
