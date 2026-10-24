# LLM Deployment with RAG

A beginner-friendly project that serves a small open-source LLM on a free Kaggle GPU and connects it to a local Retrieval-Augmented Generation (RAG) application.

## Architecture

```text
Kaggle Notebook (free GPU)
  Qwen2.5-1.5B-Instruct + vLLM
              |
              | OpenAI-compatible HTTP API (loopback only; no Kaggle tunnel)
              v
Local machine
  Streamlit UI  ->  FastAPI backend  ->  LangChain RAG chain
                                             |            |
                                             v            v
                                  Sentence Transformers  Pinecone
                                     (embeddings)      (vector DB)
                                             ^
                                             |
                                     PyMuPDF (PDF text)
```

| Part | Tool | Where it runs |
| --- | --- | --- |
| LLM | Qwen2.5-1.5B-Instruct | Kaggle GPU |
| Model serving | vLLM | Kaggle GPU |
| RAG orchestration | LangChain | Local |
| Embeddings | Sentence Transformers | Local (CPU is fine) |
| Vector database | Pinecone | Cloud |
| PDF parsing | PyMuPDF | Local |
| Backend | FastAPI | Local |
| Frontend | Streamlit | Local |

## Repository layout

```text
.
├── kaggle/
│   └── llm_server.ipynb   # GPU server notebook, uploaded to Kaggle manually
├── backend/
│   ├── __init__.py        # Local backend package
│   ├── pdf_processor.py   # PDF text extraction and chunking
│   └── embeddings.py      # Local CPU text embeddings
├── frontend/              # Streamlit app (comes later)
├── tests/
│   ├── test_remote_llm.py # Authenticated client for a future HTTPS endpoint
│   ├── test_pdf_processor.py  # Manual PDF extraction/chunking check
│   └── test_embeddings.py     # Embedding and similarity demo
├── requirements.txt       # Local dependencies only
├── .env.example           # Template for Pinecone and vLLM settings
├── .gitignore
└── README.md
```

## How the Kaggle notebook fits in

1. `kaggle/llm_server.ipynb` is written locally in VS Code. Open the `kaggle/` folder in VS Code to see it as a notebook.
2. Upload the `.ipynb` file to Kaggle and run it there with a GPU enabled. Do **not** run the GPU/vLLM cells locally.
3. Task 4 starts vLLM on `127.0.0.1:8000` and tests the API inside Kaggle.
4. Task 5 adds API-key authentication. Kaggle does not permit tunneling or public API hosting, so **no public URL is created**.
5. `tests/test_remote_llm.py` is ready for a future permitted HTTPS endpoint. It is currently untested against any remote endpoint.

vLLM is installed only inside Kaggle. It is deliberately excluded from `requirements.txt` because the local machine does not need it (and may not have a compatible GPU/CUDA setup).

## Local setup (Windows PowerShell)

Prerequisites: Python 3.11 or 3.12, Git, and a Pinecone account (for a later task).

```powershell
# 1. Create the virtual environment
py -3.12 -m venv .venv

# 2. Activate it (run this each time you open a new terminal)
.\.venv\Scripts\Activate.ps1

# If PowerShell blocks activation, allow it for this terminal only:
# Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

# 3. Install local dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

# 4. Create your local environment file and fill in real values
Copy-Item .env.example .env
```

## Verify the setup

```powershell
# Notebook is valid JSON and has the expected cells
py -3.12 -m json.tool kaggle\llm_server.ipynb > $null

# Dependencies import correctly (after installing them)
python -c "import fastapi, streamlit, langchain, pinecone, pymupdf, sentence_transformers, torch, numpy; print('OK')"
```

Then open `kaggle\llm_server.ipynb` in VS Code and confirm the notebook cells render.

## Task 5 remote-access status

Kaggle tunneling is prohibited, and Cloudflare Quick Tunnel is unauthenticated by default. This project therefore does not expose the Kaggle vLLM server. The server stays on `127.0.0.1` and uses a `VLLM_API_KEY` bearer token for `/v1` requests.

`tests/test_remote_llm.py` is a reusable client for a future permitted HTTPS endpoint. It validates HTTPS, reads `VLLM_API_BASE_URL` and `VLLM_API_KEY` from `.env`, and handles authentication, timeout, and connection failures. It has **not** been run against Kaggle or any remote endpoint.

## PDF processing and chunking

`backend/pdf_processor.py` implements the first local RAG stage:

- `extract_pdf_text(pdf_path)` uses PyMuPDF to extract the text layer from each PDF page. It returns one LangChain `Document` per nonempty page, with `source` (the PDF filename) and `page` (starting at 1) metadata.
- `split_documents(documents, chunk_size=500, chunk_overlap=100)` uses LangChain's `RecursiveCharacterTextSplitter` to create smaller overlapping chunks. Every chunk keeps the original source and page metadata.

`chunk_size` is the maximum chunk length in **characters**, not model tokens. `chunk_overlap` is the number of characters repeated between neighboring chunks. Overlap helps preserve sentences or ideas that cross a chunk boundary.

Scanned PDFs without a text layer may need OCR. OCR is outside this mini-project's scope.

Run the manual test from the project root:

```powershell
.\.venv\Scripts\Activate.ps1
python tests\test_pdf_processor.py "C:\path\to\your\sample.pdf"
```

Expected output looks like:

```text
Pages with extractable text: 3
Generated chunks: 8

First chunks:

Chunk 1
  Source: sample.pdf
  Page: 1
  Preview: The first page text starts here...

Chunk 2
  Source: sample.pdf
  Page: 1
  Preview: ...

All chunks retain source and page metadata.
```

## Text embeddings

`backend/embeddings.py` converts document chunks into numeric vectors with `sentence-transformers/all-MiniLM-L6-v2` on the CPU.

- The model produces **384-dimensional** vectors. Each dimension is a learned feature; the combination of all 384 values represents the text's meaning.
- `load_embedding_model()` uses `functools.lru_cache` so the model is loaded once per Python process. The first use may download it from Hugging Face and cache it locally.
- `generate_embeddings(documents)` extracts each Document's `page_content`, encodes the texts in batches of 32, and returns one list of Python floats per input Document in the same order.
- Keeping the order is essential: the vector at position `i` must remain paired with the chunk at position `i`, including its source filename and page metadata.
- The same model must embed both document chunks and user questions. Otherwise their vectors would not be comparable.

The embedding test also includes a small semantic-retrieval demo. It embeds the question "What is machine learning?", computes cosine similarity with NumPy, and prints the three closest chunks. Cosine similarity compares vector direction: a score near 1 means the texts are closely related; a score near 0 means they are mostly unrelated.

This is only an in-memory demonstration. Pinecone will store the vectors and their metadata so larger collections can be searched without loading every vector into local memory.

Run the embedding test:

```powershell
.\.venv\Scripts\Activate.ps1
python tests\test_embeddings.py "C:\path\to\your\sample.pdf"
```

Expected output looks like:

```text
Embedding model: sentence-transformers/all-MiniLM-L6-v2 (CPU)
Number of chunks: 8
Number of embeddings: 8
Dimension of first embedding: 384
First 10 values of first embedding: [0.0123, -0.0456, ...]

Semantic similarity demo
Question: What is machine learning?
Higher cosine similarity means the chunk is more related to the question.

1. Similarity: 0.6234
   Source: sample.pdf
   Page: 2
   Preview: Machine learning is a field of artificial intelligence...

Embedding checks passed.
```

## Current status

Tasks 1-6 are complete. Task 7 adds local CPU text embeddings and a small semantic-similarity demo. Pinecone, the RAG chain, FastAPI endpoints, and the Streamlit UI are not implemented yet. The Kaggle notebook is unchanged.
