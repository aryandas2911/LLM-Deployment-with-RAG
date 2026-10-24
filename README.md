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
│   └── __init__.py        # FastAPI backend package (endpoints come later)
├── frontend/              # Streamlit app (comes later)
├── tests/
│   └── test_remote_llm.py # Authenticated client for a future HTTPS endpoint
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
python -c "import fastapi, streamlit, langchain, pinecone, fitz, sentence_transformers; print('OK')"
```

Then open `kaggle\llm_server.ipynb` in VS Code and confirm the notebook cells render.

## Task 5 remote-access status

Kaggle tunneling is prohibited, and Cloudflare Quick Tunnel is unauthenticated by default. This project therefore does not expose the Kaggle vLLM server. The server stays on `127.0.0.1` and uses a `VLLM_API_KEY` bearer token for `/v1` requests.

`tests/test_remote_llm.py` is a reusable client for a future permitted HTTPS endpoint. It validates HTTPS, reads `VLLM_API_BASE_URL` and `VLLM_API_KEY` from `.env`, and handles authentication, timeout, and connection failures. It has **not** been run against Kaggle or any remote endpoint.

## Current status

Tasks 1-4 are complete. Task 5 adds API-key authentication, local authentication tests inside Kaggle, and a prepared remote client. Remote access is intentionally not configured because of Kaggle's platform restrictions. The FastAPI backend, Streamlit UI, RAG pipeline, and Pinecone integration are not implemented yet.
