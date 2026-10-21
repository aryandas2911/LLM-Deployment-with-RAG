# LLM Deployment with RAG

A beginner-friendly project that serves a small open-source LLM on a free Kaggle GPU and connects it to a local Retrieval-Augmented Generation (RAG) application.

## Architecture

```text
Kaggle Notebook (free GPU)
  Qwen2.5-1.5B-Instruct + vLLM
              |
              | OpenAI-compatible HTTP API (public URL added later)
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
├── tests/                 # Tests (come later)
├── requirements.txt       # Local dependencies only
├── .env.example           # Template for Pinecone and vLLM settings
├── .gitignore
└── README.md
```

## How the Kaggle notebook fits in

1. `kaggle/llm_server.ipynb` is written locally in VS Code. Open the `kaggle/` folder in VS Code to see it as a notebook.
2. Upload the `.ipynb` file to Kaggle and run it there with a GPU enabled. Do **not** run the GPU/vLLM cells locally.
3. In a later task, the notebook will start vLLM and expose it through a temporary public URL.
4. Copy that URL into your local `.env` as `VLLM_BASE_URL`. The local FastAPI backend then calls the Kaggle-hosted model.
5. The URL changes every Kaggle session, so it is never hard-coded or committed.

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

Then open `kaggle\llm_server.ipynb` in VS Code and confirm the three cells render.

## Current status

Task 1 only: project scaffolding, notebook placeholder, dependencies, environment template, Git ignore rules, and documentation. The RAG pipeline, FastAPI endpoints, Streamlit UI, Kaggle GPU setup, and vLLM installation are intentionally not implemented yet.
