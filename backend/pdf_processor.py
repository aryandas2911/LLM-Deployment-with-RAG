"""Extract PDF text and split it into overlapping chunks for the RAG pipeline.

This module is the first local RAG stage:

    PDF file -> page Documents -> overlapping chunks -> (later) embeddings

PyMuPDF reads the PDF and LangChain keeps the text plus metadata in Document
objects. Embeddings, the vector store, and the RAG chain are added later.
"""

from pathlib import Path

import pymupdf as fitz
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def extract_pdf_text(pdf_path):
    """Extract nonempty PDF pages as LangChain Document objects.

    Each Document contains the text of one page plus:

    - ``source``: the PDF filename.
    - ``page``: the page number starting at 1.

    PyMuPDF opens the file, exposes each page, and extracts its text layer.
    A page with no text layer (for example, a scanned image) is skipped and
    may need OCR, which is outside this mini-project's scope.
    """
    path = Path(pdf_path).expanduser()

    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {path}")
    if not path.is_file():
        raise ValueError(f"PDF path is not a file: {path}")
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a .pdf file, got: {path.name}")

    try:
        # The context manager closes the PDF and releases its file handle.
        with fitz.open(str(path)) as pdf:
            if not pdf.is_pdf:
                raise ValueError(f"The file is not a valid PDF: {path.name}")
            if pdf.needs_pass:
                raise ValueError(
                    f"The PDF is password-protected: {path.name}. "
                    "Remove the password and try again."
                )
            if pdf.page_count == 0:
                raise ValueError(f"The PDF has no pages: {path.name}")

            documents = []

            # PyMuPDF numbers pages from 0 internally. Store a human-readable
            # page number instead, starting at 1.
            for page_number, page in enumerate(pdf, start=1):
                text = page.get_text("text").strip()
                if not text:
                    # This can happen with scanned pages or decorative pages.
                    continue

                # A LangChain Document represents a piece of text plus metadata.
                # The metadata travels with the text when the document is split.
                documents.append(
                    Document(
                        page_content=text,
                        metadata={"source": path.name, "page": page_number},
                    )
                )

            return documents

    except (fitz.FileDataError, RuntimeError) as error:
        raise ValueError(
            f"Could not read PDF (invalid, corrupted, or unreadable): {path.name}"
        ) from error
    except OSError as error:
        raise OSError(f"Could not open PDF file: {path.name}") from error


def split_documents(documents, chunk_size=500, chunk_overlap=100):
    """Split Documents into overlapping chunks while preserving metadata.

    ``chunk_size`` and ``chunk_overlap`` are measured in characters by default,
    not model tokens. Overlap repeats some text at chunk boundaries so a sentence
    or idea that crosses a boundary is less likely to be lost during retrieval.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0.")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be 0 or greater.")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    # LangChain copies each Document's metadata onto every resulting chunk, so
    # source and page information still identify where a chunk came from.
    return splitter.split_documents(documents)
