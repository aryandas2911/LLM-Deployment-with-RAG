"""Real PDF extraction and chunking tests that do not need external services."""

import pymupdf as fitz
import pytest

from backend.pdf_processor import extract_pdf_text, split_documents


def _write_pdf(path, page_texts):
    pdf = fitz.open()
    for text in page_texts:
        page = pdf.new_page()
        page.insert_text((72, 72), text)
    pdf.save(path)
    pdf.close()


def test_valid_pdf_extracts_text_and_page_metadata(tmp_path):
    pdf_path = tmp_path / "sample.pdf"
    _write_pdf(
        pdf_path,
        [
            "Machine learning learns patterns from data.",
            "Pinecone stores vectors for semantic search.",
        ],
    )

    documents = extract_pdf_text(pdf_path)

    assert len(documents) == 2
    assert documents[0].metadata == {"source": "sample.pdf", "page": 1}
    assert documents[1].metadata == {"source": "sample.pdf", "page": 2}
    assert "Machine learning" in documents[0].page_content
    assert "Pinecone" in documents[1].page_content


def test_split_documents_preserves_source_and_page(tmp_path):
    pdf_path = tmp_path / "sample.pdf"
    _write_pdf(pdf_path, ["A sentence about retrieval. " * 20])

    documents = extract_pdf_text(pdf_path)
    chunks = split_documents(documents, chunk_size=40, chunk_overlap=10)

    assert len(chunks) > 1
    assert all(chunk.page_content.strip() for chunk in chunks)
    assert all(chunk.metadata["source"] == "sample.pdf" for chunk in chunks)
    assert all(chunk.metadata["page"] == 1 for chunk in chunks)


def test_invalid_pdf_is_rejected(tmp_path):
    pdf_path = tmp_path / "invalid.pdf"
    pdf_path.write_bytes(b"not a pdf")

    with pytest.raises(ValueError, match="Could not read PDF"):
        extract_pdf_text(pdf_path)


def test_empty_pdf_is_rejected(tmp_path):
    pdf_path = tmp_path / "empty.pdf"
    pdf_path.write_bytes(b"")

    with pytest.raises(ValueError, match="Could not read PDF"):
        extract_pdf_text(pdf_path)


def test_pdf_without_text_returns_no_documents(tmp_path):
    pdf_path = tmp_path / "no-text.pdf"
    pdf = fitz.open()
    pdf.new_page()
    pdf.save(pdf_path)
    pdf.close()

    assert extract_pdf_text(pdf_path) == []


@pytest.mark.parametrize(
    "chunk_size, chunk_overlap",
    [(0, 0), (100, -1), (100, 100), (100, 101)],
)
def test_split_documents_rejects_invalid_settings(chunk_size, chunk_overlap):
    with pytest.raises(ValueError):
        split_documents([], chunk_size=chunk_size, chunk_overlap=chunk_overlap)
