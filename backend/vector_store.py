"""Store and search document embeddings in Pinecone.

A vector database stores numeric vectors and can find the vectors closest to a
query vector. Pinecone stores the embeddings produced by Task 7, not the raw PDF
files; the chunk text, source filename, and page number travel as metadata.

The index uses cosine similarity. A score near 1 means the query and chunk
vectors point in similar directions, which usually means they are semantically
related. This is different from keyword search, which mainly matches literal
words rather than meaning.

This module uses the Pinecone SDK directly, without LangChain's wrapper, so the
storage and query flow stays visible.
"""

import hashlib
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.documents import Document
from pinecone import Pinecone, ServerlessSpec

from backend.embeddings import EMBEDDING_DIMENSION, generate_embeddings

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# A namespace is a logical partition inside one index. Using a dedicated one
# keeps this mini-project's vectors separate from unrelated data in the index.
PINECONE_NAMESPACE = "rag-mini-project"

# Upload in small batches so each request stays manageable.
UPSERT_BATCH_SIZE = 50

# Serverless AWS us-east-1 is supported by Pinecone's free plan.
PINECONE_CLOUD = "aws"
PINECONE_REGION = "us-east-1"
INDEX_WAIT_SECONDS = 120
INDEX_POLL_SECONDS = 5


def get_pinecone_index():
    """Return a ready Pinecone index, creating it only when it is missing.

    The index must have the same 384 dimensions and cosine metric as the
    embedding model. An existing incompatible index is never deleted or
    recreated automatically; a clear error is raised instead.
    """
    # Load .env without overriding variables already set in the environment.
    load_dotenv(PROJECT_ROOT / ".env")

    api_key = os.getenv("PINECONE_API_KEY", "").strip()
    index_name = os.getenv("PINECONE_INDEX_NAME", "").strip()

    if not api_key:
        raise ValueError(
            "PINECONE_API_KEY is missing. Add it to .env or the environment."
        )
    if not index_name:
        raise ValueError(
            "PINECONE_INDEX_NAME is missing. Add it to .env or the environment."
        )

    # Current SDK initialization. The old pinecone.init(api_key=...) style is
    # deprecated and is intentionally not used.
    pc = Pinecone(api_key=api_key)

    if not pc.has_index(index_name):
        print(f"Creating Pinecone index '{index_name}'...")
        try:
            pc.create_index(
                name=index_name,
                dimension=EMBEDDING_DIMENSION,
                metric="cosine",
                vector_type="dense",
                spec=ServerlessSpec(
                    cloud=PINECONE_CLOUD,
                    region=PINECONE_REGION,
                ),
                timeout=INDEX_WAIT_SECONDS,
            )
        except TimeoutError as error:
            raise TimeoutError(
                f"Pinecone index '{index_name}' was not ready within "
                f"{INDEX_WAIT_SECONDS} seconds."
            ) from error

    description = pc.describe_index(index_name)

    dimension = description.dimension
    metric = getattr(description.metric, "value", description.metric)

    if dimension != EMBEDDING_DIMENSION:
        raise RuntimeError(
            f"Pinecone index '{index_name}' has dimension {dimension}, but this "
            f"project requires {EMBEDDING_DIMENSION}. Use a different index name."
        )
    if str(metric).lower() != "cosine":
        raise RuntimeError(
            f"Pinecone index '{index_name}' uses metric '{metric}', but this "
            "project requires cosine. Use a different index name."
        )

    # An existing index may still be initializing. Wait before returning it so
    # callers never query an index that is not ready.
    deadline = time.monotonic() + INDEX_WAIT_SECONDS
    while not description.status.ready:
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"Pinecone index '{index_name}' was not ready within "
                f"{INDEX_WAIT_SECONDS} seconds."
            )
        print(f"Waiting for Pinecone index '{index_name}' to become ready...")
        time.sleep(INDEX_POLL_SECONDS)
        description = pc.describe_index(index_name)

    return pc.Index(index_name)


def store_documents(documents, embeddings, index):
    """Upsert Document chunks and their embeddings into Pinecone.

    ``upsert`` means insert new vectors or update vectors with the same IDs.
    Stable IDs are hashed from the source, page, chunk position, and text, so
    re-running the same upload updates the same records instead of creating
    duplicate vectors.

    Metadata is stored beside each vector so a search result can show its text,
    source filename, and page number. Batch uploads avoid sending one huge
    request at a time.

    Source filenames alone are not globally unique. Two different PDFs with the
    same filename, page, chunk position, and text could collide. This simple
    mini-project accepts that small risk; a production system would use a
    stronger source identifier such as a file hash or full path.
    """
    if not isinstance(documents, (list, tuple)) or not documents:
        raise ValueError("documents must be a nonempty list of Document objects.")
    if not isinstance(embeddings, (list, tuple)) or not embeddings:
        raise ValueError("embeddings must be a nonempty list of vectors.")
    if len(documents) != len(embeddings):
        raise ValueError(
            "The number of documents must equal the number of embeddings."
        )

    vectors = []

    for position, (document, embedding) in enumerate(zip(documents, embeddings)):
        text = getattr(document, "page_content", None)
        if not isinstance(text, str) or not text.strip():
            raise ValueError(
                f"Document at index {position} has no nonempty page_content."
            )

        metadata = getattr(document, "metadata", None) or {}
        source = metadata.get("source")
        page = metadata.get("page")

        if not isinstance(source, str) or not source.strip():
            raise ValueError(
                f"Document at index {position} is missing a source filename."
            )
        if not isinstance(page, int) or page < 1:
            raise ValueError(
                f"Document at index {position} is missing a valid page number."
            )

        try:
            values = [float(value) for value in embedding]
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Embedding at index {position} contains a non-numeric value."
            ) from error

        if len(values) != EMBEDDING_DIMENSION:
            raise ValueError(
                f"Embedding at index {position} has {len(values)} dimensions; "
                f"expected {EMBEDDING_DIMENSION}."
            )

        # Chunk position is part of the hash so two identical chunks on the same
        # page still receive different IDs.
        id_source = f"{source}|{page}|{position}|{text}"
        vector_id = hashlib.sha256(id_source.encode("utf-8")).hexdigest()

        vectors.append(
            {
                "id": vector_id,
                "values": values,
                "metadata": {
                    "source": source,
                    "page": page,
                    "text": text,
                },
            }
        )

    uploaded_count = 0

    for start in range(0, len(vectors), UPSERT_BATCH_SIZE):
        batch = vectors[start : start + UPSERT_BATCH_SIZE]
        response = index.upsert(
            vectors=batch,
            namespace=PINECONE_NAMESPACE,
            show_progress=False,
        )
        batch_count = getattr(response, "upserted_count", None)
        uploaded_count += batch_count if isinstance(batch_count, int) else len(batch)

    print(
        f"Uploaded {uploaded_count} vectors to Pinecone namespace "
        f"'{PINECONE_NAMESPACE}'."
    )
    return uploaded_count


def search_documents(query, index, top_k=3):
    """Return the top matching chunks for a question.

    The question is embedded with the same model used for the chunks, then
    Pinecone compares its vector against stored vectors using the index's cosine
    metric. An empty list means Pinecone returned no matches.
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a nonempty string.")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive integer.")

    # Reuse the exact embedding model used for the document chunks so the
    # question and chunks are represented in the same vector space.
    query_embedding = generate_embeddings(
        [Document(page_content=query.strip())]
    )[0]

    response = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True,
        namespace=PINECONE_NAMESPACE,
    )

    matches = getattr(response, "matches", None) or []
    results = []

    for match in matches:
        metadata = getattr(match, "metadata", None) or {}
        results.append(
            {
                "text": metadata.get("text", ""),
                "source": metadata.get("source", ""),
                "page": metadata.get("page"),
                "score": float(getattr(match, "score", 0.0)),
            }
        )

    return results
