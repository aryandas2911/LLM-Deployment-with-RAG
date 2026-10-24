"""Generate local CPU embeddings for document chunks and user questions.

An embedding model converts text into a fixed-length list of numbers. Texts with
similar meaning tend to produce vectors that point in similar directions, which
lets a RAG system compare a question with document chunks numerically.

All-MiniLM-L6-v2 is a small Sentence Transformers model that produces
384-dimensional vectors. Each of those 384 numbers is a learned feature; an
individual dimension is not usually meaningful to a human, but the pattern of
all 384 values captures the text's meaning. The same model must be used for
document chunks and questions so both kinds of text live in the same vector
space.

This module produces the vectors only. Pinecone will store and search them when
the document collection becomes too large to compare in memory.
"""

from functools import lru_cache

from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384
EMBEDDING_BATCH_SIZE = 32


@lru_cache(maxsize=1)
def load_embedding_model():
    """Load the Sentence Transformers model once per Python process.

    The model may be downloaded from Hugging Face on its first use and then
    cached locally. Later calls reuse the cached in-memory model, avoiding
    repeated downloads, disk reads, and memory use.
    """
    try:
        # CPU is explicit: this project does not depend on a local GPU.
        model = SentenceTransformer(EMBEDDING_MODEL_NAME, device="cpu")
    except Exception as error:
        raise RuntimeError(
            "Could not load the embedding model. The first use may require "
            "internet access to download it from Hugging Face. Check the "
            "connection and try again."
        ) from error

    # Sentence Transformers 5 renamed this method; support both names so the
    # project also works with the older 3.x/4.x releases allowed by requirements.
    if hasattr(model, "get_embedding_dimension"):
        actual_dimension = model.get_embedding_dimension()
    else:
        actual_dimension = model.get_sentence_embedding_dimension()

    if actual_dimension != EMBEDDING_DIMENSION:
        raise RuntimeError(
            f"Expected a {EMBEDDING_DIMENSION}-dimensional embedding model, "
            f"but {EMBEDDING_MODEL_NAME} returned {actual_dimension}."
        )

    return model


def generate_embeddings(documents):
    """Convert LangChain Documents into a list of 384-dimensional vectors.

    The output order matches the input order: ``embeddings[i]`` belongs to
    ``documents[i]``. Never filtering or reordering documents here is important
    because each vector must stay paired with the source filename and page
    metadata used later for retrieval and citations.
    """
    if not isinstance(documents, (list, tuple)):
        raise ValueError("documents must be a list of LangChain Document objects.")
    if not documents:
        raise ValueError("documents must contain at least one Document.")

    texts = []
    for index, document in enumerate(documents):
        text = getattr(document, "page_content", None)
        if not isinstance(text, str) or not text.strip():
            raise ValueError(
                f"Document at index {index} has no nonempty page_content."
            )
        texts.append(text)

    model = load_embedding_model()

    # Batch encoding keeps memory use reasonable while returning rows in the
    # same order as the input texts.
    embeddings = model.encode(
        texts,
        batch_size=EMBEDDING_BATCH_SIZE,
        show_progress_bar=False,
        convert_to_numpy=True,
    )

    if embeddings.ndim != 2 or embeddings.shape[0] != len(texts):
        raise ValueError(
            "The embedding model returned an unexpected number of vectors."
        )
    if embeddings.shape[1] != EMBEDDING_DIMENSION:
        raise ValueError(
            f"Expected {EMBEDDING_DIMENSION}-dimensional vectors, "
            f"but received {embeddings.shape[1]} dimensions."
        )

    # tolist() converts NumPy values into ordinary Python floats.
    return embeddings.tolist()
