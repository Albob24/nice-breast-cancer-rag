"""
Embedding pipeline.

Uses:
    Google GenAI SDK
    gemini-embedding-2
    768 dimensions
    ChromaDB

Builds:

    Experiment A
    Experiment B
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import chromadb

from google import genai
from google.genai import types

from config import (
    GOOGLE_API_KEY,
    EMBEDDING_MODEL,
    EMBEDDING_DIMENSION,
    EMBED_BATCH_SIZE,
    EMBED_RETRY_LIMIT,
    EMBED_RETRY_DELAY_SECONDS,
    get_chunks_file,
    get_db_path,
    get_collection_name,
)


# =========================================================
# GOOGLE CLIENT
# =========================================================

client = genai.Client(
    api_key=GOOGLE_API_KEY
)


# =========================================================
# LOAD CHUNKS
# =========================================================

def load_chunks(path):

    path = Path(path)

    if not path.exists():

        raise FileNotFoundError(
            f"{path} not found.\n"
            "Run src/chunking.py first."
        )

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:

        chunks = json.load(file)

    if not isinstance(
        chunks,
        list,
    ):

        raise ValueError(
            "Chunk file must contain a JSON list."
        )

    return chunks


# =========================================================
# VALIDATION
# =========================================================

def validate_chunks(
    chunks,
):

    seen_ids = set()

    for chunk in chunks:

        if not isinstance(
            chunk,
            dict,
        ):
            raise ValueError(
                "Invalid chunk object."
            )

        if not chunk.get(
            "text"
        ):
            raise ValueError(
                "Chunk contains empty text."
            )

        metadata = chunk.get(
            "metadata"
        )

        if not isinstance(
            metadata,
            dict,
        ):
            raise ValueError(
                "Chunk metadata missing."
            )

        chunk_id = metadata.get(
            "chunk_id"
        )

        if chunk_id in seen_ids:

            raise ValueError(
                f"Duplicate chunk ID: "
                f"{chunk_id}"
            )

        seen_ids.add(
            chunk_id
        )


# =========================================================
# METADATA CLEANING
# =========================================================

def clean_metadata(
    metadata,
):

    cleaned = {}

    for key, value in metadata.items():

        if value is None:

            cleaned[key] = ""

        elif isinstance(
            value,
            (
                str,
                int,
                float,
                bool,
            ),
        ):

            cleaned[key] = value

        else:

            cleaned[key] = str(
                value
            )

    return cleaned


# =========================================================
# DOCUMENT FORMAT
# =========================================================

def prepare_document(
    chunk,
):

    metadata = chunk[
        "metadata"
    ]

    return (
        f"Source: "
        f"{metadata.get('source', '')}\n"

        f"Section: "
        f"{metadata.get('number', '')} "
        f"{metadata.get('subheader', '')}\n"

        f"Page: "
        f"{metadata.get('page', '')}\n\n"

        f"Evidence:\n"
        f"{chunk['text']}"
    )


# =========================================================
# RETRY DELAY
# =========================================================

def get_retry_delay(
    error,
):

    error_text = str(
        error
    )

    match = re.search(
        r"retryDelay['\"]?\s*:\s*['\"]?(\d+)",
        error_text,
        re.IGNORECASE,
    )

    if match:

        return max(
            int(match.group(1)),
            65,
        )

    return 65


# =========================================================
# EMBED BATCH
# =========================================================

def embed_batch(
    texts,
):

    contents = [
        types.Content(
            parts=[
                types.Part.from_text(
                    text=text
                )
            ]
        )
        for text in texts
    ]

    for attempt in range(
        1,
        EMBED_RETRY_LIMIT + 1,
    ):

        try:

            response = (
                client.models.embed_content(
                    model=EMBEDDING_MODEL,
                    contents=contents,
                    config=types.EmbedContentConfig(
                        output_dimensionality=
                            EMBEDDING_DIMENSION
                    ),
                )
            )

            embeddings = [
                embedding.values
                for embedding
                in response.embeddings
            ]

            if len(embeddings) != len(
                texts
            ):

                raise RuntimeError(
                    "Number of embeddings "
                    "does not match number "
                    "of texts. "
                    f"Expected {len(texts)}, "
                    f"got {len(embeddings)}."
                )

            for vector in embeddings:

                if len(vector) != (
                    EMBEDDING_DIMENSION
                ):

                    raise RuntimeError(
                        "Embedding dimension "
                        "mismatch. "
                        f"Expected "
                        f"{EMBEDDING_DIMENSION}, "
                        f"got {len(vector)}."
                    )

            return embeddings

        except Exception as error:

            text = str(
                error
            )

            quota_error = (
                "429" in text
                or
                "RESOURCE_EXHAUSTED"
                in text
                or
                "quota"
                in text.lower()
            )

            if quota_error:

                delay = (
                    get_retry_delay(
                        error
                    )
                )

                print(
                    f"API quota reached. "
                    f"Waiting {delay}s..."
                )

                time.sleep(
                    delay
                )

            elif attempt < (
                EMBED_RETRY_LIMIT
            ):

                print(
                    f"Embedding batch "
                    f"failed "
                    f"(attempt "
                    f"{attempt}/"
                    f"{EMBED_RETRY_LIMIT})"
                )

                print(
                    f"Error: {error}"
                )

                time.sleep(
                    EMBED_RETRY_DELAY_SECONDS
                )

            else:

                raise

    raise RuntimeError(
        "Embedding failed."
    )


# =========================================================
# BUILD DATABASE
# =========================================================

def build_database(
    experiment,
):

    experiment = experiment.upper()

    chunks_file = get_chunks_file(
        experiment
    )

    db_path = get_db_path(
        experiment
    )

    collection_name = (
        get_collection_name(
            experiment
        )
    )

    print()
    print("=" * 70)
    print(
        f"BUILDING EXPERIMENT {experiment}"
    )
    print("=" * 70)

    chunks = load_chunks(
        chunks_file
    )

    validate_chunks(
        chunks
    )

    print(
        f"Loaded {len(chunks)} chunks."
    )

    chroma = (
        chromadb.PersistentClient(
            path=db_path
        )
    )

    # Rebuild from scratch
    try:

        chroma.delete_collection(
            collection_name
        )

    except Exception:

        pass

    collection = (
        chroma.create_collection(
            name=collection_name,
            metadata={
                "experiment":
                    experiment,

                "embedding_model":
                    EMBEDDING_MODEL,

                "embedding_dimension":
                    EMBEDDING_DIMENSION,

                "hnsw:space":
                    "cosine",
            },
        )
    )

    total = len(
        chunks
    )

    for start in range(
        0,
        total,
        EMBED_BATCH_SIZE,
    ):

        batch = chunks[
            start:
            start + EMBED_BATCH_SIZE
        ]

        documents = [
            prepare_document(
                chunk
            )
            for chunk in batch
        ]

        print()
        print(
            f"Experiment {experiment}: "
            f"embedding "
            f"{start + 1}-"
            f"{min(start + EMBED_BATCH_SIZE, total)} "
            f"of {total}..."
        )

        embeddings = embed_batch(
            documents
        )

        ids = [
            f"chunk_"
            f"{chunk['metadata']['chunk_id']}"
            for chunk in batch
        ]

        metadatas = [
            clean_metadata(
                chunk["metadata"]
            )
            for chunk in batch
        ]

        collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        print(
            f"✓ Stored "
            f"{collection.count()} vectors."
        )

    stored = (
        collection.count()
    )

    if stored != total:

        raise RuntimeError(
            f"Database verification failed. "
            f"Expected {total}, "
            f"stored {stored}."
        )

    print()
    print("=" * 70)
    print(
        f"EXPERIMENT {experiment} COMPLETE"
    )
    print("=" * 70)

    print(
        f"Chunks      : {total}"
    )

    print(
        f"Vectors     : {stored}"
    )

    print(
        f"Model       : {EMBEDDING_MODEL}"
    )

    print(
        f"Dimensions  : {EMBEDDING_DIMENSION}"
    )

    print(
        f"Collection  : {collection_name}"
    )

    print(
        f"Database    : {db_path}"
    )

    print("=" * 70)


# =========================================================
# MAIN
# =========================================================

def main():

    build_database("A")

    build_database("B")


if __name__ == "__main__":
    main()