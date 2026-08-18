"""
Retrieval engine for the NICE Breast Cancer RAG.

Supports:
    Experiment A
    Experiment B

Features:
    - Query embedding
    - Top-K retrieval
    - Cosine distance
    - Relevance guardrail
    - Citation metadata
    - Grounding context
"""

from __future__ import annotations

from dataclasses import dataclass

import chromadb

from google import genai
from google.genai import types

from config import (
    GOOGLE_API_KEY,
    EMBEDDING_MODEL,
    EMBEDDING_DIMENSION,
    TOP_K,
    RELEVANCE_THRESHOLD,
    INSUFFICIENT_CONTEXT_MESSAGE,
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
# RESULT
# =========================================================

@dataclass
class RetrievedChunk:

    chunk_id: str

    source: str

    header: str

    number: str

    subheader: str

    page: int | str

    chunk_start_page: int | str

    chunk_end_page: int | str

    section_start_page: int | str

    section_end_page: int | str

    chunk_number: int | str

    distance: float

    document: str


# =========================================================
# EMBED QUERY
# =========================================================

def embed_query(
    question: str,
):

    response = (
        client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=question,
            config=types.EmbedContentConfig(
                output_dimensionality=
                    EMBEDDING_DIMENSION
            ),
        )
    )

    if not response.embeddings:

        raise RuntimeError(
            "No query embedding returned."
        )

    vector = (
        response
        .embeddings[0]
        .values
    )

    if len(vector) != (
        EMBEDDING_DIMENSION
    ):

        raise RuntimeError(
            "Query embedding dimension "
            f"mismatch. Expected "
            f"{EMBEDDING_DIMENSION}, "
            f"got {len(vector)}."
        )

    return vector


# =========================================================
# COLLECTION
# =========================================================

def get_collection(
    experiment: str,
):

    experiment = (
        experiment.upper()
    )

    if experiment not in {
        "A",
        "B",
    }:

        raise ValueError(
            "Experiment must be A or B."
        )

    db_path = get_db_path(
        experiment
    )

    collection_name = (
        get_collection_name(
            experiment
        )
    )

    chroma = (
        chromadb.PersistentClient(
            path=db_path
        )
    )

    try:

        collection = (
            chroma.get_collection(
                name=collection_name
            )
        )

    except Exception as error:

        raise RuntimeError(
            f"Collection "
            f"{collection_name} "
            "does not exist.\n"
            "Run src/embed_chunks.py first."
        ) from error

    return collection


# =========================================================
# RETRIEVE
# =========================================================

def retrieve(
    question: str,
    experiment: str,
    top_k: int = TOP_K,
):

    question = question.strip()

    if not question:

        raise ValueError(
            "Question cannot be empty."
        )

    collection = get_collection(
        experiment
    )

    count = (
        collection.count()
    )

    if count == 0:

        raise RuntimeError(
            "Vector collection is empty."
        )

    top_k = min(
        max(1, top_k),
        count,
    )

    query_embedding = embed_query(
        question
    )

    result = collection.query(
        query_embeddings=[
            query_embedding
        ],
        n_results=top_k,
        include=[
            "documents",
            "metadatas",
            "distances",
        ],
    )

    documents = (
        result["documents"][0]
    )

    metadatas = (
        result["metadatas"][0]
    )

    distances = (
        result["distances"][0]
    )

    chunks = []

    for document, metadata, distance in zip(
        documents,
        metadatas,
        distances,
    ):

        metadata = (
            metadata or {}
        )

        chunks.append(
            RetrievedChunk(

                chunk_id=str(
                    metadata.get(
                        "chunk_id",
                        "",
                    )
                ),

                source=str(
                    metadata.get(
                        "source",
                        "",
                    )
                ),

                header=str(
                    metadata.get(
                        "header",
                        "",
                    )
                ),

                number=str(
                    metadata.get(
                        "number",
                        "",
                    )
                ),

                subheader=str(
                    metadata.get(
                        "subheader",
                        "",
                    )
                ),

                page=metadata.get(
                    "page",
                    "",
                ),

                chunk_start_page=metadata.get(
                    "chunk_start_page",
                    metadata.get(
                        "page",
                        "",
                    ),
                ),

                chunk_end_page=metadata.get(
                    "chunk_end_page",
                    metadata.get(
                        "page",
                        "",
                    ),
                ),

                section_start_page=metadata.get(
                    "section_start_page",
                    "",
                ),

                section_end_page=metadata.get(
                    "section_end_page",
                    "",
                ),

                chunk_number=metadata.get(
                    "chunk_number",
                    "",
                ),

                distance=float(
                    distance
                ),

                document=document or "",
            )
        )

    return chunks


# =========================================================
# GUARDRAIL
# =========================================================

def filter_relevant(
    chunks,
    threshold=RELEVANCE_THRESHOLD,
):

    return [
        chunk
        for chunk in chunks
        if chunk.distance <= threshold
    ]


# =========================================================
# CITATION
# =========================================================

def citation_for_chunk(
    chunk,
):

    return (
        f"({chunk.source}, "
        f"Page {chunk.page})"
    )


# =========================================================
# BUILD CONTEXT
# =========================================================

def build_context(
    chunks,
):

    if not chunks:
        return ""

    blocks = []

    for index, chunk in enumerate(
        chunks,
        start=1,
    ):

        blocks.append(
            f"""
[CONTEXT {index}]

Document:
{chunk.source}

Citation:
{citation_for_chunk(chunk)}

Page:
{chunk.page}

Chunk page range:
{chunk.chunk_start_page}-
{chunk.chunk_end_page}

Section:
{chunk.number} {chunk.subheader}

Section range:
{chunk.section_start_page}-
{chunk.section_end_page}

Evidence:
{chunk.document}
""".strip()
        )

    return "\n\n".join(
        blocks
    )


# =========================================================
# DISPLAY
# =========================================================

def display_results(
    question,
    experiment,
    chunks,
):

    print()
    print("=" * 70)
    print(
        f"QUESTION: {question}"
    )
    print(
        f"EXPERIMENT: {experiment}"
    )
    print("=" * 70)

    for rank, chunk in enumerate(
        chunks,
        start=1,
    ):

        status = (
            "RELEVANT"
            if chunk.distance
            <= RELEVANCE_THRESHOLD
            else "REJECTED"
        )

        print()
        print(
            f"#{rank}"
        )

        print(
            f"Chunk ID : "
            f"{chunk.chunk_id}"
        )

        print(
            f"Document : "
            f"{chunk.source}"
        )

        print(
            f"Page     : "
            f"{chunk.page}"
        )

        print(
            f"Section  : "
            f"{chunk.subheader}"
        )

        print(
            f"Distance : "
            f"{chunk.distance:.4f}"
        )

        print(
            f"Status   : "
            f"{status}"
        )

        preview = (
            chunk.document
            .replace("\n", " ")
            .strip()
        )

        if len(preview) > 500:

            preview = (
                preview[:500]
                + "..."
            )

        print(
            "Preview  :"
        )

        print(
            preview
        )

    relevant = filter_relevant(
        chunks
    )

    print()
    print("=" * 70)
    print(
        "GUARDRAIL RESULT"
    )
    print("=" * 70)

    if relevant:

        print(
            f"PASS - "
            f"{len(relevant)} relevant "
            "chunk(s) found."
        )

    else:

        print(
            "FAIL - No relevant evidence."
        )

        print(
            INSUFFICIENT_CONTEXT_MESSAGE
        )

    return relevant


# =========================================================
# INTERACTIVE
# =========================================================

def main():

    print()
    print("=" * 70)
    print(
        "NICE BREAST CANCER RAG"
    )
    print(
        "EXPERIMENTAL RETRIEVAL"
    )
    print("=" * 70)

    print(
        f"Embedding model : "
        f"{EMBEDDING_MODEL}"
    )

    print(
        f"Embedding dimensions : "
        f"{EMBEDDING_DIMENSION}"
    )

    print(
        f"Threshold : "
        f"{RELEVANCE_THRESHOLD}"
    )

    print()
    print(
        "Type 'exit' to quit."
    )

    while True:

        experiment = input(
            "\nChoose experiment [A/B/exit]: "
        ).strip().upper()

        if experiment in {
            "EXIT",
            "QUIT",
        }:

            print(
                "Exiting."
            )

            break

        if experiment not in {
            "A",
            "B",
        }:

            print(
                "Please choose A or B."
            )

            continue

        question = input(
            "Question: "
        ).strip()

        if not question:
            continue

        top_k_input = input(
            "Top-K [5/10, default 5]: "
        ).strip()

        try:

            top_k = (
                int(top_k_input)
                if top_k_input
                else TOP_K
            )

        except ValueError:

            print(
                "Top-K must be an integer."
            )

            continue

        try:

            chunks = retrieve(
                question,
                experiment,
                top_k,
            )

            relevant = display_results(
                question,
                experiment,
                chunks,
            )

            if relevant:

                print()
                print("=" * 70)
                print(
                    "RETRIEVED CONTEXT"
                )
                print("=" * 70)

                print(
                    build_context(
                        relevant
                    )
                )

        except Exception as error:

            print()
            print(
                f"ERROR: {error}"
            )


if __name__ == "__main__":
    main()