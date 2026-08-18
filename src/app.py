"""
NICE Breast Cancer AI
FastAPI Backend

API layer over the existing grounded RAG engine.

Architecture:

    Frontend
       |
       v
    FastAPI
       |
       v
    src.generate
       |
       +--> Gemini Embedding
       |
       +--> ChromaDB
       |
       +--> Relevance Guardrail
       |
       +--> Gemini Generation
       |
       +--> Citation Validation
       |
       v
    JSON Response
"""

from __future__ import annotations

import time
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


# =========================================================
# RAG ENGINE
# =========================================================

try:

    from . import generate

except ImportError:

    import generate


# =========================================================
# APPLICATION
# =========================================================

app = FastAPI(
    title="NICE Breast Cancer AI",
    description=(
        "Grounded medical information assistant "
        "based strictly on NICE NG101 and CG81."
    ),
    version="1.0.0",
)


# =========================================================
# CORS
# =========================================================

# Development configuration.
#
# Later, when the frontend is deployed, replace this
# with the real frontend domain.

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=[
        "*",
    ],
    allow_headers=[
        "*",
    ],
)


# =========================================================
# REQUEST MODELS
# =========================================================

class ChatRequest(BaseModel):
    """
    Request body for /api/chat.
    """

    question: str = Field(
        ...,
        min_length=1,
        max_length=4000,
        description=(
            "User question about NICE breast cancer guidelines."
        ),
    )


# =========================================================
# RESPONSE MODELS
# =========================================================

class Citation(BaseModel):
    """
    Valid citation returned by the RAG engine.
    """

    source: str
    page: str


class Source(BaseModel):
    """
    Retrieved evidence source.
    """

    rank: int
    source: str
    page: str
    section: str
    distance: float
    text: str


class RetrievalInfo(BaseModel):
    """
    Retrieval diagnostics.
    """

    top_k: int
    generation_context_k: int
    minimum_distance: float | None
    threshold: float
    vectors: int
    experiment: str
    collection: str


class ChatResponse(BaseModel):
    """
    Complete response returned to the frontend.
    """

    answer: str
    status: str

    citations: list[Citation]

    sources: list[Source]

    retrieval: RetrievalInfo

    validation: dict[str, Any]

    latency_ms: float


# =========================================================
# HEALTH RESPONSE
# =========================================================

class HealthResponse(BaseModel):

    status: str

    service: str

    experiment: str

    collection: str

    vectors: int

    embedding_model: str

    generation_model: str

    top_k: int

    generation_context_k: int

    threshold: float


# =========================================================
# ROOT
# =========================================================

@app.get(
    "/",
    tags=["System"],
)
async def root() -> dict[str, Any]:
    """
    API root.
    """

    return {
        "service": "NICE Breast Cancer AI",
        "status": "online",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/health",
    }


# =========================================================
# HEALTH
# =========================================================

@app.get(
    "/api/health",
    response_model=HealthResponse,
    tags=["System"],
)
async def health() -> HealthResponse:
    """
    Check backend and ChromaDB health.
    """

    try:

        vector_count = (
            generate.collection.count()
        )

    except Exception as exc:

        raise HTTPException(
            status_code=503,
            detail=(
                "ChromaDB is unavailable: "
                f"{exc}"
            ),
        )

    return HealthResponse(
        status="healthy",
        service="NICE Breast Cancer AI",
        experiment=generate.EXPERIMENT,
        collection=generate.COLLECTION_NAME,
        vectors=vector_count,
        embedding_model=(
            generate.EMBEDDING_MODEL
        ),
        generation_model=(
            generate.GEMINI_MODEL
        ),
        top_k=generate.TOP_K,
        generation_context_k=(
            generate.GENERATION_CONTEXT_K
        ),
        threshold=(
            generate.RELEVANCE_THRESHOLD
        ),
    )


# =========================================================
# CHAT
# =========================================================

@app.post(
    "/api/chat",
    response_model=ChatResponse,
    tags=["RAG"],
)
async def chat(
    request: ChatRequest,
) -> ChatResponse:
    """
    Run one complete grounded RAG query.

    Pipeline:

        Question
            ↓
        Embedding
            ↓
        ChromaDB
            ↓
        Guardrail
            ↓
        Evidence selection
            ↓
        Gemini
            ↓
        Citation validation
            ↓
        JSON
    """

    question = (
        request.question
        .strip()
    )

    if not question:

        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty.",
        )

    started = time.perf_counter()

    try:

        result = (
            generate.answer_question(
                question
            )
        )

    except Exception as exc:

        elapsed = (
            time.perf_counter()
            - started
        )

        print(
            f"[ERROR] /api/chat "
            f"after {elapsed:.2f}s: "
            f"{type(exc).__name__}: "
            f"{exc}"
        )

        raise HTTPException(
            status_code=500,
            detail={
                "message": (
                    "RAG pipeline failed."
                ),
                "error": str(exc),
                "type": type(
                    exc
                ).__name__,
            },
        ) from exc

    elapsed_ms = (
        time.perf_counter()
        - started
    ) * 1000

    # =====================================================
    # RESULT
    # =====================================================

    answer = result.get(
        "answer",
        "",
    )

    status = result.get(
        "status",
        "UNKNOWN",
    )

    minimum_distance = result.get(
        "minimum_distance"
    )

    # =====================================================
    # SOURCES
    # =====================================================

    generation_results = (
        result.get(
            "generation_context_results",
            [],
        )
    )

    sources: list[Source] = []

    for item in generation_results:

        sources.append(
            Source(
                rank=int(
                    item.get(
                        "rank",
                        0,
                    )
                ),
                source=str(
                    item.get(
                        "source",
                        "UNKNOWN",
                    )
                ),
                page=str(
                    item.get(
                        "page",
                        "",
                    )
                ),
                section=str(
                    item.get(
                        "section",
                        "",
                    )
                ),
                distance=float(
                    item.get(
                        "distance",
                        0.0,
                    )
                ),
                text=str(
                    item.get(
                        "text",
                        "",
                    )
                ),
            )
        )

    # =====================================================
    # CITATIONS
    # =====================================================

    citations: list[Citation] = []

    validation = result.get(
        "validation",
        {},
    )

    citation_validation = (
        validation.get(
            "citation_validation",
            {},
        )
        if isinstance(
            validation,
            dict,
        )
        else {}
    )

    valid_citations = (
        citation_validation.get(
            "valid_citations",
            [],
        )
    )

    seen: set[
        tuple[str, str]
    ] = set()

    for citation in valid_citations:

        if not isinstance(
            citation,
            (
                tuple,
                list,
            ),
        ):
            continue

        if len(citation) != 2:
            continue

        source = str(
            citation[0]
        ).upper()

        page = str(
            citation[1]
        )

        key = (
            source,
            page,
        )

        if key in seen:
            continue

        seen.add(key)

        citations.append(
            Citation(
                source=source,
                page=page,
            )
        )

    # =====================================================
    # RETRIEVAL INFO
    # =====================================================

    try:

        vector_count = (
            generate.collection.count()
        )

    except Exception:

        vector_count = 0

    retrieval = RetrievalInfo(
        top_k=generate.TOP_K,
        generation_context_k=(
            generate.GENERATION_CONTEXT_K
        ),
        minimum_distance=(
            float(
                minimum_distance
            )
            if minimum_distance is not None
            else None
        ),
        threshold=(
            generate.RELEVANCE_THRESHOLD
        ),
        vectors=vector_count,
        experiment=(
            generate.EXPERIMENT
        ),
        collection=(
            generate.COLLECTION_NAME
        ),
    )

    # =====================================================
    # FINAL RESPONSE
    # =====================================================

    return ChatResponse(
        answer=answer,
        status=status,
        citations=citations,
        sources=sources,
        retrieval=retrieval,
        validation=validation,
        latency_ms=round(
            elapsed_ms,
            2,
        ),
    )


# =========================================================
# DEVELOPMENT SERVER
# =========================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "src.app:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
    )