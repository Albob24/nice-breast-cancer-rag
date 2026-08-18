"""
NICE Breast Cancer RAG
Grounded Generation Layer

Production-oriented RAG pipeline.

Architecture:

    User Question
          |
          v
    Gemini Embedding
          |
          v
       ChromaDB
          |
          v
     Guardrail
          |
          v
   Evidence Selection
          |
          v
    Gemini Generation
          |
          v
 Citation Validation
          |
          v
      Final Answer

Features:
- Experiment A / B compatible
- Experiment B selected by default
- Retrieval Top-K = 10
- Generation Context Top-N = 5
- Relevance guardrail
- Strict NICE grounding
- Citation validation
- Gemini retry / exponential backoff
- 429 / 500 / 502 / 503 / 504 handling
- Clean CLI
"""

from __future__ import annotations

import re
import sys
import time
from typing import Any


# =========================================================
# IMPORT CONFIGURATION
# =========================================================

try:
    # Package execution:
    # python -m src.generate
    from .config import (
        GOOGLE_API_KEY,
        EMBEDDING_MODEL,
        EMBEDDING_DIMENSION,
        GEMINI_MODEL,
        RELEVANCE_THRESHOLD,
        SYSTEM_INSTRUCTIONS,
        INSUFFICIENT_CONTEXT_MESSAGE,
        get_db_path,
        get_collection_name,
    )

except ImportError:
    # Direct execution:
    # python src\generate.py
    from config import (
        GOOGLE_API_KEY,
        EMBEDDING_MODEL,
        EMBEDDING_DIMENSION,
        GEMINI_MODEL,
        RELEVANCE_THRESHOLD,
        SYSTEM_INSTRUCTIONS,
        INSUFFICIENT_CONTEXT_MESSAGE,
        get_db_path,
        get_collection_name,
    )


# =========================================================
# THIRD-PARTY IMPORTS
# =========================================================

import chromadb
from google import genai


# =========================================================
# EXPERIMENT CONFIGURATION
# =========================================================

# Retrieval evaluation showed Experiment B as the
# recommended configuration.

EXPERIMENT = "B"


# =========================================================
# RETRIEVAL CONFIGURATION
# =========================================================

# Retrieve enough candidates for robust recall.

TOP_K = 10


# Number of retrieved chunks actually supplied
# to the generation model.

GENERATION_CONTEXT_K = 5


# =========================================================
# GEMINI RETRY CONFIGURATION
# =========================================================

# Temporary API failures should not immediately terminate
# the RAG application.

GENERATION_MAX_RETRIES = 4

GENERATION_INITIAL_DELAY_SECONDS = 3

GENERATION_MAX_DELAY_SECONDS = 30


# Retryable HTTP/API conditions.

RETRYABLE_ERROR_CODES = (
    "429",
    "500",
    "502",
    "503",
    "504",
    "UNAVAILABLE",
    "RESOURCE_EXHAUSTED",
    "INTERNAL",
    "DEADLINE_EXCEEDED",
)


# =========================================================
# DATABASE CONFIGURATION
# =========================================================

DB_PATH = get_db_path(
    EXPERIMENT
)

COLLECTION_NAME = get_collection_name(
    EXPERIMENT
)


# =========================================================
# GEMINI CLIENT
# =========================================================

if not GOOGLE_API_KEY:
    raise RuntimeError(
        "GOOGLE_API_KEY is not configured."
    )


client = genai.Client(
    api_key=GOOGLE_API_KEY
)


# =========================================================
# CHROMADB
# =========================================================

chroma_client = chromadb.PersistentClient(
    path=DB_PATH
)


try:

    collection = chroma_client.get_collection(
        name=COLLECTION_NAME
    )

except Exception as exc:

    raise RuntimeError(
        "Unable to open ChromaDB collection.\n"
        f"Database   : {DB_PATH}\n"
        f"Collection : {COLLECTION_NAME}\n"
        f"Error      : {exc}"
    ) from exc


# =========================================================
# DATABASE VALIDATION
# =========================================================

def validate_database() -> None:
    """
    Validate that the selected ChromaDB collection
    exists and contains vectors.
    """

    count = collection.count()

    if count <= 0:

        raise RuntimeError(
            "ChromaDB collection is empty.\n"
            f"Database   : {DB_PATH}\n"
            f"Collection : {COLLECTION_NAME}"
        )


# =========================================================
# QUERY EMBEDDING
# =========================================================

def embed_query(
    question: str,
) -> list[float]:
    """
    Generate a Gemini embedding for the user query.

    The resulting dimensionality must match the
    vectors stored in ChromaDB.
    """

    question = question.strip()

    if not question:

        raise ValueError(
            "Question cannot be empty."
        )

    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=question,
        config={
            "output_dimensionality": (
                EMBEDDING_DIMENSION
            )
        },
    )

    if not response.embeddings:

        raise RuntimeError(
            "Gemini returned no embeddings."
        )

    embedding = (
        response.embeddings[0].values
    )

    if not embedding:

        raise RuntimeError(
            "Gemini returned an empty embedding."
        )

    if len(embedding) != EMBEDDING_DIMENSION:

        raise RuntimeError(
            "Embedding dimension mismatch.\n"
            f"Expected : {EMBEDDING_DIMENSION}\n"
            f"Received : {len(embedding)}"
        )

    return list(embedding)


# =========================================================
# RETRIEVAL
# =========================================================

def retrieve_context(
    question: str,
    top_k: int = TOP_K,
) -> dict[str, Any]:
    """
    Retrieve the most relevant chunks.

    ChromaDB distance:
        lower = better
        higher = worse

    Guardrail:
        minimum distance <= threshold
            => PASS

        minimum distance > threshold
            => REJECT
    """

    validate_database()

    if top_k <= 0:

        raise ValueError(
            "top_k must be greater than zero."
        )

    query_embedding = embed_query(
        question
    )

    vector_count = collection.count()

    actual_top_k = min(
        top_k,
        vector_count,
    )

    results = collection.query(
        query_embeddings=[
            query_embedding
        ],
        n_results=actual_top_k,
        include=[
            "documents",
            "metadatas",
            "distances",
        ],
    )

    documents = results.get(
        "documents",
        [[]],
    )[0]

    metadatas = results.get(
        "metadatas",
        [[]],
    )[0]

    distances = results.get(
        "distances",
        [[]],
    )[0]

    if not distances:

        return {
            "passed": False,
            "minimum_distance": None,
            "results": [],
        }

    retrieved: list[
        dict[str, Any]
    ] = []

    for index, distance in enumerate(
        distances
    ):

        metadata: dict[str, Any] = {}

        if index < len(metadatas):

            metadata = (
                metadatas[index]
                or {}
            )

        document = ""

        if index < len(documents):

            document = (
                documents[index]
                or ""
            )

        # -------------------------------------------------
        # SOURCE
        # -------------------------------------------------

        source = str(
            metadata.get(
                "source",
                "UNKNOWN",
            )
        )

        # -------------------------------------------------
        # PAGE
        # -------------------------------------------------

        page = metadata.get(
            "page",
            metadata.get(
                "page_number",
                metadata.get(
                    "start_page",
                    metadata.get(
                        "section_start_page",
                        "",
                    ),
                ),
            ),
        )

        # -------------------------------------------------
        # SECTION
        # -------------------------------------------------

        section = (
            metadata.get(
                "subheader"
            )
            or metadata.get(
                "section"
            )
            or metadata.get(
                "header"
            )
            or ""
        )

        retrieved.append(
            {
                "rank": index + 1,
                "distance": float(
                    distance
                ),
                "source": source,
                "page": page,
                "section": str(
                    section
                ),
                "text": document,
                "metadata": metadata,
            }
        )

    minimum_distance = min(
        item["distance"]
        for item in retrieved
    )

    passed = (
        minimum_distance
        <= RELEVANCE_THRESHOLD
    )

    return {
        "passed": passed,
        "minimum_distance": (
            minimum_distance
        ),
        "results": retrieved,
    }


# =========================================================
# GENERATION CONTEXT SELECTION
# =========================================================

def select_generation_context(
    results: list[
        dict[str, Any]
    ],
    context_k: int = GENERATION_CONTEXT_K,
) -> list[
    dict[str, Any]
]:
    """
    Select the best retrieved chunks for generation.

    Retrieval can use Top-K=10 for recall.

    Generation receives only the strongest Top-N chunks
    to reduce noise and hallucination risk.
    """

    if context_k <= 0:

        raise ValueError(
            "context_k must be greater than zero."
        )

    return results[
        :context_k
    ]


# =========================================================
# CONTEXT BUILDER
# =========================================================

def build_context(
    results: list[
        dict[str, Any]
    ],
) -> str:
    """
    Convert retrieved chunks into a structured
    evidence context for Gemini.
    """

    if not results:

        return ""

    parts: list[str] = []

    for item in results:

        source = item["source"]

        page = item["page"]

        section = item["section"]

        distance = item["distance"]

        text = item["text"]

        parts.append(
            (
                f"--- CONTEXT "
                f"{item['rank']} ---\n\n"

                f"Document: "
                f"{source}\n"

                f"Page: "
                f"{page}\n"

                f"Section: "
                f"{section}\n"

                f"Retrieval distance: "
                f"{distance:.4f}\n"

                f"Citation: "
                f"({source}, Page {page})\n\n"

                f"Evidence:\n"
                f"{text}"
            )
        )

    return "\n\n".join(
        parts
    )


# =========================================================
# CITATION EXTRACTION
# =========================================================

def extract_citations(
    answer: str,
) -> list[
    tuple[str, str]
]:
    """
    Extract citations in the exact supported format:

        (NG101, Page 35)
        (CG81, Page 20)
    """

    pattern = (
        r"\((NG101|CG81),\s*Page\s+(\d+)\)"
    )

    return re.findall(
        pattern,
        answer,
        flags=re.IGNORECASE,
    )


# =========================================================
# CITATION VALIDATION
# =========================================================

def validate_citations(
    answer: str,
    retrieved_results: list[
        dict[str, Any]
    ],
) -> dict[str, Any]:
    """
    Verify that every citation in the answer
    points to a source/page actually present
    in the retrieved evidence.

    This validates citation consistency.

    It does NOT prove semantic correctness of
    every generated claim.
    """

    citations = extract_citations(
        answer
    )

    available: set[
        tuple[str, str]
    ] = set()

    for item in retrieved_results:

        source = str(
            item["source"]
        ).upper()

        page = str(
            item["page"]
        )

        if source and page:

            available.add(
                (
                    source,
                    page,
                )
            )

    normalized: list[
        tuple[str, str]
    ] = []

    invalid: list[
        tuple[str, str]
    ] = []

    for source, page in citations:

        citation = (
            source.upper(),
            page,
        )

        normalized.append(
            citation
        )

        if citation not in available:

            invalid.append(
                citation
            )

    valid = [
        citation
        for citation in normalized
        if citation in available
    ]

    return {
        "citations_found": citations,
        "valid_citations": valid,
        "invalid_citations": invalid,
        "all_valid": (
            len(invalid) == 0
        ),
    }


# =========================================================
# GENERATION PROMPT
# =========================================================

def build_generation_prompt(
    question: str,
    context: str,
) -> str:
    """
    Build the strict grounded-generation prompt.
    """

    return f"""
{SYSTEM_INSTRUCTIONS}

============================================================
RETRIEVED CONTEXT
============================================================

{context}

============================================================
USER QUESTION
============================================================

{question}

============================================================
STRICT GROUNDING RULES
============================================================

1. Answer ONLY from the retrieved context.

2. The retrieved context is the ONLY medical source
   available to you.

3. Do NOT use general medical knowledge.

4. Do NOT use internet information.

5. Do NOT use model memory.

6. Do NOT invent facts.

7. Do NOT guess.

8. Do NOT infer unsupported treatment recommendations.

9. Every factual medical claim must be supported
   by retrieved evidence.

10. Cite factual claims using exactly:

    (NG101, Page X)

    or

    (CG81, Page X)

11. X MUST correspond to a page contained in
    the retrieved context.

12. If the retrieved context is insufficient to
    answer the question, return exactly:

    {INSUFFICIENT_CONTEXT_MESSAGE}

13. Do not add any additional text to the fallback.

14. Keep successful answers concise.

15. Do not provide personalized diagnosis.

16. Do not provide personalized treatment decisions.

17. Do not use evidence from documents other than
    NG101 and CG81.

18. Do not cite pages that are not in the context.

============================================================
SUCCESSFUL ANSWER FORMAT
============================================================

Answer the question directly.

Every medical factual claim requires a citation.

End exactly with:

This system does not replace healthcare professionals.
""".strip()


# =========================================================
# RETRYABLE ERROR DETECTION
# =========================================================

def is_retryable_gemini_error(
    exc: Exception,
) -> bool:
    """
    Determine whether a Gemini exception is likely
    temporary and safe to retry.
    """

    error_text = str(
        exc
    ).upper()

    return any(
        code in error_text
        for code in RETRYABLE_ERROR_CODES
    )


# =========================================================
# GEMINI GENERATION WITH RETRY
# =========================================================

def generate_with_retry(
    *,
    model: str,
    contents: str,
    max_retries: int = GENERATION_MAX_RETRIES,
    initial_delay: float = (
        GENERATION_INITIAL_DELAY_SECONDS
    ),
) -> Any:
    """
    Call Gemini with automatic exponential backoff.

    Retry sequence by default:

        attempt 1
        wait 3s
        attempt 2
        wait 6s
        attempt 3
        wait 12s
        attempt 4

    Handles temporary API failures such as:

        429
        500
        502
        503
        504
    """

    if max_retries <= 0:

        raise ValueError(
            "max_retries must be greater than zero."
        )

    delay = float(
        initial_delay
    )

    last_exception: Exception | None = None

    for attempt in range(
        1,
        max_retries + 1,
    ):

        try:

            return client.models.generate_content(
                model=model,
                contents=contents,
            )

        except Exception as exc:

            last_exception = exc

            # -------------------------------------------------
            # Permanent error
            # -------------------------------------------------

            if not is_retryable_gemini_error(
                exc
            ):

                raise

            # -------------------------------------------------
            # Final attempt
            # -------------------------------------------------

            if attempt >= max_retries:

                print()
                print(
                    "Gemini generation failed after "
                    f"{max_retries} attempts."
                )

                raise

            # -------------------------------------------------
            # Retry
            # -------------------------------------------------

            print()
            print(
                "Gemini temporary API error."
            )

            print(
                f"Attempt        : "
                f"{attempt}/{max_retries}"
            )

            print(
                f"Error          : "
                f"{exc}"
            )

            print(
                f"Retrying in    : "
                f"{delay:.0f} seconds..."
            )

            time.sleep(
                delay
            )

            delay = min(
                delay * 2,
                GENERATION_MAX_DELAY_SECONDS,
            )

    # Defensive fallback.

    if last_exception:

        raise last_exception

    raise RuntimeError(
        "Gemini generation failed."
    )


# =========================================================
# GEMINI GENERATION
# =========================================================

def generate_answer(
    question: str,
    context: str,
) -> str:
    """
    Generate a grounded answer using Gemini.

    Uses retry/backoff for temporary API errors.
    """

    if not question.strip():

        raise ValueError(
            "Question cannot be empty."
        )

    if not context.strip():

        return INSUFFICIENT_CONTEXT_MESSAGE

    prompt = build_generation_prompt(
        question,
        context,
    )

    response = generate_with_retry(
        model=GEMINI_MODEL,
        contents=prompt,
    )

    answer = (
        response.text
        if response.text
        else ""
    ).strip()

    if not answer:

        raise RuntimeError(
            "Gemini returned an empty answer."
        )

    return answer


# =========================================================
# ANSWER VALIDATION
# =========================================================

def validate_answer(
    answer: str,
    retrieved_results: list[
        dict[str, Any]
    ],
) -> dict[str, Any]:
    """
    Validate final-answer grounding.

    Requirements for a successful answer:

    - At least one citation
    - Every citation valid
    - Required footer present

    The fallback response is considered valid
    without citation/footer because it is intentional.
    """

    insufficient = (
        answer.strip()
        == INSUFFICIENT_CONTEXT_MESSAGE
    )

    citation_result = (
        validate_citations(
            answer,
            retrieved_results,
        )
    )

    has_citation = bool(
        citation_result[
            "citations_found"
        ]
    )

    required_footer = (
        "This system does not replace "
        "healthcare professionals."
    )

    has_footer = (
        required_footer in answer
    )

    passed = (
        insufficient
        or (
            has_citation
            and citation_result[
                "all_valid"
            ]
            and has_footer
        )
    )

    return {
        "passed": passed,
        "insufficient_context": (
            insufficient
        ),
        "has_citation": (
            has_citation
        ),
        "has_footer": (
            has_footer
        ),
        "citation_validation": (
            citation_result
        ),
    }


# =========================================================
# COMPLETE RAG PIPELINE
# =========================================================

def answer_question(
    question: str,
) -> dict[str, Any]:
    """
    Complete RAG pipeline:

        Question
           |
           v
        Embedding
           |
           v
        Retrieval Top-K
           |
           v
        Guardrail
           |
           v
        Generation Top-N
           |
           v
        Gemini
           |
           v
        Citation Validation
    """

    question = question.strip()

    if not question:

        raise ValueError(
            "Question cannot be empty."
        )

    # =====================================================
    # 1. RETRIEVAL
    # =====================================================

    retrieval = retrieve_context(
        question,
        top_k=TOP_K,
    )

    retrieved_results = (
        retrieval["results"]
    )

    # =====================================================
    # 2. GUARDRAIL
    # =====================================================

    if not retrieval["passed"]:

        return {
            "question": question,
            "status": "REJECTED",
            "minimum_distance": (
                retrieval[
                    "minimum_distance"
                ]
            ),
            "answer": (
                INSUFFICIENT_CONTEXT_MESSAGE
            ),
            "retrieved_results": (
                retrieved_results
            ),
            "generation_context_results": [],
            "validation": {
                "passed": True,
                "insufficient_context": True,
                "reason": (
                    "No sufficiently relevant "
                    "NICE guideline evidence."
                ),
            },
        }

    # =====================================================
    # 3. SELECT GENERATION CONTEXT
    # =====================================================

    generation_results = (
        select_generation_context(
            retrieved_results,
            GENERATION_CONTEXT_K,
        )
    )

    # =====================================================
    # 4. BUILD CONTEXT
    # =====================================================

    context = build_context(
        generation_results
    )

    if not context:

        return {
            "question": question,
            "status": "REJECTED",
            "minimum_distance": (
                retrieval[
                    "minimum_distance"
                ]
            ),
            "answer": (
                INSUFFICIENT_CONTEXT_MESSAGE
            ),
            "retrieved_results": (
                retrieved_results
            ),
            "generation_context_results": (
                generation_results
            ),
            "validation": {
                "passed": True,
                "insufficient_context": True,
                "reason": (
                    "No generation context "
                    "was available."
                ),
            },
        }

    # =====================================================
    # 5. GENERATION
    # =====================================================

    answer = generate_answer(
        question,
        context,
    )

    # =====================================================
    # 6. VALIDATION
    # =====================================================

    validation = validate_answer(
        answer,
        generation_results,
    )

    return {
        "question": question,
        "status": "GENERATED",
        "minimum_distance": (
            retrieval[
                "minimum_distance"
            ]
        ),
        "answer": answer,
        "retrieved_results": (
            retrieved_results
        ),
        "generation_context_results": (
            generation_results
        ),
        "validation": validation,
    }


# =========================================================
# DISPLAY RETRIEVED EVIDENCE
# =========================================================

def print_retrieved_evidence(
    results: list[
        dict[str, Any]
    ],
) -> None:
    """
    Display retrieved evidence.
    """

    print()
    print("=" * 75)
    print("RETRIEVED EVIDENCE")
    print("=" * 75)

    if not results:

        print(
            "No evidence retrieved."
        )

        return

    for item in results:

        print()

        print(
            f"#{item['rank']}"
        )

        print(
            f"Document : "
            f"{item['source']}"
        )

        print(
            f"Page     : "
            f"{item['page']}"
        )

        print(
            f"Section  : "
            f"{item['section']}"
        )

        print(
            f"Distance : "
            f"{item['distance']:.4f}"
        )

        preview = (
            item["text"]
            .replace(
                "\n",
                " ",
            )
            .strip()
        )

        if len(preview) > 350:

            preview = (
                preview[:350]
                + "..."
            )

        print(
            f"Preview  : "
            f"{preview}"
        )


# =========================================================
# DISPLAY FINAL ANSWER
# =========================================================

def print_answer_result(
    result: dict[str, Any],
) -> None:
    """
    Display final RAG answer and validation.
    """

    print()
    print("=" * 75)
    print("RAG ANSWER")
    print("=" * 75)

    print()

    print(
        f"Status           : "
        f"{result['status']}"
    )

    distance = result[
        "minimum_distance"
    ]

    if distance is not None:

        print(
            f"Minimum distance : "
            f"{distance:.4f}"
        )

    print(
        f"Threshold        : "
        f"{RELEVANCE_THRESHOLD:.4f}"
    )

    print()

    print(
        result["answer"]
    )

    # =====================================================
    # VALIDATION
    # =====================================================

    validation = result.get(
        "validation",
        {},
    )

    print()
    print("=" * 75)
    print("ANSWER VALIDATION")
    print("=" * 75)

    print(
        "Validation passed :",
        validation.get(
            "passed",
            False,
        ),
    )

    if "has_citation" in validation:

        print(
            "Citation found    :",
            validation.get(
                "has_citation",
                False,
            ),
        )

    if "has_footer" in validation:

        print(
            "Footer found      :",
            validation.get(
                "has_footer",
                False,
            ),
        )

    citation_validation = (
        validation.get(
            "citation_validation"
        )
    )

    if citation_validation:

        print(
            "Valid citations   :",
            citation_validation.get(
                "valid_citations",
                [],
            ),
        )

        print(
            "Invalid citations :",
            citation_validation.get(
                "invalid_citations",
                [],
            ),
        )


# =========================================================
# STARTUP INFORMATION
# =========================================================

def print_startup_info() -> None:
    """
    Display application configuration.
    """

    validate_database()

    vector_count = (
        collection.count()
    )

    print()
    print("=" * 75)
    print("NICE BREAST CANCER RAG")
    print("GROUNDED GENERATION")
    print("=" * 75)

    print()

    print(
        f"Experiment           : "
        f"{EXPERIMENT}"
    )

    print(
        f"Embedding model      : "
        f"{EMBEDDING_MODEL}"
    )

    print(
        f"Embedding dimensions : "
        f"{EMBEDDING_DIMENSION}"
    )

    print(
        f"Generation model     : "
        f"{GEMINI_MODEL}"
    )

    print(
        f"Database             : "
        f"{DB_PATH}"
    )

    print(
        f"Collection           : "
        f"{COLLECTION_NAME}"
    )

    print(
        f"Vectors              : "
        f"{vector_count}"
    )

    print(
        f"Retrieval Top-K      : "
        f"{TOP_K}"
    )

    print(
        f"Generation Top-N     : "
        f"{GENERATION_CONTEXT_K}"
    )

    print(
        f"Guardrail threshold  : "
        f"{RELEVANCE_THRESHOLD}"
    )

    print(
        f"Generation retries   : "
        f"{GENERATION_MAX_RETRIES}"
    )

    print(
        f"Initial retry delay  : "
        f"{GENERATION_INITIAL_DELAY_SECONDS}s"
    )

    print()

    print(
        "Type 'exit' to quit."
    )


# =========================================================
# MAIN CLI
# =========================================================

def main() -> None:
    """
    Interactive CLI.
    """

    print_startup_info()

    while True:

        try:

            print()

            question = input(
                "Question: "
            ).strip()

        except (
            KeyboardInterrupt,
            EOFError,
        ):

            print()
            print(
                "Exiting."
            )

            break

        if not question:

            continue

        if question.lower() in {
            "exit",
            "quit",
            "q",
        }:

            print(
                "Exiting."
            )

            break

        try:

            result = answer_question(
                question
            )

            print_retrieved_evidence(
                result[
                    "generation_context_results"
                ]
            )

            print_answer_result(
                result
            )

        except KeyboardInterrupt:

            print()
            print(
                "Operation cancelled."
            )

        except Exception as exc:

            print()
            print("=" * 75)
            print("ERROR")
            print("=" * 75)

            print()

            print(
                f"{type(exc).__name__}: "
                f"{exc}"
            )

            print()

            print(
                "The application remains running."
            )


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":

    main()