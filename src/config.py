"""
NICE Breast Cancer RAG
Central Configuration

Single source of truth for:

- Environment variables
- Project paths
- PDF configuration
- Chunking experiments
- Embedding configuration
- Retrieval configuration
- Generation configuration
- Guardrails
- System instructions
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


# =========================================================
# PROJECT ROOT
# =========================================================

BASE_DIR = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)


# =========================================================
# ENVIRONMENT
# =========================================================

ENV_FILE = BASE_DIR / ".env"

load_dotenv(ENV_FILE)


# =========================================================
# GOOGLE API
# =========================================================

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise RuntimeError(
        "GOOGLE_API_KEY is not set.\n\n"
        "Create a .env file in the project root:\n\n"
        "GOOGLE_API_KEY=YOUR_REAL_API_KEY"
    )


# =========================================================
# PATHS
# =========================================================

DATA_DIR = BASE_DIR / "data"

CHUNKS_FILE = (
    BASE_DIR / "chunks_metadata.json"
)

CHUNKS_FILE_A = (
    BASE_DIR / "chunks_metadata_A.json"
)

CHUNKS_FILE_B = (
    BASE_DIR / "chunks_metadata_B.json"
)


# =========================================================
# CHROMADB PATHS
# =========================================================

DB_PATH_A = (
    BASE_DIR / "nice_breast_cancer_db_A"
)

DB_PATH_B = (
    BASE_DIR / "nice_breast_cancer_db_B"
)


# =========================================================
# DATABASE HELPERS
# =========================================================

def get_db_path(
    experiment: str = "A",
) -> str:
    """
    Return the persistent ChromaDB path
    for Experiment A or B.
    """

    experiment = (
        experiment
        .strip()
        .upper()
    )

    if experiment == "A":
        return str(DB_PATH_A)

    if experiment == "B":
        return str(DB_PATH_B)

    raise ValueError(
        "Experiment must be 'A' or 'B'."
    )


def get_collection_name(
    experiment: str = "A",
) -> str:
    """
    Return the ChromaDB collection name
    for Experiment A or B.
    """

    experiment = (
        experiment
        .strip()
        .upper()
    )

    if experiment == "A":
        return (
            "breast_cancer_guidelines_A"
        )

    if experiment == "B":
        return (
            "breast_cancer_guidelines_B"
        )

    raise ValueError(
        "Experiment must be 'A' or 'B'."
    )


# =========================================================
# PDF FILES
# =========================================================

PDF_FILES = {
    "NG101":
        DATA_DIR / "NG101.pdf",

    "CG81":
        DATA_DIR / "CG81.pdf",
}


# =========================================================
# TABLE OF CONTENTS
# =========================================================

TOC_PAGES = {
    "NG101": [3, 4, 5],
    "CG81": [3, 4],
}


# =========================================================
# USEFUL GUIDELINE PAGES
# =========================================================

USEFUL_PAGES = {
    "NG101": (7, 58),
    "CG81": (7, 32),
}


# =========================================================
# CHUNKING
# =========================================================

# Legacy/default word-based chunking values.
# Experiments A and B use token-based targets below.

CHUNK_SIZE_WORDS = 300

CHUNK_OVERLAP_WORDS = 50


# =========================================================
# EXPERIMENT A
# =========================================================

# Recommendation-style chunks.

EXPERIMENT_A_TARGET_TOKENS = 500

EXPERIMENT_A_MIN_TOKENS = 400

EXPERIMENT_A_MAX_TOKENS = 600

EXPERIMENT_A_OVERLAP_RATIO = 0.12


# =========================================================
# EXPERIMENT B
# =========================================================

# Larger chunks for longer evidence and tables.

EXPERIMENT_B_TARGET_TOKENS = 800

EXPERIMENT_B_MIN_TOKENS = 700

EXPERIMENT_B_MAX_TOKENS = 900

EXPERIMENT_B_OVERLAP_RATIO = 0.12


# =========================================================
# EMBEDDING
# =========================================================

# IMPORTANT:
# Changing this model requires rebuilding both
# vector databases.

EMBEDDING_MODEL = (
    "gemini-embedding-2"
)

EMBEDDING_DIMENSION = 768

EMBED_BATCH_SIZE = 20

EMBED_RETRY_LIMIT = 3

EMBED_RETRY_DELAY_SECONDS = 5


# =========================================================
# COLLECTION NAMES
# =========================================================

# Legacy/default collection.

COLLECTION_NAME = (
    "breast_cancer_guidelines"
)

COLLECTION_NAME_A = (
    "breast_cancer_guidelines_A"
)

COLLECTION_NAME_B = (
    "breast_cancer_guidelines_B"
)


# =========================================================
# RETRIEVAL
# =========================================================

# Default Top-K used by the generation pipeline.

TOP_K = 10


# Top-K values used specifically for
# retrieval evaluation.

TOP_K_VALUES = (
    5,
    10,
)


# ChromaDB distance threshold.
#
# Lower distance = better semantic similarity.
#
# Current calibrated guardrail:
#
# distance <= 0.50 -> sufficient/relevant
# distance >  0.50 -> insufficient context

RELEVANCE_THRESHOLD = 0.50


# =========================================================
# GENERATION
# =========================================================

# Generation model is independent from
# the embedding model.
#
# Changing this does NOT require rebuilding
# the vector databases.

GEMINI_MODEL = (
    "gemini-3.6-flash"
)


# =========================================================
# OPTIONAL LOCAL FALLBACK
# =========================================================

LOCAL_FALLBACK_MODEL = (
    "TinyLlama/"
    "TinyLlama-1.1B-Chat-v1.0"
)


# =========================================================
# GUARDRAIL FALLBACK
# =========================================================

INSUFFICIENT_CONTEXT_MESSAGE = (
    "عذراً، لا توجد معلومات كافية "
    "في التوصيات المرجعية."
)


# =========================================================
# SYSTEM INSTRUCTIONS
# =========================================================

SYSTEM_INSTRUCTIONS = """
You are a grounded medical information assistant.

Your scope is STRICTLY limited to the retrieved content
from the following NICE guidelines:

- NG101
- CG81

These guidelines concern breast cancer diagnosis,
treatment and management within their documented scope.

STRICT RULES:

1. Use ONLY the retrieved context supplied to you.

2. Do NOT use outside medical knowledge.

3. Do NOT use internet information.

4. Do NOT use model memory to add medical facts.

5. Do NOT invent information.

6. Do NOT guess.

7. Do NOT infer unsupported treatment recommendations.

8. Every factual medical claim must be supported by
   retrieved evidence.

9. Cite factual claims using exactly:

   (NG101, Page X)

   or

   (CG81, Page X)

10. X must correspond to a page present in the retrieved
    context.

11. If the retrieved context is insufficient to answer
    the question, return exactly:

    عذراً، لا توجد معلومات كافية في التوصيات المرجعية.

12. Do not answer questions outside the scope of NG101
    and CG81 using outside knowledge.

13. Keep answers concise and evidence-grounded.

14. Do not provide diagnosis or personalized medical
    treatment decisions.

15. End every successful answer exactly with:

This system does not replace healthcare professionals.
""".strip()