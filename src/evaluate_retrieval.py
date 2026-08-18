"""
NICE Breast Cancer RAG
Retrieval Evaluation

Purpose
-------
Compare Experiment A and Experiment B using:

    Top-5
    Top-10

Metrics
-------
    Recall@5
    Recall@10
    MRR
    Correct Evidence Rank
    Retrieval Distance

Experiments
-----------
A = 400-600 tokens
B = 700-900 tokens

The evaluation uses the same Gemini embedding model
used when the vector databases were built.

IMPORTANT
---------
Distance is NOT accuracy.

Lower ChromaDB distance means higher semantic similarity.
The actual retrieval evaluation is based on whether the
expected evidence appears in Top-K and where it appears.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


# =========================================================
# CONFIG
# =========================================================

try:
    from .config import (
        BASE_DIR,
        GOOGLE_API_KEY,
        EMBEDDING_MODEL,
        EMBEDDING_DIMENSION,
        RELEVANCE_THRESHOLD,
        get_db_path,
        get_collection_name,
    )

except ImportError:
    from config import (
        BASE_DIR,
        GOOGLE_API_KEY,
        EMBEDDING_MODEL,
        EMBEDDING_DIMENSION,
        RELEVANCE_THRESHOLD,
        get_db_path,
        get_collection_name,
    )


# =========================================================
# THIRD-PARTY LIBRARIES
# =========================================================

import chromadb

from google import genai


# =========================================================
# EVALUATION SETTINGS
# =========================================================

# Top-K values required by the experiment.
TOP_K_VALUES = (5, 10)

# Experiments to compare.
EXPERIMENTS = ("A", "B")

# Output report.
REPORT_FILE = (
    BASE_DIR /
    "retrieval_evaluation_report.json"
)

# Small delay between embedding calls.
# Helps reduce API burst/rate-limit problems.
EMBED_DELAY_SECONDS = 1.0


# =========================================================
# GOOGLE CLIENT
# =========================================================

client = genai.Client(
    api_key=GOOGLE_API_KEY
)


# =========================================================
# TEST QUESTIONS
# =========================================================
#
# These are retrieval questions covering different areas
# of the NICE breast-cancer guideline corpus.
#
# expected_source:
#     document expected to contain the evidence.
#
# expected_terms:
#     terms used to identify the relevant section.
#
# The evaluator does NOT require an exact page number.
# It checks the retrieved chunk's source, section metadata,
# and text.
#
# This is safer than pretending that a similarity distance
# itself is ground truth.
# =========================================================

QUESTIONS: list[dict[str, Any]] = [

    {
        "id": "Q01",
        "question": (
            "What endocrine therapy is recommended for "
            "postmenopausal women with ER-positive invasive "
            "breast cancer?"
        ),
        "expected_source": "NG101",
        "expected_terms": [
            "endocrine",
            "ER-positive",
            "postmenopausal",
        ],
    },

    {
        "id": "Q02",
        "question": (
            "What is recommended for people with male "
            "reproductive organs who have ER-positive "
            "invasive breast cancer?"
        ),
        "expected_source": "NG101",
        "expected_terms": [
            "endocrine",
            "ER-positive",
            "male",
        ],
    },

    {
        "id": "Q03",
        "question": (
            "What is neoadjuvant endocrine therapy used for "
            "in postmenopausal women with ER-positive "
            "invasive breast cancer?"
        ),
        "expected_source": "NG101",
        "expected_terms": [
            "neoadjuvant",
            "endocrine",
            "postmenopausal",
        ],
    },

    {
        "id": "Q04",
        "question": (
            "What treatment options are described for "
            "hormone receptor positive HER2-negative "
            "breast cancer?"
        ),
        "expected_source": "NG101",
        "expected_terms": [
            "hormone receptor positive",
            "HER2-negative",
        ],
    },

    {
        "id": "Q05",
        "question": (
            "What recommendations are given for "
            "HER2-positive breast cancer?"
        ),
        "expected_source": "NG101",
        "expected_terms": [
            "HER2-positive",
        ],
    },

    {
        "id": "Q06",
        "question": (
            "What recommendations are given for "
            "triple-negative breast cancer?"
        ),
        "expected_source": "NG101",
        "expected_terms": [
            "triple-negative",
        ],
    },

    {
        "id": "Q07",
        "question": (
            "What is recommended regarding adjuvant endocrine "
            "therapy after menopause?"
        ),
        "expected_source": "NG101",
        "expected_terms": [
            "adjuvant",
            "endocrine",
            "menopause",
        ],
    },

    {
        "id": "Q08",
        "question": (
            "What are the recommendations concerning "
            "extended endocrine therapy?"
        ),
        "expected_source": "NG101",
        "expected_terms": [
            "extended",
            "endocrine",
        ],
    },

    {
        "id": "Q09",
        "question": (
            "What are the considerations for extended "
            "tamoxifen therapy?"
        ),
        "expected_source": "NG101",
        "expected_terms": [
            "tamoxifen",
            "extended",
        ],
    },

    {
        "id": "Q10",
        "question": (
            "What are the considerations for extended "
            "aromatase inhibitor therapy?"
        ),
        "expected_source": "NG101",
        "expected_terms": [
            "aromatase inhibitor",
            "extended",
        ],
    },

    {
        "id": "Q11",
        "question": (
            "What are the benefits and risks of endocrine "
            "therapy described in the guideline?"
        ),
        "expected_source": "NG101",
        "expected_terms": [
            "endocrine",
            "benefits",
            "risks",
        ],
    },

    {
        "id": "Q12",
        "question": (
            "What side effects are associated with "
            "extended endocrine therapy?"
        ),
        "expected_source": "NG101",
        "expected_terms": [
            "side effects",
            "endocrine",
        ],
    },

    {
        "id": "Q13",
        "question": (
            "What does the guideline say about bone density "
            "during endocrine therapy?"
        ),
        "expected_source": "NG101",
        "expected_terms": [
            "bone",
            "endocrine",
        ],
    },

    {
        "id": "Q14",
        "question": (
            "What does the guideline say about fertility "
            "and family planning during endocrine therapy?"
        ),
        "expected_source": "NG101",
        "expected_terms": [
            "fertility",
            "family planning",
        ],
    },

    {
        "id": "Q15",
        "question": (
            "What information and psychological support "
            "should be provided to people with breast cancer?"
        ),
        "expected_source": "NG101",
        "expected_terms": [
            "information",
            "psychological support",
        ],
    },

    {
        "id": "Q16",
        "question": (
            "What is recommended for adjuvant treatment of "
            "HER2-negative high-risk early breast cancer "
            "with germline BRCA mutations?"
        ),
        "expected_source": "NG101",
        "expected_terms": [
            "BRCA",
            "HER2-negative",
            "adjuvant",
        ],
    },

    {
        "id": "Q17",
        "question": (
            "What is recommended for people with ER-positive "
            "invasive breast cancer when tamoxifen is not "
            "suitable or tolerated?"
        ),
        "expected_source": "NG101",
        "expected_terms": [
            "tamoxifen",
            "ER-positive",
        ],
    },

    {
        "id": "Q18",
        "question": (
            "What does the guideline say about aromatase "
            "inhibitors for people with male reproductive organs?"
        ),
        "expected_source": "NG101",
        "expected_terms": [
            "aromatase inhibitor",
            "male",
        ],
    },

    {
        "id": "Q19",
        "question": (
            "What is recommended regarding testicular function "
            "suppression and aromatase inhibitors?"
        ),
        "expected_source": "NG101",
        "expected_terms": [
            "testicular",
            "aromatase inhibitor",
        ],
    },

    {
        "id": "Q20",
        "question": (
            "What recommendations concern ER-positive "
            "ductal carcinoma in situ and endocrine therapy?"
        ),
        "expected_source": "NG101",
        "expected_terms": [
            "ductal carcinoma in situ",
            "endocrine",
        ],
    },
]


# =========================================================
# TEXT NORMALIZATION
# =========================================================

def normalize_text(
    text: Any,
) -> str:
    """
    Normalize metadata/text for matching.
    """

    if text is None:
        return ""

    return (
        str(text)
        .lower()
        .replace("\n", " ")
        .replace("\r", " ")
        .strip()
    )


# =========================================================
# EMBEDDING
# =========================================================

def embed_query(
    question: str,
) -> list[float]:
    """
    Embed one question.

    The embedding is generated only once per question
    and reused for A/B and Top-5/Top-10 evaluation.
    """

    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=question,
        config={
            "output_dimensionality":
                EMBEDDING_DIMENSION
        },
    )

    if not response.embeddings:
        raise RuntimeError(
            "Gemini returned no embedding."
        )

    values = response.embeddings[
        0
    ].values

    if not values:
        raise RuntimeError(
            "Gemini returned an empty embedding."
        )

    if len(values) != EMBEDDING_DIMENSION:
        raise RuntimeError(
            "Embedding dimension mismatch: "
            f"expected {EMBEDDING_DIMENSION}, "
            f"received {len(values)}"
        )

    return list(values)


# =========================================================
# COLLECTION LOADING
# =========================================================

def load_collection(
    experiment: str,
):
    """
    Open the ChromaDB collection for A or B.
    """

    experiment = (
        experiment
        .strip()
        .upper()
    )

    db_path = get_db_path(
        experiment
    )

    collection_name = (
        get_collection_name(
            experiment
        )
    )

    db_path_obj = Path(
        db_path
    )

    if not db_path_obj.exists():
        raise FileNotFoundError(
            f"Database does not exist:\n"
            f"{db_path}"
        )

    chroma_client = (
        chromadb.PersistentClient(
            path=db_path
        )
    )

    try:

        collection = (
            chroma_client.get_collection(
                name=collection_name
            )
        )

    except Exception as exc:

        raise RuntimeError(
            f"Could not open collection:\n"
            f"{collection_name}\n"
            f"Database: {db_path}\n"
            f"Error: {exc}"
        ) from exc

    count = collection.count()

    if count == 0:

        raise RuntimeError(
            f"Collection is empty:\n"
            f"{collection_name}"
        )

    return collection


# =========================================================
# METADATA EXTRACTION
# =========================================================

def get_chunk_metadata(
    metadata: dict[str, Any] | None,
) -> dict[str, str]:
    """
    Normalize metadata from the different chunking
    versions.
    """

    metadata = (
        metadata or {}
    )

    source = (
        metadata.get("source")
        or metadata.get("document")
        or ""
    )

    page = (
        metadata.get("page")
        or metadata.get("page_number")
        or metadata.get("start_page")
        or metadata.get("section_start_page")
        or ""
    )

    section = (
        metadata.get("subheader")
        or metadata.get("section")
        or metadata.get("title")
        or metadata.get("header")
        or ""
    )

    header = (
        metadata.get("header")
        or ""
    )

    number = (
        metadata.get("number")
        or ""
    )

    return {
        "source":
            str(source),

        "page":
            str(page),

        "section":
            str(section),

        "header":
            str(header),

        "number":
            str(number),
    }


# =========================================================
# CHUNK RELEVANCE
# =========================================================

def is_expected_evidence(
    result: dict[str, Any],
    question: dict[str, Any],
) -> bool:
    """
    Determine whether a retrieved chunk is relevant
    according to the evaluation ground truth.

    A chunk is considered expected evidence when:

    1. It belongs to the expected document.
    2. At least one expected semantic term appears in
       the chunk metadata or document text.

    This is a transparent heuristic, not an LLM judge.
    """

    expected_source = normalize_text(
        question.get(
            "expected_source",
            "",
        )
    )

    expected_terms = [
        normalize_text(term)
        for term in question.get(
            "expected_terms",
            [],
        )
        if normalize_text(term)
    ]

    source = normalize_text(
        result.get("source")
    )

    if (
        expected_source
        and source != expected_source
    ):
        return False

    searchable = " ".join(
        [
            normalize_text(
                result.get("text")
            ),
            normalize_text(
                result.get("section")
            ),
            normalize_text(
                result.get("header")
            ),
        ]
    )

    if not expected_terms:
        return True

    return any(
        term in searchable
        for term in expected_terms
    )


# =========================================================
# RETRIEVE
# =========================================================

def retrieve(
    collection,
    query_embedding: list[float],
    k: int,
) -> list[dict[str, Any]]:
    """
    Retrieve Top-K chunks.
    """

    count = collection.count()

    actual_k = min(
        k,
        count,
    )

    results = collection.query(
        query_embeddings=[
            query_embedding
        ],
        n_results=actual_k,
        include=[
            "documents",
            "metadatas",
            "distances",
        ],
    )

    documents = (
        results.get(
            "documents",
            [[]],
        )[0]
    )

    metadatas = (
        results.get(
            "metadatas",
            [[]],
        )[0]
    )

    distances = (
        results.get(
            "distances",
            [[]],
        )[0]
    )

    retrieved = []

    for index, distance in enumerate(
        distances
    ):

        document = ""

        if index < len(documents):
            document = (
                documents[index]
                or ""
            )

        metadata = {}

        if index < len(metadatas):
            metadata = (
                metadatas[index]
                or {}
            )

        normalized = (
            get_chunk_metadata(
                metadata
            )
        )

        retrieved.append(
            {
                "rank":
                    index + 1,

                "distance":
                    float(distance),

                "source":
                    normalized[
                        "source"
                    ],

                "page":
                    normalized[
                        "page"
                    ],

                "section":
                    normalized[
                        "section"
                    ],

                "header":
                    normalized[
                        "header"
                    ],

                "number":
                    normalized[
                        "number"
                    ],

                "text":
                    document,

                "metadata":
                    metadata,
            }
        )

    return retrieved


# =========================================================
# QUESTION EVALUATION
# =========================================================

def evaluate_question(
    question: dict[str, Any],
    collection,
    query_embedding: list[float],
) -> dict[str, Any]:
    """
    Evaluate one question for Top-5 and Top-10.
    """

    evaluation = {
        "question_id":
            question["id"],

        "question":
            question["question"],

        "expected_source":
            question.get(
                "expected_source"
            ),

        "expected_terms":
            question.get(
                "expected_terms",
                [],
            ),

        "top_k": {},
    }

    for k in TOP_K_VALUES:

        retrieved = retrieve(
            collection,
            query_embedding,
            k,
        )

        matching_results = []

        for result in retrieved:

            if is_expected_evidence(
                result,
                question,
            ):

                matching_results.append(
                    result
                )

        # -------------------------------------------------
        # First relevant rank
        # -------------------------------------------------

        first_rank = None

        if matching_results:

            first_rank = (
                matching_results[0]
                ["rank"]
            )

        # -------------------------------------------------
        # Recall@K
        # -------------------------------------------------

        recall = (
            1.0
            if matching_results
            else 0.0
        )

        # -------------------------------------------------
        # Reciprocal Rank
        # -------------------------------------------------

        reciprocal_rank = (
            1.0 / first_rank
            if first_rank is not None
            else 0.0
        )

        evaluation[
            "top_k"
        ][str(k)] = {

            "recall":
                recall,

            "first_relevant_rank":
                first_rank,

            "reciprocal_rank":
                reciprocal_rank,

            "minimum_distance":
                (
                    min(
                        r["distance"]
                        for r in retrieved
                    )
                    if retrieved
                    else None
                ),

            "matching_results":
                matching_results,

            "retrieved":
                retrieved,
        }

    return evaluation


# =========================================================
# METRICS
# =========================================================

def calculate_metrics(
    evaluations: list[
        dict[str, Any]
    ],
    k: int,
) -> dict[str, float]:
    """
    Calculate aggregate Recall@K and MRR.
    """

    if not evaluations:

        return {
            "recall_at_k": 0.0,
            "mrr": 0.0,
            "questions": 0,
        }

    recalls = []
    reciprocal_ranks = []

    for item in evaluations:

        data = item[
            "top_k"
        ][
            str(k)
        ]

        recalls.append(
            float(
                data["recall"]
            )
        )

        reciprocal_ranks.append(
            float(
                data[
                    "reciprocal_rank"
                ]
            )
        )

    return {
        "recall_at_k":
            sum(recalls)
            / len(recalls),

        "mrr":
            sum(reciprocal_ranks)
            / len(
                reciprocal_ranks
            ),

        "questions":
            len(evaluations),
    }


# =========================================================
# RUN EXPERIMENT
# =========================================================

def run_experiment(
    experiment: str,
    query_embeddings: dict[
        str,
        list[float],
    ],
) -> dict[str, Any]:
    """
    Evaluate one experiment.
    """

    print()
    print("=" * 70)
    print(
        f"EXPERIMENT {experiment}"
    )
    print("=" * 70)

    collection = load_collection(
        experiment
    )

    print(
        f"Database   : "
        f"{get_db_path(experiment)}"
    )

    print(
        f"Collection : "
        f"{get_collection_name(experiment)}"
    )

    print(
        f"Vectors    : "
        f"{collection.count()}"
    )

    evaluations = []

    for index, question in enumerate(
        QUESTIONS,
        start=1,
    ):

        print(
            f"\n[{index:02d}/{len(QUESTIONS)}] "
            f"{question['id']} "
            f"{question['question']}"
        )

        embedding = query_embeddings[
            question["id"]
        ]

        result = evaluate_question(
            question,
            collection,
            embedding,
        )

        evaluations.append(
            result
        )

        for k in TOP_K_VALUES:

            data = result[
                "top_k"
            ][
                str(k)
            ]

            rank = (
                data[
                    "first_relevant_rank"
                ]
            )

            distance = (
                data[
                    "minimum_distance"
                ]
            )

            if rank is None:

                print(
                    f"  Top-{k}: "
                    f"MISS"
                )

            else:

                print(
                    f"  Top-{k}: "
                    f"HIT "
                    f"(rank={rank}, "
                    f"distance="
                    f"{distance:.4f})"
                )

    metrics = {}

    for k in TOP_K_VALUES:

        metrics[str(k)] = (
            calculate_metrics(
                evaluations,
                k,
            )
        )

    return {
        "experiment":
            experiment,

        "database":
            get_db_path(
                experiment
            ),

        "collection":
            get_collection_name(
                experiment
            ),

        "vector_count":
            collection.count(),

        "evaluations":
            evaluations,

        "metrics":
            metrics,
    }


# =========================================================
# PRINT SUMMARY
# =========================================================

def print_summary(
    results: dict[str, Any],
) -> None:

    print()
    print("=" * 70)
    print(
        "FINAL RETRIEVAL COMPARISON"
    )
    print("=" * 70)

    for experiment in EXPERIMENTS:

        data = results[
            experiment
        ]

        print()
        print(
            f"EXPERIMENT {experiment}"
        )

        if experiment == "A":

            print(
                "Chunk size: "
                "400-600 tokens"
            )

        else:

            print(
                "Chunk size: "
                "700-900 tokens"
            )

        print(
            f"Vectors: "
            f"{data['vector_count']}"
        )

        for k in TOP_K_VALUES:

            metrics = data[
                "metrics"
            ][
                str(k)
            ]

            print()

            print(
                f"Top-{k}"
            )

            print(
                f"  Recall@{k} : "
                f"{metrics['recall_at_k']:.4f}"
            )

            print(
                f"  MRR        : "
                f"{metrics['mrr']:.4f}"
            )

    # -----------------------------------------------------
    # Determine winner
    # -----------------------------------------------------

    a = results["A"]["metrics"]
    b = results["B"]["metrics"]

    a_score = (
        a["5"]["recall_at_k"]
        + a["10"]["recall_at_k"]
        + a["5"]["mrr"]
        + a["10"]["mrr"]
    )

    b_score = (
        b["5"]["recall_at_k"]
        + b["10"]["recall_at_k"]
        + b["5"]["mrr"]
        + b["10"]["mrr"]
    )

    print()
    print("=" * 70)
    print(
        "DECISION"
    )
    print("=" * 70)

    print(
        f"Experiment A score : "
        f"{a_score:.4f}"
    )

    print(
        f"Experiment B score : "
        f"{b_score:.4f}"
    )

    if a_score > b_score:

        print(
            "\nRecommended experiment: A"
        )

    elif b_score > a_score:

        print(
            "\nRecommended experiment: B"
        )

    else:

        print(
            "\nResult: TIE"
        )

    print()
    print(
        "Decision rule:"
    )

    print(
        "Choose the experiment that places "
        "the correct evidence highest across "
        "the test questions."
    )

    print("=" * 70)


# =========================================================
# SAVE REPORT
# =========================================================

def save_report(
    report: dict[str, Any],
) -> None:

    with open(
        REPORT_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            report,
            file,
            ensure_ascii=False,
            indent=2,
        )


# =========================================================
# MAIN
# =========================================================

def main() -> None:

    print()
    print("=" * 70)
    print(
        "NICE BREAST CANCER RAG"
    )
    print(
        "RETRIEVAL EVALUATION"
    )
    print("=" * 70)

    print()
    print(
        f"Embedding model : "
        f"{EMBEDDING_MODEL}"
    )

    print(
        f"Dimensions      : "
        f"{EMBEDDING_DIMENSION}"
    )

    print(
        f"Top-K values    : "
        f"{TOP_K_VALUES}"
    )

    print(
        f"Guardrail       : "
        f"{RELEVANCE_THRESHOLD}"
    )

    print(
        f"Questions       : "
        f"{len(QUESTIONS)}"
    )

    print()

    # -----------------------------------------------------
    # Validate databases before making API calls
    # -----------------------------------------------------

    for experiment in EXPERIMENTS:

        collection = load_collection(
            experiment
        )

        print(
            f"Experiment {experiment}: "
            f"{collection.count()} vectors OK"
        )

    # -----------------------------------------------------
    # Embed each question ONCE
    # -----------------------------------------------------

    print()
    print("=" * 70)
    print(
        "EMBEDDING TEST QUESTIONS"
    )
    print("=" * 70)

    query_embeddings = {}

    for index, question in enumerate(
        QUESTIONS,
        start=1,
    ):

        print(
            f"Embedding "
            f"{index}/{len(QUESTIONS)}: "
            f"{question['id']}"
        )

        query_embeddings[
            question["id"]
        ] = embed_query(
            question["question"]
        )

        if (
            index
            < len(QUESTIONS)
        ):

            time.sleep(
                EMBED_DELAY_SECONDS
            )

    # -----------------------------------------------------
    # Evaluate A
    # -----------------------------------------------------

    result_a = run_experiment(
        "A",
        query_embeddings,
    )

    # -----------------------------------------------------
    # Evaluate B
    # -----------------------------------------------------

    result_b = run_experiment(
        "B",
        query_embeddings,
    )

    # -----------------------------------------------------
    # Build report
    # -----------------------------------------------------

    report = {

        "evaluation": {
            "embedding_model":
                EMBEDDING_MODEL,

            "embedding_dimension":
                EMBEDDING_DIMENSION,

            "top_k_values":
                list(
                    TOP_K_VALUES
                ),

            "relevance_threshold":
                RELEVANCE_THRESHOLD,

            "question_count":
                len(QUESTIONS),
        },

        "experiments": {

            "A":
                result_a,

            "B":
                result_b,
        },
    }

    # -----------------------------------------------------
    # Save
    # -----------------------------------------------------

    save_report(
        report
    )

    # -----------------------------------------------------
    # Display
    # -----------------------------------------------------

    print_summary(
        {
            "A":
                result_a,

            "B":
                result_b,
        }
    )

    print()
    print(
        f"Report saved to:"
    )

    print(
        REPORT_FILE
    )


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":

    main()