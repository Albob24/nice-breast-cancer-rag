"""
NICE Breast Cancer RAG
Guardrail Threshold Calibration

Purpose
-------
Calibrate the ChromaDB relevance threshold empirically using:

    1. In-scope / relevant questions
    2. Out-of-scope / irrelevant questions

The script measures the minimum Chroma distance among the
Top-K retrieved chunks for every query.

IMPORTANT
---------
Chroma cosine distance:
    lower distance = more similar
    higher distance = less similar

A query is considered RETRIEVABLE when:

    minimum_distance <= threshold

The script DOES NOT modify config.py automatically.

It produces:
    - JSON report
    - CSV report
    - threshold comparison
    - recommended threshold

The final threshold should be reviewed before being copied
into config.py.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import chromadb
from google import genai

from config import (
    GOOGLE_API_KEY,
    EMBEDDING_MODEL,
    EMBEDDING_DIMENSION,
    RELEVANCE_THRESHOLD,
    get_db_path,
    get_collection_name,
)


# =========================================================
# PROJECT PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RESULTS_DIR = (
    PROJECT_ROOT / "evaluation_results"
)

JSON_OUTPUT = (
    RESULTS_DIR / "guardrail_calibration.json"
)

CSV_OUTPUT = (
    RESULTS_DIR / "guardrail_calibration.csv"
)


# =========================================================
# EXPERIMENT
# =========================================================

# Experiment A is currently the retrieval winner
# according to the previous benchmark.

EXPERIMENT = "A"

DB_PATH = get_db_path(
    EXPERIMENT
)

COLLECTION_NAME = get_collection_name(
    EXPERIMENT
)


# =========================================================
# RETRIEVAL
# =========================================================

TOP_K = 5


# =========================================================
# THRESHOLD SEARCH RANGE
# =========================================================

# We do not assume 0.75 is correct.
#
# These thresholds are evaluated against the actual
# distance distribution produced by the test set.

THRESHOLD_START = 0.30

THRESHOLD_END = 1.10

THRESHOLD_STEP = 0.02


# =========================================================
# TEST DATA
# =========================================================
#
# IMPORTANT:
#
# These are deliberately split into:
#
#   relevant
#   out_of_scope
#
# The calibration task is NOT answering these questions.
#
# It only measures whether the retrieval distance can
# separate the two classes.
#

RELEVANT_QUESTIONS = [
    (
        1,
        "What endocrine therapy is recommended for "
        "postmenopausal women with ER-positive invasive "
        "breast cancer?",
    ),
    (
        2,
        "What endocrine therapy options should be discussed "
        "with premenopausal or perimenopausal people with "
        "ER-positive invasive breast cancer?",
    ),
    (
        3,
        "What endocrine therapy is recommended for people "
        "with male reproductive organs who have ER-positive "
        "invasive breast cancer?",
    ),
    (
        4,
        "When should extended endocrine therapy with an "
        "aromatase inhibitor be offered after tamoxifen?",
    ),
    (
        5,
        "What is recommended for extended tamoxifen therapy "
        "beyond 5 years?",
    ),
    (
        6,
        "When should neoadjuvant endocrine therapy be "
        "considered?",
    ),
    (
        7,
        "What does NICE recommend for premenopausal women "
        "regarding neoadjuvant chemotherapy versus "
        "neoadjuvant endocrine therapy?",
    ),
    (
        8,
        "What are the recommendations for HER2-positive "
        "breast cancer?",
    ),
    (
        9,
        "What treatment is recommended for triple-negative "
        "breast cancer?",
    ),
    (
        10,
        "What is recommended for HER2-negative high-risk "
        "early breast cancer with germline BRCA1 or BRCA2 "
        "mutations?",
    ),
    (
        11,
        "What is recommended for adjuvant bisphosphonate "
        "therapy in postmenopausal women with node-positive "
        "invasive breast cancer?",
    ),
    (
        12,
        "When should bisphosphonates be considered for "
        "node-negative invasive breast cancer?",
    ),
    (
        13,
        "What information should be discussed regarding "
        "the risks of bisphosphonate treatment?",
    ),
    (
        14,
        "What are the recommendations for adjuvant endocrine "
        "therapy for ER-positive ductal carcinoma in situ?",
    ),
    (
        15,
        "What are the recommendations regarding ovarian "
        "function suppression with endocrine therapy?",
    ),
]


OUT_OF_SCOPE_QUESTIONS = [
    (
        101,
        "What treatment does NICE recommend for pancreatic "
        "cancer?",
    ),
    (
        102,
        "What are the symptoms and treatment options for "
        "lung cancer?",
    ),
    (
        103,
        "What is the recommended treatment for diabetes?",
    ),
    (
        104,
        "What antibiotics are recommended for pneumonia?",
    ),
    (
        105,
        "What is the treatment for hypertension?",
    ),
    (
        106,
        "How is Alzheimer's disease treated?",
    ),
    (
        107,
        "What treatment is recommended for prostate cancer?",
    ),
    (
        108,
        "What is the treatment for a broken leg?",
    ),
    (
        109,
        "What medication should be used for migraine?",
    ),
    (
        110,
        "What is the recommended treatment for appendicitis?",
    ),
]


# =========================================================
# GOOGLE GENAI CLIENT
# =========================================================

genai_client = genai.Client(
    api_key=GOOGLE_API_KEY
)


# =========================================================
# EMBEDDING
# =========================================================

def embed_query(
    question: str,
) -> list[float]:
    """
    Generate a Gemini embedding for one query.

    The output dimensionality MUST match the vector
    database dimensionality.
    """

    response = genai_client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=question,
        config={
            "output_dimensionality": (
                EMBEDDING_DIMENSION
            ),
        },
    )

    if not response.embeddings:
        raise RuntimeError(
            "Google GenAI returned no embeddings."
        )

    embedding = response.embeddings[0].values

    if not embedding:
        raise RuntimeError(
            "Google GenAI returned an empty embedding."
        )

    if len(embedding) != EMBEDDING_DIMENSION:
        raise RuntimeError(
            "Embedding dimension mismatch: "
            f"expected {EMBEDDING_DIMENSION}, "
            f"got {len(embedding)}."
        )

    return list(embedding)


# =========================================================
# CHROMA RETRIEVAL
# =========================================================

def retrieve_top_k(
    collection: Any,
    question: str,
) -> dict[str, Any]:
    """
    Retrieve Top-K chunks and return their distances.

    The most important value for calibration is:

        minimum_distance

    because this represents the closest retrieved evidence.
    """

    query_embedding = embed_query(
        question
    )

    result = collection.query(
        query_embeddings=[
            query_embedding
        ],
        n_results=TOP_K,
        include=[
            "documents",
            "metadatas",
            "distances",
        ],
    )

    documents = (
        result.get("documents", [[]])[0]
    )

    metadatas = (
        result.get("metadatas", [[]])[0]
    )

    distances = (
        result.get("distances", [[]])[0]
    )

    if not distances:
        raise RuntimeError(
            "ChromaDB returned no distances."
        )

    retrieved = []

    for index, distance in enumerate(
        distances
    ):

        metadata = {}

        if index < len(metadatas):
            metadata = (
                metadatas[index] or {}
            )

        document = ""

        if index < len(documents):
            document = (
                documents[index] or ""
            )

        retrieved.append(
            {
                "rank": index + 1,
                "distance": float(distance),
                "source": str(
                    metadata.get(
                        "source",
                        "",
                    )
                ),
                "page": str(
                    metadata.get(
                        "page",
                        metadata.get(
                            "section_start_page",
                            "",
                        ),
                    )
                ),
                "section": str(
                    metadata.get(
                        "subheader",
                        metadata.get(
                            "header",
                            "",
                        ),
                    )
                ),
                "document_preview": (
                    document[:250]
                    .replace("\n", " ")
                ),
            }
        )

    minimum_distance = min(
        item["distance"]
        for item in retrieved
    )

    return {
        "retrieved": retrieved,
        "minimum_distance": (
            minimum_distance
        ),
    }


# =========================================================
# COLLECT DISTANCES
# =========================================================

def collect_query_results(
    collection: Any,
) -> list[dict[str, Any]]:
    """
    Run every calibration query exactly once.

    We retrieve Top-5 for every question and store the
    resulting distances.

    This is important because threshold testing should NOT
    repeatedly call the embedding API.
    """

    all_results = []

    total = (
        len(RELEVANT_QUESTIONS)
        + len(OUT_OF_SCOPE_QUESTIONS)
    )

    current = 0

    # -----------------------------------------------------
    # Relevant
    # -----------------------------------------------------

    for question_id, question in (
        RELEVANT_QUESTIONS
    ):

        current += 1

        print()
        print(
            f"[{current}/{total}] "
            f"RELEVANT #{question_id}"
        )

        print(
            f"Question: {question}"
        )

        result = retrieve_top_k(
            collection,
            question,
        )

        minimum_distance = result[
            "minimum_distance"
        ]

        print(
            f"Minimum distance: "
            f"{minimum_distance:.4f}"
        )

        for item in result[
            "retrieved"
        ]:

            print(
                f"  #{item['rank']} "
                f"distance={item['distance']:.4f} "
                f"source={item['source']} "
                f"page={item['page']}"
            )

        all_results.append(
            {
                "id": question_id,
                "type": "relevant",
                "question": question,
                "minimum_distance": (
                    minimum_distance
                ),
                "top_k": result[
                    "retrieved"
                ],
            }
        )

    # -----------------------------------------------------
    # Out of scope
    # -----------------------------------------------------

    for question_id, question in (
        OUT_OF_SCOPE_QUESTIONS
    ):

        current += 1

        print()
        print(
            f"[{current}/{total}] "
            f"OUT-OF-SCOPE #{question_id}"
        )

        print(
            f"Question: {question}"
        )

        result = retrieve_top_k(
            collection,
            question,
        )

        minimum_distance = result[
            "minimum_distance"
        ]

        print(
            f"Minimum distance: "
            f"{minimum_distance:.4f}"
        )

        for item in result[
            "retrieved"
        ]:

            print(
                f"  #{item['rank']} "
                f"distance={item['distance']:.4f} "
                f"source={item['source']} "
                f"page={item['page']}"
            )

        all_results.append(
            {
                "id": question_id,
                "type": "out_of_scope",
                "question": question,
                "minimum_distance": (
                    minimum_distance
                ),
                "top_k": result[
                    "retrieved"
                ],
            }
        )

    return all_results


# =========================================================
# THRESHOLD GENERATOR
# =========================================================

def generate_thresholds() -> list[float]:
    """
    Generate candidate thresholds.

    Floating-point rounding is normalized to four decimals.
    """

    thresholds = []

    value = THRESHOLD_START

    while value <= (
        THRESHOLD_END + 1e-9
    ):

        thresholds.append(
            round(value, 4)
        )

        value += THRESHOLD_STEP

    return thresholds


# =========================================================
# CONFUSION MATRIX
# =========================================================

def confusion_matrix(
    results: list[dict[str, Any]],
    threshold: float,
) -> dict[str, int]:
    """
    Evaluate the threshold.

    Prediction:

        minimum_distance <= threshold
            -> RELEVANT

        minimum_distance > threshold
            -> REJECT

    Ground truth:

        relevant
        out_of_scope
    """

    tp = 0
    fp = 0
    tn = 0
    fn = 0

    for item in results:

        predicted_relevant = (
            item["minimum_distance"]
            <= threshold
        )

        actual_relevant = (
            item["type"]
            == "relevant"
        )

        if (
            actual_relevant
            and predicted_relevant
        ):

            tp += 1

        elif (
            not actual_relevant
            and predicted_relevant
        ):

            fp += 1

        elif (
            not actual_relevant
            and not predicted_relevant
        ):

            tn += 1

        elif (
            actual_relevant
            and not predicted_relevant
        ):

            fn += 1

    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


# =========================================================
# METRICS
# =========================================================

def calculate_metrics(
    matrix: dict[str, int],
) -> dict[str, float]:
    """
    Calculate standard binary classification metrics.
    """

    tp = matrix["tp"]

    fp = matrix["fp"]

    tn = matrix["tn"]

    fn = matrix["fn"]

    total = (
        tp + fp + tn + fn
    )

    accuracy = (
        (tp + tn) / total
        if total
        else 0.0
    )

    precision = (
        tp / (tp + fp)
        if (tp + fp)
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn)
        else 0.0
    )

    specificity = (
        tn / (tn + fp)
        if (tn + fp)
        else 0.0
    )

    f1 = (
        2 * precision * recall
        / (precision + recall)
        if (precision + recall)
        else 0.0
    )

    false_positive_rate = (
        fp / (fp + tn)
        if (fp + tn)
        else 0.0
    )

    false_negative_rate = (
        fn / (fn + tp)
        if (fn + tp)
        else 0.0
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "false_positive_rate": (
            false_positive_rate
        ),
        "false_negative_rate": (
            false_negative_rate
        ),
    }


# =========================================================
# THRESHOLD EVALUATION
# =========================================================

def evaluate_thresholds(
    results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Evaluate every candidate threshold.
    """

    evaluations = []

    for threshold in (
        generate_thresholds()
    ):

        matrix = confusion_matrix(
            results,
            threshold,
        )

        metrics = calculate_metrics(
            matrix
        )

        evaluation = {
            "threshold": threshold,
            **matrix,
            **metrics,
        }

        evaluations.append(
            evaluation
        )

    return evaluations


# =========================================================
# SELECT BEST THRESHOLD
# =========================================================

def choose_best_threshold(
    evaluations: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Select a practical threshold.

    Primary objective:
        maximize F1.

    Tie-breaking:
        1. Higher recall
        2. Higher specificity
        3. Lower threshold

    Why?

    The RAG must not reject too many legitimate medical
    guideline questions, while still rejecting unrelated
    questions.
    """

    ranked = sorted(
        evaluations,
        key=lambda item: (
            item["f1"],
            item["recall"],
            item["specificity"],
            -item["threshold"],
        ),
        reverse=True,
    )

    return ranked[0]


# =========================================================
# PRINT DISTRIBUTION
# =========================================================

def print_distance_distribution(
    results: list[dict[str, Any]],
) -> None:
    """
    Print the distance distribution separately for relevant
    and out-of-scope questions.
    """

    relevant = [
        item["minimum_distance"]
        for item in results
        if item["type"] == "relevant"
    ]

    out_of_scope = [
        item["minimum_distance"]
        for item in results
        if item["type"]
        == "out_of_scope"
    ]

    print()
    print("=" * 80)
    print("DISTANCE DISTRIBUTION")
    print("=" * 80)

    print()

    print("RELEVANT QUESTIONS")

    for distance in sorted(
        relevant
    ):

        print(
            f"  {distance:.4f}"
        )

    print()

    print("OUT-OF-SCOPE QUESTIONS")

    for distance in sorted(
        out_of_scope
    ):

        print(
            f"  {distance:.4f}"
        )

    print()

    if relevant:

        print(
            "Relevant minimum distance:"
        )

        print(
            f"  min = "
            f"{min(relevant):.4f}"
        )

        print(
            f"  max = "
            f"{max(relevant):.4f}"
        )

    if out_of_scope:

        print()

        print(
            "Out-of-scope minimum distance:"
        )

        print(
            f"  min = "
            f"{min(out_of_scope):.4f}"
        )

        print(
            f"  max = "
            f"{max(out_of_scope):.4f}"
        )


# =========================================================
# PRINT THRESHOLD TABLE
# =========================================================

def print_threshold_table(
    evaluations: list[dict[str, Any]],
) -> None:
    """
    Print threshold evaluation table.

    Only selected useful thresholds are displayed to avoid
    an unnecessarily huge terminal output.
    """

    print()
    print("=" * 100)
    print("THRESHOLD CALIBRATION")
    print("=" * 100)

    print()

    print(
        f"{'Threshold':>10}"
        f"{'TP':>6}"
        f"{'FP':>6}"
        f"{'TN':>6}"
        f"{'FN':>6}"
        f"{'Precision':>12}"
        f"{'Recall':>10}"
        f"{'Specificity':>13}"
        f"{'F1':>10}"
    )

    print("-" * 100)

    for item in evaluations:

        threshold = item[
            "threshold"
        ]

        # Print every 0.10 plus the current config
        # threshold.

        is_major_step = (
            abs(
                (
                    threshold * 100
                ) % 10
            ) < 1e-6
        )

        is_current = (
            abs(
                threshold
                - RELEVANCE_THRESHOLD
            )
            < 1e-6
        )

        if not (
            is_major_step
            or is_current
        ):
            continue

        print(
            f"{threshold:>10.2f}"
            f"{item['tp']:>6}"
            f"{item['fp']:>6}"
            f"{item['tn']:>6}"
            f"{item['fn']:>6}"
            f"{item['precision']:>11.2%}"
            f"{item['recall']:>9.2%}"
            f"{item['specificity']:>12.2%}"
            f"{item['f1']:>9.2%}"
        )


# =========================================================
# SAVE JSON
# =========================================================

def save_json(
    data: Any,
) -> None:

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        JSON_OUTPUT,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=2,
        )


# =========================================================
# SAVE CSV
# =========================================================

def save_csv(
    results: list[dict[str, Any]],
    evaluations: list[dict[str, Any]],
) -> None:

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -----------------------------------------------------
    # Query-level CSV
    # -----------------------------------------------------

    query_csv = (
        RESULTS_DIR
        / "guardrail_query_distances.csv"
    )

    with open(
        query_csv,
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:

        fieldnames = [
            "id",
            "type",
            "question",
            "minimum_distance",
        ]

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for item in results:

            writer.writerow(
                {
                    "id": item["id"],
                    "type": item["type"],
                    "question": item[
                        "question"
                    ],
                    "minimum_distance": (
                        item[
                            "minimum_distance"
                        ]
                    ),
                }
            )

    # -----------------------------------------------------
    # Threshold CSV
    # -----------------------------------------------------

    threshold_csv = (
        RESULTS_DIR
        / "guardrail_thresholds.csv"
    )

    with open(
        threshold_csv,
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:

        if not evaluations:
            return

        fieldnames = list(
            evaluations[0].keys()
        )

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(
            evaluations
        )


# =========================================================
# FINAL REPORT
# =========================================================

def print_final_report(
    best: dict[str, Any],
) -> None:

    print()
    print()
    print("=" * 80)
    print("GUARDRAIL CALIBRATION RESULT")
    print("=" * 80)

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
        f"Top-K                : "
        f"{TOP_K}"
    )

    print()

    print(
        f"Current threshold    : "
        f"{RELEVANCE_THRESHOLD:.4f}"
    )

    print(
        f"Recommended threshold: "
        f"{best['threshold']:.4f}"
    )

    print()

    print(
        f"TP                   : "
        f"{best['tp']}"
    )

    print(
        f"FP                   : "
        f"{best['fp']}"
    )

    print(
        f"TN                   : "
        f"{best['tn']}"
    )

    print(
        f"FN                   : "
        f"{best['fn']}"
    )

    print()

    print(
        f"Accuracy             : "
        f"{best['accuracy']:.2%}"
    )

    print(
        f"Precision            : "
        f"{best['precision']:.2%}"
    )

    print(
        f"Recall               : "
        f"{best['recall']:.2%}"
    )

    print(
        f"Specificity          : "
        f"{best['specificity']:.2%}"
    )

    print(
        f"F1                   : "
        f"{best['f1']:.2%}"
    )

    print(
        f"False positive rate  : "
        f"{best['false_positive_rate']:.2%}"
    )

    print(
        f"False negative rate  : "
        f"{best['false_negative_rate']:.2%}"
    )

    print()
    print("=" * 80)

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "The recommended threshold has NOT been written "
        "to config.py."
    )

    print(
        "Review the calibration results first."
    )

    print()
    print(
        f"JSON report : {JSON_OUTPUT}"
    )

    print(
        f"CSV report  : {CSV_OUTPUT}"
    )

    print("=" * 80)


# =========================================================
# MAIN
# =========================================================

def main() -> None:

    print()
    print("=" * 80)
    print("NICE BREAST CANCER RAG")
    print("GUARDRAIL THRESHOLD CALIBRATION")
    print("=" * 80)

    print()

    print(
        f"Experiment           : {EXPERIMENT}"
    )

    print(
        f"Database             : {DB_PATH}"
    )

    print(
        f"Collection           : {COLLECTION_NAME}"
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
        f"Top-K                : {TOP_K}"
    )

    print(
        f"Current threshold    : "
        f"{RELEVANCE_THRESHOLD}"
    )

    print()

    print(
        f"Relevant questions   : "
        f"{len(RELEVANT_QUESTIONS)}"
    )

    print(
        f"Out-of-scope        : "
        f"{len(OUT_OF_SCOPE_QUESTIONS)}"
    )

    print("=" * 80)

    # -----------------------------------------------------
    # Open Chroma
    # -----------------------------------------------------

    chroma_client = chromadb.PersistentClient(
        path=DB_PATH
    )

    try:

        collection = (
            chroma_client.get_collection(
                name=COLLECTION_NAME
            )
        )

    except Exception as exc:

        raise RuntimeError(
            "Could not open ChromaDB collection.\n"
            f"Database: {DB_PATH}\n"
            f"Collection: {COLLECTION_NAME}\n\n"
            f"Original error: {exc}"
        ) from exc

    print()

    print(
        f"Chroma vectors: "
        f"{collection.count()}"
    )

    # -----------------------------------------------------
    # Collect distances
    # -----------------------------------------------------

    results = collect_query_results(
        collection
    )

    # -----------------------------------------------------
    # Distribution
    # -----------------------------------------------------

    print_distance_distribution(
        results
    )

    # -----------------------------------------------------
    # Threshold evaluation
    # -----------------------------------------------------

    evaluations = evaluate_thresholds(
        results
    )

    print_threshold_table(
        evaluations
    )

    # -----------------------------------------------------
    # Best threshold
    # -----------------------------------------------------

    best = choose_best_threshold(
        evaluations
    )

    # -----------------------------------------------------
    # Save
    # -----------------------------------------------------

    report = {
        "experiment": EXPERIMENT,
        "database": DB_PATH,
        "collection": COLLECTION_NAME,
        "embedding_model": EMBEDDING_MODEL,
        "embedding_dimensions": (
            EMBEDDING_DIMENSION
        ),
        "top_k": TOP_K,
        "current_threshold": (
            RELEVANCE_THRESHOLD
        ),
        "recommended_threshold": (
            best["threshold"]
        ),
        "best_metrics": best,
        "query_results": results,
        "threshold_evaluations": evaluations,
    }

    save_json(
        report
    )

    save_csv(
        results,
        evaluations,
    )

    # -----------------------------------------------------
    # Final report
    # -----------------------------------------------------

    print_final_report(
        best
    )


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":
    main()