"""
NICE Breast Cancer RAG
Ground Truth Candidate Finder

Purpose
-------
Search the real chunk metadata from Experiment A and B
and produce candidate evidence for manual verification.

IMPORTANT
---------
This script does NOT automatically create final ground truth.

The final ground_truth.json must contain manually verified
evidence only.

Windows PowerShell UTF-8 output is explicitly configured
to prevent UnicodeEncodeError when guideline text contains
special characters.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any


# =========================================================
# WINDOWS UTF-8 OUTPUT
# =========================================================

def configure_utf8_output() -> None:
    """
    Force UTF-8 output on Windows/Python 3.13+.

    This prevents errors such as:

        UnicodeEncodeError: 'charmap' codec can't encode ...

    when NICE guideline text contains Unicode characters.
    """

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(
            encoding="utf-8",
            errors="replace",
        )

    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(
            encoding="utf-8",
            errors="replace",
        )


configure_utf8_output()


# =========================================================
# PROJECT PATHS
# =========================================================

BASE_DIR = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

CHUNKS_A = (
    BASE_DIR /
    "chunks_metadata_A.json"
)

CHUNKS_B = (
    BASE_DIR /
    "chunks_metadata_B.json"
)

GROUND_TRUTH = (
    BASE_DIR /
    "ground_truth.json"
)


# =========================================================
# TEST QUESTIONS
# =========================================================

QUESTIONS: dict[str, dict[str, Any]] = {

    "Q01": {
        "question": (
            "What endocrine therapy is recommended for "
            "postmenopausal women with ER-positive invasive "
            "breast cancer?"
        ),
        "terms": [
            "endocrine",
            "postmenopausal",
            "ER-positive",
            "aromatase inhibitor",
            "tamoxifen",
        ],
    },

    "Q02": {
        "question": (
            "What is recommended for people with male "
            "reproductive organs who have ER-positive "
            "invasive breast cancer?"
        ),
        "terms": [
            "male reproductive organs",
            "ER-positive",
            "tamoxifen",
            "endocrine",
        ],
    },

    "Q03": {
        "question": (
            "What is neoadjuvant endocrine therapy used for "
            "in postmenopausal women with ER-positive "
            "invasive breast cancer?"
        ),
        "terms": [
            "neoadjuvant endocrine therapy",
            "postmenopausal",
            "ER-positive",
        ],
    },

    "Q04": {
        "question": (
            "What treatment options are described for "
            "hormone receptor positive HER2-negative "
            "breast cancer?"
        ),
        "terms": [
            "hormone receptor positive",
            "HER2-negative",
        ],
    },

    "Q05": {
        "question": (
            "What recommendations are given for "
            "HER2-positive breast cancer?"
        ),
        "terms": [
            "HER2-positive",
        ],
    },

    "Q06": {
        "question": (
            "What recommendations are given for "
            "triple-negative breast cancer?"
        ),
        "terms": [
            "triple-negative",
        ],
    },

    "Q07": {
        "question": (
            "What is recommended regarding adjuvant endocrine "
            "therapy after menopause?"
        ),
        "terms": [
            "adjuvant endocrine therapy",
            "postmenopausal",
            "menopause",
        ],
    },

    "Q08": {
        "question": (
            "What are the recommendations concerning "
            "extended endocrine therapy?"
        ),
        "terms": [
            "extended endocrine therapy",
            "endocrine therapy",
        ],
    },

    "Q09": {
        "question": (
            "What are the considerations for extended "
            "tamoxifen therapy?"
        ),
        "terms": [
            "extended tamoxifen",
            "tamoxifen",
        ],
    },

    "Q10": {
        "question": (
            "What are the considerations for extended "
            "aromatase inhibitor therapy?"
        ),
        "terms": [
            "extended",
            "aromatase inhibitor",
        ],
    },

    "Q11": {
        "question": (
            "What are the benefits and risks of endocrine "
            "therapy described in the guideline?"
        ),
        "terms": [
            "benefits",
            "risks",
            "endocrine therapy",
        ],
    },

    "Q12": {
        "question": (
            "What side effects are associated with extended "
            "endocrine therapy?"
        ),
        "terms": [
            "side effects",
            "extended",
            "endocrine therapy",
        ],
    },

    "Q13": {
        "question": (
            "What does the guideline say about bone density "
            "during endocrine therapy?"
        ),
        "terms": [
            "bone density",
            "endocrine",
        ],
    },

    "Q14": {
        "question": (
            "What does the guideline say about fertility "
            "and family planning during endocrine therapy?"
        ),
        "terms": [
            "fertility",
            "family planning",
            "endocrine",
        ],
    },

    "Q15": {
        "question": (
            "What information and psychological support "
            "should be provided to people with breast cancer?"
        ),
        "terms": [
            "information",
            "psychological support",
        ],
    },

    "Q16": {
        "question": (
            "What is recommended for adjuvant treatment of "
            "HER2-negative high-risk early breast cancer "
            "with germline BRCA mutations?"
        ),
        "terms": [
            "BRCA",
            "HER2-negative",
            "adjuvant",
        ],
    },

    "Q17": {
        "question": (
            "What is recommended for people with ER-positive "
            "invasive breast cancer when tamoxifen is not "
            "suitable or tolerated?"
        ),
        "terms": [
            "tamoxifen",
            "not tolerated",
            "ER-positive",
        ],
    },

    "Q18": {
        "question": (
            "What does the guideline say about aromatase "
            "inhibitors for people with male reproductive organs?"
        ),
        "terms": [
            "aromatase inhibitor",
            "male reproductive organs",
        ],
    },

    "Q19": {
        "question": (
            "What is recommended regarding testicular function "
            "suppression and aromatase inhibitors?"
        ),
        "terms": [
            "testicular function suppression",
            "aromatase inhibitor",
        ],
    },

    "Q20": {
        "question": (
            "What recommendations concern ER-positive "
            "ductal carcinoma in situ and endocrine therapy?"
        ),
        "terms": [
            "ductal carcinoma in situ",
            "ER-positive",
            "endocrine therapy",
        ],
    },
}


# =========================================================
# LOAD JSON
# =========================================================

def load_chunks(
    path: Path,
) -> list[dict[str, Any]]:
    """Load chunk metadata from JSON."""

    if not path.exists():
        raise FileNotFoundError(
            f"Chunk metadata file not found:\n{path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError(
            f"Expected a JSON list in:\n{path}"
        )

    return data


# =========================================================
# NORMALIZATION
# =========================================================

def normalize(
    text: Any,
) -> str:
    """
    Normalize text for case-insensitive matching.
    """

    if text is None:
        return ""

    return re.sub(
        r"\s+",
        " ",
        str(text).lower(),
    ).strip()


# =========================================================
# SAFE INTEGER
# =========================================================

def safe_page(
    value: Any,
) -> int:
    """
    Convert page value to an integer for sorting.

    Missing/invalid pages are pushed to the end.
    """

    try:
        return int(value)
    except (
        TypeError,
        ValueError,
    ):
        return 999999


# =========================================================
# SCORE CHUNK
# =========================================================

def score_chunk(
    chunk: dict[str, Any],
    terms: list[str],
) -> int:
    """
    Score a chunk based on occurrence of question terms.

    This is ONLY a candidate-ranking heuristic.
    It is NOT the final retrieval metric.
    """

    text = normalize(
        chunk.get("text", "")
    )

    metadata = chunk.get(
        "metadata",
        {},
    )

    if not isinstance(
        metadata,
        dict,
    ):
        metadata = {}

    searchable = " ".join(
        [
            text,
            normalize(
                metadata.get(
                    "header",
                    "",
                )
            ),
            normalize(
                metadata.get(
                    "subheader",
                    "",
                )
            ),
        ]
    )

    score = 0

    for term in terms:

        normalized_term = normalize(term)

        if (
            normalized_term
            and normalized_term in searchable
        ):
            score += 1

    return score


# =========================================================
# FIND CANDIDATES
# =========================================================

def find_candidates(
    chunks: list[dict[str, Any]],
    question: dict[str, Any],
    limit: int = 5,
) -> list[dict[str, Any]]:
    """
    Find top candidate chunks using lexical evidence.

    Results are candidates only and require manual verification.
    """

    scored: list[dict[str, Any]] = []

    terms = question.get(
        "terms",
        [],
    )

    for chunk in chunks:

        if not isinstance(
            chunk,
            dict,
        ):
            continue

        score = score_chunk(
            chunk,
            terms,
        )

        if score <= 0:
            continue

        metadata = chunk.get(
            "metadata",
            {},
        )

        if not isinstance(
            metadata,
            dict,
        ):
            metadata = {}

        scored.append(
            {
                "score": score,

                "chunk_id":
                    metadata.get(
                        "chunk_id"
                    ),

                "source":
                    metadata.get(
                        "source"
                    ),

                "page":
                    metadata.get(
                        "page"
                    ),

                "chunk_start_page":
                    metadata.get(
                        "chunk_start_page"
                    ),

                "chunk_end_page":
                    metadata.get(
                        "chunk_end_page"
                    ),

                "section_start_page":
                    metadata.get(
                        "section_start_page"
                    ),

                "section_end_page":
                    metadata.get(
                        "section_end_page"
                    ),

                "header":
                    metadata.get(
                        "header"
                    ),

                "subheader":
                    metadata.get(
                        "subheader"
                    ),

                "text":
                    chunk.get(
                        "text",
                        "",
                    ),
            }
        )

    scored.sort(
        key=lambda item: (
            -item["score"],
            safe_page(
                item.get("page")
            ),
            safe_page(
                item.get(
                    "chunk_id"
                )
            ),
        )
    )

    return scored[:limit]


# =========================================================
# PREVIEW
# =========================================================

def make_preview(
    text: Any,
    max_length: int = 500,
) -> str:
    """
    Create a safe single-line preview.
    """

    preview = normalize(text)

    if len(preview) > max_length:
        preview = (
            preview[:max_length]
            + "..."
        )

    return preview


# =========================================================
# PRINT CANDIDATE
# =========================================================

def print_candidate(
    candidate: dict[str, Any],
    rank: int,
) -> None:

    print()
    print(
        f"  Candidate #{rank}"
    )

    print(
        f"  Score       : "
        f"{candidate.get('score')}"
    )

    print(
        f"  Chunk ID    : "
        f"{candidate.get('chunk_id')}"
    )

    print(
        f"  Source      : "
        f"{candidate.get('source')}"
    )

    print(
        f"  Page        : "
        f"{candidate.get('page')}"
    )

    print(
        f"  Page range  : "
        f"{candidate.get('chunk_start_page')}"
        f"-"
        f"{candidate.get('chunk_end_page')}"
    )

    print(
        f"  Section     : "
        f"{candidate.get('subheader')}"
    )

    preview = make_preview(
        candidate.get(
            "text",
            "",
        )
    )

    print(
        f"  Preview     : "
        f"{preview}"
    )


# =========================================================
# PRINT QUESTION RESULTS
# =========================================================

def print_question_results(
    qid: str,
    question: dict[str, Any],
    chunks_a: list[dict[str, Any]],
    chunks_b: list[dict[str, Any]],
) -> None:

    print()
    print("=" * 70)

    print(
        f"{qid}: "
        f"{question['question']}"
    )

    print("=" * 70)

    # -----------------------------------------------------
    # Experiment A
    # -----------------------------------------------------

    print()
    print(
        "Experiment A candidates:"
    )

    candidates_a = find_candidates(
        chunks_a,
        question,
    )

    if not candidates_a:

        print(
            "  No candidates found."
        )

    else:

        for rank, candidate in enumerate(
            candidates_a,
            start=1,
        ):
            print_candidate(
                candidate,
                rank,
            )

    # -----------------------------------------------------
    # Experiment B
    # -----------------------------------------------------

    print()
    print(
        "Experiment B candidates:"
    )

    candidates_b = find_candidates(
        chunks_b,
        question,
    )

    if not candidates_b:

        print(
            "  No candidates found."
        )

    else:

        for rank, candidate in enumerate(
            candidates_b,
            start=1,
        ):
            print_candidate(
                candidate,
                rank,
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
        "GROUND TRUTH CANDIDATE FINDER"
    )
    print("=" * 70)

    print()
    print(
        f"Project root : {BASE_DIR}"
    )

    print(
        f"Experiment A : {CHUNKS_A}"
    )

    print(
        f"Experiment B : {CHUNKS_B}"
    )

    # -----------------------------------------------------
    # Validate files
    # -----------------------------------------------------

    if not CHUNKS_A.exists():
        raise FileNotFoundError(
            "\nExperiment A metadata not found:\n"
            f"{CHUNKS_A}"
        )

    if not CHUNKS_B.exists():
        raise FileNotFoundError(
            "\nExperiment B metadata not found:\n"
            f"{CHUNKS_B}"
        )

    # -----------------------------------------------------
    # Load chunks
    # -----------------------------------------------------

    chunks_a = load_chunks(
        CHUNKS_A
    )

    chunks_b = load_chunks(
        CHUNKS_B
    )

    print()
    print(
        f"Experiment A chunks: "
        f"{len(chunks_a)}"
    )

    print(
        f"Experiment B chunks: "
        f"{len(chunks_b)}"
    )

    print()
    print(
        f"Questions: "
        f"{len(QUESTIONS)}"
    )

    # -----------------------------------------------------
    # Process all questions
    # -----------------------------------------------------

    for qid, question in QUESTIONS.items():

        print_question_results(
            qid,
            question,
            chunks_a,
            chunks_b,
        )

    # -----------------------------------------------------
    # Final warning
    # -----------------------------------------------------

    print()
    print("=" * 70)
    print(
        "IMPORTANT"
    )
    print("=" * 70)

    print(
        "These are CANDIDATES only."
    )

    print(
        "Do NOT automatically copy them "
        "into ground_truth.json."
    )

    print(
        "Each question must be manually verified "
        "against the actual guideline evidence."
    )

    print()
    print(
        "Ground truth file:"
    )

    print(
        f"  {GROUND_TRUTH}"
    )

    print()
    print(
        "Candidate generation completed successfully."
    )


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":
    main()