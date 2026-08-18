"""
NICE Breast Cancer RAG
Chunking Experiments

Experiment A:
    400-600 tokens
    Target = 500
    Overlap = 12%

Experiment B:
    700-900 tokens
    Target = 800
    Overlap = 12%
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from pypdf import PdfReader

from config import (
    PDF_FILES,
    TOC_PAGES,
    USEFUL_PAGES,
    CHUNK_EXPERIMENTS,
    CHUNKS_FILE_A,
    CHUNKS_FILE_B,
)


# =========================================================
# TOKEN ESTIMATION
# =========================================================

def estimate_tokens(text: str) -> int:
    """
    Approximate English token count.

    Approximation:
        tokens ~= words / 0.75
    """

    words = len(text.split())

    return max(
        1,
        round(words / 0.75),
    )


def tokens_to_words(tokens: int) -> int:
    """
    Approximate token count as words.
    """

    return max(
        1,
        round(tokens * 0.75),
    )


# =========================================================
# TEXT CLEANING
# =========================================================

def clean_text(text: str) -> str:

    text = re.sub(
        r"© NICE \d{4}\.\s*"
        r"All rights reserved\.\s*"
        r"Subject to Notice of rights\s*"
        r"\(https://www\.nice\.org\.uk/"
        r"terms-and-\s*conditions#notice-of-rights\)\.",
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"Page\s+\d+\s+of\s+\d+",
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"https?://\S+",
        "",
        text,
    )

    text = re.sub(
        r"[ \t]+",
        " ",
        text,
    )

    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text,
    )

    return text.strip()


# =========================================================
# READ PDFS
# =========================================================

def read_pdfs():

    pages = []

    for source, pdf_path in PDF_FILES.items():

        pdf_path = Path(pdf_path)

        if not pdf_path.exists():
            raise FileNotFoundError(
                f"PDF not found: {pdf_path}"
            )

        reader = PdfReader(
            str(pdf_path)
        )

        print(
            f"Read {source}: "
            f"{len(reader.pages)} pages"
        )

        for page_number, page in enumerate(
            reader.pages,
            start=1,
        ):

            text = page.extract_text() or ""

            pages.append(
                {
                    "source": source,
                    "page": page_number,
                    "text": text,
                }
            )

    return pages


# =========================================================
# TOC PARSER
# =========================================================

def parse_toc(toc_text: str):

    entries = []

    lines = toc_text.splitlines()

    merged_lines = []

    i = 0

    while i < len(lines):

        line = lines[i].strip()

        if re.match(
            r"^\d+(?:\.\d+)*\s+",
            line,
        ):

            while (
                not re.search(
                    r"\.{2,}\s*\d+\s*$",
                    line,
                )
                and i + 1 < len(lines)
            ):

                i += 1

                line += (
                    " "
                    + lines[i].strip()
                )

        merged_lines.append(line)

        i += 1

    pattern = (
        r"(?:(\d+(?:\.\d+)*)\s+)?"
        r"(.+?)\s*\.{2,}\s*(\d+)"
    )

    for line in merged_lines:

        for match in re.finditer(
            pattern,
            line,
        ):

            number = match.group(1)

            title = (
                match.group(2)
                .strip()
            )

            page_number = int(
                match.group(3)
            )

            entries.append(
                {
                    "number": number,
                    "title": title,
                    "start_page": page_number,
                }
            )

    return entries


# =========================================================
# BUILD TOC MAP
# =========================================================

def build_toc_maps(all_pages):

    toc_maps = {}

    for source, toc_pages in TOC_PAGES.items():

        toc_text = ""

        for page in all_pages:

            if (
                page["source"] == source
                and page["page"] in toc_pages
            ):

                toc_text += (
                    page["text"]
                    + "\n"
                )

        toc_maps[source] = parse_toc(
            clean_text(toc_text)
        )

    return toc_maps


# =========================================================
# SECTION MAP
# =========================================================

def build_section_map(
    entries,
    start_page,
    end_page,
):

    sections = []

    current_header = ""

    useful_entries = [
        entry
        for entry in entries
        if (
            start_page
            <= entry["start_page"]
            <= end_page
        )
    ]

    for entry in useful_entries:

        if entry["number"] is None:

            current_header = (
                entry["title"]
            )

        else:

            sections.append(
                {
                    "header":
                        current_header,

                    "number":
                        entry["number"],

                    "subheader":
                        entry["title"],

                    "start_page":
                        entry["start_page"],
                }
            )

    for index in range(
        len(sections)
    ):

        if index + 1 < len(sections):

            sections[index][
                "end_page"
            ] = sections[
                index + 1
            ]["start_page"]

        else:

            sections[index][
                "end_page"
            ] = end_page

    return sections


# =========================================================
# SECTION PAGE DATA
# =========================================================

def get_section_pages(
    all_pages,
    source,
    start_page,
    end_page,
):

    result = []

    for page in all_pages:

        if (
            page["source"] != source
            or page["page"] < start_page
            or page["page"] > end_page
        ):
            continue

        text = clean_text(
            page["text"]
        )

        if not text:
            continue

        result.append(
            {
                "page": page["page"],
                "text": text,
                "words": text.split(),
            }
        )

    return result


# =========================================================
# CHUNK SECTION
# =========================================================

def chunk_section(
    page_blocks,
    target_tokens,
    overlap_ratio,
):

    target_words = tokens_to_words(
        target_tokens
    )

    overlap_words = max(
        1,
        round(
            target_words
            * overlap_ratio
        ),
    )

    step = max(
        1,
        target_words
        - overlap_words,
    )

    words = []

    page_map = []

    for block in page_blocks:

        for word in block["words"]:

            words.append(word)

            page_map.append(
                block["page"]
            )

    chunks = []

    for start in range(
        0,
        len(words),
        step,
    ):

        end = min(
            start + target_words,
            len(words),
        )

        selected = words[
            start:end
        ]

        if not selected:
            break

        text = " ".join(
            selected
        ).strip()

        start_page = (
            page_map[start]
        )

        end_page = (
            page_map[end - 1]
        )

        chunks.append(
            {
                "text": text,
                "start_page":
                    start_page,
                "end_page":
                    end_page,
                "estimated_tokens":
                    estimate_tokens(text),
            }
        )

        if end >= len(words):
            break

    return chunks


# =========================================================
# BUILD EXPERIMENT
# =========================================================

def build_experiment(
    experiment,
    all_pages,
    sections_by_source,
):

    settings = CHUNK_EXPERIMENTS[
        experiment
    ]

    documents = []

    chunk_id = 1

    for source, sections in (
        sections_by_source.items()
    ):

        useful_start, useful_end = (
            USEFUL_PAGES[source]
        )

        for section in sections:

            section_start = max(
                section["start_page"],
                useful_start,
            )

            section_end = min(
                section["end_page"],
                useful_end,
            )

            page_blocks = (
                get_section_pages(
                    all_pages,
                    source,
                    section_start,
                    section_end,
                )
            )

            if not page_blocks:
                continue

            chunks = chunk_section(
                page_blocks,
                settings["target_tokens"],
                settings["overlap_ratio"],
            )

            for chunk_number, chunk in enumerate(
                chunks,
                start=1,
            ):

                documents.append(
                    {
                        "text":
                            chunk["text"],

                        "metadata":
                            {
                                "chunk_id":
                                    chunk_id,

                                "source":
                                    source,

                                "header":
                                    section["header"]
                                    or "",

                                "number":
                                    section["number"]
                                    or "",

                                "subheader":
                                    section["subheader"]
                                    or "",

                                "page":
                                    chunk[
                                        "start_page"
                                    ],

                                "chunk_start_page":
                                    chunk[
                                        "start_page"
                                    ],

                                "chunk_end_page":
                                    chunk[
                                        "end_page"
                                    ],

                                "section_start_page":
                                    section[
                                        "start_page"
                                    ],

                                "section_end_page":
                                    section[
                                        "end_page"
                                    ],

                                "chunk_number":
                                    chunk_number,

                                "experiment":
                                    experiment,

                                "target_tokens":
                                    settings[
                                        "target_tokens"
                                    ],

                                "min_tokens":
                                    settings[
                                        "min_tokens"
                                    ],

                                "max_tokens":
                                    settings[
                                        "max_tokens"
                                    ],

                                "estimated_tokens":
                                    chunk[
                                        "estimated_tokens"
                                    ],

                                "overlap_ratio":
                                    settings[
                                        "overlap_ratio"
                                    ],
                            },
                    }
                )

                chunk_id += 1

    return documents


# =========================================================
# SAVE
# =========================================================

def save_chunks(
    chunks,
    output_path,
):

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            chunks,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print(
        f"Wrote {len(chunks)} chunks "
        f"to {output_path}"
    )


# =========================================================
# MAIN
# =========================================================

def main():

    print()
    print("=" * 70)
    print(
        "NICE BREAST CANCER RAG"
    )
    print(
        "CHUNKING EXPERIMENTS"
    )
    print("=" * 70)

    all_pages = read_pdfs()

    toc_maps = build_toc_maps(
        all_pages
    )

    sections_by_source = {}

    for source, entries in (
        toc_maps.items()
    ):

        start_page, end_page = (
            USEFUL_PAGES[source]
        )

        sections = build_section_map(
            entries,
            start_page,
            end_page,
        )

        sections_by_source[source] = (
            sections
        )

        print(
            f"{source}: found "
            f"{len(sections)} sections"
        )

    print()
    print(
        "Building Experiment A..."
    )

    chunks_a = build_experiment(
        "A",
        all_pages,
        sections_by_source,
    )

    save_chunks(
        chunks_a,
        CHUNKS_FILE_A,
    )

    print()
    print(
        "Building Experiment B..."
    )

    chunks_b = build_experiment(
        "B",
        all_pages,
        sections_by_source,
    )

    save_chunks(
        chunks_b,
        CHUNKS_FILE_B,
    )

    print()
    print("=" * 70)
    print(
        "CHUNKING COMPLETE"
    )
    print("=" * 70)

    print(
        f"Experiment A: "
        f"{len(chunks_a)} chunks"
    )

    print(
        f"Experiment B: "
        f"{len(chunks_b)} chunks"
    )

    print()
    print(
        "A = 400-600 tokens "
        "(target 500, overlap 12%)"
    )

    print(
        "B = 700-900 tokens "
        "(target 800, overlap 12%)"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()