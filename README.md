# 🩺 NICE Breast Cancer RAG

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Next.js](https://img.shields.io/badge/Next.js-15-black?logo=next.js&logoColor=white)](https://nextjs.org/)
[![ChromaDB](https://img.shields.io/badge/Vector_DB-ChromaDB-orange?logo=chroma&logoColor=white)](https://www.trychroma.com/)
[![Google Gemini](https://img.shields.io/badge/LLM-Gemini_Flash-8E75B2?logo=google&logoColor=white)](https://ai.google.dev/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **An evidence-grounded Clinical Decision Support RAG system** strictly scoped to the UK National Institute for Health and Care Excellence ([NICE](https://www.nice.org.uk/)) Guidelines for early and advanced breast cancer diagnosis and management (**NG101** & **CG81**).

---

## 🌟 Key Features

* **Strict Medical Guardrails & Grounding:** Zero outside hallucination. If evidence is missing or ambiguous, requests are deterministically rejected with standard fallback responses.
* **Exact Multi-Level Citation Verification:** Outputs are validated post-generation to enforce explicit page-level citations (`(NG101, Page X)` / `(CG81, Page X)`).
* **Calibrated Semantic Retrieval:** Dynamic distance-threshold filtering using `gemini-embedding-2` and ChromaDB vector search to eliminate irrelevant context.
* **Structured Chunking Architecture:** TOC-aware chunking pipeline preserving clinical sections, guideline hierarchy, and recommendation numbers.
* **Full-Stack Implementation:** High-performance async FastAPI backend paired with a modern, responsive Next.js / TypeScript chat interface with error parsing.

---

## 🏗️ System Architecture

```text
  [ User Query ]
        │
        ▼
┌──────────────────┐
│ Semantic Search  │ ◄── Vector Store (ChromaDB + gemini-embedding-2)
└─────────┬────────┘
          │ (Top-K Chunks + Distance Filter)
          ▼
┌──────────────────┐
│ Guardrail Engine │ ──── If distance > threshold ──► Return Insufficient Context
└─────────┬────────┘
          │ (Valid Context)
          ▼
┌──────────────────┐
│ Gemini Flash LLM │ ◄── Strict Clinical System Instructions
└─────────┬────────┘
          │ (Generated Draft)
          ▼
┌──────────────────┐
│ Citation Checker │ ──── Validates (Guideline, Page #) & Disclaimer Footer
└─────────┬────────┘
          │
          ▼
   [ JSON Response ] ──► Next.js Frontend (Citations & Sources Modal)
