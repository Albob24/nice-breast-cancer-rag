# NICE Breast Cancer RAG

A RAG system that answers questions about breast cancer **treatment and
management** in adults, grounded only in two official NICE guidelines:

- **NG101** — Early and locally advanced breast cancer: diagnosis and management
- **CG81** — Advanced breast cancer: diagnosis and treatment

Out-of-scope questions (prevention, other cancers, pediatric cases,
personalized diagnosis) are rejected **before generation**, based on
retrieval-distance scoring — not on hoping the language model refuses.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

1. Copy `.env.example` to `.env` and add your Google AI Studio API key.
2. Put `NG101.pdf` and `CG81.pdf` in `data/`.

## Pipeline (run in order)

```bash
cd src
python chunking.py       # PDFs -> chunks_metadata.json
python embed_chunks.py   # chunks_metadata.json -> local ChromaDB
python retrieve.py       # sanity-check retrieval scores (tune RELEVANCE_THRESHOLD in config.py)
python generate.py       # test end-to-end Q&A in the terminal
```

## Running the API

```bash
cd src
python app.py
```

```bash
curl -X POST http://localhost:5000/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "What treatments are recommended for early breast cancer?"}'
```

## Project structure

```
nice-breast-cancer-rag/
├── data/                  # place NG101.pdf and CG81.pdf here (not included)
├── src/
│   ├── config.py          # all paths, model names, tunable parameters
│   ├── chunking.py        # PDF -> section-aware, page-accurate chunks
│   ├── embed_chunks.py     # chunks -> ChromaDB embeddings
│   ├── retrieve.py        # semantic search + scope guardrail
│   ├── generate.py        # Gemini generation with local fallback
│   └── app.py              # Flask API
├── requirements.txt
├── .env.example
└── README.md
```

## Notes on deployment

Don't use Google Colab as the production backend for the demo — sessions
time out and there's no stable public endpoint. Deploy `app.py` on a free
tier of Render, Railway, or Fly.io instead so the API stays reachable
during judging.

## Tuning the guardrail

`RELEVANCE_THRESHOLD` in `config.py` controls how strict the out-of-scope
rejection is. Run `python src/retrieve.py` after building the DB to see
real distance scores for in-scope vs. out-of-scope test queries, then set
the threshold between the two clusters.
