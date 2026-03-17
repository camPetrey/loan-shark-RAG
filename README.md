# Loan Shark

Loan Shark is an early-stage Retrieval-Augmented Generation (RAG) backend for mortgage underwriting guideline analysis. The goal is to help loan officers validate loan applications against official guideline documents and receive structured, citation-backed explanations.

This first scaffold focuses on:

- a clean FastAPI backend
- modular ingestion components for PDF parsing and chunking
- OpenAI-ready configuration
- a vector-store abstraction with a `pgvector`-first path for local/personal-project development

Full retrieval and answer generation are intentionally not implemented yet.

## Why `pgvector` first?

For a personal project, `pgvector` is the better default because it keeps the stack simple:

- one datastore for metadata and embeddings
- easier local development
- lower operational overhead than adding a hosted vector service early

The codebase includes a vector-store abstraction so Pinecone can be added later without rewriting the ingestion pipeline.

## Project structure

```text
.
├── main.py
├── requirements.txt
├── data/
├── artifacts/
└── src/
    ├── api/
    ├── cli/
    ├── core/
    ├── domain/
    ├── embeddings/
    ├── ingestion/
    ├── services/
    └── vectorstores/
```

## Environment variables

Create a `.env` file in the project root:

```bash
OPENAI_API_KEY=your_openai_api_key
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
OPENAI_CHAT_MODEL=gpt-4.1-mini
APP_NAME=Loan Shark API
APP_ENV=development
LOG_LEVEL=INFO
DATA_DIR=data
ARTIFACTS_DIR=artifacts
PGVECTOR_DATABASE_URL=postgresql+psycopg://postgres:postgres@localhost:5432/loanshark
VECTOR_STORE_BACKEND=pgvector
CHUNK_SIZE_TOKENS=800
CHUNK_OVERLAP_TOKENS=120
```

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the API

```bash
uvicorn main:app --reload
```

Open:

- `http://127.0.0.1:8000/health`
- `http://127.0.0.1:8000/docs`

## Run ingestion

The ingestion command currently parses PDFs, creates citation-friendly chunks, and writes a preview artifact. It does not embed or store vectors yet.

Preview a single PDF:

```bash
python -m src.cli.ingest --input data/02042026_Single_Family_Selling_Guide_highlighting.pdf
```

Write output to a specific file:

```bash
python -m src.cli.ingest \
  --input data/02042026_Single_Family_Selling_Guide_highlighting.pdf \
  --output artifacts/fannie_preview.json
```

Ingest every PDF inside the default `data/` folder:

```bash
python -m src.cli.ingest --input data
```

## What ingestion does today

1. Reads PDF text page by page with PyMuPDF
2. Normalizes and splits the text into token-aware chunks
3. Preserves metadata needed for future citations
4. Writes a JSON artifact for inspection and debugging

## What is intentionally deferred

- embedding generation
- vector persistence
- retrieval
- answer generation
- guideline-to-loan decision logic

Those belong in the next iterations after we validate the ingestion shape and metadata model.
