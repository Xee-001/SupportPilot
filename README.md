# SupportPilot

SupportPilot is an AI support assistant that answers questions from a product's documentation. It uses Retrieval-Augmented Generation (RAG): it finds the most relevant parts of the docs using hybrid search (vector search + BM25 keyword search, merged with Reciprocal Rank Fusion), then gives them to an LLM (GPT-OSS 20B via Groq) to write an answer. Every answer comes back with the source files it was based on. It is currently trained on the FastAPI documentation and served through a FastAPI endpoint.

## How it works

```
   FastAPI docs (Markdown files)
            |
            v
   +-----------------+
   |   ingest.py     |  Split docs into chunks, turn each chunk into an
   |                 |  embedding, store everything in ChromaDB
   +-----------------+
            |
            v
      chroma_db/
            |
            v
   +-----------------+
   |    query.py     |  Question -> vector search + BM25 search
   |                 |  -> merge with RRF -> top chunks -> Groq LLM
   +-----------------+
            |
            v
   +-----------------+
   |    main.py      |  POST /ask  ->  { answer, sources }
   +-----------------+
```

## Project structure

| File | What it does |
|------|--------------|
| `ingest.py` | Reads the docs, splits them into chunks (500 characters, 75 overlap), embeds them with `all-MiniLM-L6-v2`, and stores them in ChromaDB. |
| `query.py` | Finds the best chunks for a question using hybrid search and asks the LLM to answer using them. |
| `main.py` | FastAPI app that exposes `POST /ask`. |
| `requirements.txt` | Python dependencies. |
| `SupportPilot.md` | The original project plan and roadmap. |
| `CHANGES.md` | Plain-language explanation of each file and a log of changes. |

## Setup

**Requirements:** Python 3.10+ and a free [Groq API key](https://console.groq.com).

1. Clone this repo and go into it:
```bash
   git clone https://github.com/Xee-001/SupportPilot.git
   cd SupportPilot
```

2. Create and activate a virtual environment:
```bash
   python -m venv .venv
   source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
   pip install -r requirements.txt
```

4. Add your Groq API key. Create a file named `.env` in the project root:
```
   GROQ_API_KEY=your_key_here
```

5. Download the FastAPI docs that SupportPilot learns from (they are not stored in this repo):
```bash
   git clone --depth 1 https://github.com/fastapi/fastapi.git
```
   `ingest.py` reads the Markdown files from `fastapi/docs/en/docs`.

## Run

**Step 1: build the search database (once, or whenever the docs change):**
```bash
python ingest.py
```
This creates the `chroma_db/` folder. Re-running it is safe: existing chunks are updated, not duplicated.

**Step 2: start the API:**
```bash
uvicorn main:app --reload
```

**Step 3: ask a question.** Open http://127.0.0.1:8000/docs and try the `/ask` endpoint, or use curl:
```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is FastAPI?"}'
```

You can also test the answering logic without the API by running `python query.py`.

## Tech stack

- **Embeddings:** sentence-transformers (`all-MiniLM-L6-v2`)
- **Vector database:** ChromaDB
- **Keyword search:** rank-bm25
- **LLM:** GPT-OSS 20B (`openai/gpt-oss-20b`) via Groq
- **API:** FastAPI + Uvicorn

## Roadmap

- [x] Phase 1: core RAG pipeline (ingest, retrieve, answer, API)
- [x] Hybrid search (BM25 + vector, merged with RRF)
- [ ] Phase 2: streaming responses, conversation memory, retries, Docker, tests
- [ ] Phase 3: LangGraph agent with escalation
- [ ] Phase 4: AWS deployment and CI/CD
- [ ] Phase 5: evaluation and observability
- [ ] Phase 6: forecasting

See `SupportPilot.md` for the full plan.