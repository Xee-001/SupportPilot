# Flagship Project: "SupportPilot" — AI Product-Support Assistant

## What it is
An AI assistant for a product company's docs and tickets: users ask questions, it retrieves from product documentation (RAG), answers with citations, escalates complex queries through an agentic workflow, and exposes everything via a FastAPI service — Dockerized, deployed, monitored, and evaluated.
**Why this project**: it covers most tech — RAG, AI assistant, agentic workflow, FastAPI, vector DB, embeddings, Docker, CI/CD, AWS, observability, evaluation. And the domain (support assistant) is something instantly understood.

## JD coverage map

| JD requirement | Where it's covered |
|---|---|
| RAG, embeddings, vector DB, semantic search | Phase 1 |
| Backend APIs with Python/FastAPI | Phase 1–2 |
| AI assistants, agentic workflows | Phase 3 |
| Docker, Linux, Git, CI/CD | Phase 2, 4 |
| AWS cloud-native deployment | Phase 4 |
| Observability, monitoring, evaluation | Phase 5 |
| LangChain/LangGraph, pgvector/Chroma | Phase 1, 3 |
| Forecasting/analytics (good-to-have) | Phase 6 (optional) |

---

## Phase 1 (Week 1): Core RAG pipeline — *the foundation*
**Build:**
- Pick a docs corpus: open-source project docs (e.g., FastAPI's own docs, or Stripe API docs) — real, messy, public.
- Ingestion script: load → chunk (recursive, ~512 tokens, 15% overlap) → embed (OpenAI `text-embedding-3-small` or free: `bge-small` via sentence-transformers) → store in **Chroma** (local, zero setup; migrate to pgvector in Phase 4).
- Retrieval + answer: top-k retrieve → prompt with context → answer **with citations** (chunk sources).
- FastAPI app: `POST /ask` endpoint, pydantic models, basic error handling.

**Rules for this phase:** No LangChain yet — raw OpenAI/Anthropic SDK + Chroma client, so you understand every moving part. Use AI as a *tutor* (explain concepts), not an author (write your code). You must be able to rewrite any file from memory.

**You can now answer:** "Explain a RAG pipeline." "How does chunking affect quality?" "How do embeddings work?"
**Resume bullet earned:** *Built a RAG-based documentation assistant with semantic search over N docs using FastAPI, ChromaDB, and OpenAI embeddings, with cited answers.*

## Phase 2 (Week 2): Production-grade API
**Build:**
- **Streaming responses** (SSE) — top interview topic.
- Conversation memory: session ID → chat history in SQLite/Postgres.
- Retry with exponential backoff on LLM calls; request timeouts; rate limiting.
- Hybrid search: add BM25 (`rank-bm25`) + reciprocal rank fusion with vector results.
- Dockerfile (multi-stage, slim image) + docker-compose (app + db). Git from day 1, meaningful commits.
- pytest: unit tests for chunking/retrieval, mocked LLM tests for the endpoint.

**You can now answer:** "How do you stream LLM output?" "How do you handle provider failures?" "Why hybrid search?"
**Resume bullets:** streaming SSE endpoints; hybrid retrieval (BM25 + dense, RRF); containerized with multi-stage Docker builds; tested with pytest.

## Phase 3 (Week 3): Agentic workflow
**Build (now introduce LangGraph):**
- Router agent: classifies query → *docs question* (RAG path) / *needs live data* (tool call) / *complex* (escalate).
- 2–3 tools: search docs, check "ticket status" (mock API you write), create escalation ticket.
- Human-in-the-loop: escalation requires confirmation.
- Guardrail: answer only from context; abstain ("I don't know") when retrieval is weak — log these.

**You can now answer:** "What's an agentic workflow?" "Explain tool/function calling." "When are agents overkill?" — and you'll have *opinions*, because you built both raw and framework versions.
**Resume bullet:** *Designed an agentic support workflow with LangGraph — query routing, tool use, and human-in-the-loop escalation.*

## Phase 4 (Week 4): Deploy to AWS + CI/CD
**Build:**
- Migrate Chroma → **pgvector** on RDS (or Postgres container on EC2 to stay in free tier).
- Deploy: ECS Fargate (preferred talking point) or a small EC2 with docker-compose (cheaper). Secrets in AWS Secrets Manager. S3 for the docs corpus.
- GitHub Actions: lint (ruff) → test → build image → push to ECR → deploy.

**You can now answer:** "How do you deploy an LLM app?" — with a real architecture you own.
**Resume bullet:** *Deployed on AWS (ECS/ECR/RDS pgvector/S3/Secrets Manager) with GitHub Actions CI/CD.*

## Phase 5 (Week 5): Evaluation + observability — *what separates you from other career-switchers*
**Build:**
- Eval set: 30–50 question/expected-answer pairs from your corpus.
- **RAGAS**: faithfulness, answer relevance, context recall. Run in CI — a prompt change that drops scores fails the build.
- Tracing with **LangSmith** or **Arize Phoenix**: every call logged with latency, tokens, cost.
- Simple cost/latency dashboard (even a notebook).

**You can now answer:** "How do you evaluate RAG?" "How would you catch a bad prompt change before prod?" — most 2–4 YOE candidates can't.
**Resume bullet:** *Built automated RAG evaluation (RAGAS) gating CI deploys; full LLM observability with tracing, token/cost monitoring.*

## Phase 6 (optional, +1 week): Forecasting module
Ticket-volume forecasting on a public support-tickets dataset (Prophet or simple ML baseline + backtesting) exposed as `/forecast`. Directly hits the JD's "AI-driven forecasting" line.

---

## Rules of engagement (non-negotiable for interview survival)
1. **Type every line yourself.** AI explains; you write. The interview tests your fingers' memory, not your reading comprehension.
2. **Break it on purpose** weekly: delete the retry logic and watch it fail; feed it a 200-page PDF; ask it questions outside the corpus. The war stories from this ARE your interview answers.
3. **README with architecture diagram** — interviewers open this first. Include eval scores and a 2-min demo GIF.
4. **Public GitHub, daily commits** — green squares tell the self-driven story for you.
5. **Budget:** <$10/month — small models (gpt-4o-mini / Claude Haiku), free embedding models, AWS free tier.

## Your interview pitch after Phase 2 (memorize the shape)
*"I have 3 years of backend engineering on enterprise production systems. Over the last year I've moved into AI engineering — I built and deployed SupportPilot, a RAG-based support assistant: FastAPI with streaming, hybrid retrieval over pgvector, an agentic escalation workflow in LangGraph, Dockerized and deployed on AWS with CI/CD and automated RAG evaluation. Happy to walk through any part of it."*

Every sentence is checkable, and you can survive any follow-up — because you built it.




