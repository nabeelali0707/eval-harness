# 04 — Provider Strategy

Goal: keep costs near-zero and keep everything swappable, since the whole point of the project is comparing components, not committing to one vendor.

## Embeddings (dense retrieval)

| Option | Type | Notes |
|---|---|---|
| `sentence-transformers/all-MiniLM-L6-v2` | Local, free | Good default for v1 — fast, no API cost, no rate limits |
| `BAAI/bge-large-en-v1.5` | Local, free | Stronger quality than MiniLM, slower |
| `voyage-code-2` | API | Only needed if you extend to a code-search variant |
| `text-embedding-3-large` (OpenAI) | API | Optional stretch — use only for a final "does a paid model change results" comparison row |

**Default for v1: local `bge-large-en-v1.5` via `sentence-transformers`.** No API key needed, fully reproducible, runs on CPU (slow but fine for 100–300 questions).

## Sparse retrieval (BM25)

- Default: `rank_bm25` (pure Python, zero infra) for v1 — simplest possible setup
- Stretch: OpenSearch/Elasticsearch if you want to show you can operate a real search engine, but this adds real infra overhead — not worth it for v1

## Vector store

| Option | Notes |
|---|---|
| **Qdrant** (local, Docker or embedded mode) | Recommended — native hybrid search support, easy local setup, good docs |
| Weaviate | Also fine, similar tradeoffs |
| FAISS (local, in-memory) | Simplest possible option if you don't want to run a DB at all — good for v1 if you want zero infra |

**Default for v1: FAISS in-memory** for the dense index (zero setup), with BM25 run separately via `rank_bm25`, fused manually via RRF in your own code. Upgrade to Qdrant only if you want the "I operated a real vector DB" line on the CV — worth doing for v1.1.

## Reranker (cross-encoder)

| Option | Type | Notes |
|---|---|---|
| `BAAI/bge-reranker-v2-m3` | Local, free | Recommended default — solid quality, runs locally |
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | Local, free | Lighter/faster fallback if the above is too slow on your hardware |
| Cohere Rerank API | API, has free tier | Optional comparison row — "local cross-encoder vs hosted reranker" is a nice extra ablation |

## Query rewriting / HyDE / generation / LLM-judge

You need an LLM for: query rewriting, final answer generation, and faithfulness/relevance judging.

| Option | Notes |
|---|---|
| **Claude (via Anthropic API)** | Recommended — you already have this set up; use `claude-sonnet-4-6` for generation/judging, cheap and capable |
| Local open model (e.g. via Ollama) | Only if you want a fully-offline, zero-cost version — slower iteration |

**Important:** use the *same* generation model across all 5 modes. The only variable under test should be the retrieval pipeline, not the generator — otherwise your ablation isn't clean.

## Eval framework

- **RAGAS** — for faithfulness, answer relevance, context precision/recall. Well-documented, plugs into LangChain-style pipelines but works standalone too.
- Supplement with your own **hand-labeled 50-question subset** for ground-truth sanity checking — don't rely solely on LLM-judged metrics, note this explicitly in your README (shows evaluation maturity).

## Cost control notes for the agent

- Cap eval set size at 100–200 questions for iteration; only run the full 300 once configs are stable
- Cache embeddings and retrieval results to disk (`cache/`) so re-running the generator/judge doesn't require re-embedding the whole corpus
- Batch LLM judge calls where the API supports it
