# 05 — Tech Stack

## Language & runtime
- Python 3.11+

## Core libraries

| Purpose | Library |
|---|---|
| Dense embeddings | `sentence-transformers` |
| Vector index | `faiss-cpu` (v1) → optional `qdrant-client` (v1.1) |
| Sparse/BM25 | `rank_bm25` |
| Reranking | `sentence-transformers` (cross-encoder) or `FlagEmbedding` for bge-reranker |
| LLM calls | `anthropic` SDK |
| Eval metrics | `ragas` |
| Config parsing | `pyyaml` |
| Data handling | `pandas` |
| CLI | `argparse` (stdlib — no need for click/typer at this scale) |
| Progress bars | `tqdm` |
| Testing | `pytest` |
| Env management | `python-dotenv` for API keys |

## Optional (stretch goal, v1.1+)

- `streamlit` — for the optional visual dashboard over results
- `qdrant-client` + Docker — for a "real" vector DB setup instead of FAISS

## Dev tooling

- `ruff` — linting/formatting
- `pytest` — unit tests for scorer functions (Recall@k, MRR, RRF math) — these are pure functions and should be tested directly, since a bug here silently invalidates your whole comparison table

## requirements.txt (draft)

```
sentence-transformers>=2.7
faiss-cpu>=1.8
rank_bm25>=0.2.2
anthropic>=0.34
ragas>=0.1
pyyaml>=6.0
pandas>=2.2
tqdm>=4.66
python-dotenv>=1.0
pytest>=8.0
ruff>=0.5
```

## Why not LangChain/LlamaIndex

Deliberately avoided for the core pipeline. The whole point of this project is to show you understand what's happening at each retrieval stage — RRF math, rerank scoring, prompt construction for rewriting — not that you can call a framework's `.retrieve()` method. Plain Python keeps every component inspectable and testable. (Fine to mention in the README as a deliberate design choice — it reads well in interviews.)
