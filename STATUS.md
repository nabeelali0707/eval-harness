# Project Status — RAG Evaluation & Ablation Harness

**Date:** 2026-09-10
**Repo:** https://github.com/nabeelali0707/eval-harness.git

## What was done

1. **Connected repo**
   - Initialized `D:\eval_harness` as a Git repository.
   - Added `origin` remote pointing to `https://github.com/nabeelali0707/eval-harness.git`.
   - Pushed all completed milestones to GitHub on the `master` branch.

2. **Project scaffold and data**
   - Created the project structure, configuration, linting, and test setup.
   - Converted a HotpotQA distractor development-set sample into 150 eval questions and 1,496 corpus chunks.
   - Excluded the raw 59 MB dataset source from Git.

3. **Retrieval baseline (Modes A/B/C)**
   - Built FAISS dense retrieval with `sentence-transformers/all-MiniLM-L6-v2`, BM25 retrieval, RRF fusion, and Recall@5/MRR scoring.
   - Added a config-driven pipeline, CSV output, and comparison-table aggregation.
   - Recorded the 150-question retrieval-only baseline:

     | mode       | recall@5 |   mrr | avg_latency_ms |
     |:-----------|---------:|------:|---------------:|
     | bm25_only  |    0.893 | 0.725 |          10.01 |
     | dense_only |    1.000 | 0.905 |          20.92 |
     | hybrid_rrf |    0.973 | 0.859 |          33.09 |

4. **Reranking and query rewriting (Modes D/E)**
   - Added the `cross-encoder/ms-marco-MiniLM-L-6-v2` reranker.
   - Added multi-query expansion and HyDE behind the `rewriter.enabled` configuration flag.
   - Enforced configured `top_k` after multi-query RRF fusion before reranking.

5. **Local Ollama LLM migration**
   - Replaced all Anthropic API usage with a standard-library local Ollama client in `src/ollama_client.py`.
   - Standardized generation, rewriting, and judging on installed `qwen2.5-coder:7b` across all five YAML configs.
   - Added clear errors for an unavailable Ollama service, missing local model, malformed responses, and cloud-tagged model names.
   - Preserved the judge's JSON parsing and 0–1 score normalization safeguards.
   - Removed the Anthropic dependency and API-key requirement; `.env.example` now documents optional `OLLAMA_HOST` configuration.

6. **Quality checks**
   - `python -m pytest` passes (22 tests), including Ollama transport, model availability, cloud-model rejection, and shared pipeline-client coverage.
   - `python -m ruff check .` passes.
   - Local generation smoke test answered “Paris” for a France-capital context.
   - Local JSON judge smoke test returned `{"faithfulness": 1.0, "answer_relevance": 1.0}` for a fully supported answer.
   - One-question BM25 and hybrid-rewrite-rerank evaluations completed locally, exercising generation, judging, rewriting, and all latency fields.

## Deliberate deviations from the handoff plan

- Embedding model switched from `BAAI/bge-large-en-v1.5` to `sentence-transformers/all-MiniLM-L6-v2` for faster CPU iteration.
- Reranker switched from `BAAI/bge-reranker-v2-m3` to `cross-encoder/ms-marco-MiniLM-L-6-v2` for speed.
- The planned hosted generator/judge was replaced by local Ollama using `qwen2.5-coder:7b`; no API key or network LLM provider is needed.
- The hand-rolled judge remains instead of RAGAS to avoid unrelated provider dependencies.

## What is NOT done

- The complete 150-question, five-mode local ablation has not run yet.
- The final comparison table does not yet include all five modes with faithfulness and answer-relevance averages.
- LLM stages are slow on the current local hardware: the one-question full BM25 and rewrite/rerank smoke runs took about 184 and 210 seconds respectively.

## How to finish

1. Ensure the selected local model is present and Ollama is available:
   ```bash
   ollama pull qwen2.5-coder:7b
   ollama list
   ```
2. Run all five modes with the same local model:
   ```bash
   python run_eval.py --config configs/dense_only.yaml
   python run_eval.py --config configs/bm25_only.yaml
   python run_eval.py --config configs/hybrid_rrf.yaml
   python run_eval.py --config configs/hybrid_rerank.yaml
   python run_eval.py --config configs/hybrid_rewrite_rerank.yaml
   ```
3. Generate the final table:
   ```bash
   python compare_results.py results/*.csv
   ```
4. Copy the updated table into `README.md`.
