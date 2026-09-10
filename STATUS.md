# Project Status — RAG Evaluation & Ablation Harness

**Date:** 2026-09-10
**Repo:** https://github.com/nabeelali0707/eval-harness.git

## What was done

1. **Connected repo, scaffold, and data**
   - Initialized `D:\eval_harness` as a Git repository and connected it to `origin`.
   - Created the project configuration, linting, test setup, and five pipeline configs.
   - Converted a HotpotQA distractor development-set sample into 150 eval questions and 1,496 corpus chunks.
   - Excluded the raw 59 MB dataset source from Git.

2. **Retrieval baseline (Modes A/B/C)**
   - Built FAISS dense retrieval with `sentence-transformers/all-MiniLM-L6-v2`, BM25 retrieval, RRF fusion, and Recall@5/MRR scoring.
   - Added a config-driven pipeline, per-question CSV output, and comparison-table aggregation.
   - Recorded the 150-question retrieval-only baseline:

     | mode       | recall@5 |   mrr | avg_latency_ms |
     |:-----------|---------:|------:|---------------:|
     | bm25_only  |    0.893 | 0.725 |          10.01 |
     | dense_only |    1.000 | 0.905 |          20.92 |
     | hybrid_rrf |    0.973 | 0.859 |          33.09 |

3. **Reranking, rewriting, and local Ollama**
   - Added the `cross-encoder/ms-marco-MiniLM-L-6-v2` reranker plus multi-query expansion and HyDE.
   - Replaced all Anthropic API usage with a standard-library local Ollama client.
   - Standardized generation, rewriting, and judging on installed `qwen2.5-coder:7b` across all five YAML configs.
   - Preserved JSON judge parsing and 0–1 score normalization; removed API-key and Anthropic dependency requirements.

4. **Resumable final-ablation infrastructure**
   - Added atomic per-question JSON checkpoints to `run_eval.py`.
   - Added `--resume`, `--restart`, `--max-retries`, and `--retry-delay` for safe recovery from local Ollama failures.
   - Validates run config/question fingerprints before resuming, records successful `attempt_count`, and preserves an existing completed CSV until replacement output is complete.
   - Increased local Ollama request timeout from 120 to 300 seconds for slow CPU inference.

5. **Quality checks**
   - `python -m pytest` passes (27 tests), including checkpoint resume/restart, retry, and output-preservation coverage.
   - `python -m ruff check .` passes.
   - Real one-question preflight evaluations completed for all five modes through generation, judging, reranking, and rewriting using resumable checkpoints.

## Deliberate deviations from the handoff plan

- Embedding model switched from `BAAI/bge-large-en-v1.5` to `sentence-transformers/all-MiniLM-L6-v2` for faster CPU iteration.
- Reranker switched from `BAAI/bge-reranker-v2-m3` to `cross-encoder/ms-marco-MiniLM-L-6-v2` for speed.
- The planned hosted generator/judge was replaced by local Ollama using `qwen2.5-coder:7b`; no API key or network LLM provider is needed.
- The hand-rolled judge remains instead of RAGAS to avoid unrelated provider dependencies.

## What is NOT done

- The complete 150-question, five-mode local ablation has not run yet.
- The final comparison table does not yet include all five modes with faithfulness and answer-relevance averages.
- LLM stages are slow on the current local hardware; each full one-question preflight took roughly 2–4 minutes.

## How to finish

1. Ensure the selected local model is present and Ollama is available:
   ```bash
   ollama pull qwen2.5-coder:7b
   ollama list
   ```
2. Run each mode sequentially into the same output directory; rerun an interrupted command with `--resume`:
   ```bash
   python run_eval.py --config configs/dense_only.yaml --output results/final_local_ollama_150q --resume
   python run_eval.py --config configs/bm25_only.yaml --output results/final_local_ollama_150q --resume
   python run_eval.py --config configs/hybrid_rrf.yaml --output results/final_local_ollama_150q --resume
   python run_eval.py --config configs/hybrid_rerank.yaml --output results/final_local_ollama_150q --resume
   python run_eval.py --config configs/hybrid_rewrite_rerank.yaml --output results/final_local_ollama_150q --resume
   ```
3. Generate the final table from explicit mode files:
   ```bash
   python compare_results.py results/final_local_ollama_150q/dense_only.csv results/final_local_ollama_150q/bm25_only.csv results/final_local_ollama_150q/hybrid_rrf.csv results/final_local_ollama_150q/hybrid_rerank.csv results/final_local_ollama_150q/hybrid_rewrite_rerank.csv --output results/final_local_ollama_150q
   ```
4. Copy the updated table into `README.md`.
