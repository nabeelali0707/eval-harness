# Project Status — RAG Evaluation & Ablation Harness

**Date:** 2026-09-20
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

5. **2026-09-20 session: run protection, orchestration, and judge validation tooling**
   - Investigated the low judge averages in the partial dense_only checkpoint (faithfulness 0.416): sampled rows show the 0.0 scores correspond to genuinely weak answers (abstentions, unsupported claims), so the judge behaves plausibly; no fix needed.
   - Benchmarked `qwen2.5-coder:7b` (~2.25 tok/s warm) vs `llama3.2:3b` (~3.2 tok/s warm) on this 4-core i7-10510U: only ~1.4x faster, so switching models was rejected — it would invalidate the existing checkpoint and break the same-generator ablation constraint. Kept `qwen2.5-coder:7b`.
   - Added `src/keep_awake.py`: sleep prevention for long runs (`SetThreadExecutionState` on Windows, `caffeinate` on macOS, no-op elsewhere) with unit tests.
   - Added `run_ablation.py`: runs all five modes sequentially through `run_eval.main` with resume, skips completed mode CSVs, stops the sequence on unrecoverable Ollama failures, and holds the keep-awake inhibitor; unit tested.
   - Added `label_judge_sample.py`: exports a reproducible random sample of judged rows (with retrieved passages) to a JSONL hand-labeling worksheet and scores human-vs-judge MAE, agreement rate, and Pearson correlation; unit tested.
   - Pruned the 11.5-hour machine-sleep outlier row (`hotpotqa_0009`) from the dense_only checkpoint; it regenerates on resume.
   - Launched the full five-mode ablation via `run_ablation.py` into `results/final_local_ollama_150q` (dense_only resumed at 36/150).

## Quality checks

- `python -m pytest` passes (52 tests, including keep-awake, ablation runner, and labeling tool coverage).
- `python -m ruff check .` passes.
- Real one-question preflight evaluations completed for all five modes through generation, judging, reranking, and rewriting using resumable checkpoints.

## Deliberate deviations from the handoff plan

- Embedding model switched from `BAAI/bge-large-en-v1.5` to `sentence-transformers/all-MiniLM-L6-v2` for faster CPU iteration.
- Reranker switched from `BAAI/bge-reranker-v2-m3` to `cross-encoder/ms-marco-MiniLM-L-6-v2` for speed.
- The planned hosted generator/judge was replaced by local Ollama using `qwen2.5-coder:7b`; no API key or network LLM provider is needed.
- The hand-rolled judge remains instead of RAGAS to avoid unrelated provider dependencies.
- Model swap to `llama3.2:3b` for speed was evaluated and rejected (2026-09-20): only ~1.4x faster warm and it would invalidate completed checkpoint work plus the fixed-generator ablation design.

## What is NOT done

- The complete 150-question, five-mode local ablation is running now; dense_only is partially complete (resumed at 36/150).
- The final comparison table does not yet include all five modes with faithfulness and answer-relevance averages.
- The 50-question hand-labeling worksheet has been exported tooling-wise but not yet labeled; agreement numbers are pending the finished ablation.
- LLM stages remain slow on this hardware (~20+ min per question); the full run takes days even with sleep prevention.

## How to finish

1. Ensure the selected local model is present and Ollama is available:
   ```bash
   ollama pull qwen2.5-coder:7b
   ollama list
   ```
2. Run all five modes sequentially (resumable; skips completed CSVs) — this is the command currently running:
   ```bash
   python run_ablation.py --output results/final_local_ollama_150q
   ```
   If it is interrupted, rerun the same command; if Ollama was down, restart `ollama serve` first.
3. Generate the final table from explicit mode files:
   ```bash
   python compare_results.py results/final_local_ollama_150q/dense_only.csv results/final_local_ollama_150q/bm25_only.csv results/final_local_ollama_150q/hybrid_rrf.csv results/final_local_ollama_150q/hybrid_rerank.csv results/final_local_ollama_150q/hybrid_rewrite_rerank.csv --output results/final_local_ollama_150q
   ```
4. Hand-validate the judge (roadmap Week 4):
   ```bash
   python label_judge_sample.py export results/final_local_ollama_150q/checkpoints/dense_only.json --output results/judge_worksheet.jsonl --n 50
   # fill human_faithfulness / human_relevance in the JSONL, then:
   python label_judge_sample.py score results/judge_worksheet.jsonl
   ```
   Report the agreement numbers alongside the LLM-judge averages in the README.
5. Copy the updated table and judge-agreement paragraph into `README.md`.
