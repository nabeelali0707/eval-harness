# Project Status — RAG Evaluation & Ablation Harness

**Date:** 2026-09-10
**Repo:** https://github.com/nabeelali0707/eval-harness.git

## What was done

1. **Connected repo**
   - Initialized `D:\eval_harness` as a Git repository.
   - Added `origin` remote pointing to `https://github.com/nabeelali0707/eval-harness.git`.
   - Pushed all commits to GitHub on the `master` branch.

2. **Project scaffold**
   - Created directory structure: `configs/`, `data/`, `src/`, `tests/`, `cache/`, `results/`.
   - Added `requirements.txt`, `.env.example`, `.gitignore`, `pyproject.toml` (ruff config).
   - Added five YAML configs: `dense_only`, `bm25_only`, `hybrid_rrf`, `hybrid_rerank`, `hybrid_rewrite_rerank`.

3. **Dataset**
   - Downloaded the HotpotQA distractor development set.
   - Created `prepare_hotpotqa.py` to convert the raw JSON into:
     - `data/eval_questions.json` — 150 questions with gold doc IDs and expected answers.
     - `data/corpus.json` — 1,496 corpus chunks.
   - Raw 59 MB source file is excluded from git via `.gitignore`.

4. **Indexing**
   - `build_index.py` builds a FAISS dense index and a `rank_bm25` index from `corpus.json`.
   - Used `sentence-transformers/all-MiniLM-L6-v2` for fast CPU embeddings.

5. **Retrieval baseline (Modes A/B/C)**
   - `src/retriever.py` implements dense, BM25, reciprocal-rank-fusion, and hybrid search.
   - `src/scorer.py` implements Recall@k and MRR with unit tests.
   - `src/pipeline.py` is a config-driven `EvalPipeline`.
   - `run_eval.py` runs one mode and writes per-question CSVs.
   - `compare_results.py` aggregates mode CSVs into a markdown/CSV comparison table.
   - Week 2 results on 150 HotpotQA questions:

     | mode       | recall@5 |   mrr | avg_latency_ms |
     |:-----------|---------:|------:|---------------:|
     | bm25_only  |    0.893 | 0.725 |          10.01 |
     | dense_only |    1.000 | 0.905 |          20.92 |
     | hybrid_rrf |    0.973 | 0.859 |          33.09 |

6. **Reranker, generator, judge, rewriter (Modes D/E code)**
   - `src/reranker.py` — cross-encoder reranking (`cross-encoder/ms-marco-MiniLM-L-6-v2`).
   - `src/generator.py` — Claude answer generation from retrieved passages.
   - `src/judge.py` — Claude-based faithfulness and answer-relevance scoring.
   - `src/rewriter.py` — Claude-based multi-query expansion and HyDE.
   - `EvalPipeline` wires retrieval → optional rewrite → optional rerank → generation → judging.

7. **Quality checks**
   - `python -m pytest` passes (17 tests): scorer, RRF, dense/BM25 retrieval, API-free pipeline stages, multi-query candidate limits, judge parsing, comparison aggregation, and eval CSV output.
   - `python -m ruff check .` passes.
   - API-free BM25 and hybrid-rerank smoke tests pass with `--no-generator`.
   - `run_eval.py` now writes `rewrite_ms` for per-question rewrite latency tracing.
   - Retrieval-only runs record generation metrics as unavailable, not artificial zero scores.
   - Multi-query RRF retrieval now enforces configured `top_k` before reranking.
   - Judge responses are normalized to valid 0–1 metrics.
   - `README.md` distinguishes retrieval-only runs from the full Claude-backed ablation.
   - `handoff/10_BUILD_PROGRESS.md` updated with milestones and deviations.

## Deliberate deviations from the handoff plan

- Embedding model switched from `BAAI/bge-large-en-v1.5` to `sentence-transformers/all-MiniLM-L6-v2` for faster CPU iteration.
- Reranker switched from `BAAI/bge-reranker-v2-m3` to `cross-encoder/ms-marco-MiniLM-L-6-v2` for speed.
- Hand-rolled Claude judge used instead of RAGAS to avoid an OpenAI dependency.

## What is NOT done

- Modes D and E have not been run end-to-end because `ANTHROPIC_API_KEY` is not set in the environment or in a `.env` file.
- Final comparison table does not yet include faithfulness, answer relevance, or Mode D/E rows.

## How to finish

1. Create `D:\eval_harness\.env` with:
   ```env
   ANTHROPIC_API_KEY=your_key_here
   ```
2. Run all five modes with the same generator/judge model:
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
