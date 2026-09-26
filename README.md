# RAG Evaluation & Ablation Harness

A small, reproducible experiment framework that runs the same question set through five retrieval pipelines and compares them with standard IR and RAG metrics.

**One-line pitch:** dense-only → BM25-only → hybrid (RRF) → hybrid + rerank → hybrid + rewrite + rerank, measured with Recall@5, MRR, faithfulness, answer relevance, and latency.

## Architecture

```
eval_questions.json ──▶ run_eval.py --config configs/<mode>.yaml
                              │
                              ▼
                        ┌──────────────┐
                        │ EvalPipeline │
                        └──────────────┘
                              │
        ┌─────────┬──────┴───────┬───────────────┐
        ▼         ▼              ▼                ▼
   [Rewriter]  [Retriever]   [Reranker]      [Generator/Judge]
   (optional)  dense/BM25/    (optional)      (Ollama)
                hybrid+RRF
                              │
                              ▼
                   results/<mode>.csv
                              │
                              ▼
                  compare_results.py
                              │
                              ▼
                   final comparison table
```

## Reproduction

```bash
pip install -r requirements.txt

# 1. Install and start the local Ollama model used by every LLM stage.
ollama pull qwen2.5-coder:7b
ollama list
# Run `ollama serve` only when the Ollama service is not already running.

# 2. Build indices (downloads sentence-transformer model on first run)
python build_index.py --corpus data/corpus.json --output cache

# 3. Run retrieval-only baselines without LLM stages.
python run_eval.py --config configs/dense_only.yaml --no-generator
python run_eval.py --config configs/bm25_only.yaml --no-generator
python run_eval.py --config configs/hybrid_rrf.yaml --no-generator

# 4. Run the complete five-mode ablation locally.
# `run_ablation.py` runs all five modes sequentially into one directory,
# resuming checkpoints, skipping finished modes, and holding OS sleep off.
# On Windows, `start_ablation.ps1` does the same but fully detached from the
# terminal: it refuses to start a second runner, starts Ollama if it is down,
# and survives closing the shell or rebooting the machine.
powershell -ExecutionPolicy Bypass -File start_ablation.ps1
# Or run it in the foreground instead:
python run_ablation.py --output results/final_local_ollama_150q

# Monitor progress and ETA at any time:
python progress_report.py

# Equivalent manual alternative (each mode resumes its own checkpoint):
python run_eval.py --config configs/dense_only.yaml --output results/final_local_ollama_150q --resume
python run_eval.py --config configs/bm25_only.yaml --output results/final_local_ollama_150q --resume
python run_eval.py --config configs/hybrid_rrf.yaml --output results/final_local_ollama_150q --resume
python run_eval.py --config configs/hybrid_rerank.yaml --output results/final_local_ollama_150q --resume
python run_eval.py --config configs/hybrid_rewrite_rerank.yaml --output results/final_local_ollama_150q --resume

# 5. Compare only the five completed mode outputs.
python compare_results.py results/final_local_ollama_150q/dense_only.csv results/final_local_ollama_150q/bm25_only.csv results/final_local_ollama_150q/hybrid_rrf.csv results/final_local_ollama_150q/hybrid_rerank.csv results/final_local_ollama_150q/hybrid_rewrite_rerank.csv --output results/final_local_ollama_150q
```

`--resume` validates the config and selected-question fingerprint before continuing a checkpoint. Use `--restart` only when intentionally replacing that mode's checkpoint; a completed CSV is not replaced until a new run finishes successfully. `--max-retries` defaults to three retries for local Ollama failures, and `--retry-delay` defaults to five seconds.

No API key is required. Ollama defaults to `http://localhost:11434`; set `OLLAMA_HOST` in a local `.env` file only when your local service uses a different address:

```env
OLLAMA_HOST=http://localhost:11434
```

The harness intentionally rejects cloud-tagged Ollama models so all generation, rewriting, and judging remain local. `qwen2.5-coder:7b` was selected from the installed local models because it is the strongest available option with a 32K context window; it occupies about 4.7 GB.

## Data

This repo uses a 150-question sample from the [HotpotQA](https://hotpotqa.github.io/) distractor development set, converted to `data/eval_questions.json` and `data/corpus.json` via `prepare_hotpotqa.py`. The raw source file is excluded from git because of its size; the script can regenerate the processed files from any HotpotQA JSON download.

## Current results

Retrieval-only baseline on 150 HotpotQA questions:

| mode       | recall@5 |   mrr | avg_latency_ms |
|:-----------|---------:|------:|---------------:|
| bm25_only  |    0.893 | 0.725 |          10.01 |
| dense_only |    1.000 | 0.905 |          20.92 |
| hybrid_rrf |    0.973 | 0.859 |          33.09 |

On this sample, dense retrieval alone already finds at least one gold document in the top-5 for every question, so hybrid fusion and reranking are not expected to improve Recall@5. Local generation, JSON judging, and the rewrite/rerank route have passed one-question preflight runs with resumable checkpoints; run the complete five-mode local ablation to produce final faithfulness, answer-relevance, and latency comparisons.

## Project structure

```
├── configs/              # YAML pipeline configs (one per mode)
├── data/                 # eval_questions.json + corpus.json
├── src/                  # retrieval, local Ollama LLM stages, pipeline, and scoring
├── tests/                # unit tests
├── build_index.py        # build FAISS + BM25 indices
├── run_eval.py           # run one mode over the eval set
├── run_ablation.py       # run all five modes sequentially with resume
├── start_ablation.ps1    # detached Windows relauncher (duplicate-safe)
├── progress_report.py    # per-mode checkpoint progress + ETA
├── label_judge_sample.py # judge hand-labeling worksheet export/scoring
├── compare_results.py    # aggregate mode CSVs into a table
├── prepare_hotpotqa.py   # convert raw HotpotQA JSON to corpus/eval format
└── handoff/              # original project docs
```

## Design notes

- **Config-driven:** adding a sixth mode only requires a new YAML file in `configs/`.
- **Local by default:** Ollama keeps answer generation, rewriting, and judging on the machine without credentials.
- **Same generator across all modes:** the only variable under test is retrieval, not generation.
- **Recoverable evaluation:** atomic per-question checkpoints let multi-day local runs continue after interruption; `run_ablation.py` also prevents OS sleep for the duration.
- **Judge validation:** `label_judge_sample.py` exports a reproducible hand-labeling sample and reports human-vs-judge MAE, agreement rate, and Pearson correlation.
- **No LangChain/LlamaIndex:** plain Python functions so each stage is inspectable and unit-testable.
- **Cached indices:** `cache/` holds the FAISS dense index and BM25 index so re-runs are fast.
