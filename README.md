# RAG Evaluation & Ablation Harness

A small, reproducible experiment framework that runs the same question set through five retrieval pipelines and compares them with standard IR and RAG metrics.

**One-line pitch:** dense-only → BM25-only → hybrid (RRF) → hybrid + rerank → hybrid + rewrite + rerank, measured with Recall@5, MRR, faithfulness, answer relevance, and latency.

## Architecture

```
eval_questions.json ──▶ run_eval.py --config configs/<mode>.yaml
                              │
                              ▼
                        ┌──────────────┐
                        │  EvalPipeline │
                        └──────────────┘
                              │
        ┌─────────┬──────┴───────┬───────────────┐
        ▼         ▼              ▼                ▼
   [Rewriter]  [Retriever]   [Reranker]      [Generator/Judge]
   (optional)  dense/BM25/    (optional)      (Claude)
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

# 1. Build indices (downloads sentence-transformer model on first run)
python build_index.py --corpus data/corpus.json --output cache

# 2. Run retrieval-only modes (no API key needed)
python run_eval.py --config configs/dense_only.yaml
python run_eval.py --config configs/bm25_only.yaml
python run_eval.py --config configs/hybrid_rrf.yaml

# 3. Run modes that use Claude for generation/judging (requires ANTHROPIC_API_KEY)
python run_eval.py --config configs/hybrid_rerank.yaml
python run_eval.py --config configs/hybrid_rewrite_rerank.yaml

# 4. Compare all modes
python compare_results.py results/*.csv
```

Set your API key before running Modes D or E:

```bash
# Linux/macOS
export ANTHROPIC_API_KEY=your_key_here

# Windows
set ANTHROPIC_API_KEY=your_key_here
```

Or create a `.env` file in the repo root:

```env
ANTHROPIC_API_KEY=your_key_here
```

## Data

This repo uses a 150-question sample from the [HotpotQA](https://hotpotqa.github.io/) distractor development set, converted to `data/eval_questions.json` and `data/corpus.json` via `prepare_hotpotqa.py`. The raw source file is excluded from git because of its size; the script can regenerate the processed files from any HotpotQA JSON download.

## Current results

Retrieval-only baseline on 150 HotpotQA questions:

| mode       | recall@5 |   mrr | avg_latency_ms |
|:-----------|---------:|------:|---------------:|
| bm25_only  |    0.893 | 0.725 |          10.01 |
| dense_only |    1.000 | 0.905 |          20.92 |
| hybrid_rrf |    0.973 | 0.859 |          33.09 |

On this sample, dense retrieval alone already finds at least one gold document in the top-5 for every question, so hybrid fusion and reranking are not expected to improve Recall@5. The full ablation including generation, faithfulness, and the rewrite mode will be added once an Anthropic API key is supplied.

## Project structure

```
├── configs/              # YAML pipeline configs (one per mode)
├── data/                 # eval_questions.json + corpus.json
├── src/                  # retriever, reranker, rewriter, generator, judge, pipeline, scorer
├── tests/                # unit tests for scorer and RRF
├── build_index.py        # build FAISS + BM25 indices
├── run_eval.py           # run one mode over the eval set
├── compare_results.py    # aggregate mode CSVs into a table
├── prepare_hotpotqa.py   # convert raw HotpotQA JSON to corpus/eval format
└── handoff/              # original project docs
```

## Design notes

- **Config-driven:** adding a sixth mode only requires a new YAML file in `configs/`.
- **No LangChain/LlamaIndex:** plain Python functions so each stage is inspectable and unit-testable.
- **Same generator across all modes:** the only variable under test is retrieval, not generation.
- **Cached indices:** `cache/` holds the FAISS dense index and BM25 index so re-runs are fast.
