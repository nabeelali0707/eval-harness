# 02 — Architecture

## High-level flow

```
eval_questions.json ──▶ run_eval.py --config configs/<mode>.yaml
                              │
                              ▼
                        ┌──────────────┐
                        │  Pipeline     │
                        │  (per mode)   │
                        └──────────────┘
                              │
        ┌─────────────┬──────┴───────┬───────────────┐
        ▼             ▼              ▼                ▼
   [Rewriter]   [Retriever]     [Reranker]      [Generator/LLM]
   (optional)   dense/BM25/     (optional)      (answer synth,
                hybrid+RRF                       used for
                                                  faithfulness eval)
                              │
                              ▼
                   results/<mode>.csv (per-question scores)
                              │
                              ▼
                  compare_results.py results/*.csv
                              │
                              ▼
                   final comparison table (README/report)
```

## Pipeline stages

### 1. Rewriter (used only in Mode E)
- Input: raw user question
- Output: 1 rewritten query, or N expanded query variants
- Techniques to implement:
  - **Multi-query expansion**: ask an LLM to generate 3 paraphrases of the question, retrieve for each, merge results
  - **HyDE**: ask an LLM to write a *hypothetical answer passage*, embed that instead of the raw question, retrieve on that embedding
- Keep this as a pluggable interface (`rewriter.py`) so you can swap strategies without touching the rest of the pipeline

### 2. Retriever
- **Dense**: embed query, cosine/dot-product search against a vector index
- **BM25**: classic sparse lexical search (`rank_bm25` or an OpenSearch/Elasticsearch BM25 index)
- **Hybrid**: run both, merge with **Reciprocal Rank Fusion (RRF)** — no ML needed, just rank math:
  ```
  score(doc) = sum over each ranked list: 1 / (k + rank_in_that_list)
  ```
  (k is a constant, typically 60)

### 3. Reranker (Modes D and E)
- Input: top-k candidates from retriever (e.g. top 20–50)
- A cross-encoder scores each (query, passage) pair jointly (more accurate than embedding similarity, but slower — only run on a shortlist)
- Output: top-n reranked (e.g. top 5) passed to the generator/scorer

### 4. Generator (all modes, for faithfulness scoring)
- Given the final top-n passages + the question, generate an answer
- This answer is what gets scored for faithfulness/answer-relevance
- Keep this consistent across all 5 modes so the *only* variable being tested is retrieval quality, not generation quality

### 5. Scorer / Evaluator
- Per-question metrics:
  - **Recall@k** — is the gold passage present in the top-k retrieved?
  - **MRR (Mean Reciprocal Rank)** — how high up is the first relevant passage?
  - **Faithfulness** — does the generated answer only claim things supported by retrieved context? (LLM-judge based, e.g. via RAGAS)
  - **Answer relevance** — does the answer actually address the question?
  - **Latency** — wall-clock time per question, per stage
- Aggregate into a per-mode summary row

## Config-driven design

Every mode is just a YAML config selecting which components are active:

```yaml
# configs/hybrid_rerank.yaml
name: hybrid_rerank
retriever:
  mode: hybrid        # dense | bm25 | hybrid
  rrf_k: 60
  top_k: 20
rewriter:
  enabled: false
reranker:
  enabled: true
  model: bge-reranker-v2-m3
  top_n: 5
generator:
  model: claude-sonnet-4-6
```

This means `pipeline.py` reads a config and assembles the right components — no branching logic scattered across the codebase, and adding a Mode F later is just a new YAML file.

## Folder-to-architecture mapping

- `src/retriever.py` → stage 2
- `src/reranker.py` → stage 3
- `src/rewriter.py` → stage 1
- `src/pipeline.py` → orchestrates 1→2→3→4 based on config
- `src/scorer.py` → stage 5
- `run_eval.py` → CLI entrypoint, loads config + data, calls pipeline + scorer per question, writes CSV
- `compare_results.py` → reads all CSVs, builds final table
