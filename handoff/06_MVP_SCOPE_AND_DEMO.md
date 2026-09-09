# 06 — MVP Scope & Demo

## What "v1 done" means

You can run, from a clean clone:

```bash
pip install -r requirements.txt
python build_index.py                          # builds dense + bm25 indices from data/corpus.json
python run_eval.py --config configs/dense_only.yaml
python run_eval.py --config configs/bm25_only.yaml
python run_eval.py --config configs/hybrid_rrf.yaml
python run_eval.py --config configs/hybrid_rerank.yaml
python run_eval.py --config configs/hybrid_rewrite_rerank.yaml
python compare_results.py results/*.csv
```

...and get a printed + saved (`results/comparison_table.csv` and `.md`) table showing all 5 modes with Recall@5, MRR, Faithfulness, and Avg Latency.

## In scope for v1

- 100–200 question eval set (subset of HotpotQA or NQ)
- 5 pipeline modes as defined in `02_ARCHITECTURE.md`
- FAISS + rank_bm25, local reranker, Claude for generation/judging
- CSV + Markdown table output
- A README with the final comparison table and a short written interpretation ("hybrid+rerank improved Recall@5 by X points over dense-only, at Yx latency cost")
- Unit tests for scorer functions (RRF, Recall@k, MRR)

## Explicitly out of scope for v1

- Web UI / dashboard (stretch goal only, see below)
- Real vector DB (Qdrant/Weaviate) — FAISS is enough for v1
- Multi-hop query decomposition — only single-query rewriting/HyDE for v1
- Production concerns: auth, multi-tenancy, deployment, monitoring

## Demo plan (what you show in an interview / on GitHub)

1. **README hero image**: the final comparison table, rendered as a markdown table right at the top
2. **One paragraph**: plain-English interpretation of what the numbers show
3. **Architecture diagram**: the pipeline flow (from `02_ARCHITECTURE.md`) — this shows you can communicate a system, not just build one
4. **A "how to reproduce" section**: the exact commands above — reviewers/interviewers can rerun it themselves if curious
5. Optional: a 2–3 minute screen recording running `run_eval.py` for one mode and showing the output — good for a portfolio site or LinkedIn post

## Stretch goals (v1.1+, only after v1 works end-to-end)

- Streamlit dashboard: dropdown to pick mode, see live comparison table + per-question drill-down
- Swap FAISS → Qdrant to demonstrate operating a real vector DB
- Add a 6th mode: hosted reranker (Cohere) vs local cross-encoder, as an extra ablation
- Multi-hop query decomposition for HotpotQA's harder questions
