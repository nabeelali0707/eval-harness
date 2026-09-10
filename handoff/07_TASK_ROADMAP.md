# 07 — Task Roadmap (Week by Week)

Assumes ~5–8 hours/week. Adjust pace as needed — order matters more than the calendar dates.

## Week 1 — Data & baseline retrieval

- [ ] Pick and download dataset subset (HotpotQA or NQ, 100–200 questions + their gold docs)
- [ ] Write `data/eval_questions.json` and `data/corpus.json` in the schemas from `03_DATA_MODELS.md`
- [ ] Build `build_index.py`: chunk corpus, build FAISS dense index + BM25 index, cache both to disk
- [ ] Build `src/retriever.py` with `dense_search()` and `bm25_search()` functions
- [ ] Sanity check: manually inspect top-5 results for 5 questions — do they look reasonable?

**Milestone:** you can retrieve top-k docs for a question via either method.

## Week 2 — Hybrid fusion + scoring + Mode A/B/C working end to end

- [ ] Implement RRF fusion in `src/retriever.py` (`hybrid_search()`)
- [ ] Implement `src/scorer.py`: `recall_at_k()`, `mrr()` — write unit tests first (these are pure functions, easy to get subtly wrong)
- [ ] Build `src/pipeline.py` that reads a YAML config and runs retrieval only (no generation yet)
- [ ] Build `run_eval.py` CLI: loads config + questions, runs pipeline, writes results CSV (retrieval metrics only for now)
- [ ] Run modes A (dense_only), B (bm25_only), C (hybrid_rrf) — confirm hybrid ≥ either baseline on Recall@5

**Milestone:** 3 of 5 modes produce a retrieval-only comparison table.

## Week 3 — Reranking + generation + faithfulness

- [ ] Build `src/reranker.py`: load cross-encoder, score (query, passage) pairs, return top-n
- [ ] Wire reranker into `pipeline.py` behind the `reranker.enabled` config flag
- [ ] Build the generator step: given top-n passages + question, call local Ollama to produce an answer
- [ ] Integrate RAGAS (or hand-rolled LLM-judge prompts) for faithfulness + answer relevance
- [ ] Run Mode D (hybrid_rerank) end to end, full metric set

**Milestone:** 4 of 5 modes done, full metric set (retrieval + generation quality + latency).

## Week 4 — Rewriting + final comparison + polish

- [ ] Build `src/rewriter.py`: multi-query expansion and/or HyDE
- [ ] Wire into pipeline behind `rewriter.enabled` flag, run Mode E (hybrid_rewrite_rerank)
- [ ] Build `compare_results.py`: aggregate all CSVs into one table (CSV + Markdown output)
- [ ] Hand-label a 50-question ground-truth subset independently, spot-check LLM-judge scores against it, note agreement/disagreement in README
- [ ] Write the README: comparison table, architecture diagram, reproduction steps, interpretation paragraph
- [ ] Add `pytest` tests for scorer + RRF logic, add `ruff` config, clean up repo structure

**Milestone:** v1 complete per `06_MVP_SCOPE_AND_DEMO.md`. Push to GitHub, add to CV.

## Week 5+ (optional stretch)

- [ ] Streamlit dashboard
- [ ] Swap FAISS → Qdrant
- [ ] Add hosted-reranker comparison row
- [ ] Multi-hop query decomposition for harder HotpotQA questions
