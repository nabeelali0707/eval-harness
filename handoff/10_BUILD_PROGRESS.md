# 10 — Build Progress

Living log. The agent (or you) should append a dated entry after each work session or milestone from `07_TASK_ROADMAP.md`. Keep entries short — a few bullets each. Newest entries at the top.

---

## Template for new entries

```
## YYYY-MM-DD — <milestone or session summary>

**Completed:**
- ...

**Tested:**
- ...

**Deviations from plan:**
- ...

**Assumptions made:**
- ...

**Next up:**
- ...
```

---

## 2026-09-10 — Retrieval-only evaluation hardened

**Completed:**
- Added `--no-generator` documentation for API-free retrieval baselines.
- Added `rewrite_ms` to per-question CSV output for Mode E latency traceability.
- Enforced configured `top_k` after multi-query RRF fusion before Mode E reranking.
- Removed unused RAGAS dependency; the implemented Claude judge remains the metrics path.
- Added API-free tests for dense/BM25 retrieval, disabled optional pipeline stages, multi-query candidate limits, and `run_eval.py` CSV behavior.

**Tested:**
- `python -m pytest` passes (13 tests).
- `python -m ruff check .` passes.
- `python run_eval.py --config configs/bm25_only.yaml --no-generator --sample 5` completes.

**Deviations from plan:**
- None beyond the previously recorded hand-rolled Claude judge.

**Assumptions made:**
- Full faithfulness/relevance comparisons will rerun all five modes using the same configured Claude generator once credentials are available.

**Next up:**
- Add `ANTHROPIC_API_KEY` to `.env`.
- Run the complete five-mode ablation and publish final metrics.

---

## 2026-09-09 — Weeks 1–2 complete; reranker/generator/judge/rewriter implemented

**Completed:**
- Scaffolded repo structure, configs, tests, requirements, and .gitignore.
- Downloaded HotpotQA distractor dev set and created 150-question eval set + 1,496-chunk corpus.
- Built `build_index.py` (FAISS dense + BM25) and `src/retriever.py` (dense, BM25, RRF, hybrid).
- Implemented `src/scorer.py` with unit tests for Recall@k, MRR, and RRF.
- Built config-driven `src/pipeline.py`, `run_eval.py`, and `compare_results.py`.
- Ran Modes A/B/C: dense_only recall@5=1.000, bm25_only=0.893, hybrid_rrf=0.973.
- Added `src/reranker.py`, `src/generator.py`, `src/judge.py`, and `src/rewriter.py`.
- Added README with reproduction steps and ruff config.
- Pushed all commits to GitHub.

**Tested:**
- `python -m pytest tests/test_scorer.py` passes (8 tests).
- `python build_index.py` builds indices successfully.
- `python run_eval.py --config configs/{dense_only,bm25_only,hybrid_rrf}.yaml` completes.
- `python compare_results.py results/*.csv` produces comparison table.
- `python -m ruff check` passes.

**Deviations from plan:**
- Switched embedding model from `bge-large-en-v1.5` to `all-MiniLM-L6-v2` for faster CPU iteration.
- Switched reranker from `bge-reranker-v2-m3` to `cross-encoder/ms-marco-MiniLM-L-6-v2` for speed.
- Used a hand-rolled Claude judge instead of RAGAS to avoid an OpenAI dependency.

**Assumptions made:**
- HotpotQA distractor dev set is acceptable for v1 despite dense retrieval ceiling on Recall@5.
- User will provide `ANTHROPIC_API_KEY` to run Modes D and E.

**Next up:**
- Add `ANTHROPIC_API_KEY` to `.env`.
- Run Mode D (hybrid_rerank) and Mode E (hybrid_rewrite_rerank) end to end.
- Finalize comparison table and README.

---

## 2026-09-09 — Project docs created

**Completed:**
- Full project doc set written (00–09): brief, architecture, data models, provider strategy, tech stack, MVP scope, roadmap, agent build instructions, risks/open questions
- Decisions locked for v1: FAISS + rank_bm25, local bge-reranker, Claude for generation/judging, RAGAS for eval, no LangChain/LlamaIndex, CLI-only (no dashboard)

**Tested:**
- N/A — planning phase only, no code written yet

**Deviations from plan:**
- None yet

**Assumptions made:**
- Dataset choice (HotpotQA vs NQ) left open per `09_RISKS_AND_OPEN_QUESTIONS.md` — recommend NQ first for a faster working v1

**Next up:**
- Week 1 of `07_TASK_ROADMAP.md`: acquire dataset subset, build `data/eval_questions.json` + `data/corpus.json`, scaffold repo structure from `08_AGENT_BUILD_INSTRUCTIONS.md`
