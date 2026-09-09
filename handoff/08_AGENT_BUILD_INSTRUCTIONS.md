# 08 — Agent Build Instructions

These are direct instructions for the coding agent (e.g. Claude Code) building this project. Read all other docs in this folder first, especially `02_ARCHITECTURE.md`, `03_DATA_MODELS.md`, and `07_TASK_ROADMAP.md`.

## Build order

Follow `07_TASK_ROADMAP.md` week-by-week order exactly. Do not build the generator/reranker before retrieval + scoring work correctly and are unit-tested. Each stage should be independently runnable and verifiable before moving to the next — this project's entire value is in trustworthy numbers, so correctness of each stage matters more than speed of delivery.

## Non-negotiable conventions

1. **Every pipeline mode must be config-driven**, not hardcoded. Adding a 6th mode should only require a new YAML file in `configs/`, never a code change to `pipeline.py`.
2. **Use the same generator model across all modes.** The independent variable under test is retrieval, not generation. If this is violated, the ablation is invalid — flag it loudly if you're tempted to change it.
3. **Write unit tests for `src/scorer.py` before wiring it into the pipeline.** Recall@k, MRR, and RRF fusion are pure functions with well-known edge cases (empty result lists, no gold doc retrieved, ties). Get these right first — a bug here silently corrupts every downstream result.
4. **Cache expensive steps.** Embeddings, BM25 index, and retrieval results should be cached to `cache/` so re-running generation/judging doesn't require re-embedding the corpus. Use a simple content-hash or config-name based cache key.
5. **Never commit API keys.** Use `.env` + `python-dotenv`, and add `.env` to `.gitignore` immediately when scaffolding the repo.
6. **Keep components swappable, not abstracted into a framework.** Plain functions/classes with clear interfaces (see `src/retriever.py`, `src/reranker.py`, `src/rewriter.py` boundaries in `02_ARCHITECTURE.md`). Do not introduce LangChain/LlamaIndex — this is a deliberate project constraint, not an oversight.
7. **Every result row must be traceable.** `results/<mode>.csv` should let someone reconstruct exactly what was retrieved and generated for every question — no silent averaging without the per-question detail also being saved.

## Repo structure to scaffold first

```
rag-eval-harness/
├── configs/
│   ├── dense_only.yaml
│   ├── bm25_only.yaml
│   ├── hybrid_rrf.yaml
│   ├── hybrid_rerank.yaml
│   └── hybrid_rewrite_rerank.yaml
├── data/
│   ├── eval_questions.json
│   └── corpus.json
├── src/
│   ├── retriever.py
│   ├── reranker.py
│   ├── rewriter.py
│   ├── pipeline.py
│   └── scorer.py
├── cache/              # gitignored
├── results/
├── tests/
│   └── test_scorer.py
├── build_index.py
├── run_eval.py
├── compare_results.py
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

## After each milestone in the roadmap

Update `10_BUILD_PROGRESS.md` with: what was completed, what was tested, any deviations from the plan, and what's next. Keep entries short (a few bullet points), dated.

## When something in these docs is ambiguous or wrong

State the assumption you're making and proceed — don't block on it. Note the assumption in `10_BUILD_PROGRESS.md` and in `09_RISKS_AND_OPEN_QUESTIONS.md` if it's a decision worth the user revisiting later.

## Definition of done for the whole project

Matches `06_MVP_SCOPE_AND_DEMO.md` exactly: a clean clone + the documented commands produce a final comparison table across all 5 modes, with tests passing and a README that stands on its own.
