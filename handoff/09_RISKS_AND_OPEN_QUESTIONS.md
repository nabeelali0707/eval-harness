# 09 — Risks & Open Questions

## Risks

| Risk | Impact | Mitigation |
|---|---|---|
| LLM-judge metrics (faithfulness, relevance) are noisy/inconsistent | Comparison table looks less credible | Hand-label a 50-question ground-truth subset and report agreement rate alongside LLM-judge scores |
| Local reranker/embedding models are slow on CPU | Full eval run takes too long to iterate on | Keep eval set small (100–200) during development; cache aggressively; only run full set for final numbers |
| Dataset gold-document IDs don't map cleanly onto your own chunking | Recall@k becomes unreliable | Chunk at a granularity that preserves the original dataset's document boundaries where possible, or map gold docs to your chunk IDs during ingestion and verify a sample by hand |
| Hybrid/rerank doesn't actually outperform baselines on this dataset | Feels like a "failed" result | This is still a valid, honest finding — report it as-is. A well-reasoned "here's why hybrid didn't help on this dataset" is more impressive than a suspiciously perfect table |
| Scope creep toward a full production RAG app | v1 never finishes | Hold the line on `06_MVP_SCOPE_AND_DEMO.md` — no UI, no new dataset, no 6th mode until v1's 5 modes are done and documented |
| RAGAS library version/API changes | Broken eval step | Pin the version in `requirements.txt`; if RAGAS breaks, fall back to a small set of hand-written LLM-judge prompts — document the swap in `10_BUILD_PROGRESS.md` |

## Open questions (decide before or during Week 1)

- [ ] **Dataset choice**: HotpotQA (multi-hop, harder, more impressive if it works) vs NQ (simpler, faster to get right)? Recommendation: start with NQ for a working v1, add a HotpotQA slice as a stretch comparison if time allows.
- [ ] **Chunk size**: fixed token window (e.g. 300 tokens) vs paragraph-based chunking? Recommendation: fixed window for v1 simplicity; note as a limitation in README.
- [ ] **Eval set size**: 100 vs 200 vs 300 questions — bigger is more statistically credible but slower/costlier to iterate on. Recommendation: 150 for development, decide final size once per-question latency is known from Week 1–2.
- [ ] **How strict should faithfulness scoring be?** Binary pass/fail vs 0–1 continuous score from the LLM judge. Recommendation: continuous score (RAGAS default), it's more informative for the comparison table.
- [ ] **Do you want a Streamlit dashboard for v1 or purely CLI?** Recommendation: CLI-only for v1 (per `06_MVP_SCOPE_AND_DEMO.md`), revisit as stretch goal.

## Things to explicitly call out in the final README (don't let the agent skip this)

- Sample size and its limitations (this is not a large-scale benchmark run)
- That the same generator model was used across all modes (validity of the ablation depends on this)
- Any hand-labeled ground-truth agreement numbers, alongside the LLM-judge numbers
