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
