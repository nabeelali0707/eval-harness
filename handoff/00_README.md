# RAG Evaluation & Ablation Harness — Project Docs

This folder is the single source of truth for building the project with an AI coding agent (Claude Code or similar). Give the agent this whole folder before it writes any code.

## Read order

1. **01_PROJECT_BRIEF.md** — what we're building and why, in plain terms
2. **02_ARCHITECTURE.md** — the 5 retrieval modes, pipeline stages, how they connect
3. **03_DATA_MODELS.md** — schemas for eval questions, configs, results
4. **04_PROVIDER_STRATEGY.md** — which embedding/rerank/LLM providers to use, and fallbacks
5. **05_TECH_STACK.md** — libraries, tools, versions
6. **06_MVP_SCOPE_AND_DEMO.md** — what "done" looks like for v1, and how it's demoed
7. **07_TASK_ROADMAP.md** — week-by-week build plan
8. **08_AGENT_BUILD_INSTRUCTIONS.md** — direct instructions for the coding agent: build order, conventions, guardrails
9. **09_RISKS_AND_OPEN_QUESTIONS.md** — known risks and decisions not yet made
10. **10_BUILD_PROGRESS.md** — living log the agent updates as it builds

## One-line pitch

A CLI-based evaluation harness that runs the same question set through 5 progressively smarter retrieval pipelines (dense-only → BM25-only → hybrid → hybrid+rerank → hybrid+rewrite+rerank) and produces a single comparison table proving, with numbers, which techniques actually help.

## Project name (working)

`rag-eval-harness`

## Status

Planning complete. Ready for agent-driven build. See `10_BUILD_PROGRESS.md` for live status.
