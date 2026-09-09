# 01 — Project Brief

## Motivation

Most "RAG projects" on a CV are a thin wrapper around a vector DB and an LLM call. They don't demonstrate understanding of *why* retrieval works or fails. This project exists to prove — with measured numbers, not vibes — that specific retrieval techniques (hybrid search, query rewriting, reranking) measurably improve retrieval quality and answer faithfulness.

## What we're building

An **evaluation and ablation harness**: a tool that runs the same set of benchmark questions through multiple retrieval pipeline configurations and scores each one on standard IR and RAG metrics, producing a single comparison table.

This is not "one RAG app." It's a small experiment framework applied to a real, publicly available QA dataset, so the results are reproducible and defensible.

## The 5 modes being compared

| Mode | Name | What it adds |
|---|---|---|
| A | Dense only | Baseline: embedding similarity search |
| B | BM25 only | Baseline: lexical/keyword search |
| C | Hybrid (RRF) | Combines A + B via Reciprocal Rank Fusion |
| D | Hybrid + Rerank | Adds a cross-encoder reranker on top-k candidates |
| E | Hybrid + Rewrite + Rerank | Adds query rewriting/expansion before retrieval |

## Why this is CV-strong

- Shows understanding of retrieval internals, not just API plumbing
- Produces a real artifact (a results table / mini report) that's easy to show in an interview
- Demonstrates eval-driven thinking — a skill senior ML/AI engineers specifically look for
- Reproducible: anyone can clone the repo and rerun it

## Target dataset

A subset of a public QA benchmark with known correct answers/documents — e.g. a slice of **HotpotQA** or **Natural Questions (NQ)**, small enough to run cheaply (target: 100–300 questions for v1).

## Success criteria (plain language)

By the end, we can say: *"On a 200-question subset of HotpotQA, hybrid search + reranking improved Recall@5 from X% to Y% and MRR from X to Y over vanilla dense retrieval, at Z× the latency cost."* — with a table and a README to back it up.

## Non-goals (for v1)

- No production deployment, auth, or multi-user support
- No fancy UI required (CLI + a results table is enough; a Streamlit dashboard is a stretch goal)
- Not trying to beat published SOTA — the goal is a clean, correct, well-instrumented ablation, not leaderboard chasing
