"""Config-driven pipeline orchestration."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import yaml
from sentence_transformers import SentenceTransformer

from src.generator import ClaudeGenerator
from src.judge import ClaudeJudge
from src.reranker import CrossEncoderReranker
from src.retriever import (
    bm25_search,
    dense_search,
    hybrid_search,
    load_bm25_index,
    load_corpus,
    load_dense_index,
    load_doc_ids,
)
from src.rewriter import ClaudeRewriter


def load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


class EvalPipeline:
    """Loads indices/corpus once and runs retrieval + optional rerank + generate + judge."""

    def __init__(self, config: dict[str, Any], cache_dir: Path = Path("cache")) -> None:
        self.config = config
        self.cache_dir = cache_dir
        self.corpus = load_corpus(Path("data/corpus.json"))
        self.corpus_by_id = {doc["doc_id"]: doc for doc in self.corpus}
        self.doc_ids = load_doc_ids(cache_dir / "doc_ids.json")
        self.dense_index = load_dense_index(cache_dir / "dense.index")
        self.bm25 = load_bm25_index(cache_dir / "bm25.pkl")

        retriever_cfg = config["retriever"]
        self.mode = retriever_cfg["mode"]
        self.top_k = retriever_cfg["top_k"]
        self.rrf_k = retriever_cfg.get("rrf_k", 60)

        if self.mode in ("dense", "hybrid"):
            model_name = retriever_cfg.get(
                "dense_model", "sentence-transformers/all-MiniLM-L6-v2"
            )
            self.encoder = SentenceTransformer(model_name)
        else:
            self.encoder = None

        reranker_cfg = config.get("reranker", {})
        self.use_reranker = reranker_cfg.get("enabled", False)
        if self.use_reranker:
            self.reranker = CrossEncoderReranker(reranker_cfg["model"])
            self.rerank_top_n = reranker_cfg.get("top_n", 5)
        else:
            self.reranker = None

        generator_cfg = config.get("generator", {})
        self.generator_model = generator_cfg.get("model", "claude-sonnet-4-6")

        rewriter_cfg = config.get("rewriter", {})
        self.use_rewriter = rewriter_cfg.get("enabled", False)
        if self.use_rewriter:
            self.rewriter = ClaudeRewriter(
                model=self.generator_model,
                max_tokens=256,
            )
            self.rewrite_strategy = rewriter_cfg.get("strategy", "multi_query")
        else:
            self.rewriter = None
        self.use_generator = generator_cfg.get("enabled", True)
        if self.use_generator:
            self.generator = ClaudeGenerator(
                model=self.generator_model,
                max_tokens=generator_cfg.get("max_tokens", 512),
            )
            self.judge = ClaudeJudge(
                model=self.generator_model,
                max_tokens=256,
            )
        else:
            self.generator = None
            self.judge = None

    def _get_passages(self, doc_ids: list[str]) -> list[str]:
        return [self.corpus_by_id[did]["text"] for did in doc_ids if did in self.corpus_by_id]

    def rewrite(self, query: str) -> tuple[list[str], dict[str, float]]:
        if not self.use_rewriter:
            return [query], {}

        t0 = time.perf_counter()
        queries = self.rewriter.rewrite(query, self.rewrite_strategy)
        latencies = {"rewrite_ms": (time.perf_counter() - t0) * 1000}
        return queries, latencies

    def retrieve(self, query: str) -> tuple[list[str], dict[str, float]]:
        latencies: dict[str, float] = {}

        t0 = time.perf_counter()
        if self.mode == "dense":
            results = dense_search(
                query, self.encoder, self.dense_index, self.doc_ids, self.top_k
            )
        elif self.mode == "bm25":
            results = bm25_search(query, self.bm25, self.doc_ids, self.top_k)
        elif self.mode == "hybrid":
            results = hybrid_search(
                query,
                self.encoder,
                self.dense_index,
                self.bm25,
                self.doc_ids,
                self.top_k,
                self.rrf_k,
            )
        else:
            raise ValueError(f"Unknown retriever mode: {self.mode}")
        latencies["retrieve_ms"] = (time.perf_counter() - t0) * 1000

        retrieved_ids = [doc_id for doc_id, _ in results]
        return retrieved_ids, latencies

    def retrieve_multi(
        self, queries: list[str]
    ) -> tuple[list[str], dict[str, float]]:
        """Retrieve for multiple queries and fuse results with RRF."""
        latencies: dict[str, float] = {}
        all_results: list[list[tuple[str, float]]] = []

        t0 = time.perf_counter()
        for query in queries:
            if self.mode == "dense":
                results = dense_search(
                    query, self.encoder, self.dense_index, self.doc_ids, self.top_k
                )
            elif self.mode == "bm25":
                results = bm25_search(query, self.bm25, self.doc_ids, self.top_k)
            elif self.mode == "hybrid":
                results = hybrid_search(
                    query,
                    self.encoder,
                    self.dense_index,
                    self.bm25,
                    self.doc_ids,
                    self.top_k,
                    self.rrf_k,
                )
            else:
                raise ValueError(f"Unknown retriever mode: {self.mode}")
            all_results.append(results)

        from src.retriever import reciprocal_rank_fusion

        fused = reciprocal_rank_fusion(all_results, k=self.rrf_k)
        latencies["retrieve_ms"] = (time.perf_counter() - t0) * 1000
        retrieved_ids = [doc_id for doc_id, _ in fused]
        return retrieved_ids, latencies

    def rerank(
        self, query: str, retrieved_ids: list[str]
    ) -> tuple[list[str], dict[str, float]]:
        if not self.use_reranker:
            return retrieved_ids, {}

        t0 = time.perf_counter()
        candidates = [
            (did, self.corpus_by_id[did]["text"])
            for did in retrieved_ids
            if did in self.corpus_by_id
        ]
        reranked = self.reranker.rerank(query, candidates, self.rerank_top_n)
        latencies = {"rerank_ms": (time.perf_counter() - t0) * 1000}
        return [doc_id for doc_id, _ in reranked], latencies

    def generate(
        self, question: str, final_ids: list[str]
    ) -> tuple[str, dict[str, float]]:
        if not self.use_generator or not self.generator:
            return "", {}

        t0 = time.perf_counter()
        passages = self._get_passages(final_ids)
        answer = self.generator.generate(question, passages)
        latencies = {"generate_ms": (time.perf_counter() - t0) * 1000}
        return answer, latencies

    def judge_answer(
        self, question: str, answer: str, final_ids: list[str]
    ) -> tuple[dict[str, float], dict[str, float]]:
        if not self.use_generator or not self.judge:
            return {"faithfulness": 0.0, "answer_relevance": 0.0}, {}

        t0 = time.perf_counter()
        passages = self._get_passages(final_ids)
        scores = self.judge.score(question, answer, passages)
        latencies = {"judge_ms": (time.perf_counter() - t0) * 1000}
        return scores, latencies

    def run(self, question: dict[str, Any]) -> dict[str, Any]:
        query = question["question"]
        queries, rewrite_latencies = self.rewrite(query)

        if self.use_rewriter:
            retrieved_ids, retrieve_latencies = self.retrieve_multi(queries)
        else:
            retrieved_ids, retrieve_latencies = self.retrieve(query)

        final_ids, rerank_latencies = self.rerank(query, retrieved_ids)
        answer, generate_latencies = self.generate(query, final_ids)
        scores, judge_latencies = self.judge_answer(query, answer, final_ids)

        latencies: dict[str, float] = {}
        latencies.update(rewrite_latencies)
        latencies.update(retrieve_latencies)
        latencies.update(rerank_latencies)
        latencies.update(generate_latencies)
        latencies.update(judge_latencies)

        return {
            "question_id": question["id"],
            "mode": self.config["name"],
            "question": query,
            "expected_answer": question.get("expected_answer", ""),
            "gold_doc_ids": question.get("gold_doc_ids", []),
            "retrieved_doc_ids": final_ids,
            "generated_answer": answer,
            "faithfulness": scores.get("faithfulness", 0.0),
            "answer_relevance": scores.get("answer_relevance", 0.0),
            **latencies,
            "latency_ms": sum(latencies.values()),
        }


def run_pipeline(question: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    """Convenience function for a single question."""
    pipeline = EvalPipeline(config)
    return pipeline.run(question)
