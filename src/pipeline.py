"""Config-driven pipeline orchestration."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import yaml
from sentence_transformers import SentenceTransformer

from src.retriever import (
    bm25_search,
    dense_search,
    hybrid_search,
    load_bm25_index,
    load_corpus,
    load_dense_index,
    load_doc_ids,
)


def load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


class RetrievalPipeline:
    """Loads indices/corpus once and runs retrieval for a config."""

    def __init__(self, config: dict[str, Any], cache_dir: Path = Path("cache")) -> None:
        self.config = config
        self.cache_dir = cache_dir
        self.corpus = load_corpus(Path("data/corpus.json"))
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

    def retrieve(self, query: str) -> tuple[list[str], dict[str, float]]:
        """Run retrieval and return (doc_ids, stage_latencies_ms)."""
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

    def run(self, question: dict[str, Any]) -> dict[str, Any]:
        """Run one eval question through the pipeline."""
        retrieved_ids, latencies = self.retrieve(question["question"])
        return {
            "question_id": question["id"],
            "mode": self.config["name"],
            "question": question["question"],
            "expected_answer": question.get("expected_answer", ""),
            "gold_doc_ids": question.get("gold_doc_ids", []),
            "retrieved_doc_ids": retrieved_ids,
            **latencies,
            "latency_ms": sum(latencies.values()),
        }


def run_pipeline(question: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    """Convenience function for a single question."""
    pipeline = RetrievalPipeline(config)
    return pipeline.run(question)
