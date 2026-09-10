"""Tests for aggregated comparison-table output."""
from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd

import compare_results


def write_retrieval_only_results(path: Path, mode: str) -> None:
    fieldnames = [
        "question_id",
        "mode",
        "recall_at_5",
        "mrr",
        "faithfulness",
        "answer_relevance",
        "latency_ms",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                "question_id": "q1",
                "mode": mode,
                "recall_at_5": 1.0,
                "mrr": 1.0,
                "faithfulness": "",
                "answer_relevance": "",
                "latency_ms": 10.0,
            }
        )


def test_comparison_omits_unavailable_generation_metrics(tmp_path: Path):
    dense_path = tmp_path / "dense.csv"
    bm25_path = tmp_path / "bm25.csv"
    output_dir = tmp_path / "results"
    write_retrieval_only_results(dense_path, "dense_only")
    write_retrieval_only_results(bm25_path, "bm25_only")

    compare_results.main([dense_path, bm25_path], output_dir)

    summary = pd.read_csv(output_dir / "comparison_table.csv")
    assert list(summary.columns) == ["mode", "recall@5", "mrr", "avg_latency_ms"]
