"""Unit tests for scorer functions."""
from __future__ import annotations

from src.retriever import reciprocal_rank_fusion
from src.scorer import mean_reciprocal_rank, recall_at_k


def test_recall_at_k_happy_path():
    assert recall_at_k(["a", "b", "c"], ["b"], k=3) == 1.0


def test_recall_at_k_miss():
    assert recall_at_k(["a", "b", "c"], ["d"], k=3) == 0.0


def test_recall_at_k_truncation():
    assert recall_at_k(["a", "b", "c"], ["c"], k=2) == 0.0


def test_recall_at_k_empty():
    assert recall_at_k([], ["a"], k=3) == 0.0
    assert recall_at_k(["a", "b"], [], k=3) == 0.0


def test_mrr_happy_path():
    assert mean_reciprocal_rank(["a", "b", "c"], ["b"]) == 0.5


def test_mrr_no_match():
    assert mean_reciprocal_rank(["a", "b", "c"], ["d"]) == 0.0


def test_mrr_empty():
    assert mean_reciprocal_rank([], ["a"]) == 0.0
    assert mean_reciprocal_rank(["a"], []) == 0.0


def test_rrf_fusion():
    dense = [("a", 0.9), ("b", 0.8), ("c", 0.7)]
    bm25 = [("b", 1.0), ("c", 0.6), ("d", 0.5)]
    fused = reciprocal_rank_fusion([dense, bm25], k=60)
    ids = [doc_id for doc_id, _ in fused]
    assert ids[0] == "b"  # appears in both lists near the top
    assert "a" in ids
    assert "d" in ids
