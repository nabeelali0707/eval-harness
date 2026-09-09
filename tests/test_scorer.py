"""Unit tests for scorer functions."""
from __future__ import annotations

from src.scorer import mean_reciprocal_rank, recall_at_k


def test_recall_at_k_happy_path():
    assert recall_at_k(["a", "b", "c"], ["b"], k=3) == 1.0


def test_recall_at_k_miss():
    assert recall_at_k(["a", "b", "c"], ["d"], k=3) == 0.0


def test_recall_at_k_truncation():
    assert recall_at_k(["a", "b", "c"], ["c"], k=2) == 0.0


def test_mrr_happy_path():
    assert mean_reciprocal_rank(["a", "b", "c"], ["b"]) == 0.5


def test_mrr_no_match():
    assert mean_reciprocal_rank(["a", "b", "c"], ["d"]) == 0.0
