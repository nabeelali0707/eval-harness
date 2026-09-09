"""Scoring functions for retrieval and generation quality."""
from __future__ import annotations


def recall_at_k(retrieved_ids: list[str], gold_ids: list[str], k: int) -> float:
    """Return 1.0 if any gold doc appears in the top-k retrieved, else 0.0."""
    raise NotImplementedError


def mean_reciprocal_rank(retrieved_ids: list[str], gold_ids: list[str]) -> float:
    """Return MRR for a single ranked list."""
    raise NotImplementedError
