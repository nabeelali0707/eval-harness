"""Scoring functions for retrieval and generation quality."""
from __future__ import annotations


def recall_at_k(retrieved_ids: list[str], gold_ids: list[str], k: int) -> float:
    """Return 1.0 if any gold doc appears in the top-k retrieved, else 0.0."""
    if not retrieved_ids or not gold_ids or k <= 0:
        return 0.0
    top_k = retrieved_ids[:k]
    return 1.0 if any(doc_id in top_k for doc_id in gold_ids) else 0.0


def mean_reciprocal_rank(retrieved_ids: list[str], gold_ids: list[str]) -> float:
    """Return MRR for a single ranked list."""
    if not retrieved_ids or not gold_ids:
        return 0.0
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in gold_ids:
            return 1.0 / rank
    return 0.0
