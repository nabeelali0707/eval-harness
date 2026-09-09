"""Cross-encoder reranking stage."""
from __future__ import annotations


def rerank(query: str, candidates: list[tuple[str, str]], model_name: str, top_n: int) -> list[tuple[str, float]]:
    """Rerank (doc_id, passage) candidates and return top_n (doc_id, score)."""
    raise NotImplementedError
