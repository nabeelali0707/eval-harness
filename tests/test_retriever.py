"""Unit tests for dense and BM25 retrieval helpers."""
from __future__ import annotations

import numpy as np

from src.retriever import bm25_search, dense_search


class FakeEncoder:
    def encode(self, query: str, **kwargs: object) -> np.ndarray:
        assert query == "example query"
        assert kwargs == {"convert_to_numpy": True, "normalize_embeddings": True}
        return np.array([0.25, 0.75])


class FakeDenseIndex:
    def search(self, embedding: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        assert embedding.shape == (1, 2)
        assert top_k == 4
        return (
            np.array([[0.95, 0.80, 0.75, 0.20]]),
            np.array([[2, -1, 0, 5]]),
        )


class FakeBm25:
    def __init__(self) -> None:
        self.tokens: list[str] | None = None

    def get_scores(self, tokens: list[str]) -> np.ndarray:
        self.tokens = tokens
        return np.array([0.25, 1.25, 0.75])


def test_dense_search_skips_invalid_ids_and_preserves_rank():
    results = dense_search(
        "example query",
        FakeEncoder(),  # type: ignore[arg-type]
        FakeDenseIndex(),  # type: ignore[arg-type]
        ["doc_a", "doc_b", "doc_c"],
        top_k=4,
    )

    assert results == [("doc_c", 0.95), ("doc_a", 0.75)]


def test_bm25_search_lowercases_query_and_returns_ranked_ids():
    bm25 = FakeBm25()

    results = bm25_search("Example QUERY", bm25, ["doc_a", "doc_b", "doc_c"], top_k=2)

    assert bm25.tokens == ["example", "query"]
    assert results == [("doc_b", 1.25), ("doc_c", 0.75)]
