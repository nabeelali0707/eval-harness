"""Retrieval stage: dense, BM25, and hybrid RRF search."""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


def load_corpus(corpus_path: Path) -> list[dict[str, Any]]:
    with corpus_path.open(encoding="utf-8") as f:
        return json.load(f)


def load_dense_index(index_path: Path) -> faiss.Index:
    return faiss.read_index(str(index_path))


def load_doc_ids(doc_ids_path: Path) -> list[str]:
    with doc_ids_path.open(encoding="utf-8") as f:
        return json.load(f)


def load_bm25_index(index_path: Path) -> BM25Okapi:
    with index_path.open("rb") as f:
        return pickle.load(f)


def _corpus_by_id(corpus: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {doc["doc_id"]: doc for doc in corpus}


def dense_search(
    query: str,
    encoder: SentenceTransformer,
    index: faiss.Index,
    doc_ids: list[str],
    top_k: int,
) -> list[tuple[str, float]]:
    """Return list of (doc_id, score) ranked by dense cosine similarity."""
    embedding = encoder.encode(
        query,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)
    embedding = embedding.reshape(1, -1)
    scores, indices = index.search(embedding, top_k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or idx >= len(doc_ids):
            continue
        results.append((doc_ids[idx], float(score)))
    return results


def bm25_search(
    query: str,
    bm25: BM25Okapi,
    doc_ids: list[str],
    top_k: int,
) -> list[tuple[str, float]]:
    """Return list of (doc_id, score) ranked by BM25."""
    tokens = query.lower().split()
    scores = bm25.get_scores(tokens)
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [(doc_ids[int(i)], float(scores[i])) for i in top_indices]


def reciprocal_rank_fusion(
    ranked_lists: list[list[tuple[str, float]]],
    k: int = 60,
) -> list[tuple[str, float]]:
    """Fuse multiple ranked lists into one RRF score list."""
    scores: dict[str, float] = {}
    for ranked_list in ranked_lists:
        for rank, (doc_id, _) in enumerate(ranked_list, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)

    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_scores


def hybrid_search(
    query: str,
    encoder: SentenceTransformer,
    dense_index: faiss.Index,
    bm25: BM25Okapi,
    doc_ids: list[str],
    top_k: int,
    rrf_k: int = 60,
) -> list[tuple[str, float]]:
    """Run dense + BM25 and merge with RRF."""
    # Retrieve a generous candidate pool from each method so RRF has enough overlap.
    pool_size = max(top_k * 4, 50)
    dense_results = dense_search(query, encoder, dense_index, doc_ids, pool_size)
    bm25_results = bm25_search(query, bm25, doc_ids, pool_size)
    fused = reciprocal_rank_fusion([dense_results, bm25_results], k=rrf_k)
    return fused[:top_k]
