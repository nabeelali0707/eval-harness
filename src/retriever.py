"""Retrieval stage: dense, BM25, and hybrid RRF search."""
from __future__ import annotations

import json
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


def load_bm25_index(index_path: Path) -> BM25Okapi:
    with index_path.open("rb") as f:
        return BM25Okapi.load(f)


def dense_search(
    query: str,
    encoder: SentenceTransformer,
    index: faiss.Index,
    corpus: list[dict[str, Any]],
    top_k: int,
) -> list[tuple[str, float]]:
    """Return list of (doc_id, score) ranked by dense similarity."""
    raise NotImplementedError


def bm25_search(
    query: str,
    bm25: BM25Okapi,
    corpus: list[dict[str, Any]],
    top_k: int,
) -> list[tuple[str, float]]:
    """Return list of (doc_id, score) ranked by BM25."""
    raise NotImplementedError


def reciprocal_rank_fusion(
    ranked_lists: list[list[tuple[str, float]]],
    k: int = 60,
) -> list[tuple[str, float]]:
    """Fuse multiple ranked lists into one RRF score list."""
    raise NotImplementedError


def hybrid_search(
    query: str,
    encoder: SentenceTransformer,
    dense_index: faiss.Index,
    bm25: BM25Okapi,
    corpus: list[dict[str, Any]],
    top_k: int,
    rrf_k: int = 60,
) -> list[tuple[str, float]]:
    """Run dense + BM25 and merge with RRF."""
    raise NotImplementedError
