"""Build dense (FAISS) and BM25 indices from data/corpus.json."""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


def tokenize(text: str) -> list[str]:
    return text.lower().split()


def main(corpus_path: Path, output_dir: Path, model_name: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading corpus from {corpus_path} ...")
    with corpus_path.open(encoding="utf-8") as f:
        corpus = json.load(f)

    doc_ids = [doc["doc_id"] for doc in corpus]
    texts = [doc["text"] for doc in corpus]

    print(f"Building dense index with {model_name} ...")
    encoder = SentenceTransformer(model_name)
    embeddings = encoder.encode(
        texts,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    dim = embeddings.shape[1]

    # Cosine similarity via normalized vectors + inner product
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype(np.float32))

    dense_path = output_dir / "dense.index"
    faiss.write_index(index, str(dense_path))

    doc_ids_path = output_dir / "doc_ids.json"
    doc_ids_path.write_text(json.dumps(doc_ids), encoding="utf-8")

    print(f"Saved dense index to {dense_path}")

    print("Building BM25 index ...")
    tokenized = [tokenize(text) for text in tqdm(texts, desc="Tokenizing")]
    bm25 = BM25Okapi(tokenized)

    bm25_path = output_dir / "bm25.pkl"
    with bm25_path.open("wb") as f:
        pickle.dump(bm25, f)

    print(f"Saved BM25 index to {bm25_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default="data/corpus.json")
    parser.add_argument("--output", default="cache")
    parser.add_argument(
        "--model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Sentence-transformer model for dense embeddings",
    )
    args = parser.parse_args()
    main(Path(args.corpus), Path(args.output), args.model)
