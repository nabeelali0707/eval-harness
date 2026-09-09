"""Build dense (FAISS) and BM25 indices from data/corpus.json."""
from __future__ import annotations

import argparse
from pathlib import Path


def main(corpus_path: Path, output_dir: Path) -> None:
    raise NotImplementedError


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default="data/corpus.json")
    parser.add_argument("--output", default="cache")
    args = parser.parse_args()
    main(Path(args.corpus), Path(args.output))
