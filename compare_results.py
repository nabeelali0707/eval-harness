"""Aggregate per-mode CSVs into a final comparison table."""
from __future__ import annotations

import argparse
from pathlib import Path


def main(result_paths: list[Path], output_dir: Path) -> None:
    raise NotImplementedError


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs="+")
    parser.add_argument("--output", default="results")
    args = parser.parse_args()
    main([Path(p) for p in args.results], Path(args.output))
