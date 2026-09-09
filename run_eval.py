"""CLI entrypoint: run evaluation for one pipeline config."""
from __future__ import annotations

import argparse
from pathlib import Path


def main(config_path: Path, questions_path: Path, output_dir: Path) -> None:
    raise NotImplementedError


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--questions", default="data/eval_questions.json")
    parser.add_argument("--output", default="results")
    args = parser.parse_args()
    main(Path(args.config), Path(args.questions), Path(args.output))
