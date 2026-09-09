"""CLI entrypoint: run evaluation for one pipeline config."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from tqdm import tqdm

from src.pipeline import RetrievalPipeline, load_config
from src.scorer import mean_reciprocal_rank, recall_at_k


def main(config_path: Path, questions_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    config = load_config(config_path)
    mode_name = config["name"]

    print(f"Loading questions from {questions_path} ...")
    with questions_path.open(encoding="utf-8") as f:
        questions = json.load(f)

    print(f"Running {mode_name} on {len(questions)} questions ...")
    pipeline = RetrievalPipeline(config)

    rows = []
    for question in tqdm(questions, desc=mode_name):
        result = pipeline.run(question)
        retrieved_ids = result["retrieved_doc_ids"]
        gold_ids = result["gold_doc_ids"]
        result["recall_at_5"] = recall_at_k(retrieved_ids, gold_ids, k=5)
        result["mrr"] = mean_reciprocal_rank(retrieved_ids, gold_ids)
        rows.append(result)

    output_path = output_dir / f"{mode_name}.csv"
    fieldnames = [
        "question_id",
        "mode",
        "recall_at_5",
        "mrr",
        "latency_ms",
        "retrieve_ms",
        "generated_answer",
        "retrieved_doc_ids",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    avg_recall = sum(r["recall_at_5"] for r in rows) / len(rows)
    avg_mrr = sum(r["mrr"] for r in rows) / len(rows)
    avg_latency = sum(r["latency_ms"] for r in rows) / len(rows)
    print(f"\n{mode_name}: recall@5={avg_recall:.3f}, mrr={avg_mrr:.3f}, avg_latency={avg_latency:.1f}ms")
    print(f"Wrote per-question results to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--questions", default="data/eval_questions.json")
    parser.add_argument("--output", default="results")
    args = parser.parse_args()
    main(Path(args.config), Path(args.questions), Path(args.output))
