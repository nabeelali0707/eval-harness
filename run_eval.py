"""CLI entrypoint: run evaluation for one pipeline config."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

from tqdm import tqdm

from src.ollama_client import OllamaError
from src.pipeline import EvalPipeline, load_config
from src.scorer import mean_reciprocal_rank, recall_at_k

CHECKPOINT_SCHEMA_VERSION = 1
FIELDNAMES = [
    "question_id",
    "mode",
    "recall_at_5",
    "mrr",
    "faithfulness",
    "answer_relevance",
    "latency_ms",
    "rewrite_ms",
    "retrieve_ms",
    "rerank_ms",
    "generate_ms",
    "judge_ms",
    "attempt_count",
    "generated_answer",
    "retrieved_doc_ids",
]


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        text=True,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as temporary_file:
            temporary_file.write(text)
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def _write_checkpoint(path: Path, checkpoint: dict[str, Any]) -> None:
    _atomic_write_text(path, json.dumps(checkpoint, ensure_ascii=False, indent=2) + "\n")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        text=True,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as temporary_file:
            writer = csv.DictWriter(temporary_file, fieldnames=FIELDNAMES, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def _fingerprint(config: dict[str, Any], questions: list[dict[str, Any]]) -> str:
    payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "config": config,
        "questions": questions,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_checkpoint(
    path: Path,
    fingerprint: str,
    mode_name: str,
    questions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    try:
        checkpoint = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Checkpoint {path} is unreadable. Use --restart to replace it.") from error

    if not isinstance(checkpoint, dict):
        raise RuntimeError(f"Checkpoint {path} is invalid. Use --restart to replace it.")
    if checkpoint.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise RuntimeError(f"Checkpoint {path} has an unsupported schema. Use --restart to replace it.")
    if checkpoint.get("fingerprint") != fingerprint or checkpoint.get("mode") != mode_name:
        raise RuntimeError(
            f"Checkpoint {path} does not match this run. Use --restart to replace it."
        )

    expected_ids = [question["id"] for question in questions]
    if checkpoint.get("question_ids") != expected_ids:
        raise RuntimeError(f"Checkpoint {path} has unexpected question IDs. Use --restart to replace it.")

    rows = checkpoint.get("rows")
    if not isinstance(rows, list):
        raise RuntimeError(f"Checkpoint {path} has invalid rows. Use --restart to replace it.")

    seen_ids: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            raise RuntimeError(f"Checkpoint {path} has an invalid row. Use --restart to replace it.")
        question_id = row.get("question_id")
        if question_id not in expected_ids or question_id in seen_ids:
            raise RuntimeError(f"Checkpoint {path} has invalid question IDs. Use --restart to replace it.")
        if row.get("mode") != mode_name:
            raise RuntimeError(f"Checkpoint {path} has an invalid mode. Use --restart to replace it.")
        if not isinstance(row.get("attempt_count"), int) or row["attempt_count"] < 1:
            raise RuntimeError(f"Checkpoint {path} has invalid retry metadata. Use --restart to replace it.")
        seen_ids.add(question_id)

    return rows


def _checkpoint_data(
    fingerprint: str,
    mode_name: str,
    questions: list[dict[str, Any]],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "fingerprint": fingerprint,
        "mode": mode_name,
        "question_ids": [question["id"] for question in questions],
        "rows": rows,
    }


def _score_result(result: dict[str, Any]) -> dict[str, Any]:
    result["recall_at_5"] = recall_at_k(
        result["retrieved_doc_ids"], result["gold_doc_ids"], k=5
    )
    result["mrr"] = mean_reciprocal_rank(result["retrieved_doc_ids"], result["gold_doc_ids"])
    return result


def _summarize(mode_name: str, rows: list[dict[str, Any]], output_path: Path) -> None:
    avg_recall = sum(row["recall_at_5"] for row in rows) / len(rows)
    avg_mrr = sum(row["mrr"] for row in rows) / len(rows)
    faithfulness_scores = [row["faithfulness"] for row in rows if row["faithfulness"] is not None]
    relevance_scores = [row["answer_relevance"] for row in rows if row["answer_relevance"] is not None]
    avg_faith = sum(faithfulness_scores) / len(faithfulness_scores) if faithfulness_scores else None
    avg_rel = sum(relevance_scores) / len(relevance_scores) if relevance_scores else None
    avg_latency = sum(row["latency_ms"] for row in rows) / len(rows)
    faithfulness_label = f"{avg_faith:.3f}" if avg_faith is not None else "n/a"
    relevance_label = f"{avg_rel:.3f}" if avg_rel is not None else "n/a"
    print(
        f"\n{mode_name}: recall@5={avg_recall:.3f}, mrr={avg_mrr:.3f}, "
        f"faithfulness={faithfulness_label}, relevance={relevance_label}, "
        f"avg_latency={avg_latency:.1f}ms"
    )
    print(f"Wrote per-question results to {output_path}")


def main(
    config_path: Path,
    questions_path: Path,
    output_dir: Path,
    sample_size: int | None = None,
    no_generator: bool = False,
    resume: bool = False,
    restart: bool = False,
    max_retries: int = 3,
    retry_delay: float = 5.0,
) -> None:
    if max_retries < 0:
        raise ValueError("max_retries must be non-negative")
    if retry_delay < 0:
        raise ValueError("retry_delay must be non-negative")

    output_dir.mkdir(parents=True, exist_ok=True)
    config = load_config(config_path)
    if no_generator:
        config.setdefault("generator", {})["enabled"] = False
    mode_name = config["name"]

    print(f"Loading questions from {questions_path} ...")
    with questions_path.open(encoding="utf-8") as f:
        questions = json.load(f)
    if sample_size is not None:
        questions = questions[:sample_size]
    if not questions:
        raise ValueError("No questions selected for evaluation")

    checkpoint_path = output_dir / "checkpoints" / f"{mode_name}.json"
    fingerprint = _fingerprint(config, questions)
    if checkpoint_path.exists() and not restart:
        if not resume:
            raise RuntimeError(
                f"Checkpoint {checkpoint_path} already exists. Use --resume or --restart."
            )
        rows = _load_checkpoint(checkpoint_path, fingerprint, mode_name, questions)
    else:
        rows = []
        _write_checkpoint(
            checkpoint_path,
            _checkpoint_data(fingerprint, mode_name, questions, rows),
        )

    rows_by_id = {row["question_id"]: row for row in rows}
    pending_questions = [question for question in questions if question["id"] not in rows_by_id]
    print(
        f"Running {mode_name} on {len(questions)} questions "
        f"({len(rows_by_id)} complete, {len(pending_questions)} remaining) ..."
    )

    if pending_questions:
        pipeline = EvalPipeline(config)
        for question in tqdm(pending_questions, desc=mode_name):
            for attempt_count in range(1, max_retries + 2):
                try:
                    result = _score_result(pipeline.run(question))
                    result["attempt_count"] = attempt_count
                    break
                except OllamaError as error:
                    if attempt_count > max_retries:
                        raise
                    print(
                        f"Ollama failed for {question['id']} "
                        f"(attempt {attempt_count}/{max_retries + 1}): {error}"
                    )
                    time.sleep(retry_delay)
            rows_by_id[question["id"]] = result
            rows = [
                rows_by_id[question_id]
                for question_id in (q["id"] for q in questions)
                if question_id in rows_by_id
            ]
            _write_checkpoint(
                checkpoint_path,
                _checkpoint_data(fingerprint, mode_name, questions, rows),
            )

    rows = [rows_by_id[question["id"]] for question in questions]
    output_path = output_dir / f"{mode_name}.csv"
    _write_csv(output_path, rows)
    _summarize(mode_name, rows, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--questions", default="data/eval_questions.json")
    parser.add_argument("--output", default="results")
    parser.add_argument("--sample", type=int, default=None, help="Run on first N questions only")
    parser.add_argument("--no-generator", action="store_true", help="Skip generation and judging")
    checkpoint_group = parser.add_mutually_exclusive_group()
    checkpoint_group.add_argument("--resume", action="store_true", help="Resume a matching checkpoint")
    checkpoint_group.add_argument("--restart", action="store_true", help="Replace an existing checkpoint")
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Additional retries for each Ollama failure",
    )
    parser.add_argument(
        "--retry-delay",
        type=float,
        default=5.0,
        help="Seconds to wait between Ollama retries",
    )
    args = parser.parse_args()
    main(
        Path(args.config),
        Path(args.questions),
        Path(args.output),
        args.sample,
        args.no_generator,
        args.resume,
        args.restart,
        args.max_retries,
        args.retry_delay,
    )
