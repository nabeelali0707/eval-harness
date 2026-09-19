"""Export a hand-labeling worksheet and score human vs LLM-judge agreement.

Export picks a reproducible random sample of judged rows from a checkpoint,
joins the retrieved passage texts, and writes JSONL with empty
``human_faithfulness`` / ``human_relevance`` fields. Fill those in (0.0-1.0)
and run the ``score`` subcommand to get agreement statistics for the README.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path
from typing import Any

WORKSHEET_FIELDS = [
    "question_id",
    "question",
    "expected_answer",
    "generated_answer",
    "retrieved_doc_ids",
    "passages",
    "judge_faithfulness",
    "judge_answer_relevance",
    "human_faithfulness",
    "human_relevance",
]


def _load_corpus_texts(corpus_path: Path) -> dict[str, str]:
    with corpus_path.open(encoding="utf-8") as f:
        corpus = json.load(f)
    return {doc["doc_id"]: doc["text"] for doc in corpus}


def export_worksheet(
    checkpoint_path: Path,
    output_path: Path,
    sample_size: int,
    seed: int,
    corpus_path: Path = Path("data/corpus.json"),
) -> int:
    """Sample judged rows from a checkpoint and write a labeling worksheet."""
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    rows = [row for row in checkpoint.get("rows", []) if row.get("faithfulness") is not None]
    if not rows:
        raise ValueError(f"Checkpoint {checkpoint_path} has no judged rows to label.")

    rng = random.Random(seed)
    sample = sorted(rng.sample(rows, min(sample_size, len(rows))), key=lambda r: r["question_id"])

    corpus_texts = _load_corpus_texts(corpus_path)
    records: list[dict[str, Any]] = []
    for row in sample:
        records.append(
            {
                "question_id": row["question_id"],
                "question": row["question"],
                "expected_answer": row.get("expected_answer", ""),
                "generated_answer": row.get("generated_answer", ""),
                "retrieved_doc_ids": row.get("retrieved_doc_ids", []),
                "passages": [
                    corpus_texts[doc_id]
                    for doc_id in row.get("retrieved_doc_ids", [])
                    if doc_id in corpus_texts
                ],
                "judge_faithfulness": row.get("faithfulness"),
                "judge_answer_relevance": row.get("answer_relevance"),
                "human_faithfulness": None,
                "human_relevance": None,
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Wrote {len(records)} rows to {output_path}")
    print("Fill human_faithfulness / human_relevance (0.0-1.0), then run:")
    print(f"  python label_judge_sample.py score {output_path}")
    return len(records)


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(set(xs)) == 1 or len(set(ys)) == 1:
        return None
    mean_x, mean_y = statistics.mean(xs), statistics.mean(ys)
    covariance = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    variance_x = sum((x - mean_x) ** 2 for x in xs)
    variance_y = sum((y - mean_y) ** 2 for y in ys)
    denominator = (variance_x * variance_y) ** 0.5
    return covariance / denominator if denominator else None


def score_worksheet(worksheet_path: Path, tolerance: float = 0.5) -> dict[str, Any]:
    """Compare human labels against judge scores and print a summary block."""
    records = [
        json.loads(line)
        for line in worksheet_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    labeled = [
        r
        for r in records
        if isinstance(r.get("human_faithfulness"), (int, float))
        and isinstance(r.get("human_relevance"), (int, float))
        and r.get("judge_faithfulness") is not None
    ]
    if not labeled:
        raise ValueError(
            f"{worksheet_path} has no fully labeled rows with judge scores."
        )

    metrics: dict[str, dict[str, float | int | None]] = {}
    for human_key, judge_key, name in [
        ("human_faithfulness", "judge_faithfulness", "faithfulness"),
        ("human_relevance", "judge_answer_relevance", "answer_relevance"),
    ]:
        human = [float(r[human_key]) for r in labeled]
        judge = [float(r[judge_key]) for r in labeled]
        metrics[name] = {
            "n": len(labeled),
            "mae": statistics.mean(abs(h - j) for h, j in zip(human, judge)),
            "agreement_rate": sum(
                1 for h, j in zip(human, judge) if abs(h - j) <= tolerance
            )
            / len(labeled),
            "pearson": _pearson(human, judge),
        }

    print(f"\nJudge agreement ({len(labeled)} labeled rows, tolerance ±{tolerance}):\n")
    for name, values in metrics.items():
        pearson = values["pearson"]
        pearson_label = f"{pearson:.3f}" if pearson is not None else "n/a"
        print(
            f"  {name}: MAE={values['mae']:.3f}, "
            f"agreement={values['agreement_rate']:.1%}, r={pearson_label}"
        )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    export_parser = subparsers.add_parser("export", help="Create a labeling worksheet")
    export_parser.add_argument("checkpoint", type=Path)
    export_parser.add_argument("--output", type=Path, required=True)
    export_parser.add_argument("--n", type=int, default=50)
    export_parser.add_argument("--seed", type=int, default=42)
    export_parser.add_argument("--corpus", type=Path, default=Path("data/corpus.json"))

    score_parser = subparsers.add_parser("score", help="Score a filled worksheet")
    score_parser.add_argument("worksheet", type=Path)
    score_parser.add_argument("--tolerance", type=float, default=0.5)

    args = parser.parse_args()
    if args.command == "export":
        export_worksheet(
            args.checkpoint,
            args.output,
            args.n,
            args.seed,
            args.corpus,
        )
    else:
        score_worksheet(args.worksheet, args.tolerance)


if __name__ == "__main__":
    main()
