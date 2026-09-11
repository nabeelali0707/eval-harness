"""Remove machine-sleep outlier rows from an evaluation checkpoint."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from run_eval import _atomic_write_text


def prune_checkpoint(path: Path, max_latency_ms: float) -> list[str]:
    """Drop rows whose latency exceeds max_latency_ms and rewrite the checkpoint."""
    checkpoint: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    rows = checkpoint.get("rows", [])
    if not isinstance(rows, list):
        raise ValueError(f"Checkpoint {path} has invalid rows.")

    kept: list[dict[str, Any]] = []
    pruned_ids: list[str] = []
    for row in rows:
        latency = row.get("latency_ms")
        if isinstance(latency, (int, float)) and latency > max_latency_ms:
            pruned_ids.append(row["question_id"])
        else:
            kept.append(row)

    if pruned_ids:
        checkpoint["rows"] = kept
        _atomic_write_text(path, json.dumps(checkpoint, ensure_ascii=False, indent=2) + "\n")
    return pruned_ids


def main(checkpoint_path: Path, max_latency_ms: float) -> None:
    pruned_ids = prune_checkpoint(checkpoint_path, max_latency_ms)
    if pruned_ids:
        print(f"Pruned {len(pruned_ids)} outlier rows from {checkpoint_path}:")
        for question_id in pruned_ids:
            print(f"  {question_id}")
        print("Re-run the same mode with --resume to regenerate them.")
    else:
        print(f"No outlier rows in {checkpoint_path}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument(
        "--max-latency-ms",
        type=float,
        default=1_800_000.0,
        help="Remove rows whose total latency exceeds this (default: 30 minutes)",
    )
    args = parser.parse_args()
    main(args.checkpoint, args.max_latency_ms)
