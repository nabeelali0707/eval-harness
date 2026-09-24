"""Report per-mode checkpoint completion and estimated remaining runtime.

Reads the atomic per-question checkpoints written by ``run_eval.py`` and
prints, for each mode: questions completed, median per-question latency
(robust to sleep outliers), and an ETA for the remaining questions. A mode
counts as finished once its CSV exists.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

ABLATION_MODES = [
    "dense_only",
    "bm25_only",
    "hybrid_rrf",
    "hybrid_rerank",
    "hybrid_rewrite_rerank",
]


def mode_status(checkpoint_path: Path) -> dict[str, Any]:
    """Summarize one mode's checkpoint: progress and latency-based ETA."""
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    question_ids = checkpoint.get("question_ids", [])
    total = len(question_ids) if isinstance(question_ids, list) else 0
    rows = checkpoint.get("rows", [])
    if not isinstance(rows, list):
        raise ValueError(f"Checkpoint {checkpoint_path} has invalid rows.")

    latencies = sorted(
        row["latency_ms"]
        for row in rows
        if isinstance(row.get("latency_ms"), (int, float))
    )
    median_ms = statistics.median(latencies) if latencies else None
    remaining = max(total - len(rows), 0)
    eta_minutes = (
        remaining * median_ms / 60_000 if median_ms is not None and total else None
    )

    return {
        "mode": checkpoint.get("mode", checkpoint_path.stem),
        "done": len(rows),
        "total": total,
        "median_ms": median_ms,
        "remaining": remaining,
        "eta_minutes": eta_minutes,
    }


def format_eta(minutes: float | None) -> str:
    """Render an ETA in minutes, hours, or days."""
    if minutes is None:
        return "n/a"
    if minutes < 60:
        return f"~{minutes:.0f}m"
    if minutes < 60 * 48:
        return f"~{minutes / 60:.1f}h"
    return f"~{minutes / 60 / 24:.1f}d"


def report(output_dir: Path) -> list[dict[str, Any]]:
    """Print one line per mode checkpoint and return the statuses."""
    checkpoint_paths = sorted((output_dir / "checkpoints").glob("*.json"))
    if not checkpoint_paths:
        print(f"No checkpoints found in {output_dir / 'checkpoints'}")
        return []

    print(f"{'mode':<24}{'done':>10}{'median/q':>12}{'eta':>10}")
    statuses = []
    for path in checkpoint_paths:
        status = mode_status(path)
        statuses.append(status)
        complete = (output_dir / f"{status['mode']}.csv").exists()
        progress = (
            "complete"
            if complete
            else f"{status['done']}/{status['total']}"
        )
        median_label = (
            f"{status['median_ms'] / 1000:.0f}s"
            if status["median_ms"] is not None
            else "n/a"
        )
        print(
            f"{status['mode']:<24}{progress:>10}{median_label:>12}"
            f"{format_eta(status['eta_minutes']):>10}"
        )

    pending = [s for s in statuses if not (output_dir / f"{s['mode']}.csv").exists()]
    if pending and all(s["eta_minutes"] is not None for s in pending):
        total_eta = sum(s["eta_minutes"] for s in pending)
        print(f"\nEstimated remaining across unfinished modes: {format_eta(total_eta)}")
    finished = [s["mode"] for s in statuses if (output_dir / f"{s['mode']}.csv").exists()]
    if finished:
        print(f"Completed CSVs: {', '.join(finished)}")
    not_started = [
        mode
        for mode in ABLATION_MODES
        if not (output_dir / "checkpoints" / f"{mode}.json").exists()
    ]
    if not_started:
        print(f"No checkpoints yet for: {', '.join(not_started)}")
    return statuses


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="results/final_local_ollama_150q",
        help="Output directory shared by the ablation modes",
    )
    args = parser.parse_args()
    report(Path(args.output))


if __name__ == "__main__":
    main()
