"""Aggregate per-mode CSVs into a final comparison table."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main(result_paths: list[Path], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    frames = [pd.read_csv(p) for p in result_paths]
    combined = pd.concat(frames, ignore_index=True)

    # Build aggregation dict from columns that exist in the CSV.
    agg_map = {
        "recall@5": ("recall_at_5", "mean"),
        "mrr": ("mrr", "mean"),
        "avg_latency_ms": ("latency_ms", "mean"),
    }
    if "faithfulness" in combined.columns:
        agg_map["faithfulness"] = ("faithfulness", "mean")
    if "answer_relevance" in combined.columns:
        agg_map["answer_relevance"] = ("answer_relevance", "mean")

    summary = combined.groupby("mode").agg(**agg_map).reset_index()  # type: ignore[arg-type]
    summary = summary.round(3)
    for metric in ("faithfulness", "answer_relevance"):
        if metric in summary and summary[metric].isna().all():
            summary = summary.drop(columns=metric)

    print("\nComparison Table:\n")
    print(summary.to_markdown(index=False))

    csv_path = output_dir / "comparison_table.csv"
    md_path = output_dir / "comparison_table.md"
    summary.to_csv(csv_path, index=False)
    md_path.write_text(summary.to_markdown(index=False), encoding="utf-8")

    print(f"\nWrote {csv_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs="+")
    parser.add_argument("--output", default="results")
    args = parser.parse_args()
    main([Path(p) for p in args.results], Path(args.output))
