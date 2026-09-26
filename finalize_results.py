"""Build the final comparison table and splice the results into the README.

Runs after all five mode CSVs exist: aggregates them via the same logic as
``compare_results.py``, computes an interpretation paragraph from the table
(recall deltas, rerank/rewrite faithfulness lifts, latency ratios), and
replaces the README block between the RESULTS markers. Idempotent: safe to
rerun with refreshed numbers at any time.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ABLATION_MODES = [
    "dense_only",
    "bm25_only",
    "hybrid_rrf",
    "hybrid_rerank",
    "hybrid_rewrite_rerank",
]

START_MARKER = "<!-- RESULTS:START -->"
END_MARKER = "<!-- RESULTS:END -->"


def require_mode_csvs(results_dir: Path) -> list[Path]:
    """Return the five mode CSV paths or fail loudly with what is missing."""
    missing = [
        mode for mode in ABLATION_MODES if not (results_dir / f"{mode}.csv").exists()
    ]
    if missing:
        raise FileNotFoundError(
            f"Missing completed mode CSVs in {results_dir}: {', '.join(missing)}. "
            "Resume the ablation (python run_ablation.py) until all five exist."
        )
    return [results_dir / f"{mode}.csv" for mode in ABLATION_MODES]


def _fmt(value: object, spec: str = ".3f") -> str:
    if value is None:
        return "n/a"
    try:
        if pd.isna(value):  # type: ignore[arg-type]
            return "n/a"
    except TypeError:
        pass
    return format(value, spec)


def build_interpretation(summary: pd.DataFrame) -> str:
    """Write the plain-English paragraph from the aggregated mode table."""
    rows = summary.set_index("mode")

    def val(mode: str, column: str) -> float | None:
        try:
            value = rows.loc[mode, column]
        except KeyError:
            return None
        return None if pd.isna(value) else float(value)

    sentences: list[str] = []

    dense_recall = val("dense_only", "recall@5")
    hybrid_recall = val("hybrid_rrf", "recall@5")
    if dense_recall is not None:
        sentence = (
            f"On this 150-question HotpotQA sample, dense retrieval alone already "
            f"reaches recall@5 of {_fmt(dense_recall)}"
        )
        if hybrid_recall is not None:
            sentence += (
                f", leaving no recall headroom for hybrid RRF fusion "
                f"({_fmt(hybrid_recall)})"
            )
        sentences.append(sentence + ".")

    faith_hybrid = val("hybrid_rrf", "faithfulness")
    faith_rerank = val("hybrid_rerank", "faithfulness")
    if faith_hybrid is not None and faith_rerank is not None:
        sentences.append(
            f"Reranking shows up in generation quality instead: faithfulness moves "
            f"from {_fmt(faith_hybrid)} (hybrid RRF) to {_fmt(faith_rerank)} "
            f"({faith_rerank - faith_hybrid:+.3f})."
        )

    faith_rewrite = val("hybrid_rewrite_rerank", "faithfulness")
    if faith_rerank is not None and faith_rewrite is not None:
        sentences.append(
            f"Multi-query rewriting then shifts faithfulness to {_fmt(faith_rewrite)} "
            f"({faith_rewrite - faith_rerank:+.3f} vs rerank-only)."
        )

    latency_hybrid = val("hybrid_rrf", "avg_latency_ms")
    latency_rewrite = val("hybrid_rewrite_rerank", "avg_latency_ms")
    if latency_hybrid and latency_rewrite:
        sentences.append(
            f"The full rewrite+rerank route costs {_fmt(latency_rewrite, '.0f')}ms per "
            f"question versus {_fmt(latency_hybrid, '.0f')}ms for plain hybrid "
            f"({latency_rewrite / latency_hybrid:.1f}x)."
        )

    sentences.append(
        "Faithfulness and answer relevance are LLM-judge scores produced by the same "
        "local generator used for answers, so treat differences below 0.05 as noise "
        "at n=150; recall and MRR are exact retrieval metrics."
    )
    return " ".join(sentences)


def update_readme(
    readme_path: Path,
    table_markdown: str,
    interpretation: str,
    agreement_markdown: str | None,
) -> None:
    """Replace the README block between the RESULTS markers (idempotent)."""
    text = readme_path.read_text(encoding="utf-8")
    if START_MARKER not in text or END_MARKER not in text:
        raise ValueError(
            f"{readme_path} must contain {START_MARKER} and {END_MARKER} markers "
            "around the results section."
        )
    before, rest = text.split(START_MARKER, 1)
    _, after = rest.split(END_MARKER, 1)

    block = f"\n{table_markdown}\n\n{interpretation}\n"
    if agreement_markdown:
        block += f"\n{agreement_markdown}\n"

    readme_path.write_text(before + START_MARKER + block + END_MARKER + after, encoding="utf-8")


def finalize(results_dir: Path, readme_path: Path) -> Path:
    """Aggregate mode CSVs, write the comparison table, and update the README."""
    from compare_results import main as compare_main

    result_paths = require_mode_csvs(results_dir)
    compare_main(result_paths, results_dir)

    table_path = results_dir / "comparison_table.md"
    table_markdown = table_path.read_text(encoding="utf-8").strip()
    summary = pd.read_csv(results_dir / "comparison_table.csv")

    agreement_path = results_dir / "judge_agreement.md"
    agreement_markdown = (
        agreement_path.read_text(encoding="utf-8").strip()
        if agreement_path.exists()
        else None
    )

    update_readme(readme_path, table_markdown, build_interpretation(summary), agreement_markdown)
    return table_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="results/final_local_ollama_150q")
    parser.add_argument("--readme", default="README.md")
    args = parser.parse_args()

    table_path = finalize(Path(args.results), Path(args.readme))
    print(f"Comparison table written to {table_path}")
    print(f"README results section updated between {START_MARKER} / {END_MARKER}.")


if __name__ == "__main__":
    main()
