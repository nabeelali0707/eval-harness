"""Tests for the results finalizer."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from finalize_results import (
    END_MARKER,
    START_MARKER,
    build_interpretation,
    finalize,
    require_mode_csvs,
    update_readme,
)


def _write_mode_csv(results_dir: Path, mode: str, faithfulness: float) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "question_id": [f"{mode}_{i}" for i in range(3)],
            "mode": [mode] * 3,
            "recall_at_5": [1.0] * 3,
            "mrr": [0.9] * 3,
            "faithfulness": [faithfulness] * 3,
            "answer_relevance": [0.8] * 3,
            "latency_ms": [100.0] * 3,
        }
    ).to_csv(results_dir / f"{mode}.csv", index=False)


@pytest.fixture()
def results_dir(tmp_path: Path) -> Path:
    path = tmp_path / "results"
    _write_mode_csv(path, "dense_only", 0.40)
    _write_mode_csv(path, "bm25_only", 0.38)
    _write_mode_csv(path, "hybrid_rrf", 0.42)
    _write_mode_csv(path, "hybrid_rerank", 0.50)
    _write_mode_csv(path, "hybrid_rewrite_rerank", 0.52)
    return path


def test_require_mode_csvs_lists_missing(results_dir: Path) -> None:
    (results_dir / "bm25_only.csv").unlink()
    with pytest.raises(FileNotFoundError, match="bm25_only"):
        require_mode_csvs(results_dir)


def test_require_mode_csvs_passes_when_complete(results_dir: Path) -> None:
    paths = require_mode_csvs(results_dir)
    assert [p.name for p in paths] == [
        "dense_only.csv",
        "bm25_only.csv",
        "hybrid_rrf.csv",
        "hybrid_rerank.csv",
        "hybrid_rewrite_rerank.csv",
    ]


def test_interpretation_mentions_rerank_faithfulness_lift() -> None:
    summary = pd.DataFrame(
        {
            "mode": [
                "dense_only",
                "bm25_only",
                "hybrid_rrf",
                "hybrid_rerank",
                "hybrid_rewrite_rerank",
            ],
            "recall@5": [1.0, 0.9, 0.97, 1.0, 1.0],
            "mrr": [0.9, 0.7, 0.86, 0.92, 0.93],
            "faithfulness": [0.40, 0.38, 0.42, 0.50, 0.52],
            "answer_relevance": [0.4, 0.4, 0.4, 0.5, 0.5],
            "avg_latency_ms": [100.0, 50.0, 150.0, 500.0, 5000.0],
        }
    )

    text = build_interpretation(summary)

    assert "recall@5 of 1.000" in text
    assert "from 0.420 (hybrid RRF) to 0.500 (+0.080)" in text
    assert "0.520 (+0.020 vs rerank-only)" in text
    assert "5000" in text and "33.3x" in text


def test_interpretation_handles_missing_llm_columns() -> None:
    summary = pd.DataFrame(
        {
            "mode": ["dense_only", "bm25_only"],
            "recall@5": [1.0, 0.9],
            "mrr": [0.9, 0.7],
            "avg_latency_ms": [100.0, 50.0],
        }
    )

    text = build_interpretation(summary)

    assert "recall@5 of 1.000" in text
    assert "faithfulness" not in text


def test_update_readme_replaces_block_between_markers(tmp_path: Path) -> None:
    readme = tmp_path / "README.md"
    readme.write_text(
        f"# Title\n\n{START_MARKER}\nold content\n{END_MARKER}\n\nfooter\n",
        encoding="utf-8",
    )

    update_readme(readme, "| mode |", "New interpretation.", "Agreement: 0.9")

    text = readme.read_text(encoding="utf-8")
    assert "old content" not in text
    assert "| mode |" in text
    assert "New interpretation." in text
    assert "Agreement: 0.9" in text
    assert text.startswith("# Title\n\n") and "footer\n" in text


def test_update_readme_requires_markers(tmp_path: Path) -> None:
    readme = tmp_path / "README.md"
    readme.write_text("# No markers here\n", encoding="utf-8")

    with pytest.raises(ValueError, match="markers"):
        update_readme(readme, "t", "i", None)


def test_finalize_end_to_end(results_dir: Path, tmp_path: Path) -> None:
    readme = tmp_path / "README.md"
    readme.write_text(
        f"## Current results\n\n{START_MARKER}\n{END_MARKER}\n",
        encoding="utf-8",
    )

    table_path = finalize(results_dir, readme)

    assert table_path.exists()
    assert (results_dir / "comparison_table.csv").exists()
    text = readme.read_text(encoding="utf-8")
    assert "dense_only" in text
    assert "recall@5" in text
    assert "recall@5 of 1.000" in text  # interpretation paragraph present


def test_finalize_reuses_compare_results_logic(
    results_dir: Path, tmp_path: Path, monkeypatch
) -> None:
    called: list[list[Path]] = []

    def fake_compare_main(paths, out_dir):
        called.append(paths)
        out_dir.mkdir(parents=True, exist_ok=True)
        (Path(out_dir) / "comparison_table.md").write_text(
            "| mode | recall@5 |\n|:---|---:|\n| dense_only | 1.0 |", encoding="utf-8"
        )
        pd.DataFrame(
            {
                "mode": ["dense_only"],
                "recall@5": [1.0],
                "mrr": [0.9],
                "avg_latency_ms": [100.0],
            }
        ).to_csv(Path(out_dir) / "comparison_table.csv", index=False)

    monkeypatch.setattr(
        "compare_results.main", fake_compare_main, raising=False
    )
    readme = tmp_path / "README.md"
    readme.write_text(f"{START_MARKER}\n{END_MARKER}", encoding="utf-8")

    finalize(results_dir, readme)

    assert len(called) == 1
