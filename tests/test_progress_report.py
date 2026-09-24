"""Tests for the checkpoint progress reporter."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from progress_report import format_eta, mode_status, report


def _write_checkpoint(
    path: Path,
    mode: str,
    done: int,
    total: int,
    latency_ms: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "mode": mode,
                "question_ids": [f"q{i:04d}" for i in range(total)],
                "rows": [
                    {
                        "question_id": f"q{i:04d}",
                        "mode": mode,
                        "latency_ms": latency_ms,
                    }
                    for i in range(done)
                ],
            }
        ),
        encoding="utf-8",
    )


def test_mode_status_computes_remaining_and_eta(tmp_path: Path) -> None:
    path = tmp_path / "checkpoints" / "dense_only.json"
    _write_checkpoint(path, "dense_only", done=10, total=150, latency_ms=1_200_000)

    status = mode_status(path)

    assert status["mode"] == "dense_only"
    assert status["done"] == 10
    assert status["total"] == 150
    assert status["remaining"] == 140
    assert status["median_ms"] == 1_200_000
    assert status["eta_minutes"] == pytest.approx(140 * 1_200_000 / 60_000)


def test_mode_status_handles_empty_checkpoint(tmp_path: Path) -> None:
    path = tmp_path / "checkpoints" / "bm25_only.json"
    _write_checkpoint(path, "bm25_only", done=0, total=150, latency_ms=0)

    status = mode_status(path)

    assert status["done"] == 0
    assert status["median_ms"] is None
    assert status["eta_minutes"] is None


def test_format_eta_buckets() -> None:
    assert format_eta(None) == "n/a"
    assert format_eta(45) == "~45m"
    assert format_eta(90) == "~1.5h"
    assert format_eta(60 * 72) == "~3.0d"


def test_report_marks_completed_csvs(tmp_path: Path, capsys) -> None:
    checkpoint = tmp_path / "checkpoints" / "dense_only.json"
    _write_checkpoint(checkpoint, "dense_only", done=150, total=150, latency_ms=60_000)
    (tmp_path / "dense_only.csv").write_text("question_id\n", encoding="utf-8")

    statuses = report(tmp_path)

    assert len(statuses) == 1
    out = capsys.readouterr().out
    assert "complete" in out
    assert "Completed CSVs: dense_only" in out


def test_report_empty_dir(tmp_path: Path, capsys) -> None:
    statuses = report(tmp_path)

    assert statuses == []
    assert "No checkpoints" in capsys.readouterr().out


def test_report_lists_not_started_ablation_modes(tmp_path: Path, capsys) -> None:
    checkpoint = tmp_path / "checkpoints" / "dense_only.json"
    _write_checkpoint(checkpoint, "dense_only", done=10, total=150, latency_ms=60_000)

    report(tmp_path)

    out = capsys.readouterr().out
    assert "No checkpoints yet for: bm25_only, hybrid_rrf, hybrid_rerank, hybrid_rewrite_rerank" in out
