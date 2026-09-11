"""Tests for checkpoint outlier pruning."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import prune_sleep_outliers


def make_row(question_id: str, latency_ms: float) -> dict[str, Any]:
    return {
        "question_id": question_id,
        "mode": "test_mode",
        "latency_ms": latency_ms,
        "attempt_count": 1,
    }


def write_checkpoint(path: Path, rows: list[dict[str, Any]]) -> None:
    checkpoint = {
        "schema_version": 1,
        "fingerprint": "abc",
        "mode": "test_mode",
        "question_ids": ["q1", "q2", "q3"],
        "rows": rows,
    }
    path.write_text(json.dumps(checkpoint), encoding="utf-8")


def test_prune_removes_only_outlier_rows(tmp_path: Path):
    path = tmp_path / "test_mode.json"
    write_checkpoint(
        path,
        [make_row("q1", 150_000.0), make_row("q2", 41_650_891.0), make_row("q3", 200_000.0)],
    )

    pruned = prune_sleep_outliers.prune_checkpoint(path, 1_800_000.0)

    assert pruned == ["q2"]
    checkpoint = json.loads(path.read_text(encoding="utf-8"))
    assert [row["question_id"] for row in checkpoint["rows"]] == ["q1", "q3"]
    assert checkpoint["fingerprint"] == "abc"


def test_prune_is_noop_without_outliers(tmp_path: Path):
    path = tmp_path / "test_mode.json"
    write_checkpoint(path, [make_row("q1", 150_000.0)])
    before = path.read_text(encoding="utf-8")

    pruned = prune_sleep_outliers.prune_checkpoint(path, 1_800_000.0)

    assert pruned == []
    assert path.read_text(encoding="utf-8") == before


def test_prune_keeps_rows_without_numeric_latency(tmp_path: Path):
    path = tmp_path / "test_mode.json"
    row = make_row("q1", 150_000.0)
    row["latency_ms"] = None
    write_checkpoint(path, [row])

    pruned = prune_sleep_outliers.prune_checkpoint(path, 1_800_000.0)

    assert pruned == []
    checkpoint = json.loads(path.read_text(encoding="utf-8"))
    assert len(checkpoint["rows"]) == 1
