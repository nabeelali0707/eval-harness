"""Tests for the sequential ablation runner."""
from __future__ import annotations

from pathlib import Path

import pytest

from run_ablation import MODE_ORDER, pending_modes, run_ablation


def test_pending_modes_skips_completed_csvs(tmp_path: Path) -> None:
    (tmp_path / "dense_only.csv").write_text("question_id\n", encoding="utf-8")
    (tmp_path / "hybrid_rrf.csv").write_text("question_id\n", encoding="utf-8")

    pending = pending_modes(tmp_path, MODE_ORDER)

    assert pending == ["bm25_only", "hybrid_rerank", "hybrid_rewrite_rerank"]


def test_run_ablation_skips_existing_and_runs_remaining(
    tmp_path: Path, monkeypatch
) -> None:
    (tmp_path / "bm25_only.csv").write_text("question_id\n", encoding="utf-8")
    called: list[str] = []

    def fake_main(*args, **kwargs) -> None:
        config_path = args[0]
        (tmp_path / f"{config_path.stem}.csv").write_text(
            "question_id\n", encoding="utf-8"
        )
        called.append(config_path.stem)

    monkeypatch.setattr("run_eval.main", fake_main)

    completed = run_ablation(MODE_ORDER, tmp_path)

    # bm25_only skipped via existing CSV; every other mode ran to completion.
    assert called == [
        "dense_only",
        "hybrid_rrf",
        "hybrid_rerank",
        "hybrid_rewrite_rerank",
    ]
    assert completed == MODE_ORDER


def test_run_ablation_stops_on_ollama_error(tmp_path: Path, monkeypatch) -> None:
    from src.ollama_client import OllamaError

    def fake_main(*args, **kwargs) -> None:
        raise OllamaError("cannot reach Ollama")

    monkeypatch.setattr("run_eval.main", fake_main)

    completed = run_ablation(["dense_only", "bm25_only"], tmp_path)

    assert completed == []


def test_run_ablation_passes_no_generator_flag(tmp_path: Path, monkeypatch) -> None:
    captured: dict = {}

    def fake_main(*args, **kwargs) -> None:
        captured["no_generator"] = args[4]

    monkeypatch.setattr("run_eval.main", fake_main)

    run_ablation(["dense_only"], tmp_path, no_generator=True)

    assert captured["no_generator"] is True


def test_mode_order_matches_configs() -> None:
    for mode in MODE_ORDER:
        assert Path("configs") / f"{mode}.yaml"


@pytest.mark.parametrize("mode", MODE_ORDER)
def test_pending_modes_returns_all_when_dir_empty(tmp_path: Path, mode: str) -> None:
    assert pending_modes(tmp_path, [mode]) == [mode]
