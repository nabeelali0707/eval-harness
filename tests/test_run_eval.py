"""Tests for run_eval CLI behavior and CSV output."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import run_eval


class FakePipeline:
    configs: list[dict[str, Any]] = []

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.configs.append(config)

    def run(self, question: dict[str, Any]) -> dict[str, Any]:
        return {
            "question_id": question["id"],
            "mode": self.config["name"],
            "retrieved_doc_ids": ["doc_gold"],
            "gold_doc_ids": ["doc_gold"],
            "faithfulness": 0.0,
            "answer_relevance": 0.0,
            "latency_ms": 12.5,
            "rewrite_ms": 4.25,
            "retrieve_ms": 8.25,
            "generated_answer": "",
        }


def test_main_no_generator_overrides_config_and_writes_rewrite_latency(
    monkeypatch, tmp_path: Path
):
    config_path = tmp_path / "config.yaml"
    questions_path = tmp_path / "questions.json"
    output_dir = tmp_path / "results"
    config_path.write_text("name: test_mode\n", encoding="utf-8")
    questions_path.write_text(
        json.dumps(
            [
                {"id": "q1", "question": "Question 1", "gold_doc_ids": ["doc_gold"]},
                {"id": "q2", "question": "Question 2", "gold_doc_ids": ["doc_gold"]},
            ]
        ),
        encoding="utf-8",
    )

    FakePipeline.configs = []
    monkeypatch.setattr(
        run_eval,
        "load_config",
        lambda _: {"name": "test_mode", "generator": {"enabled": True}},
    )
    monkeypatch.setattr(run_eval, "EvalPipeline", FakePipeline)

    run_eval.main(config_path, questions_path, output_dir, sample_size=1, no_generator=True)

    assert FakePipeline.configs == [{"name": "test_mode", "generator": {"enabled": False}}]
    with (output_dir / "test_mode.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 1
    assert rows[0]["rewrite_ms"] == "4.25"
    assert rows[0]["recall_at_5"] == "1.0"
    assert rows[0]["mrr"] == "1.0"
