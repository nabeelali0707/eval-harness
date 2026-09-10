"""Tests for run_eval checkpointing, retries, and CSV output."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pytest

import run_eval
from src.ollama_client import OllamaError


def result_for(question: dict[str, Any], mode: str) -> dict[str, Any]:
    return {
        "question_id": question["id"],
        "mode": mode,
        "retrieved_doc_ids": ["doc_gold"],
        "gold_doc_ids": ["doc_gold"],
        "faithfulness": 0.0,
        "answer_relevance": 0.0,
        "latency_ms": 12.5,
        "rewrite_ms": 4.25,
        "retrieve_ms": 8.25,
        "generated_answer": "",
    }


def write_inputs(tmp_path: Path) -> tuple[Path, Path]:
    config_path = tmp_path / "config.yaml"
    questions_path = tmp_path / "questions.json"
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
    return config_path, questions_path


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_main_no_generator_overrides_config_and_writes_rewrite_latency(
    monkeypatch, tmp_path: Path
):
    config_path, questions_path = write_inputs(tmp_path)
    output_dir = tmp_path / "results"
    configs: list[dict[str, Any]] = []

    class FakePipeline:
        def __init__(self, config: dict[str, Any]) -> None:
            configs.append(config)
            self.config = config

        def run(self, question: dict[str, Any]) -> dict[str, Any]:
            return result_for(question, self.config["name"])

    monkeypatch.setattr(
        run_eval,
        "load_config",
        lambda _: {"name": "test_mode", "generator": {"enabled": True}},
    )
    monkeypatch.setattr(run_eval, "EvalPipeline", FakePipeline)

    run_eval.main(
        config_path,
        questions_path,
        output_dir,
        sample_size=1,
        no_generator=True,
    )

    assert configs == [{"name": "test_mode", "generator": {"enabled": False}}]
    rows = read_rows(output_dir / "test_mode.csv")
    assert len(rows) == 1
    assert rows[0]["rewrite_ms"] == "4.25"
    assert rows[0]["recall_at_5"] == "1.0"
    assert rows[0]["mrr"] == "1.0"
    assert rows[0]["attempt_count"] == "1"


def test_resume_skips_checkpointed_questions_after_interruption(monkeypatch, tmp_path: Path):
    config_path, questions_path = write_inputs(tmp_path)
    output_dir = tmp_path / "results"
    interrupted_calls: list[str] = []

    class InterruptingPipeline:
        def __init__(self, config: dict[str, Any]) -> None:
            self.config = config

        def run(self, question: dict[str, Any]) -> dict[str, Any]:
            interrupted_calls.append(question["id"])
            if question["id"] == "q2":
                raise OllamaError("temporary failure")
            return result_for(question, self.config["name"])

    monkeypatch.setattr(run_eval, "EvalPipeline", InterruptingPipeline)
    with pytest.raises(OllamaError, match="temporary failure"):
        run_eval.main(
            config_path,
            questions_path,
            output_dir,
            resume=True,
            max_retries=0,
            retry_delay=0,
        )

    checkpoint_path = output_dir / "checkpoints" / "test_mode.json"
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    assert [row["question_id"] for row in checkpoint["rows"]] == ["q1"]
    assert not (output_dir / "test_mode.csv").exists()
    assert interrupted_calls == ["q1", "q2"]

    resumed_calls: list[str] = []

    class ResumingPipeline:
        def __init__(self, config: dict[str, Any]) -> None:
            self.config = config

        def run(self, question: dict[str, Any]) -> dict[str, Any]:
            resumed_calls.append(question["id"])
            return result_for(question, self.config["name"])

    monkeypatch.setattr(run_eval, "EvalPipeline", ResumingPipeline)
    run_eval.main(
        config_path,
        questions_path,
        output_dir,
        resume=True,
        retry_delay=0,
    )

    rows = read_rows(output_dir / "test_mode.csv")
    assert [row["question_id"] for row in rows] == ["q1", "q2"]
    assert resumed_calls == ["q2"]


def test_resume_rejects_mismatched_checkpoint_and_restart_replaces_it(monkeypatch, tmp_path: Path):
    config_path, questions_path = write_inputs(tmp_path)
    output_dir = tmp_path / "results"
    calls: list[str] = []

    class FakePipeline:
        def __init__(self, config: dict[str, Any]) -> None:
            self.config = config

        def run(self, question: dict[str, Any]) -> dict[str, Any]:
            calls.append(question["id"])
            return result_for(question, self.config["name"])

    monkeypatch.setattr(run_eval, "EvalPipeline", FakePipeline)
    run_eval.main(config_path, questions_path, output_dir, sample_size=1, resume=True)

    with pytest.raises(RuntimeError, match="does not match this run"):
        run_eval.main(config_path, questions_path, output_dir, resume=True)

    calls.clear()
    run_eval.main(config_path, questions_path, output_dir, restart=True)
    assert calls == ["q1", "q2"]
    assert len(read_rows(output_dir / "test_mode.csv")) == 2


def test_retries_ollama_failures_and_records_successful_attempt(monkeypatch, tmp_path: Path):
    config_path, questions_path = write_inputs(tmp_path)
    output_dir = tmp_path / "results"
    attempts = 0

    class FlakyPipeline:
        def __init__(self, config: dict[str, Any]) -> None:
            self.config = config

        def run(self, question: dict[str, Any]) -> dict[str, Any]:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise OllamaError("temporary failure")
            return result_for(question, self.config["name"])

    monkeypatch.setattr(run_eval, "EvalPipeline", FlakyPipeline)
    run_eval.main(
        config_path,
        questions_path,
        output_dir,
        sample_size=1,
        max_retries=1,
        retry_delay=0,
    )

    assert attempts == 2
    assert read_rows(output_dir / "test_mode.csv")[0]["attempt_count"] == "2"


def test_failed_run_preserves_existing_completed_csv(monkeypatch, tmp_path: Path):
    config_path, questions_path = write_inputs(tmp_path)
    output_dir = tmp_path / "results"
    output_dir.mkdir()
    output_path = output_dir / "test_mode.csv"
    output_path.write_text("previous completed output\n", encoding="utf-8")

    class FailingPipeline:
        def __init__(self, config: dict[str, Any]) -> None:
            self.config = config

        def run(self, question: dict[str, Any]) -> dict[str, Any]:
            raise OllamaError("unavailable")

    monkeypatch.setattr(run_eval, "EvalPipeline", FailingPipeline)
    with pytest.raises(OllamaError, match="unavailable"):
        run_eval.main(
            config_path,
            questions_path,
            output_dir,
            max_retries=0,
            retry_delay=0,
        )

    assert output_path.read_text(encoding="utf-8") == "previous completed output\n"
