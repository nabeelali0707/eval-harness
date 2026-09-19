"""Tests for the judge hand-labeling worksheet tool."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from label_judge_sample import export_worksheet, score_worksheet


def _write_checkpoint(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"schema_version": 1, "mode": "dense_only", "rows": rows}),
        encoding="utf-8",
    )


@pytest.fixture()
def corpus(tmp_path: Path) -> Path:
    corpus_path = tmp_path / "corpus.json"
    corpus_path.write_text(
        json.dumps(
            [
                {"doc_id": "a", "text": "Passage A text."},
                {"doc_id": "b", "text": "Passage B text."},
            ]
        ),
        encoding="utf-8",
    )
    return corpus_path


@pytest.fixture()
def checkpoint(tmp_path: Path) -> Path:
    path = tmp_path / "checkpoints" / "dense_only.json"
    rows = [
        {
            "question_id": f"hotpotqa_{i:04d}",
            "question": f"Question {i}?",
            "expected_answer": f"answer {i}",
            "generated_answer": f"Answer {i}.",
            "retrieved_doc_ids": ["a", "b"],
            "faithfulness": 0.5,
            "answer_relevance": 0.75,
        }
        for i in range(10)
    ]
    _write_checkpoint(path, rows)
    return path


def test_export_samples_and_joins_passages(
    tmp_path: Path, checkpoint: Path, corpus: Path
) -> None:
    output = tmp_path / "worksheet.jsonl"

    count = export_worksheet(checkpoint, output, sample_size=5, seed=42, corpus_path=corpus)

    assert count == 5
    records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 5
    assert records[0]["question_id"] < records[-1]["question_id"]  # sorted
    for record in records:
        assert record["passages"] == ["Passage A text.", "Passage B text."]
        assert record["judge_faithfulness"] == 0.5
        assert record["human_faithfulness"] is None
        assert record["human_relevance"] is None


def test_export_is_reproducible_for_seed(
    tmp_path: Path, checkpoint: Path, corpus: Path
) -> None:
    out1 = tmp_path / "w1.jsonl"
    out2 = tmp_path / "w2.jsonl"
    export_worksheet(checkpoint, out1, 5, 42, corpus)
    export_worksheet(checkpoint, out2, 5, 42, corpus)
    assert out1.read_text(encoding="utf-8") == out2.read_text(encoding="utf-8")


def test_export_rejects_checkpoint_with_no_judged_rows(tmp_path: Path, corpus: Path) -> None:
    path = tmp_path / "cp.json"
    _write_checkpoint(path, [{"question_id": "x", "faithfulness": None}])
    with pytest.raises(ValueError, match="no judged rows"):
        export_worksheet(path, tmp_path / "w.jsonl", 5, 42, corpus)


def _fill_worksheet(path: Path) -> None:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    for i, record in enumerate(records):
        record["human_faithfulness"] = min(1.0, record["judge_faithfulness"] + 0.1)
        record["human_relevance"] = max(0.0, record["judge_answer_relevance"] - 0.1)
        path.write_text(
            "\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8"
        )


def test_score_computes_agreement(tmp_path: Path, checkpoint: Path, corpus: Path) -> None:
    worksheet = tmp_path / "w.jsonl"
    export_worksheet(checkpoint, worksheet, 10, 42, corpus)

    records = [json.loads(line) for line in worksheet.read_text(encoding="utf-8").splitlines()]
    for record in records:
        record["human_faithfulness"] = record["judge_faithfulness"]  # perfect match
        record["human_relevance"] = record["judge_answer_relevance"]
    worksheet.write_text(
        "\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8"
    )

    metrics = score_worksheet(worksheet, tolerance=0.5)

    assert metrics["faithfulness"]["n"] == 10
    assert metrics["faithfulness"]["mae"] == 0.0
    assert metrics["faithfulness"]["agreement_rate"] == 1.0


def test_score_skips_unlabeled_rows(tmp_path: Path, checkpoint: Path, corpus: Path) -> None:
    worksheet = tmp_path / "w.jsonl"
    export_worksheet(checkpoint, worksheet, 10, 42, corpus)

    records = [json.loads(line) for line in worksheet.read_text(encoding="utf-8").splitlines()]
    for i, record in enumerate(records):
        if i < 4:  # leave half unlabeled
            record["human_faithfulness"] = record["judge_faithfulness"]
            record["human_relevance"] = record["judge_answer_relevance"]
    worksheet.write_text(
        "\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8"
    )

    metrics = score_worksheet(worksheet)

    assert metrics["faithfulness"]["n"] == 4


def test_score_rejects_empty_worksheet(tmp_path: Path) -> None:
    path = tmp_path / "w.jsonl"
    path.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="no fully labeled"):
        score_worksheet(path)
