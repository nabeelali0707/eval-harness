"""Unit tests for parsing Claude judge responses without API calls."""
from __future__ import annotations

from src.judge import ClaudeJudge


def parse_scores(text: str) -> dict[str, float]:
    judge = ClaudeJudge.__new__(ClaudeJudge)
    return judge._parse_scores(text)


def test_judge_parses_fenced_json_and_clamps_scores():
    scores = parse_scores(
        "```json\n"
        '{"faithfulness": 1.25, "answer_relevance": -0.5}'
        "\n```"
    )

    assert scores == {"faithfulness": 1.0, "answer_relevance": 0.0}


def test_judge_defaults_missing_or_invalid_fields_to_zero():
    scores = parse_scores('{"faithfulness": null, "answer_relevance": "invalid"}')

    assert scores == {"faithfulness": 0.0, "answer_relevance": 0.0}


def test_judge_rejects_malformed_or_non_object_json():
    assert parse_scores("not json") == {"faithfulness": 0.0, "answer_relevance": 0.0}
    assert parse_scores("[0.5, 0.9]") == {"faithfulness": 0.0, "answer_relevance": 0.0}
