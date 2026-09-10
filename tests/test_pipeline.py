"""Unit tests for optional EvalPipeline stages."""
from __future__ import annotations

from typing import Any

import src.pipeline as pipeline_module


def unexpected_constructor(*args: object, **kwargs: object) -> None:
    raise AssertionError("disabled optional stage was initialized")


def test_bm25_pipeline_skips_disabled_optional_stages(monkeypatch):
    corpus = [
        {"doc_id": "doc_a", "text": "First passage"},
        {"doc_id": "doc_b", "text": "Second passage"},
    ]
    config: dict[str, Any] = {
        "name": "bm25_only",
        "retriever": {"mode": "bm25", "top_k": 2},
        "rewriter": {"enabled": False},
        "reranker": {"enabled": False},
        "generator": {"enabled": False},
    }

    monkeypatch.setattr(pipeline_module, "load_corpus", lambda _: corpus)
    monkeypatch.setattr(pipeline_module, "load_doc_ids", lambda _: ["doc_a", "doc_b"])
    monkeypatch.setattr(pipeline_module, "load_dense_index", lambda _: object())
    monkeypatch.setattr(pipeline_module, "load_bm25_index", lambda _: object())
    monkeypatch.setattr(
        pipeline_module,
        "bm25_search",
        lambda *args: [("doc_b", 2.0), ("doc_a", 1.0)],
    )
    monkeypatch.setattr(pipeline_module, "ClaudeGenerator", unexpected_constructor)
    monkeypatch.setattr(pipeline_module, "ClaudeJudge", unexpected_constructor)
    monkeypatch.setattr(pipeline_module, "ClaudeRewriter", unexpected_constructor)
    monkeypatch.setattr(pipeline_module, "CrossEncoderReranker", unexpected_constructor)

    result = pipeline_module.EvalPipeline(config).run(
        {
            "id": "question_1",
            "question": "Which passage is second?",
            "expected_answer": "Second passage",
            "gold_doc_ids": ["doc_b"],
        }
    )

    assert result["retrieved_doc_ids"] == ["doc_b", "doc_a"]
    assert result["generated_answer"] == ""
    assert result["faithfulness"] == 0.0
    assert result["answer_relevance"] == 0.0
    assert "retrieve_ms" in result
    assert "rewrite_ms" not in result
    assert "rerank_ms" not in result
    assert "generate_ms" not in result
    assert "judge_ms" not in result
