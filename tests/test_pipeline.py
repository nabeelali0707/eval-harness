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
    monkeypatch.setattr(pipeline_module, "OllamaClient", unexpected_constructor)
    monkeypatch.setattr(pipeline_module, "OllamaGenerator", unexpected_constructor)
    monkeypatch.setattr(pipeline_module, "OllamaJudge", unexpected_constructor)
    monkeypatch.setattr(pipeline_module, "OllamaRewriter", unexpected_constructor)
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
    assert result["faithfulness"] is None
    assert result["answer_relevance"] is None
    assert "retrieve_ms" in result
    assert "rewrite_ms" not in result
    assert "rerank_ms" not in result
    assert "generate_ms" not in result
    assert "judge_ms" not in result


def test_multi_query_fusion_respects_retriever_top_k(monkeypatch):
    corpus = [
        {"doc_id": "doc_a", "text": "A"},
        {"doc_id": "doc_b", "text": "B"},
        {"doc_id": "doc_c", "text": "C"},
        {"doc_id": "doc_d", "text": "D"},
    ]
    config: dict[str, Any] = {
        "name": "hybrid_rewrite_rerank",
        "retriever": {
            "mode": "hybrid",
            "dense_model": "test-model",
            "top_k": 2,
            "rrf_k": 60,
        },
        "rewriter": {"enabled": False},
        "reranker": {"enabled": False},
        "generator": {"enabled": False},
    }

    monkeypatch.setattr(pipeline_module, "load_corpus", lambda _: corpus)
    monkeypatch.setattr(
        pipeline_module,
        "load_doc_ids",
        lambda _: ["doc_a", "doc_b", "doc_c", "doc_d"],
    )
    monkeypatch.setattr(pipeline_module, "load_dense_index", lambda _: object())
    monkeypatch.setattr(pipeline_module, "load_bm25_index", lambda _: object())
    monkeypatch.setattr(pipeline_module, "SentenceTransformer", lambda _: object())
    monkeypatch.setattr(
        pipeline_module,
        "hybrid_search",
        lambda query, *args: (
            [("doc_a", 1.0), ("doc_b", 0.9)]
            if query == "original"
            else [("doc_c", 1.0), ("doc_d", 0.9)]
        ),
    )

    retrieved_ids, _ = pipeline_module.EvalPipeline(config).retrieve_multi(
        ["original", "variant"]
    )

    assert retrieved_ids == ["doc_a", "doc_c"]


def test_pipeline_shares_one_ollama_client_across_llm_stages(monkeypatch):
    corpus = [{"doc_id": "doc_a", "text": "First passage"}]
    config: dict[str, Any] = {
        "name": "bm25_with_llm",
        "retriever": {"mode": "bm25", "top_k": 1},
        "rewriter": {"enabled": True},
        "reranker": {"enabled": False},
        "generator": {"enabled": True, "model": "test-model", "max_tokens": 64},
    }
    stage_clients: list[object] = []

    class FakeOllamaClient:
        def __init__(self, model: str) -> None:
            self.model = model
            self.available = False

        def ensure_model_available(self) -> None:
            self.available = True

    class FakeRewriter:
        def __init__(self, **kwargs: object) -> None:
            stage_clients.append(kwargs["client"])

    class FakeGenerator:
        def __init__(self, **kwargs: object) -> None:
            stage_clients.append(kwargs["client"])

    class FakeJudge:
        def __init__(self, **kwargs: object) -> None:
            stage_clients.append(kwargs["client"])

    monkeypatch.setattr(pipeline_module, "load_corpus", lambda _: corpus)
    monkeypatch.setattr(pipeline_module, "load_doc_ids", lambda _: ["doc_a"])
    monkeypatch.setattr(pipeline_module, "load_dense_index", lambda _: object())
    monkeypatch.setattr(pipeline_module, "load_bm25_index", lambda _: object())
    monkeypatch.setattr(pipeline_module, "OllamaClient", FakeOllamaClient)
    monkeypatch.setattr(pipeline_module, "OllamaRewriter", FakeRewriter)
    monkeypatch.setattr(pipeline_module, "OllamaGenerator", FakeGenerator)
    monkeypatch.setattr(pipeline_module, "OllamaJudge", FakeJudge)

    pipeline = pipeline_module.EvalPipeline(config)

    assert pipeline.llm_client.model == "test-model"
    assert pipeline.llm_client.available is True
    assert stage_clients == [pipeline.llm_client, pipeline.llm_client, pipeline.llm_client]
