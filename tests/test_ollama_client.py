"""Unit tests for the local Ollama HTTP client."""
from __future__ import annotations

import json
from urllib.error import URLError

import pytest

import src.ollama_client as ollama_client


class FakeResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


def test_client_checks_model_and_sends_generate_payload(monkeypatch):
    requests: list[tuple[str, dict[str, object]]] = []

    def fake_urlopen(request, timeout: float):
        payload = json.loads(request.data) if request.data else {}
        requests.append((request.full_url, payload))
        if request.full_url.endswith("/api/tags"):
            return FakeResponse({"models": [{"name": "test-model"}]})
        return FakeResponse({"response": '{"faithfulness": 1.0}'})

    monkeypatch.setattr(ollama_client, "urlopen", fake_urlopen)
    client = ollama_client.OllamaClient("test-model", host="ollama.test")

    client.ensure_model_available()
    response = client.generate("Judge this", 42, json_mode=True)

    assert response == '{"faithfulness": 1.0}'
    assert requests[0][0] == "http://ollama.test/api/tags"
    assert requests[1][1] == {
        "model": "test-model",
        "prompt": "Judge this",
        "stream": False,
        "options": {"num_predict": 42, "temperature": 0},
        "format": "json",
    }


def test_client_explains_missing_model(monkeypatch):
    monkeypatch.setattr(
        ollama_client,
        "urlopen",
        lambda *args, **kwargs: FakeResponse({"models": []}),
    )

    with pytest.raises(ollama_client.OllamaError, match=r"ollama pull missing-model"):
        ollama_client.OllamaClient("missing-model").ensure_model_available()


def test_client_explains_unreachable_service(monkeypatch):
    def unreachable(*args: object, **kwargs: object) -> None:
        raise URLError("connection refused")

    monkeypatch.setattr(ollama_client, "urlopen", unreachable)

    with pytest.raises(ollama_client.OllamaError, match=r"ollama serve"):
        ollama_client.OllamaClient("test-model").ensure_model_available()


def test_client_rejects_cloud_models():
    with pytest.raises(ValueError, match="Cloud Ollama models"):
        ollama_client.OllamaClient("qwen3.5:cloud")
