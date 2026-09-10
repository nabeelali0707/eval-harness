"""Minimal local Ollama HTTP client."""
from __future__ import annotations

import json
import os
import socket
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from dotenv import load_dotenv

DEFAULT_OLLAMA_MODEL = "qwen2.5-coder:7b"

load_dotenv()


class OllamaError(RuntimeError):
    """Raised when the local Ollama service cannot complete a request."""


class OllamaClient:
    def __init__(
        self,
        model: str = DEFAULT_OLLAMA_MODEL,
        host: str | None = None,
        timeout: float = 300.0,
    ) -> None:
        if model.endswith(":cloud"):
            raise ValueError("Cloud Ollama models are not supported; choose a local model instead.")

        configured_host = host or os.getenv("OLLAMA_HOST", "http://localhost:11434")
        if not configured_host.startswith(("http://", "https://")):
            configured_host = f"http://{configured_host}"

        self.model = model
        self.host = configured_host.rstrip("/")
        self.timeout = timeout

    def ensure_model_available(self) -> None:
        response = self._request("GET", "/api/tags")
        models = response.get("models", [])
        names = {
            model.get("name")
            for model in models
            if isinstance(model, dict) and isinstance(model.get("name"), str)
        }
        if self.model not in names:
            raise OllamaError(
                f"Ollama model `{self.model}` is not installed. "
                f"Run `ollama pull {self.model}` and try again."
            )

    def generate(self, prompt: str, max_tokens: int, *, json_mode: bool = False) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"num_predict": max_tokens},
        }
        if json_mode:
            payload["format"] = "json"
            payload["options"]["temperature"] = 0

        response = self._request("POST", "/api/generate", payload)
        text = response.get("response")
        if not isinstance(text, str):
            raise OllamaError("Ollama returned a response without generated text.")
        return text.strip()

    def _request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        request = Request(
            f"{self.host}{path}",
            data=data,
            headers={"Content-Type": "application/json"},
            method=method,
        )

        try:
            with urlopen(request, timeout=self.timeout) as response:
                raw_response = response.read().decode("utf-8")
        except HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace").strip()
            raise OllamaError(
                f"Ollama request to {self.host}{path} failed with HTTP {error.code}: {detail}"
            ) from error
        except (URLError, TimeoutError, socket.timeout) as error:
            raise OllamaError(
                f"Cannot reach Ollama at {self.host}. Start it with `ollama serve` and try again."
            ) from error

        try:
            response_data = json.loads(raw_response)
        except json.JSONDecodeError as error:
            raise OllamaError("Ollama returned invalid JSON.") from error

        if not isinstance(response_data, dict):
            raise OllamaError("Ollama returned an unexpected response format.")
        if isinstance(response_data.get("error"), str):
            raise OllamaError(f"Ollama request failed: {response_data['error']}")
        return response_data
