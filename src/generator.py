"""LLM answer generation from retrieved context."""
from __future__ import annotations

from src.ollama_client import DEFAULT_OLLAMA_MODEL, OllamaClient


class OllamaGenerator:
    def __init__(
        self,
        model: str = DEFAULT_OLLAMA_MODEL,
        max_tokens: int = 512,
        client: OllamaClient | None = None,
    ) -> None:
        self.client = client or OllamaClient(model)
        self.model = model
        self.max_tokens = max_tokens

    def generate(self, question: str, passages: list[str]) -> str:
        context = "\n\n".join(
            f"Passage {i + 1}:\n{passage}" for i, passage in enumerate(passages)
        )
        prompt = (
            "You are a helpful question-answering assistant. "
            "Use only the provided passages to answer the question. "
            "If the passages do not contain enough information, say so.\n\n"
            f"{context}\n\n"
            f"Question: {question}\n\n"
            "Answer concisely:"
        )
        return self.client.generate(prompt, self.max_tokens)
