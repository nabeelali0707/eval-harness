"""LLM answer generation from retrieved context."""
from __future__ import annotations

import os

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()


class ClaudeGenerator:
    def __init__(self, model: str = "claude-sonnet-4-6", max_tokens: int = 512) -> None:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY not found. Set it in a .env file or environment."
            )
        self.client = Anthropic(api_key=api_key)
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
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text.strip()
