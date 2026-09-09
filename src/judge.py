"""LLM-as-judge for faithfulness and answer relevance."""
from __future__ import annotations

import json
import os
import re

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()


class ClaudeJudge:
    def __init__(self, model: str = "claude-sonnet-4-6", max_tokens: int = 256) -> None:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY not found. Set it in a .env file or environment."
            )
        self.client = Anthropic(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens

    def score(
        self,
        question: str,
        answer: str,
        passages: list[str],
    ) -> dict[str, float]:
        context = "\n\n".join(
            f"Passage {i + 1}:\n{passage}" for i, passage in enumerate(passages)
        )
        prompt = (
            "You are an evaluator for retrieval-augmented QA systems. "
            "Rate the answer below on two metrics using the provided passages.\n\n"
            f"Passages:\n{context}\n\n"
            f"Question: {question}\n\n"
            f"Answer: {answer}\n\n"
            "Respond with ONLY a JSON object in this exact format:\n"
            '{"faithfulness": 0.0, "answer_relevance": 0.0}\n\n'
            "Definitions:\n"
            "- faithfulness: does the answer contain only claims supported by the passages? (0 = unsupported, 1 = fully supported)\n"
            "- answer_relevance: does the answer address the question? (0 = irrelevant, 1 = directly answers)\n\n"
            "JSON:"
        )
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.content[0].text.strip()
        return self._parse_scores(text)

    def _parse_scores(self, text: str) -> dict[str, float]:
        # Try to extract JSON from markdown code fences or raw text.
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            text = match.group(0)
        try:
            parsed = json.loads(text)
            return {
                "faithfulness": float(parsed.get("faithfulness", 0.0)),
                "answer_relevance": float(parsed.get("answer_relevance", 0.0)),
            }
        except (json.JSONDecodeError, ValueError):
            return {"faithfulness": 0.0, "answer_relevance": 0.0}
