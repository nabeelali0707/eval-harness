"""Query rewriting / expansion stage."""
from __future__ import annotations

import os

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()


class ClaudeRewriter:
    def __init__(self, model: str = "claude-sonnet-4-6", max_tokens: int = 256) -> None:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY not found. Set it in a .env file or environment."
            )
        self.client = Anthropic(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens

    def rewrite(self, query: str, strategy: str) -> list[str]:
        if strategy == "multi_query":
            return self._multi_query_expansion(query)
        if strategy == "hyde":
            return self._hyde(query)
        raise ValueError(f"Unknown rewrite strategy: {strategy}")

    def _multi_query_expansion(self, query: str) -> list[str]:
        prompt = (
            "Generate 3 short paraphrases of the following search query. "
            "Each paraphrase should retrieve the same kind of documents but use different wording. "
            "Return one per line, no numbering.\n\n"
            f"Query: {query}\n\n"
            "Paraphrases:"
        )
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        variants = [
            line.strip("-• ").strip()
            for line in response.content[0].text.strip().splitlines()
            if line.strip()
        ]
        # Always include the original query so we don't lose signal.
        return [query] + variants[:3]

    def _hyde(self, query: str) -> list[str]:
        prompt = (
            "Write a short hypothetical passage that would answer the following question. "
            "The passage should look like a real Wikipedia excerpt.\n\n"
            f"Question: {query}\n\n"
            "Passage:"
        )
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return [query, response.content[0].text.strip()]
