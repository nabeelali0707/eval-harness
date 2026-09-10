"""Query rewriting / expansion stage."""
from __future__ import annotations

from src.ollama_client import DEFAULT_OLLAMA_MODEL, OllamaClient


class OllamaRewriter:
    def __init__(
        self,
        model: str = DEFAULT_OLLAMA_MODEL,
        max_tokens: int = 256,
        client: OllamaClient | None = None,
    ) -> None:
        self.client = client or OllamaClient(model)
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
        variants = [
            line.strip("-• ").strip()
            for line in self.client.generate(prompt, self.max_tokens).splitlines()
            if line.strip()
        ]
        return [query] + variants[:3]

    def _hyde(self, query: str) -> list[str]:
        prompt = (
            "Write a short hypothetical passage that would answer the following question. "
            "The passage should look like a real Wikipedia excerpt.\n\n"
            f"Question: {query}\n\n"
            "Passage:"
        )
        return [query, self.client.generate(prompt, self.max_tokens)]
