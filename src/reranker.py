"""Cross-encoder reranking stage."""
from __future__ import annotations

from sentence_transformers import CrossEncoder


class CrossEncoderReranker:
    def __init__(self, model_name: str) -> None:
        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        candidates: list[tuple[str, str]],
        top_n: int,
    ) -> list[tuple[str, float]]:
        """Rerank (doc_id, passage) candidates and return top_n (doc_id, score)."""
        if not candidates:
            return []

        pairs = [(query, passage) for _, passage in candidates]
        scores = self.model.predict(pairs, show_progress_bar=False)
        scored = [
            (doc_id, float(score)) for (doc_id, _), score in zip(candidates, scores)
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_n]
