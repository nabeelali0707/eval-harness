"""Query rewriting / expansion stage."""
from __future__ import annotations


def rewrite(query: str, strategy: str) -> list[str]:
    """Return rewritten or expanded queries."""
    raise NotImplementedError
