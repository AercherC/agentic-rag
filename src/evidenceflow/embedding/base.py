"""Defines the unified in-project Dense Embedding contract."""

from typing import Protocol, Sequence, runtime_checkable


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Maps document bodies and queries into the same Dense Vector space."""

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed a batch of documents and return plain Python float lists."""

        ...

    def embed_query(self, query: str) -> list[float]:
        """Embed a single query and return a plain Python float list."""

        ...
