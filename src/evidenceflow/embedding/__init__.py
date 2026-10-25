"""Public Dense Embedding interface decoupled from any specific model library."""

from evidenceflow.embedding.base import EmbeddingProvider
from evidenceflow.embedding.bge_m3 import (
    BgeM3EmbeddingProvider,
    EmbeddingError,
    EmbeddingLoadError,
    EmbeddingNormalizationError,
)

__all__ = [
    "BgeM3EmbeddingProvider",
    "EmbeddingError",
    "EmbeddingLoadError",
    "EmbeddingNormalizationError",
    "EmbeddingProvider",
]
