"""提供与具体模型库解耦的 Dense Embedding 公共接口。"""

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
