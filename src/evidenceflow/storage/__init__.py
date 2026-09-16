"""提供与底层向量数据库解耦的持久化 Vector Store 接口。"""

from evidenceflow.storage.base import VectorStore
from evidenceflow.storage.chroma import (
    ChromaVectorStore,
    serialize_chroma_record_metadata,
    serialize_chunk_metadata,
    VectorStoreConfigurationError,
    VectorStoreError,
    VectorStoreQueryError,
    VectorStoreWriteError,
)

__all__ = [
    "ChromaVectorStore",
    "serialize_chroma_record_metadata",
    "serialize_chunk_metadata",
    "VectorStore",
    "VectorStoreConfigurationError",
    "VectorStoreError",
    "VectorStoreQueryError",
    "VectorStoreWriteError",
]
