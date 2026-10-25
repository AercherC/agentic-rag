"""Persistent Vector Store interface decoupled from the underlying vector database."""

from evidenceflow.storage.base import VectorStore
from evidenceflow.storage.chroma import (
    ChromaVectorStore,
    read_collection_metadata,
    serialize_chroma_record_metadata,
    serialize_chunk_metadata,
    VectorStoreConfigurationError,
    VectorStoreError,
    VectorStoreQueryError,
    VectorStoreWriteError,
)

__all__ = [
    "ChromaVectorStore",
    "read_collection_metadata",
    "serialize_chroma_record_metadata",
    "serialize_chunk_metadata",
    "VectorStore",
    "VectorStoreConfigurationError",
    "VectorStoreError",
    "VectorStoreQueryError",
    "VectorStoreWriteError",
]
