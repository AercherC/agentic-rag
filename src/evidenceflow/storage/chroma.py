"""Persist and query DocumentChunk through the local Chroma PersistentClient."""

from __future__ import annotations

import json
import math
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any, Mapping

import chromadb
from chromadb.errors import NotFoundError

from evidenceflow.domain.chunk import DocumentChunk, RetrievedChunk

DOMAIN_METADATA_KEY = "domain_metadata_json"
COSINE_CONFIGURATION = {"hnsw": {"space": "cosine"}}


class VectorStoreError(RuntimeError):
    """Common base exception for Vector Store configuration, write, or query failures."""


class VectorStoreConfigurationError(VectorStoreError):
    """Indicates that the local database or Collection configuration violates the project contract."""


class VectorStoreWriteError(VectorStoreError):
    """Indicates that a Chunk write, delete, or clear operation failed."""


class VectorStoreQueryError(VectorStoreError):
    """Indicates that a Chroma query result cannot be converted into a project Domain Model."""


def read_collection_metadata(
    persist_path: str | Path, collection_name: str
) -> dict[str, Any] | None:
    """Read a Collection's metadata read-only; return None when the Collection does not exist.

    Used by lifecycle validation and ingest protection: this must run before any model is loaded, and it must never create a Collection.
    When the persistence directory does not exist it returns None directly, rather than creating a Chroma directory on disk just to "query".
    """

    _require_non_empty_text(collection_name, field_name="collection_name")
    path = Path(persist_path)
    if not path.is_dir():
        return None
    try:
        client = chromadb.PersistentClient(path=str(path))
        # Use get_collection rather than get_or_create_collection: a query must not produce side
        # effects.
        collection = client.get_collection(name=collection_name)
    except NotFoundError:
        return None
    except Exception as exc:
        raise VectorStoreConfigurationError(
            f"Failed to read metadata of collection {collection_name!r} at: {path}"
        ) from exc
    return dict(collection.metadata or {})


def _require_non_empty_text(value: str, *, field_name: str) -> None:
    """Validate that the filter value and Collection name are usable non-empty strings."""

    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must not be empty")


def _coerce_embedding(
    embedding: Sequence[float],
    *,
    field_name: str,
) -> list[float]:
    """Convert the vector into finite Python floats; the Storage layer does not normalise it again."""

    # Chroma's read-only API returns NumPy ndarray; convert it to a plain sequence at the Storage
    # boundary
    # so third-party numeric objects do not keep propagating into the Domain/retrieval layers.
    raw_embedding = embedding.tolist() if hasattr(embedding, "tolist") else embedding
    if isinstance(raw_embedding, (str, bytes)) or not isinstance(raw_embedding, Sequence):
        raise TypeError(f"{field_name} must be a numeric sequence")
    if not raw_embedding:
        raise ValueError(f"{field_name} must not be empty")
    try:
        values = [float(value) for value in raw_embedding]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} contains non-numeric values") from exc
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"{field_name} contains non-finite values")
    return values


def _serialize_domain_metadata(metadata: dict[str, Any]) -> str:
    """Serialize Domain metadata deterministically and reject business objects that cannot be represented."""

    try:
        # Sorted keys keep the persisted representation stable; NaN is forbidden so non-standard
        # JSON cannot be written.
        return json.dumps(
            metadata,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise VectorStoreWriteError(
            "DocumentChunk metadata must be JSON serializable"
        ) from exc


def _deserialize_domain_metadata(metadata_json: Any) -> dict[str, Any]:
    """Restore Domain metadata and reject corrupted or wrongly typed stored values."""

    if not isinstance(metadata_json, str):
        raise VectorStoreQueryError("Stored domain metadata is not JSON text")
    try:
        metadata = json.loads(metadata_json)
    except (TypeError, ValueError) as exc:
        raise VectorStoreQueryError("Stored domain metadata JSON is invalid") from exc
    if not isinstance(metadata, dict):
        raise VectorStoreQueryError("Stored domain metadata must decode to an object")
    return metadata


def serialize_chroma_record_metadata(record: Mapping[str, Any]) -> dict[str, Any]:
    """Serialize a Chunk record into scalar Metadata that Chroma accepts.

    Chroma's native metadata only accepts scalar values, so the complete Domain metadata is stored as a deterministic
    JSON string, while the remaining common retrieval/diagnostic fields stay as top-level scalars for filtering and inventory.
    Production Domain Chunks are written through this function, and experiment records write their existing fields through the
    same function, avoiding half-persisted records that "can query vectors but cannot restore the Chunk".
    """

    domain_metadata = record.get("metadata") or {}
    if not isinstance(domain_metadata, dict):
        raise VectorStoreWriteError("Chunk metadata must be a JSON object")
    metadata: dict[str, Any] = {
        "chunk_id": str(record.get("chunk_id", "")),
        "document_id": str(record["document_id"]),
        "chunk_index": int(record.get("chunk_index", record.get("seq", 0))),
        "token_count": int(record["token_count"]),
        DOMAIN_METADATA_KEY: _serialize_domain_metadata(domain_metadata),
    }
    # 13C evaluation records have no file_name/content_hash; missing fields are not faked with
    # production provenance.
    for key in ("content_hash", "file_name"):
        value = record.get(key)
        if value is not None:
            metadata[key] = str(value)
    # Chroma metadata does not rely on None; text without page numbers simply omits both fields.
    page_start = record.get("page_start")
    page_end = record.get("page_end")
    if page_start is not None:
        metadata["page_start"] = int(page_start)
        metadata["page_end"] = int(page_end)
    return metadata


def serialize_chunk_metadata(chunk: DocumentChunk) -> dict[str, Any]:
    """Delegate a complete Domain Chunk to the unified record serializer."""

    return serialize_chroma_record_metadata(
        {
            "document_id": chunk.document_id,
            "chunk_id": chunk.chunk_id,
            "content_hash": chunk.content_hash,
            "file_name": chunk.file_name,
            "chunk_index": chunk.chunk_index,
            "token_count": chunk.token_count,
            "page_start": chunk.page_start,
            "page_end": chunk.page_end,
            "metadata": chunk.metadata,
        }
    )


# Keep the internal legacy name so historical experiment scripts or external diagnostic code do not
# break on upgrade.
_build_chroma_metadata = serialize_chunk_metadata


def _restore_chunk(
    chunk_id: Any,
    document: Any,
    metadata: Any,
) -> DocumentChunk:
    """Restore one Chroma Record into the project's internal DocumentChunk."""

    if not isinstance(chunk_id, str) or not isinstance(document, str):
        raise VectorStoreQueryError("Stored chunk ID or document text is invalid")
    if not isinstance(metadata, dict):
        raise VectorStoreQueryError("Stored chunk metadata is invalid")
    try:
        return DocumentChunk(
            chunk_id=chunk_id,
            document_id=metadata["document_id"],
            content=document,
            token_count=metadata["token_count"],
            chunk_index=metadata["chunk_index"],
            page_start=metadata.get("page_start"),
            page_end=metadata.get("page_end"),
            file_name=metadata["file_name"],
            content_hash=metadata["content_hash"],
            metadata=_deserialize_domain_metadata(metadata.get(DOMAIN_METADATA_KEY)),
        )
    except VectorStoreQueryError:
        raise
    except Exception as exc:
        raise VectorStoreQueryError("Stored chunk record is incomplete or invalid") from exc


class ChromaVectorStore:
    """Persist Dense Vectors and DocumentChunks into a cosine Collection."""

    def __init__(
        self,
        persist_path: str | Path,
        collection_name: str,
        *,
        collection_metadata: Mapping[str, Any] | None = None,
        create_if_missing: bool = True,
    ) -> None:
        """Open the local PersistentClient and ensure the Collection uses cosine.

        `collection_metadata` is only used for lifecycle marking of newly created Collections such as staging;
        when not provided, the metadata and runtime semantics of the existing Collection stay unchanged.
        With `create_if_missing=False` only an existing Collection is opened (as used online when reading through the active pointer);
        if it does not exist, `VectorStoreConfigurationError` is raised and no empty Collection is left behind.
        """

        _require_non_empty_text(collection_name, field_name="collection_name")
        self.persist_path = Path(persist_path)
        self.collection_name = collection_name
        self.collection_metadata = dict(collection_metadata or {})
        self.create_if_missing = create_if_missing
        try:
            self.persist_path.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=str(self.persist_path))
            self._collection = self._create_collection()
        except VectorStoreConfigurationError:
            raise
        except Exception as exc:
            raise VectorStoreConfigurationError(
                f"Failed to initialize Chroma at: {self.persist_path}"
            ) from exc

    def _create_collection(self):
        """Create or open a cosine Collection that binds no Chroma Embedding Function."""

        if not self.create_if_missing:
            try:
                collection = self._client.get_collection(
                    name=self.collection_name, embedding_function=None
                )
            except NotFoundError as exc:
                raise VectorStoreConfigurationError(
                    f"Collection {self.collection_name!r} does not exist at: {self.persist_path}"
                ) from exc
            self._require_cosine(collection)
            return collection
        kwargs: dict[str, Any] = {
            "name": self.collection_name,
            "configuration": COSINE_CONFIGURATION,
            "embedding_function": None,
        }
        if self.collection_metadata:
            # metadata is supplied only when the Collection is created, so opening the active one
            # cannot accidentally overwrite identity information.
            kwargs["metadata"] = self.collection_metadata
        collection = self._client.get_or_create_collection(**kwargs)
        self._require_cosine(collection)
        return collection

    def _require_cosine(self, collection: Any) -> None:
        """The project's similarity-score semantics depend on cosine; refuse to use the Collection if the configuration differs."""

        hnsw_configuration = collection.configuration.get("hnsw") or {}
        if hnsw_configuration.get("space") != "cosine":
            raise VectorStoreConfigurationError(
                f"Collection {self.collection_name!r} is not configured for cosine"
            )

    @property
    def metadata(self) -> dict[str, Any]:
        """Return a read-only copy of the Collection metadata for lifecycle validation audits."""

        return dict(self._collection.metadata or {})

    @property
    def count(self) -> int:
        """Return the current Collection record count without changing any stored content."""

        return self._collection.count()

    @property
    def embedding_dimension(self) -> int | None:
        """Return the existing vector dimension; an empty Collection has no dimension that can be inferred."""

        return self._stored_dimension()

    def _stored_dimension(self) -> int | None:
        """Read one existing vector to validate the dimension of later writes and queries."""

        try:
            result = self._collection.get(limit=1, include=["embeddings"])
        except Exception as exc:
            raise VectorStoreQueryError(
                "Failed to inspect stored embedding dimension"
            ) from exc
        embeddings = result.get("embeddings")
        if embeddings is None or len(embeddings) == 0:
            return None
        return len(embeddings[0])

    def add_chunks(
        self,
        chunks: Sequence[DocumentChunk],
        embeddings: Sequence[Sequence[float]],
    ) -> None:
        """Validate, then write Chunks and externally generated Dense Embeddings in a batch."""

        chunk_list = list(chunks)
        embedding_list = list(embeddings)
        if len(chunk_list) != len(embedding_list):
            raise ValueError("chunks and embeddings must have the same length")
        if not chunk_list:
            return

        chunk_ids = [chunk.chunk_id for chunk in chunk_list]
        if len(set(chunk_ids)) != len(chunk_ids):
            raise VectorStoreWriteError("Duplicate chunk IDs exist in the input batch")

        vectors = [
            _coerce_embedding(vector, field_name=f"embeddings[{index}]")
            for index, vector in enumerate(embedding_list)
        ]
        dimensions = {len(vector) for vector in vectors}
        if len(dimensions) != 1:
            raise ValueError("All embeddings in one batch must have the same dimension")
        stored_dimension = self._stored_dimension()
        vector_dimension = len(vectors[0])
        if stored_dimension is not None and stored_dimension != vector_dimension:
            raise ValueError(
                "Embedding dimension does not match the existing collection"
            )

        try:
            # Check duplicate IDs actively instead of relying on Chroma add()'s silent behaviour
            # for duplicates.
            existing = self._collection.get(ids=chunk_ids, include=[])
            if existing.get("ids"):
                raise VectorStoreWriteError(
                    "One or more chunk IDs already exist in the collection"
                )
            # metadata is written in the same batch as ids/documents, guaranteeing the Domain Chunk
            # can be fully deserialized.
            metadatas = [serialize_chunk_metadata(chunk) for chunk in chunk_list]
            self._collection.add(
                ids=chunk_ids,
                embeddings=vectors,
                documents=[chunk.content for chunk in chunk_list],
                metadatas=metadatas,
            )
        except VectorStoreWriteError:
            raise
        except Exception as exc:
            raise VectorStoreWriteError("Failed to add chunks to Chroma") from exc

    def search(
        self,
        query_embedding: Sequence[float],
        top_k: int,
    ) -> list[RetrievedChunk]:
        """Run a cosine query and convert distance into the project's similarity_score."""

        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k must be greater than 0")
        vector = _coerce_embedding(query_embedding, field_name="query_embedding")
        try:
            record_count = self._collection.count()
            if record_count == 0:
                return []
            stored_dimension = self._stored_dimension()
            if stored_dimension is not None and stored_dimension != len(vector):
                raise ValueError(
                    "Query embedding dimension does not match the collection"
                )
            result = self._collection.query(
                query_embeddings=[vector],
                n_results=min(top_k, record_count),
                include=["documents", "metadatas", "distances"],
            )
        except (ValueError, VectorStoreQueryError):
            raise
        except Exception as exc:
            raise VectorStoreQueryError("Failed to query Chroma") from exc

        try:
            ids = result["ids"][0]
            documents = result["documents"][0]
            metadatas = result["metadatas"][0]
            distances = result["distances"][0]
            if not (
                len(ids) == len(documents) == len(metadatas) == len(distances)
            ):
                raise VectorStoreQueryError("Chroma query result fields are misaligned")

            retrieved: list[RetrievedChunk] = []
            for rank, (chunk_id, document, metadata, distance) in enumerate(
                zip(ids, documents, metadatas, distances),
                start=1,
            ):
                # cosine_distance = 1 - cosine_similarity; no additional scaling or normalisation
                # is allowed.
                similarity_score = 1.0 - float(distance)
                retrieved.append(
                    RetrievedChunk(
                        chunk=_restore_chunk(chunk_id, document, metadata),
                        rank=rank,
                        similarity_score=similarity_score,
                    )
                )
            return retrieved
        except VectorStoreQueryError:
            raise
        except Exception as exc:
            raise VectorStoreQueryError(
                "Failed to restore Chroma query results"
            ) from exc

    def contains_document(
        self,
        document_id: str,
        content_hash: str | None = None,
    ) -> bool:
        """Query whether at least one Chunk exists by document identity and an optional content Hash."""

        _require_non_empty_text(document_id, field_name="document_id")
        if content_hash is not None:
            _require_non_empty_text(content_hash, field_name="content_hash")
            where = {
                "$and": [
                    {"document_id": document_id},
                    {"content_hash": content_hash},
                ]
            }
        else:
            where = {"document_id": document_id}
        try:
            result = self._collection.get(where=where, limit=1, include=[])
            return bool(result.get("ids"))
        except Exception as exc:
            raise VectorStoreQueryError("Failed to inspect document existence") from exc

    def delete_document(self, document_id: str) -> None:
        """Delete all old Chunks of the given document in one call using a metadata filter."""

        _require_non_empty_text(document_id, field_name="document_id")
        try:
            self._collection.delete(where={"document_id": document_id})
        except Exception as exc:
            raise VectorStoreWriteError(
                f"Failed to delete document: {document_id}"
            ) from exc

    def iter_all_chunks(self) -> Iterator[DocumentChunk]:
        """Iterate over every official Chunk in the current Collection in ascending chunk_id order.

        Used for index lifecycle validation and for the full read before building the Exact Dense index, so this method is read-only
        and deterministic, returning chunks in ascending chunk_id order so callers cannot drift apart on sorting.
        """

        try:
            # Iterate with offset/limit pagination so all Chunk content is never loaded into memory
            # at once.
            page_size = 1024
            offset = 0
            while True:
                result = self._collection.get(
                    limit=page_size,
                    offset=offset,
                    include=["documents", "metadatas"],
                )
                ids = result.get("ids") or []
                documents = result.get("documents") or []
                metadatas = result.get("metadatas") or []
                if not ids:
                    break
                if not (
                    len(ids) == len(documents) == len(metadatas)
                ):
                    raise VectorStoreQueryError(
                        "Chroma iter_all_chunks returned misaligned fields"
                    )
                restored = [
                    _restore_chunk(chunk_id, document, metadata)
                    for chunk_id, document, metadata in zip(
                        ids, documents, metadatas
                    )
                ]
                # Sorting within a page guarantees the overall chunk_id lexicographic order; the
                # paging offset also advances along the id sequence.
                restored.sort(key=lambda chunk: chunk.chunk_id)
                for chunk in restored:
                    yield chunk
                if len(ids) < page_size:
                    break
                offset += page_size
        except VectorStoreQueryError:
            raise
        except Exception as exc:
            raise VectorStoreQueryError(
                "Failed to iterate Chroma collection chunks"
            ) from exc

    def iter_all_chunks_with_embeddings(
        self,
    ) -> Iterator[tuple[DocumentChunk, list[float]]]:
        """Restore Chunks and vectors in ascending Chunk ID order for in-process Exact Index construction."""

        try:
            # Uses the same paging contract as iter_all_chunks; vectors are read out only and never
            # copied into a second persisted source of truth.
            page_size = 1024
            offset = 0
            while True:
                result = self._collection.get(
                    limit=page_size,
                    offset=offset,
                    include=["documents", "metadatas", "embeddings"],
                )
                ids = result.get("ids") or []
                documents = result.get("documents") or []
                metadatas = result.get("metadatas") or []
                raw_embeddings = result.get("embeddings")
                embeddings = [] if raw_embeddings is None else raw_embeddings
                if not ids:
                    break
                if not (
                    len(ids)
                    == len(documents)
                    == len(metadatas)
                    == len(embeddings)
                ):
                    raise VectorStoreQueryError(
                        "Chroma iter_all_chunks_with_embeddings returned misaligned fields"
                    )
                restored = [
                    (
                        _restore_chunk(chunk_id, document, metadata),
                        _coerce_embedding(
                            embedding,
                            field_name=f"embedding[{index}]",
                        ),
                    )
                    for index, (chunk_id, document, metadata, embedding) in enumerate(
                        zip(ids, documents, metadatas, embeddings)
                    )
                ]
                # Chroma's physical return order is not a business contract; fix a globally
                # reproducible ID order here.
                restored.sort(key=lambda item: item[0].chunk_id)
                yield from restored
                if len(ids) < page_size:
                    break
                offset += page_size
        except VectorStoreQueryError:
            raise
        except Exception as exc:
            raise VectorStoreQueryError(
                "Failed to iterate Chroma collection chunks with embeddings"
            ) from exc

    def clear(self) -> None:
        """Rebuild the current cosine Collection without deleting the whole persistence directory."""

        try:
            self._client.delete_collection(name=self.collection_name)
            self._collection = self._create_collection()
        except Exception as exc:
            raise VectorStoreWriteError(
                f"Failed to clear collection: {self.collection_name}"
            ) from exc
