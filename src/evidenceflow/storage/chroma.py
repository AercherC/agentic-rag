"""使用本地 Chroma PersistentClient 持久化和查询 DocumentChunk。"""

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
    """Vector Store 配置、写入或查询失败的统一基础异常。"""


class VectorStoreConfigurationError(VectorStoreError):
    """表示本地数据库或 Collection 配置不符合项目合同。"""


class VectorStoreWriteError(VectorStoreError):
    """表示 Chunk 写入、删除或清空操作失败。"""


class VectorStoreQueryError(VectorStoreError):
    """表示 Chroma 查询结果无法转换为项目 Domain Model。"""


def read_collection_metadata(
    persist_path: str | Path, collection_name: str
) -> dict[str, Any] | None:
    """只读查询 Collection 的 metadata；Collection 不存在时返回 None。

    用于 lifecycle 校验与 ingest 保护：必须在加载任何模型之前执行，且绝不创建 Collection。
    持久化目录不存在时直接返回 None，不为了“查询”而在磁盘上新建 Chroma 目录。
    """

    _require_non_empty_text(collection_name, field_name="collection_name")
    path = Path(persist_path)
    if not path.is_dir():
        return None
    try:
        client = chromadb.PersistentClient(path=str(path))
        # 用 get_collection 而不是 get_or_create_collection：查询不能产生副作用。
        collection = client.get_collection(name=collection_name)
    except NotFoundError:
        return None
    except Exception as exc:
        raise VectorStoreConfigurationError(
            f"Failed to read metadata of collection {collection_name!r} at: {path}"
        ) from exc
    return dict(collection.metadata or {})


def _require_non_empty_text(value: str, *, field_name: str) -> None:
    """校验过滤条件和 Collection 名称是可用的非空字符串。"""

    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must not be empty")


def _coerce_embedding(
    embedding: Sequence[float],
    *,
    field_name: str,
) -> list[float]:
    """将向量转换为有限 Python float；Storage 不重复执行归一化。"""

    # Chroma 的只读接口返回 NumPy ndarray；在 Storage 边界转成普通序列，避免
    # 第三方数值对象继续向 Domain/检索层传播。
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
    """确定性序列化 Domain metadata，并拒绝不可表示的业务对象。"""

    try:
        # 排序键使持久化表示稳定；禁止 NaN，避免写入非标准 JSON。
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
    """恢复 Domain metadata，并拒绝损坏或类型不正确的存储值。"""

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
    """把 Chunk 记录序列化为可被 Chroma 接受的标量 Metadata。

    Chroma 的原生 metadata 只接受标量值，因此完整的 Domain metadata 以确定性
    JSON 字符串保存；其余常用检索/诊断字段保留为顶层标量，便于过滤和盘点。
    生产 Domain Chunk 通过本函数写入，实验记录也通过同一函数写入其已有字段，避免
    出现“能查向量但不能恢复 Chunk”的半持久化记录。
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
    # 13C 评测记录没有 file_name/content_hash；缺失字段不伪造生产来源信息。
    for key in ("content_hash", "file_name"):
        value = record.get(key)
        if value is not None:
            metadata[key] = str(value)
    # Chroma metadata 不依赖 None；无页码文本直接省略两个字段。
    page_start = record.get("page_start")
    page_end = record.get("page_end")
    if page_start is not None:
        metadata["page_start"] = int(page_start)
        metadata["page_end"] = int(page_end)
    return metadata


def serialize_chunk_metadata(chunk: DocumentChunk) -> dict[str, Any]:
    """把完整 Domain Chunk 委托给统一记录序列化器。"""

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


# 保留内部旧名称，避免历史实验脚本或外部诊断代码在升级时失效。
_build_chroma_metadata = serialize_chunk_metadata


def _restore_chunk(
    chunk_id: Any,
    document: Any,
    metadata: Any,
) -> DocumentChunk:
    """将一条 Chroma Record 恢复为项目内部 DocumentChunk。"""

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
    """使用 cosine Collection 持久化 Dense Vector 和 DocumentChunk。"""

    def __init__(
        self,
        persist_path: str | Path,
        collection_name: str,
        *,
        collection_metadata: Mapping[str, Any] | None = None,
        create_if_missing: bool = True,
    ) -> None:
        """打开本地 PersistentClient，并确保 Collection 使用 cosine。

        `collection_metadata` 只用于新建 staging 等 Collection 的生命周期标记；
        未提供时保持既有 Collection 的 metadata 和运行语义不变。
        `create_if_missing=False` 时只打开已存在的 Collection（线上按 active 指针读取时使用），
        不存在则抛 `VectorStoreConfigurationError`，不会留下一个空 Collection。
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
        """创建或打开不绑定 Chroma Embedding Function 的 cosine Collection。"""

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
            # metadata 只在 Collection 创建时提供，避免打开 active 时意外覆盖身份信息。
            kwargs["metadata"] = self.collection_metadata
        collection = self._client.get_or_create_collection(**kwargs)
        self._require_cosine(collection)
        return collection

    def _require_cosine(self, collection: Any) -> None:
        """项目的相似度分数语义依赖 cosine；配置不符时拒绝使用该 Collection。"""

        hnsw_configuration = collection.configuration.get("hnsw") or {}
        if hnsw_configuration.get("space") != "cosine":
            raise VectorStoreConfigurationError(
                f"Collection {self.collection_name!r} is not configured for cosine"
            )

    @property
    def metadata(self) -> dict[str, Any]:
        """返回 Collection metadata 的只读副本，供 lifecycle validation 审计。"""

        return dict(self._collection.metadata or {})

    @property
    def count(self) -> int:
        """返回当前 Collection 记录数，不改变任何存储内容。"""

        return self._collection.count()

    @property
    def embedding_dimension(self) -> int | None:
        """返回已有向量维度；空 Collection 没有可推断的维度。"""

        return self._stored_dimension()

    def _stored_dimension(self) -> int | None:
        """读取一条现有向量以校验后续写入和 Query 的维度。"""

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
        """校验后批量写入 Chunk 与外部生成的 Dense Embedding。"""

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
            # 主动检查重复 ID，避免依赖 Chroma add() 对重复项的静默行为。
            existing = self._collection.get(ids=chunk_ids, include=[])
            if existing.get("ids"):
                raise VectorStoreWriteError(
                    "One or more chunk IDs already exist in the collection"
                )
            # metadata 与 ids/documents 同批写入，保证 Domain Chunk 可完整反序列化。
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
        """执行 cosine 查询，并将 distance 转换为项目 similarity_score。"""

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
                # cosine_distance = 1 - cosine_similarity，禁止额外缩放或归一化。
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
        """按文档身份和可选内容 Hash 查询是否至少存在一个 Chunk。"""

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
        """使用 metadata 过滤一次删除指定文档的全部旧 Chunk。"""

        _require_non_empty_text(document_id, field_name="document_id")
        try:
            self._collection.delete(where={"document_id": document_id})
        except Exception as exc:
            raise VectorStoreWriteError(
                f"Failed to delete document: {document_id}"
            ) from exc

    def iter_all_chunks(self) -> Iterator[DocumentChunk]:
        """按 chunk_id 升序遍历当前 Collection 中的全部正式 Chunk。

        Hybrid 集成（§17.3.1）需要把同一份 Chunk 喂给 BM25Index，所以该方法只读、
        确定性，并且按 chunk_id 升序返回，避免调用方在排序上各自漂移。
        """

        try:
            # offset/limit 分页遍历，避免一次性把全部 Chunk 内容加载到内存。
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
                # 单页内排序保证整体 chunk_id 字典序；分页 offset 也会按 id 序列前进。
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
        """按 Chunk ID 升序恢复 Chunk 与向量，供进程内 Exact Index 构建。"""

        try:
            # 与 iter_all_chunks 使用相同分页合同；向量只读出，不复制为第二份持久化真源。
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
                # Chroma 的物理返回顺序不是业务合同；这里固定全局可复现的 ID 顺序。
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
        """重建当前 cosine Collection，不删除整个持久化目录。"""

        try:
            self._client.delete_collection(name=self.collection_name)
            self._collection = self._create_collection()
        except Exception as exc:
            raise VectorStoreWriteError(
                f"Failed to clear collection: {self.collection_name}"
            ) from exc
