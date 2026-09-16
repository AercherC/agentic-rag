"""Phase 15A 的 staging validation、active pointer、activation 与 rollback。"""

from __future__ import annotations

import json
import os
import re
import tempfile
from collections import Counter
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any, Literal, Protocol

from pydantic import BaseModel, Field, model_validator

from evidenceflow.domain.chunk import DocumentChunk
from evidenceflow.knowledge.job import BuildJob, BuildJobStatus, BuildJobStore, BuildJobTracker
from evidenceflow.knowledge.manifest import KnowledgeBaseManifest

LifecyclePointerStatus = Literal["ACTIVE", "STAGING", "PREVIOUS"]


class CollectionReader(Protocol):
    """生命周期校验所需的 Vector Store 只读能力。"""

    def iter_all_chunks(self) -> Iterator[DocumentChunk]:
        """返回当前 Collection 的全部可恢复 Chunk。"""

        ...

    def search(self, query_embedding: Sequence[float], top_k: int) -> list[Any]:
        """执行一次只读 Dense smoke query。"""

        ...


class IndexPointer(BaseModel):
    """active/staging/previous 指针中的不可变索引身份摘要。"""

    knowledge_base_id: str = Field(min_length=1, description="所属知识库 ID。")
    collection_name: str = Field(min_length=1, description="绑定的 Chroma Collection 名称。")
    index_signature: str = Field(min_length=1, description="绑定的 indexed representation 身份。")
    manifest_path: str = Field(min_length=1, description="对应 manifest 的相对或绝对路径。")
    document_count: int = Field(ge=0, description="该索引的逻辑文档数量。")
    chunk_count: int = Field(ge=0, description="该索引的 Chunk 数量。")
    status: LifecyclePointerStatus = Field(description="ACTIVE、STAGING 或 PREVIOUS。")

    @classmethod
    def from_manifest(
        cls,
        manifest: KnowledgeBaseManifest,
        *,
        status: LifecyclePointerStatus | None = None,
        manifest_path: str | None = None,
    ) -> "IndexPointer":
        """从 manifest 生成指针，默认沿用 manifest 的 index 状态。"""

        pointer_status = status or manifest.index.status
        if pointer_status not in {"ACTIVE", "STAGING", "PREVIOUS"}:
            raise ValueError(f"manifest status cannot become lifecycle pointer: {pointer_status}")
        return cls(
            knowledge_base_id=manifest.knowledge_base_id,
            collection_name=manifest.index.collection_name,
            index_signature=manifest.index.index_signature,
            manifest_path=manifest_path or f"manifests/{manifest.index.index_signature}.json",
            document_count=manifest.index.document_count,
            chunk_count=manifest.index.chunk_count,
            status=pointer_status,
        )


class LifecycleState(BaseModel):
    """单机知识库的 active/previous/staging 单一状态记录。"""

    schema_version: str = Field(default="index-lifecycle-v1", min_length=1)
    knowledge_base_id: str = Field(min_length=1, description="状态所属知识库 ID。")
    active: IndexPointer = Field(description="当前唯一 active index。")
    previous: IndexPointer | None = Field(default=None, description="最近一次已知良好旧 active。")
    staging: IndexPointer | None = Field(default=None, description="尚未 activation 的新 index。")

    @model_validator(mode="after")
    def validate_pointer_ownership(self) -> "LifecycleState":
        """确保指针不会跨知识库或携带错误生命周期状态。"""

        if self.active.status != "ACTIVE":
            raise ValueError("active pointer must have status ACTIVE")
        pointers = [("active", self.active, "ACTIVE"), ("previous", self.previous, "PREVIOUS"), ("staging", self.staging, "STAGING")]
        for name, pointer, expected_status in pointers:
            if pointer is None:
                continue
            if pointer.knowledge_base_id != self.knowledge_base_id:
                raise ValueError(f"{name} pointer belongs to another knowledge base")
            if pointer.status != expected_status:
                raise ValueError(f"{name} pointer must have status {expected_status}")
        return self


class CollectionValidationReport(BaseModel):
    """staging Collection 在 activation 前的完整性验证结果。"""

    passed: bool = Field(description="所有启用的 validation checks 是否通过。")
    manifest_document_count: int = Field(ge=0)
    actual_document_count: int = Field(ge=0)
    manifest_chunk_count: int = Field(ge=0)
    actual_chunk_count: int = Field(ge=0)
    missing_document_ids: tuple[str, ...] = Field(default=())
    unexpected_document_ids: tuple[str, ...] = Field(default=())
    duplicate_chunk_ids: tuple[str, ...] = Field(default=())
    missing_chunk_ids: tuple[str, ...] = Field(default=())
    unexpected_chunk_ids: tuple[str, ...] = Field(default=())
    missing_metadata: tuple[str, ...] = Field(default=())
    embedding_dimension: int | None = Field(default=None, ge=1)
    expected_embedding_dimension: int | None = Field(default=None, ge=1)
    embedding_dimension_consistent: bool = Field(default=False)
    collection_readable: bool = Field(default=False)
    dense_retrieval_smoke: bool = Field(default=False)
    errors: tuple[str, ...] = Field(default=())


class StagingBuildCoordinator:
    """驱动一次同步 staging build，并把真实阶段写入 BuildJob。"""

    def __init__(self, lifecycle: "IndexLifecycle", job_store: BuildJobStore) -> None:
        """注入 lifecycle pointer 和 Job 持久化器，不创建后台任务。"""

        self.lifecycle = lifecycle
        self.job_store = job_store

    def build(
        self,
        manifest: KnowledgeBaseManifest,
        build: Callable[[str, Callable[[BuildJobStatus], None]], CollectionReader],
        *,
        expected_chunk_ids: Sequence[str] | None = None,
        smoke_query_embedding: Sequence[float] | None = None,
        expected_embedding_dimension: int | None = None,
    ) -> tuple[BuildJob, CollectionValidationReport | None]:
        """在新 Collection 上执行受回调阶段约束的 build，再进行 validation。

        `build` 负责在传入的 Collection 名称上复用既有 IngestionService；它必须在实际
        进入 PARSING/CHUNKING/EMBEDDING/INDEXING 时调用阶段回调。Coordinator 不会复制
        ingestion pipeline，也不会在 validation 失败时调用 activation。
        """

        tracker = BuildJobTracker.create(
            knowledge_base_id=manifest.knowledge_base_id,
            source_corpus_signature=manifest.corpus_signature,
            target_index_signature=manifest.index.index_signature,
            target_collection_name=manifest.index.collection_name,
            store=self.job_store,
        )
        try:
            self.lifecycle.prepare_staging(manifest)
            tracker.advance("SCANNING")

            def report_stage(status: BuildJobStatus) -> None:
                """将 builder 报告的真实阶段推进到同一个持久化 Job。"""

                if status not in {"PARSING", "CHUNKING", "EMBEDDING", "INDEXING"}:
                    raise ValueError(f"invalid staging build stage: {status}")
                tracker.advance(status)

            store = build(manifest.index.collection_name, report_stage)
            if tracker.job.status != "INDEXING":
                raise ValueError("staging builder returned before INDEXING stage")
            tracker.advance("VALIDATING")
            report = validate_collection(
                manifest,
                store,
                expected_chunk_ids=expected_chunk_ids,
                smoke_query_embedding=smoke_query_embedding,
                expected_embedding_dimension=expected_embedding_dimension,
            )
            if not report.passed:
                tracker.fail("VALIDATION_ERROR", "; ".join(report.errors) or "staging validation failed")
                return tracker.job, report
            tracker.advance(
                "READY",
                document_count=report.actual_document_count,
                chunk_count=report.actual_chunk_count,
            )
            return tracker.job, report
        except Exception as exc:
            # 失败只落 Job 诊断，不改 active pointer；staging 可由调用方按生命周期策略清理。
            if tracker.job.status not in {"FAILED", "BLOCKED", "ACTIVATED"}:
                tracker.fail("BUILD_ERROR", f"{type(exc).__name__}: {exc}")
            return tracker.job, None


def build_staging_collection_name(base_collection: str, index_signature: str) -> str:
    """为 staging 生成稳定、可读且不依赖随机 UUID 的 Collection 名称。"""

    if not base_collection.strip() or not index_signature.strip():
        raise ValueError("base_collection and index_signature must not be empty")
    # Chroma 名称长度有限；保留稳定 base 前缀和 signature 前缀即可完成审计绑定。
    safe_base = re.sub(r"[^A-Za-z0-9._-]", "-", base_collection)[:32].strip(".-") or "evidenceflow"
    return f"{safe_base}__staging__{index_signature[:16]}"


def _read_embedding_dimension(store: Any) -> int | None:
    """从具体 Store 的只读 inspection 能力读取向量维度。"""

    value = getattr(store, "embedding_dimension", None)
    if callable(value):
        value = value()
    return int(value) if value is not None else None


def validate_collection(
    manifest: KnowledgeBaseManifest,
    store: CollectionReader,
    *,
    expected_chunk_ids: Sequence[str] | None = None,
    smoke_query_embedding: Sequence[float] | None = None,
    expected_embedding_dimension: int | None = None,
    required_metadata_keys: Sequence[str] = ("chunk_strategy", "element_ids"),
) -> CollectionValidationReport:
    """校验 staging 的文档/Chunk/metadata/维度/可读性和 Dense smoke。"""

    errors: list[str] = []
    try:
        chunks = list(store.iter_all_chunks())
        collection_readable = True
    except Exception as exc:
        chunks = []
        collection_readable = False
        errors.append(f"collection read failed: {type(exc).__name__}: {exc}")

    actual_document_ids = {chunk.document_id for chunk in chunks}
    expected_document_ids = {source.document_id for source in manifest.sources}
    chunk_ids = [chunk.chunk_id for chunk in chunks]
    duplicate_chunk_ids = tuple(sorted(chunk_id for chunk_id, count in Counter(chunk_ids).items() if count > 1))
    expected_ids = set(expected_chunk_ids) if expected_chunk_ids is not None else None
    missing_chunk_ids = tuple(sorted(expected_ids - set(chunk_ids))) if expected_ids is not None else ()
    unexpected_chunk_ids = tuple(sorted(set(chunk_ids) - expected_ids)) if expected_ids is not None else ()
    missing_metadata = tuple(
        f"{chunk.chunk_id}:{key}"
        for chunk in chunks
        for key in required_metadata_keys
        if key not in chunk.metadata or (key == "element_ids" and not chunk.metadata[key])
    )
    dimension = _read_embedding_dimension(store)
    dimension_consistent = (
        expected_embedding_dimension is None
        or dimension == expected_embedding_dimension
    ) and (expected_embedding_dimension is not None or dimension is not None)
    dense_smoke = False
    if smoke_query_embedding is not None and collection_readable:
        try:
            dense_smoke = bool(store.search(smoke_query_embedding, top_k=1))
        except Exception as exc:
            errors.append(f"dense retrieval smoke failed: {type(exc).__name__}: {exc}")

    if duplicate_chunk_ids:
        errors.append("duplicate chunk IDs")
    if missing_metadata:
        errors.append("required chunk metadata is missing")
    if not collection_readable:
        errors.append("collection is not readable")
    if not dimension_consistent:
        errors.append("embedding dimensionality is inconsistent or unavailable")
    if not dense_smoke:
        errors.append("dense retrieval smoke did not pass")

    passed = (
        collection_readable
        and len(actual_document_ids) == manifest.index.document_count
        and len(chunks) == manifest.index.chunk_count
        and not (expected_document_ids - actual_document_ids)
        and not (actual_document_ids - expected_document_ids)
        and not duplicate_chunk_ids
        and not missing_chunk_ids
        and not unexpected_chunk_ids
        and not missing_metadata
        and dimension_consistent
        and dense_smoke
    )
    return CollectionValidationReport(
        passed=passed,
        manifest_document_count=manifest.index.document_count,
        actual_document_count=len(actual_document_ids),
        manifest_chunk_count=manifest.index.chunk_count,
        actual_chunk_count=len(chunks),
        missing_document_ids=tuple(sorted(expected_document_ids - actual_document_ids)),
        unexpected_document_ids=tuple(sorted(actual_document_ids - expected_document_ids)),
        duplicate_chunk_ids=duplicate_chunk_ids,
        missing_chunk_ids=missing_chunk_ids,
        unexpected_chunk_ids=unexpected_chunk_ids,
        missing_metadata=missing_metadata,
        embedding_dimension=dimension,
        expected_embedding_dimension=expected_embedding_dimension,
        embedding_dimension_consistent=dimension_consistent,
        collection_readable=collection_readable,
        dense_retrieval_smoke=dense_smoke,
        errors=tuple(dict.fromkeys(errors)),
    )


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """原子替换生命周期 JSON，失败时保留原 active pointer。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


class IndexLifecycle:
    """管理单机 active/staging/previous 指针，不执行索引重建或检索。"""

    def __init__(self, state_path: str | Path, knowledge_base_id: str) -> None:
        """注入状态文件路径和知识库身份；所有写入仅限该状态文件。"""

        if not knowledge_base_id.strip():
            raise ValueError("knowledge_base_id must not be empty")
        self.state_path = Path(state_path)
        self.knowledge_base_id = knowledge_base_id

    def load_state(self) -> LifecycleState:
        """读取并校验当前 active pointer state。"""

        state = LifecycleState.model_validate_json(self.state_path.read_text(encoding="utf-8"))
        if state.knowledge_base_id != self.knowledge_base_id:
            raise ValueError("lifecycle state belongs to another knowledge base")
        return state

    def initialize_active(
        self,
        manifest: KnowledgeBaseManifest,
        *,
        manifest_path: str | None = None,
    ) -> LifecycleState:
        """以一个已知良好 manifest 建立初始 active state。"""

        if manifest.knowledge_base_id != self.knowledge_base_id:
            raise ValueError("manifest belongs to another knowledge base")
        if manifest.index.status != "ACTIVE":
            raise ValueError("initial active manifest must have status ACTIVE")
        state = LifecycleState(
            knowledge_base_id=self.knowledge_base_id,
            active=IndexPointer.from_manifest(
                manifest,
                status="ACTIVE",
                manifest_path=manifest_path,
            ),
        )
        _write_json_atomic(self.state_path, state.model_dump(mode="json"))
        return state

    def prepare_staging(self, manifest: KnowledgeBaseManifest) -> LifecycleState:
        """登记 staging identity；此步骤不触碰 active Collection。"""

        if manifest.knowledge_base_id != self.knowledge_base_id:
            raise ValueError("manifest belongs to another knowledge base")
        if manifest.index.status != "STAGING":
            raise ValueError("staging manifest must have status STAGING")
        state = self.load_state()
        next_state = state.model_copy(
            update={"staging": IndexPointer.from_manifest(manifest, status="STAGING")}
        )
        _write_json_atomic(self.state_path, next_state.model_dump(mode="json"))
        return next_state

    def activate(
        self,
        manifest: KnowledgeBaseManifest,
        validation: CollectionValidationReport,
        *,
        manifest_path: str | None = None,
    ) -> LifecycleState:
        """仅在 validation PASS 后把 staging 变为 active，并保留旧 active。"""

        if manifest.index.status != "STAGING":
            raise ValueError("activation requires a staging manifest")
        if not validation.passed:
            raise ValueError("cannot activate a staging collection that failed validation")
        state = self.load_state()
        if state.staging is None:
            raise ValueError("no staging pointer is registered")
        if state.staging.index_signature != manifest.index.index_signature or state.staging.collection_name != manifest.index.collection_name:
            raise ValueError("staging pointer does not match manifest")
        if manifest.knowledge_base_id != self.knowledge_base_id:
            raise ValueError("manifest belongs to another knowledge base")
        next_state = LifecycleState(
            knowledge_base_id=self.knowledge_base_id,
            active=IndexPointer.from_manifest(manifest, status="ACTIVE", manifest_path=manifest_path),
            previous=state.active.model_copy(update={"status": "PREVIOUS"}),
            staging=None,
        )
        _write_json_atomic(self.state_path, next_state.model_dump(mode="json"))
        return next_state

    def rollback(self) -> LifecycleState:
        """直接恢复 previous pointer，不重新解析、Embedding 或构建旧索引。"""

        state = self.load_state()
        if state.previous is None:
            raise ValueError("no previous known-good index is available")
        next_state = LifecycleState(
            knowledge_base_id=self.knowledge_base_id,
            active=state.previous.model_copy(update={"status": "ACTIVE"}),
            previous=state.active.model_copy(update={"status": "PREVIOUS"}),
            staging=None,
        )
        _write_json_atomic(self.state_path, next_state.model_dump(mode="json"))
        return next_state
