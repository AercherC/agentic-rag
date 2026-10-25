"""Phase 15A staging validation, active pointer, activation and rollback."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections import Counter
from collections.abc import Callable, Collection, Iterator, Sequence
from pathlib import Path
from typing import Any, Literal, Protocol

from pydantic import BaseModel, Field, model_validator

from evidenceflow.domain.chunk import DocumentChunk
from evidenceflow.knowledge.job import BuildJob, BuildJobStatus, BuildJobStore, BuildJobTracker
from evidenceflow.knowledge.manifest import KnowledgeBaseManifest

LifecyclePointerStatus = Literal["ACTIVE", "STAGING", "PREVIOUS"]


class ActiveCollectionError(RuntimeError):
    """Resolving the serving collection or verifying its identity failed: configuration conflict, corrupt state file or mismatching lifecycle flag."""


class ActiveIndexProtectedError(RuntimeError):
    """The target collection is the index the lifecycle active pointer references; ingesting into it directly is forbidden."""


class CollectionReader(Protocol):
    """The read-only Vector Store capabilities that lifecycle validation needs."""

    def iter_all_chunks(self) -> Iterator[DocumentChunk]:
        """Return every recoverable Chunk in the current Collection."""

        ...

    def search(self, query_embedding: Sequence[float], top_k: int) -> list[Any]:
        """Run one read-only Dense smoke query."""

        ...


class IndexPointer(BaseModel):
    """Immutable index identity summary of the active/staging/previous pointers."""

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
        """Build a pointer from a manifest, defaulting to the manifest's index status."""

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
    """The single active/previous/staging state record of a single-machine knowledge base."""

    schema_version: str = Field(default="index-lifecycle-v1", min_length=1)
    knowledge_base_id: str = Field(min_length=1, description="状态所属知识库 ID。")
    active: IndexPointer = Field(description="当前唯一 active index。")
    previous: IndexPointer | None = Field(default=None, description="最近一次已知良好旧 active。")
    staging: IndexPointer | None = Field(default=None, description="尚未 activation 的新 index。")

    @model_validator(mode="after")
    def validate_pointer_ownership(self) -> "LifecycleState":
        """Ensure pointers never cross knowledge bases or carry the wrong lifecycle status."""

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
    """Result of validating a staging Collection before activation."""

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
    """Drive one synchronous staging build and write the real stages into the BuildJob."""

    def __init__(self, lifecycle: "IndexLifecycle", job_store: BuildJobStore) -> None:
        """Inject the lifecycle pointer and the Job persistence backend; creates no background task."""

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
        """Run a build on the new Collection under the callback-driven stage contract, then validate.

        `build` is responsible for reusing the existing IngestionService against the Collection name it
        is given, and it must invoke the stage callback when it actually enters
        PARSING/CHUNKING/EMBEDDING/INDEXING. The Coordinator does not duplicate the ingestion pipeline,
        and it never calls activation when validation fails.
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
                """Advance the same persisted Job to the real stage reported by the builder."""

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
            # A failure only writes Job diagnostics and never touches the active pointer; staging
            # can be
            # cleaned up by the caller per the lifecycle policy.
            if tracker.job.status not in {"FAILED", "BLOCKED", "ACTIVATED"}:
                tracker.fail("BUILD_ERROR", f"{type(exc).__name__}: {exc}")
            return tracker.job, None


# Total length of a staging name: base prefix 24 + "-" + base short hash 8 + "__staging__" 11 +
# signature prefix 16 = 60, below Chroma's 63-character name limit.
_STAGING_BASE_PREFIX = 24
_STAGING_BASE_HASH = 8
_STAGING_SIGNATURE_PREFIX = 16
# Maximum number of candidates tried when avoiding reserved names; going beyond it means the
# input is anomalous, so raise instead of retrying forever.
_STAGING_NAME_ATTEMPTS = 8


def build_staging_collection_name(
    base_collection: str,
    index_signature: str,
    *,
    reserved: Collection[str] = (),
) -> str:
    """Generate a deterministic, readable Collection name for staging that does not rely on a random UUID.

    Name = truncated base prefix + short hash of the complete base + ``__staging__`` + signature
    prefix. The short hash is computed from the **complete** base: with only the truncated prefix,
    generating a name from a generated name would return that name itself (found in the A3 review),
    so a staging name rebuilt after activation would equal active. With the hash of the complete base
    included, generating from a generated result necessarily differs from itself (barring a 32-bit
    hash collision).

    Names in ``reserved`` (usually active and previous) are steered around: when rebuilding with the
    same base + signature after a rollback, the preferred name happens to equal previous (it was the
    last staging name), so the code falls back to candidates salted with an index in a fixed order,
    and the result is still decided only by the input. Raises ``ValueError`` when every candidate is taken.
    """

    if not base_collection.strip() or not index_signature.strip():
        raise ValueError("base_collection and index_signature must not be empty")
    # Keep only the characters Chroma allows and truncate to the name-length limit; leading and
    # trailing . and - are not allowed, so strip them.
    safe_base = (
        re.sub(r"[^A-Za-z0-9._-]", "-", base_collection)[:_STAGING_BASE_PREFIX].strip(".-")
        or "evidenceflow"
    )
    taken = set(reserved)
    for attempt in range(_STAGING_NAME_ATTEMPTS):
        # Candidate 0 hashes the base itself, making the first build's name a pure function of
        # (base, signature); later candidates append an index to the hash input only to avoid
        # reserved
        # names.
        salt = "" if attempt == 0 else f"#{attempt}"
        digest = hashlib.sha256(f"{base_collection}{salt}".encode("utf-8")).hexdigest()
        name = (
            f"{safe_base}-{digest[:_STAGING_BASE_HASH]}"
            f"__staging__{index_signature[:_STAGING_SIGNATURE_PREFIX]}"
        )
        if name not in taken:
            return name
    raise ValueError(
        f"cannot derive a staging collection name for {base_collection!r} that avoids {sorted(taken)}"
    )


def _read_embedding_dimension(store: Any) -> int | None:
    """Read the vector dimension from the concrete Store's read-only inspection capability."""

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
    """Validate the staging document/Chunk/metadata/dimension/readability and the Dense smoke query."""

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
    """Atomically replace the lifecycle JSON, preserving the original active pointer on failure."""

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
    """Manage the single-machine active/staging/previous pointers; it performs no index rebuild or retrieval."""

    def __init__(self, state_path: str | Path, knowledge_base_id: str) -> None:
        """Inject the state file path and the knowledge base identity; all writes are confined to that state file."""

        if not knowledge_base_id.strip():
            raise ValueError("knowledge_base_id must not be empty")
        self.state_path = Path(state_path)
        self.knowledge_base_id = knowledge_base_id

    def load_state(self) -> LifecycleState:
        """Read and validate the current active pointer state."""

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
        """Establish the initial active state from one known-good manifest."""

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
        """Register the staging identity; this step does not touch the active Collection."""

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
        """Turn staging into active only after validation PASSES, keeping the old active index."""

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
        """Restore the previous pointer directly, without re-parsing, re-Embedding or rebuilding the old index."""

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

    def discard_staging(self) -> LifecycleState:
        """Clear only the staging pointer; active/previous are untouched and no Chroma collection is deleted.

        The caller is responsible for deleting the staging collection through the Chroma API first and
        for confirming beforehand that it is neither active nor previous; this method only maintains the
        pointer, avoiding hand-edited JSON. When staging is already null it returns immediately without
        rewriting the state file.
        """

        state = self.load_state()
        if state.staging is None:
            return state
        next_state = state.model_copy(update={"staging": None})
        _write_json_atomic(self.state_path, next_state.model_dump(mode="json"))
        return next_state
