"""Deterministic Domain logic for the Knowledge Base Manifest, its signatures, and source diffs."""

from __future__ import annotations

import hashlib
import json
import posixpath
import os
import tempfile
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any, Literal, Sequence

from pydantic import BaseModel, Field, model_validator

from evidenceflow.domain.document import Document

# Version tokens must be maintained explicitly in code, so a parser/chunker behaviour change is
# never silently treated as the same index.
CURRENT_PARSER_VERSION = "evidenceflow.p0-structure.v1"
CURRENT_CLEANER_VERSION = "evidenceflow.cleaner.v1"
CURRENT_CHUNKER_VERSION = "evidenceflow.hybrid-chunker.v1"
CURRENT_EMBEDDING_VERSION = "evidenceflow.bge-m3-dense-l2-normalized.v1"

ManifestIndexStatus = Literal["ACTIVE", "STAGING", "PREVIOUS", "FAILED"]
SourceChangeStatus = Literal["UNCHANGED", "ADDED", "UPDATED", "DELETED", "RENAMED"]


def canonical_json_bytes(value: Any) -> bytes:
    """Serialise JSON-compatible values by fixed rules, providing the single input representation behind every signature."""

    # Sorted keys, compact separators, and the NaN ban together remove drift caused by the platform
    # and by floating-point special values.
    try:
        serialized = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("value must be deterministically JSON serializable") from exc
    return serialized.encode("utf-8")


def stable_hash(value: Any) -> str:
    """Return the SHA-256 of a deterministic JSON payload, used for corpus/config/index identity."""

    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def save_manifest(path: str | os.PathLike[str], manifest: "KnowledgeBaseManifest") -> None:
    """Persist the manifest atomically in a stable, auditable JSON format."""

    destination = os.fspath(path)
    parent = os.path.dirname(os.path.abspath(destination))
    os.makedirs(parent, exist_ok=True)
    # The file format keeps indentation for reviewability, but ordering and field content still
    # match the canonical manifest.
    text = json.dumps(
        manifest.canonical_dict(),
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    ) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{os.path.basename(destination)}.",
        suffix=".tmp",
        dir=parent,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, destination)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def load_manifest(path: str | os.PathLike[str]) -> "KnowledgeBaseManifest":
    """Read the manifest and re-validate its derived signatures."""

    return KnowledgeBaseManifest.model_validate_json(
        Path(os.fspath(path)).read_text(encoding="utf-8")
    )


def _normalize_relative_path(value: str) -> str:
    """Normalise a source relative path inside the manifest and reject paths outside the root."""

    if not isinstance(value, str) or not value.strip():
        raise ValueError("relative_path must not be empty")
    path = unicodedata.normalize("NFC", value).replace("\\", "/")
    normalized = posixpath.normpath(path)
    if normalized.startswith("/") or normalized in {"", ".", ".."} or normalized.startswith("../"):
        raise ValueError("relative_path must stay within the ingestion root")
    return normalized


class SourceRecord(BaseModel):
    """Minimal record describing one source file's identity and content version inside the Manifest."""

    document_id: str = Field(min_length=1, description="稳定逻辑文档 ID。")
    relative_path: str = Field(min_length=1, description="相对 ingestion root 的 NFC/POSIX 路径。")
    content_hash: str = Field(min_length=1, description="原始文件 bytes 的 SHA-256。")
    size_bytes: int | None = Field(default=None, ge=0, description="文件字节数，仅用于审计。")

    @model_validator(mode="after")
    def normalize_path(self) -> "SourceRecord":
        """Guarantee that the written form of the source path matches the path convention of document_id."""

        # The source path is both displayed and part of the corpus signature, so it must be unified
        # at the model boundary.
        self.relative_path = _normalize_relative_path(self.relative_path)
        return self


class PipelineIdentity(BaseModel):
    """Contains only the parser/chunker/embedding identity that changes the indexed representation."""

    parser_version: str = Field(min_length=1, description="结构解析实现版本 token。")
    cleaner_version: str = Field(min_length=1, description="文本清洗实现版本 token。")
    chunker_version: str = Field(min_length=1, description="Chunker 实现版本 token。")
    chunker_config: dict[str, Any] = Field(description="影响 Chunk 文本表示的配置。")
    chunker_config_hash: str | None = Field(
        default=None,
        description="chunker_config 的确定性 JSON SHA-256；缺省时自动计算。",
    )
    embedding_model: str = Field(min_length=1, description="Embedding 模型标识。")
    embedding_version: str = Field(min_length=1, description="Embedding 输出协议版本 token。")

    @model_validator(mode="after")
    def validate_config_hash(self) -> "PipelineIdentity":
        """Ensure the config hash really corresponds to the config content, so a manual typo never causes wrong reuse."""

        computed = stable_hash(self.chunker_config)
        if self.chunker_config_hash is not None and self.chunker_config_hash != computed:
            raise ValueError("chunker_config_hash does not match chunker_config")
        self.chunker_config_hash = computed
        return self


class IndexIdentity(BaseModel):
    """Describes the status and size of one concrete indexed representation inside the Manifest."""

    index_signature: str = Field(min_length=1, description="由 corpus 与 indexing pipeline 生成的身份。")
    collection_name: str = Field(min_length=1, description="该 manifest 绑定的 Chroma Collection。")
    document_count: int = Field(ge=0, description="Collection 中逻辑文档数量。")
    chunk_count: int = Field(ge=0, description="Collection 中 Chunk 数量。")
    status: ManifestIndexStatus = Field(description="ACTIVE/STAGING/PREVIOUS/FAILED 生命周期状态。")
    parent_index_signature: str | None = Field(
        default=None,
        description="本次 staging 从哪个已知索引演进而来。",
    )


class KnowledgeBaseManifest(BaseModel):
    """A persistable knowledge-base manifest; every identity field is derived deterministically from its content."""

    schema_version: str = Field(default="knowledge-base-manifest-v1", min_length=1)
    knowledge_base_id: str = Field(min_length=1, description="稳定的知识库逻辑身份。")
    corpus_signature: str = Field(min_length=1, description="source 集合及内容的签名。")
    sources: list[SourceRecord] = Field(description="按稳定路径排序的 source records。")
    pipeline: PipelineIdentity = Field(description="影响 indexed representation 的 pipeline 身份。")
    index: IndexIdentity = Field(description="当前 manifest 绑定的索引状态。")

    @model_validator(mode="after")
    def validate_derived_signatures(self) -> "KnowledgeBaseManifest":
        """Reject a forged state where the corpus/index signatures in the manifest disagree with the content."""

        expected_corpus = build_corpus_signature(self.sources)
        if self.corpus_signature != expected_corpus:
            raise ValueError("corpus_signature does not match sources")
        expected_index = build_index_signature(expected_corpus, self.pipeline)
        if self.index.index_signature != expected_index:
            raise ValueError("index_signature does not match corpus and pipeline")
        # Storing sources sorted guarantees the same source set is exported in the same order and
        # callers cannot drift by sorting on their own.
        self.sources = sorted(self.sources, key=lambda item: (item.document_id, item.relative_path, item.content_hash))
        return self

    def canonical_dict(self) -> dict[str, Any]:
        """Return a stable dict suitable for persistence and hashing."""

        return self.model_dump(mode="json", exclude_none=False)

    def canonical_bytes(self) -> bytes:
        """Return the manifest's deterministic JSON bytes."""

        return canonical_json_bytes(self.canonical_dict())

    def with_index_state(
        self,
        *,
        collection_name: str | None = None,
        document_count: int | None = None,
        chunk_count: int | None = None,
        status: ManifestIndexStatus | None = None,
        parent_index_signature: str | None = None,
    ) -> "KnowledgeBaseManifest":
        """Produce a new lifecycle view without changing the corpus/index signatures."""

        updated = self.index.model_copy(
            update={
                key: value
                for key, value in {
                    "collection_name": collection_name,
                    "document_count": document_count,
                    "chunk_count": chunk_count,
                    "status": status,
                    "parent_index_signature": parent_index_signature,
                }.items()
                if value is not None
            }
        )
        return self.model_copy(update={"index": updated})


def _source_signature_records(sources: Sequence[SourceRecord]) -> list[dict[str, str]]:
    """Extract the fields allowed into the corpus signature, excluding the audit field size_bytes."""

    # File size is not part of content identity; for the same bytes, size is already fully
    # determined by content_hash.
    records = [
        {
            "document_id": source.document_id,
            "relative_path": source.relative_path,
            "content_hash": source.content_hash,
        }
        for source in sources
    ]
    return sorted(records, key=lambda item: (item["document_id"], item["relative_path"], item["content_hash"]))


def build_corpus_signature(sources: Sequence[SourceRecord]) -> str:
    """Build a pipeline-independent corpus signature from source identity/content."""

    return stable_hash(_source_signature_records(sources))


def build_index_signature(corpus_signature: str, pipeline: PipelineIdentity) -> str:
    """Build the index signature from the corpus and the indexing-relevant pipeline."""

    # List the fields explicitly so runtime parameters such as reranker, TopK, and generator never
    # leak into the index identity.
    payload = {
        "corpus_signature": corpus_signature,
        "parser_version": pipeline.parser_version,
        "cleaner_version": pipeline.cleaner_version,
        "chunker_version": pipeline.chunker_version,
        "chunker_config_hash": pipeline.chunker_config_hash,
        "embedding_model": pipeline.embedding_model,
        "embedding_version": pipeline.embedding_version,
    }
    return stable_hash(payload)


def current_pipeline_from_settings(settings: Any) -> PipelineIdentity:
    """Assemble the production pipeline identity from the current Settings without reading retrieval-only parameters."""

    # The separator and special-token convention is part of the current hybrid/fixed Chunk
    # representation, so it is frozen explicitly.
    chunker_config = {
        "strategy": settings.chunk_strategy,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "separator": "\n\n",
        "tokenizer_add_special_tokens": False,
    }
    return PipelineIdentity(
        parser_version=CURRENT_PARSER_VERSION,
        cleaner_version=CURRENT_CLEANER_VERSION,
        chunker_version=CURRENT_CHUNKER_VERSION,
        chunker_config=chunker_config,
        embedding_model=settings.embedding_model,
        embedding_version=CURRENT_EMBEDDING_VERSION,
    )


def build_manifest(
    *,
    knowledge_base_id: str,
    sources: Sequence[SourceRecord],
    pipeline: PipelineIdentity,
    collection_name: str,
    document_count: int,
    chunk_count: int,
    status: ManifestIndexStatus = "STAGING",
    parent_index_signature: str | None = None,
    schema_version: str = "knowledge-base-manifest-v1",
) -> KnowledgeBaseManifest:
    """Create and validate a manifest from real source/pipeline data."""

    source_list = sorted(list(sources), key=lambda item: (item.document_id, item.relative_path, item.content_hash))
    corpus_signature = build_corpus_signature(source_list)
    index_signature = build_index_signature(corpus_signature, pipeline)
    return KnowledgeBaseManifest(
        schema_version=schema_version,
        knowledge_base_id=knowledge_base_id,
        corpus_signature=corpus_signature,
        sources=source_list,
        pipeline=pipeline,
        index=IndexIdentity(
            index_signature=index_signature,
            collection_name=collection_name,
            document_count=document_count,
            chunk_count=chunk_count,
            status=status,
            parent_index_signature=parent_index_signature,
        ),
    )


def manifest_sources_from_documents(documents: Sequence[Document]) -> list[SourceRecord]:
    """Convert Documents already loaded by the Loader into manifest source records."""

    return [
        SourceRecord(
            document_id=document.document_id,
            relative_path=document.source_path,
            content_hash=document.content_hash,
            size_bytes=document.metadata.get("file_size"),
        )
        for document in documents
    ]


def compare_sources(
    previous: Sequence[SourceRecord],
    current: Sequence[SourceRecord],
) -> SourceDiff:
    """Compare two explicit scans, returning BLOCKED when a rename cannot be determined uniquely."""

    previous_by_path = _index_sources(previous, "previous")
    current_by_path = _index_sources(current, "current")
    changes: list[SourceChange] = []
    unmatched_previous: list[SourceRecord] = []
    unmatched_current: list[SourceRecord] = []

    # Check by path first, so a hash change on the same path is stably classified as UPDATED.
    for path in sorted(set(previous_by_path) | set(current_by_path)):
        old = previous_by_path.get(path)
        new = current_by_path.get(path)
        if old is not None and new is not None:
            if old.content_hash == new.content_hash and old.document_id == new.document_id:
                changes.append(SourceChange(status="UNCHANGED", relative_path=path, document_id=new.document_id, content_hash=new.content_hash))
            else:
                changes.append(SourceChange(status="UPDATED", relative_path=path, document_id=new.document_id, content_hash=new.content_hash, old_document_id=old.document_id))
        elif old is not None:
            unmatched_previous.append(old)
        elif new is not None:
            unmatched_current.append(new)

    old_by_hash: dict[str, list[SourceRecord]] = defaultdict(list)
    new_by_hash: dict[str, list[SourceRecord]] = defaultdict(list)
    for source in unmatched_previous:
        old_by_hash[source.content_hash].append(source)
    for source in unmatched_current:
        new_by_hash[source.content_hash].append(source)

    ambiguous_hashes = sorted(
        content_hash
        for content_hash in set(old_by_hash) & set(new_by_hash)
        if len(old_by_hash[content_hash]) != 1 or len(new_by_hash[content_hash]) != 1
    )
    if ambiguous_hashes:
        return SourceDiff(
            changes=tuple(sorted(changes, key=lambda item: (item.relative_path, item.status))),
            blocked=True,
            blocked_reason="ambiguous rename content hash: " + ", ".join(ambiguous_hashes),
        )

    consumed_old: set[str] = set()
    consumed_new: set[str] = set()
    for content_hash in sorted(set(old_by_hash) & set(new_by_hash)):
        old = old_by_hash[content_hash][0]
        new = new_by_hash[content_hash][0]
        changes.append(
            SourceChange(
                status="RENAMED",
                relative_path=new.relative_path,
                old_relative_path=old.relative_path,
                document_id=new.document_id,
                old_document_id=old.document_id,
                content_hash=new.content_hash,
            )
        )
        consumed_old.add(old.relative_path)
        consumed_new.add(new.relative_path)

    for old in unmatched_previous:
        if old.relative_path not in consumed_old:
            changes.append(SourceChange(status="DELETED", relative_path=old.relative_path, old_document_id=old.document_id, content_hash=old.content_hash))
    for new in unmatched_current:
        if new.relative_path not in consumed_new:
            changes.append(SourceChange(status="ADDED", relative_path=new.relative_path, document_id=new.document_id, content_hash=new.content_hash))

    return SourceDiff(changes=tuple(sorted(changes, key=lambda item: (item.relative_path, item.status))))


class SourceChange(BaseModel):
    """One source change together with its old/new path identity."""

    status: SourceChangeStatus = Field(description="五种 source 同步状态之一。")
    relative_path: str = Field(min_length=1, description="当前路径或删除前路径。")
    old_relative_path: str | None = Field(default=None, description="RENAMED 的旧路径。")
    document_id: str | None = Field(default=None, description="当前文档 ID。")
    old_document_id: str | None = Field(default=None, description="旧版本文档 ID。")
    content_hash: str | None = Field(default=None, description="当前或被删除文件内容 hash。")


class SourceDiff(BaseModel):
    """Stable diff result of one source scan; guessing a rename is forbidden while BLOCKED."""

    changes: tuple[SourceChange, ...] = Field(description="按路径稳定排序的变更。")
    blocked: bool = Field(default=False, description="是否因 source identity 歧义而停止。")
    blocked_reason: str | None = Field(default=None, description="BLOCKED 的可诊断原因。")


def _index_sources(sources: Sequence[SourceRecord], label: str) -> dict[str, SourceRecord]:
    """Validate the path uniqueness of one source inventory."""

    indexed: dict[str, SourceRecord] = {}
    for source in sources:
        if source.relative_path in indexed:
            raise ValueError(f"duplicate {label} source path: {source.relative_path}")
        indexed[source.relative_path] = source
    return indexed


class RebuildDecision(BaseModel):
    """A deterministic dense-rebuild decision and the reasons behind it."""

    rebuild_required: bool = Field(description="是否需要新建 staging dense index。")
    blocked: bool = Field(default=False, description="是否因 source diff 无法安全决策。")
    reasons: tuple[str, ...] = Field(default=(), description="按固定顺序排列的决策原因。")


def decide_rebuild(
    source_diff: SourceDiff,
    *,
    previous_manifest: KnowledgeBaseManifest | None = None,
    current_manifest: KnowledgeBaseManifest | None = None,
) -> RebuildDecision:
    """Decide whether to rebuild from source/index identity, ignoring reranker and query-side configuration."""

    if source_diff.blocked:
        return RebuildDecision(rebuild_required=False, blocked=True, reasons=(source_diff.blocked_reason or "ambiguous source identity",))
    if previous_manifest is None:
        return RebuildDecision(rebuild_required=True, reasons=("initial index",))

    reasons: list[str] = []
    source_statuses = {change.status for change in source_diff.changes}
    for status in ("ADDED", "UPDATED", "DELETED", "RENAMED"):
        if status in source_statuses:
            reasons.append(f"source {status.lower()}")
    if current_manifest is not None and current_manifest.index.index_signature != previous_manifest.index.index_signature:
        reasons.append("index-affecting pipeline changed")
    # An empty diff with an unchanged signature clearly means no rebuild; the reasons are output
    # for auditability rather than to introduce implicit behaviour.
    return RebuildDecision(rebuild_required=bool(reasons), reasons=tuple(reasons))
