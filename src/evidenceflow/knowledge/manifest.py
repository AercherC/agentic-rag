"""Knowledge Base Manifest、签名和 source diff 的确定性 Domain 逻辑。"""

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

# 版本 token 必须由代码显式维护；这样 parser/chunker 行为变更不会被静默视为同一索引。
CURRENT_PARSER_VERSION = "evidenceflow.p0-structure.v1"
CURRENT_CLEANER_VERSION = "evidenceflow.cleaner.v1"
CURRENT_CHUNKER_VERSION = "evidenceflow.hybrid-chunker.v1"
CURRENT_EMBEDDING_VERSION = "evidenceflow.bge-m3-dense-l2-normalized.v1"

ManifestIndexStatus = Literal["ACTIVE", "STAGING", "PREVIOUS", "FAILED"]
SourceChangeStatus = Literal["UNCHANGED", "ADDED", "UPDATED", "DELETED", "RENAMED"]


def canonical_json_bytes(value: Any) -> bytes:
    """将 JSON 兼容值按固定规则序列化，作为所有签名的唯一输入表示。"""

    # 排序键、紧凑分隔符和禁止 NaN 共同消除平台及浮点特殊值造成的漂移。
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
    """返回确定性 JSON 载荷的 SHA-256，用于 corpus/config/index 身份。"""

    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def save_manifest(path: str | os.PathLike[str], manifest: "KnowledgeBaseManifest") -> None:
    """以稳定、可审计的 JSON 格式原子保存 manifest。"""

    destination = os.fspath(path)
    parent = os.path.dirname(os.path.abspath(destination))
    os.makedirs(parent, exist_ok=True)
    # 文件格式保留缩进便于审阅，但排序和字段内容仍与 canonical manifest 一致。
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
    """读取并重新校验 manifest 的派生 signature。"""

    return KnowledgeBaseManifest.model_validate_json(
        Path(os.fspath(path)).read_text(encoding="utf-8")
    )


def _normalize_relative_path(value: str) -> str:
    """规范化 manifest 中的 source 相对路径并拒绝 root 外路径。"""

    if not isinstance(value, str) or not value.strip():
        raise ValueError("relative_path must not be empty")
    path = unicodedata.normalize("NFC", value).replace("\\", "/")
    normalized = posixpath.normpath(path)
    if normalized.startswith("/") or normalized in {"", ".", ".."} or normalized.startswith("../"):
        raise ValueError("relative_path must stay within the ingestion root")
    return normalized


class SourceRecord(BaseModel):
    """Manifest 中描述一个 source 文件身份和内容版本的最小记录。"""

    document_id: str = Field(min_length=1, description="稳定逻辑文档 ID。")
    relative_path: str = Field(min_length=1, description="相对 ingestion root 的 NFC/POSIX 路径。")
    content_hash: str = Field(min_length=1, description="原始文件 bytes 的 SHA-256。")
    size_bytes: int | None = Field(default=None, ge=0, description="文件字节数，仅用于审计。")

    @model_validator(mode="after")
    def normalize_path(self) -> "SourceRecord":
        """保证 source path 的写入形式与 document_id 的路径口径一致。"""

        # source path 既用于展示也参与 corpus signature，必须在模型边界统一。
        self.relative_path = _normalize_relative_path(self.relative_path)
        return self


class PipelineIdentity(BaseModel):
    """只包含会改变 indexed representation 的 parser/chunker/embedding 身份。"""

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
        """确保配置 hash 真正对应配置内容，避免手工填错造成错误复用。"""

        computed = stable_hash(self.chunker_config)
        if self.chunker_config_hash is not None and self.chunker_config_hash != computed:
            raise ValueError("chunker_config_hash does not match chunker_config")
        self.chunker_config_hash = computed
        return self


class IndexIdentity(BaseModel):
    """Manifest 中描述一个具体 indexed representation 的状态和规模。"""

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
    """可持久化的知识库清单；所有身份字段都由其内容确定性派生。"""

    schema_version: str = Field(default="knowledge-base-manifest-v1", min_length=1)
    knowledge_base_id: str = Field(min_length=1, description="稳定的知识库逻辑身份。")
    corpus_signature: str = Field(min_length=1, description="source 集合及内容的签名。")
    sources: list[SourceRecord] = Field(description="按稳定路径排序的 source records。")
    pipeline: PipelineIdentity = Field(description="影响 indexed representation 的 pipeline 身份。")
    index: IndexIdentity = Field(description="当前 manifest 绑定的索引状态。")

    @model_validator(mode="after")
    def validate_derived_signatures(self) -> "KnowledgeBaseManifest":
        """拒绝 manifest 中 corpus/index signature 与内容不一致的伪造状态。"""

        expected_corpus = build_corpus_signature(self.sources)
        if self.corpus_signature != expected_corpus:
            raise ValueError("corpus_signature does not match sources")
        expected_index = build_index_signature(expected_corpus, self.pipeline)
        if self.index.index_signature != expected_index:
            raise ValueError("index_signature does not match corpus and pipeline")
        # 排序存储确保同一份 source 以同一顺序导出，避免调用方自行排序漂移。
        self.sources = sorted(self.sources, key=lambda item: (item.document_id, item.relative_path, item.content_hash))
        return self

    def canonical_dict(self) -> dict[str, Any]:
        """返回适合落盘和 hash 的稳定字典。"""

        return self.model_dump(mode="json", exclude_none=False)

    def canonical_bytes(self) -> bytes:
        """返回 manifest 的确定性 JSON bytes。"""

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
        """在不改变 corpus/index signature 的前提下生成新的生命周期视图。"""

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
    """提取 corpus signature 允许使用的字段，排除审计字段 size_bytes。"""

    # 文件大小不是内容身份的一部分；同一 bytes 的 size 已由 content_hash 完整决定。
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
    """根据 source identity/content 生成与 pipeline 无关的 corpus signature。"""

    return stable_hash(_source_signature_records(sources))


def build_index_signature(corpus_signature: str, pipeline: PipelineIdentity) -> str:
    """根据 corpus 与 indexing-relevant pipeline 生成 index signature。"""

    # 明确列出字段，防止 reranker、TopK、generator 等运行时参数意外进入索引身份。
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
    """从当前 Settings 组装正式 pipeline identity，不读取 retrieval-only 参数。"""

    # separator 与 special-token 口径是当前 hybrid/fixed Chunk 表示的一部分，显式冻结。
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
    """从真实 source/pipeline 数据创建并校验一个 manifest。"""

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
    """将 Loader 已完成的 Document 转为 manifest source records。"""

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
    """比较两次 explicit scan，并在 rename 无法唯一判断时返回 BLOCKED。"""

    previous_by_path = _index_sources(previous, "previous")
    current_by_path = _index_sources(current, "current")
    changes: list[SourceChange] = []
    unmatched_previous: list[SourceRecord] = []
    unmatched_current: list[SourceRecord] = []

    # 先按路径判断，确保同路径 hash 变化稳定地归类为 UPDATED。
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
    """一条 source 变更及其旧/新路径身份。"""

    status: SourceChangeStatus = Field(description="五种 source 同步状态之一。")
    relative_path: str = Field(min_length=1, description="当前路径或删除前路径。")
    old_relative_path: str | None = Field(default=None, description="RENAMED 的旧路径。")
    document_id: str | None = Field(default=None, description="当前文档 ID。")
    old_document_id: str | None = Field(default=None, description="旧版本文档 ID。")
    content_hash: str | None = Field(default=None, description="当前或被删除文件内容 hash。")


class SourceDiff(BaseModel):
    """一次 source scan 的稳定差异结果，BLOCKED 时禁止自动猜测 rename。"""

    changes: tuple[SourceChange, ...] = Field(description="按路径稳定排序的变更。")
    blocked: bool = Field(default=False, description="是否因 source identity 歧义而停止。")
    blocked_reason: str | None = Field(default=None, description="BLOCKED 的可诊断原因。")


def _index_sources(sources: Sequence[SourceRecord], label: str) -> dict[str, SourceRecord]:
    """校验一个 source inventory 的路径唯一性。"""

    indexed: dict[str, SourceRecord] = {}
    for source in sources:
        if source.relative_path in indexed:
            raise ValueError(f"duplicate {label} source path: {source.relative_path}")
        indexed[source.relative_path] = source
    return indexed


class RebuildDecision(BaseModel):
    """确定性 dense rebuild 决策及其原因。"""

    rebuild_required: bool = Field(description="是否需要新建 staging dense index。")
    blocked: bool = Field(default=False, description="是否因 source diff 无法安全决策。")
    reasons: tuple[str, ...] = Field(default=(), description="按固定顺序排列的决策原因。")


def decide_rebuild(
    source_diff: SourceDiff,
    *,
    previous_manifest: KnowledgeBaseManifest | None = None,
    current_manifest: KnowledgeBaseManifest | None = None,
) -> RebuildDecision:
    """按 source/index identity 判断是否重建，忽略 reranker 和查询侧配置。"""

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
    # 空 diff 且 signature 不变明确表示不需要重建；输出理由便于审计而非引入隐式行为。
    return RebuildDecision(rebuild_required=bool(reasons), reasons=tuple(reasons))
