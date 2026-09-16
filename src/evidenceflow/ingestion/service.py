"""组合摄取 primitives，提供确定性的增量知识库更新工作流。"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from evidenceflow.embedding.base import EmbeddingProvider
from evidenceflow.ingestion.chunker import chunk_document, chunk_document_structured
from evidenceflow.ingestion.cleaner import clean_document
from evidenceflow.ingestion.loader import SUPPORTED_FILE_TYPES, load_document
from evidenceflow.ingestion.structure import parse_structured_document
from evidenceflow.ingestion.tokenizer import TokenizerProtocol
from evidenceflow.storage.base import VectorStore

IngestionStatus = Literal["inserted", "skipped", "updated", "failed"]


@dataclass(frozen=True)
class DocumentIngestionResult:
    """记录单个文件的增量处理状态及可诊断失败阶段。"""

    source_path: str
    document_id: str | None
    status: IngestionStatus
    chunk_count: int = 0
    embedding_count: int = 0
    failed_stage: str | None = None
    error_message: str | None = None


@dataclass(frozen=True)
class IngestionSummary:
    """汇总一次单文件或目录摄取的确定性结果。"""

    files_discovered: int
    inserted: int
    updated: int
    skipped: int
    failed: int
    chunks_written: int
    embeddings_created: int
    results: tuple[DocumentIngestionResult, ...]


def _relative_source_path(path: Path, root: Path) -> str:
    """返回与 document_id 同源的 NFC、POSIX 相对路径。"""

    return unicodedata.normalize("NFC", path.resolve().relative_to(root.resolve()).as_posix())


def discover_source_files(input_path: Path, ingestion_root: Path) -> list[Path]:
    """在稳定 root 内发现受支持文件，并按规范化相对路径排序。"""

    root = Path(ingestion_root).resolve()
    target = Path(input_path).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Ingestion root does not exist: {ingestion_root}")
    if not target.exists():
        raise FileNotFoundError(f"Ingestion input does not exist: {input_path}")
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Ingestion input is outside configured root: {input_path}") from exc

    if target.is_file():
        if target.suffix.lower() not in SUPPORTED_FILE_TYPES:
            raise ValueError(f"Unsupported ingestion file type: {target.suffix or '<none>'}")
        return [target]
    if not target.is_dir():
        raise ValueError(f"Ingestion input is neither a file nor directory: {input_path}")

    # 目录中的非知识文件直接忽略；排序不依赖操作系统遍历顺序。
    files = [path for path in target.rglob("*") if path.is_file() and path.suffix.lower() in SUPPORTED_FILE_TYPES]
    return sorted(files, key=lambda path: _relative_source_path(path, root))


class IngestionService:
    """以固定 ingestion root 串行插入、跳过或替换知识文档。"""

    def __init__(
        self,
        ingestion_root: Path,
        tokenizer: TokenizerProtocol,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStore,
        *,
        chunk_size: int,
        chunk_overlap: int,
        chunk_strategy: str = "fixed",
    ) -> None:
        """注入已验证 primitive，并冻结当前服务的 Chunk 配置与策略。"""

        if chunk_strategy not in {"hybrid", "fixed"}:
            raise ValueError("chunk_strategy must be 'hybrid' or 'fixed'")
        self.ingestion_root = Path(ingestion_root).resolve()
        self.tokenizer = tokenizer
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunk_strategy = chunk_strategy

    def _ingest_file(self, path: Path) -> DocumentIngestionResult:
        """处理一个文件；更新时确保 Embed 成功后才删除旧版本。"""

        relative_path = _relative_source_path(path, self.ingestion_root)
        document = None
        stage = "load"
        try:
            document = load_document(path, self.ingestion_root)
            stage = "storage_check"
            exists = self.vector_store.contains_document(document.document_id)
            if exists and self.vector_store.contains_document(document.document_id, document.content_hash):
                return DocumentIngestionResult(relative_path, document.document_id, "skipped")

            stage = "clean"
            cleaned = clean_document(document)
            stage = "chunk"
            if self.chunk_strategy == "hybrid" and document.file_type == "pdf":
                # §9.3：hybrid + PDF 走结构解析 → 混合切分；解析在原始页文本的
                # block 视图上执行，不依赖 Cleaner 输出。
                stage = "structure"
                structured = parse_structured_document(document, path)
                stage = "chunk"
                chunks = chunk_document_structured(
                    document, structured.elements, self.tokenizer, self.chunk_size
                )
            else:
                # fixed 策略与 Markdown/TXT（无结构信息）保持原固定窗口路径不变。
                chunks = chunk_document(cleaned, self.tokenizer, self.chunk_size, self.chunk_overlap)
            stage = "embedding"
            # 一个文档一次 Batch，避免逐 Chunk 启动模型推理。
            embeddings = self.embedding_provider.embed_documents([chunk.content for chunk in chunks])
            if len(chunks) != len(embeddings):
                raise ValueError("chunk and embedding counts do not match")

            if exists:
                # 新版本已经完整准备好后才触碰旧数据，保住 Embed 失败时的旧版本。
                stage = "delete"
                self.vector_store.delete_document(document.document_id)
            stage = "store"
            self.vector_store.add_chunks(chunks, embeddings)
            return DocumentIngestionResult(
                relative_path,
                document.document_id,
                "updated" if exists else "inserted",
                chunk_count=len(chunks),
                embedding_count=len(embeddings),
            )
        except Exception as exc:
            # Batch 层需要继续后续文件，因此把异常转成带阶段的结果而不是吞掉诊断信息。
            return DocumentIngestionResult(
                source_path=relative_path,
                document_id=document.document_id if document is not None else None,
                status="failed",
                failed_stage=stage,
                error_message=f"{type(exc).__name__}: {exc}",
            )

    def ingest(self, input_path: Path | None = None) -> IngestionSummary:
        """串行摄取单文件或目录，并在单文件失败后继续处理。"""

        files = discover_source_files(input_path or self.ingestion_root, self.ingestion_root)
        results = tuple(self._ingest_file(path) for path in files)
        return IngestionSummary(
            files_discovered=len(files),
            inserted=sum(item.status == "inserted" for item in results),
            updated=sum(item.status == "updated" for item in results),
            skipped=sum(item.status == "skipped" for item in results),
            failed=sum(item.status == "failed" for item in results),
            chunks_written=sum(item.chunk_count for item in results if item.status in {"inserted", "updated"}),
            embeddings_created=sum(item.embedding_count for item in results if item.status in {"inserted", "updated"}),
            results=results,
        )
