"""Compose the ingestion primitives into a deterministic incremental knowledge-base update workflow."""

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
    """Incremental processing status of a single file, plus the diagnosable failure stage."""

    source_path: str
    document_id: str | None
    status: IngestionStatus
    chunk_count: int = 0
    embedding_count: int = 0
    failed_stage: str | None = None
    error_message: str | None = None


@dataclass(frozen=True)
class IngestionSummary:
    """Deterministic result summary of one single-file or directory ingestion."""

    files_discovered: int
    inserted: int
    updated: int
    skipped: int
    failed: int
    chunks_written: int
    embeddings_created: int
    results: tuple[DocumentIngestionResult, ...]


def _relative_source_path(path: Path, root: Path) -> str:
    """Return the NFC, POSIX relative path that shares its basis with document_id."""

    return unicodedata.normalize("NFC", path.resolve().relative_to(root.resolve()).as_posix())


def discover_source_files(input_path: Path, ingestion_root: Path) -> list[Path]:
    """Discover supported files inside the stable root and sort them by normalized relative path."""

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

    # Non-knowledge files in the directory are ignored; the sort does not depend on the OS
    # traversal order.
    files = [path for path in target.rglob("*") if path.is_file() and path.suffix.lower() in SUPPORTED_FILE_TYPES]
    return sorted(files, key=lambda path: _relative_source_path(path, root))


class IngestionService:
    """Insert, skip, or replace knowledge documents one at a time under a fixed ingestion root."""

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
        """Inject the already-validated primitives and freeze this service's Chunk configuration and strategy."""

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
        """Process one file; on update, delete the old version only after Embedding has succeeded."""

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
                # §9.3: hybrid + PDF goes through structural parsing -> hybrid chunking; parsing
                # runs on the
                # block view of the original page text and does not depend on Cleaner output.
                stage = "structure"
                structured = parse_structured_document(document, path)
                stage = "chunk"
                chunks = chunk_document_structured(
                    document, structured.elements, self.tokenizer, self.chunk_size
                )
            else:
                # The fixed strategy and Markdown/TXT (no structural information) keep the original
                # fixed-window path unchanged.
                chunks = chunk_document(cleaned, self.tokenizer, self.chunk_size, self.chunk_overlap)
            stage = "embedding"
            # One Batch per document, avoiding a model inference startup per Chunk.
            embeddings = self.embedding_provider.embed_documents([chunk.content for chunk in chunks])
            if len(chunks) != len(embeddings):
                raise ValueError("chunk and embedding counts do not match")

            if exists:
                # Touch the old data only after the new version is fully prepared, preserving the
                # old version if Embedding fails.
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
            # The Batch layer must continue with later files, so the exception becomes a staged
            # result rather than swallowed diagnostics.
            return DocumentIngestionResult(
                source_path=relative_path,
                document_id=document.document_id if document is not None else None,
                status="failed",
                failed_stage=stage,
                error_message=f"{type(exc).__name__}: {exc}",
            )

    def ingest(self, input_path: Path | None = None) -> IngestionSummary:
        """Ingest a single file or a directory serially, continuing after an individual file fails."""

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
