"""Convert PDF, Markdown, and TXT files into the project's internal Document."""

import unicodedata
from pathlib import Path
from typing import Literal

import pymupdf

from evidenceflow.domain.document import Document, DocumentSegment
from evidenceflow.domain.ids import build_content_hash, build_document_id

FileType = Literal["pdf", "md", "txt"]

# Extension matching is always lower-cased, so files such as README.MD are still recognized.
SUPPORTED_FILE_TYPES: dict[str, FileType] = {
    ".pdf": "pdf",
    ".md": "md",
    ".txt": "txt",
}


def _resolve_source_path(path: Path, ingestion_root: Path) -> tuple[Path, str]:
    """Validate the file boundary and return the real file path plus its normalized relative path."""

    if not path.exists():
        raise FileNotFoundError(f"File does not exist: {path}")
    if not path.is_file():
        raise IsADirectoryError(f"Path is not a file: {path}")
    if not ingestion_root.exists():
        raise FileNotFoundError(f"Ingestion root does not exist: {ingestion_root}")
    if not ingestion_root.is_dir():
        raise NotADirectoryError(f"Ingestion root is not a directory: {ingestion_root}")

    # resolve() expands symlinks; the subsequent relative_to() check then prevents a link
    # that escapes the ingestion root from still being used to build a document_id.
    resolved_path = path.resolve()
    resolved_root = ingestion_root.resolve()
    try:
        relative_path = resolved_path.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"File is outside ingestion root: {path}") from exc

    # source_path and document_id use the same NFC + POSIX representation to avoid platform
    # differences.
    normalized_relative_path = unicodedata.normalize("NFC", relative_path.as_posix())
    return resolved_path, normalized_relative_path


def _load_pdf(path: Path) -> tuple[list[DocumentSegment], dict[str, int]]:
    """Extract PDF text in original page order, preserving the real 1-based page numbers."""

    segments: list[DocumentSegment] = []
    with pymupdf.open(path) as pdf:
        page_count = pdf.page_count
        for page_index, page in enumerate(pdf):
            content = page.get_text()
            # Empty pages produce no Segment, but page_index is never renumbered, so a later page
            # with text still keeps its true page number from the original PDF.
            if not content.strip():
                continue
            segments.append(
                DocumentSegment(
                    content=content,
                    page_number=page_index + 1,
                    metadata={},
                )
            )

    # OCR is not implemented at this stage; a PDF with no text at all must fail loudly rather than
    # yield an empty document.
    if not segments:
        raise ValueError("PDF contains no extractable text")
    return segments, {"page_count": page_count}


def _load_utf8_text(file_bytes: bytes) -> list[DocumentSegment]:
    """Return the whole UTF-8 text as a single page-less Segment."""

    # Decode the raw bytes read on the first pass: this avoids duplicate IO and guarantees
    # content_hash and
    # the parsed content come from the same file snapshot; a decode failure keeps Python's clear
    # UTF-8 exception.
    content = file_bytes.decode("utf-8")
    return [DocumentSegment(content=content, page_number=None, metadata={})]


def load_document(path: Path, ingestion_root: Path) -> Document:
    """Load one supported file and convert it into the unified Document Domain Model.

        Args:
            path: Path of the file to load; may be relative or absolute.
            ingestion_root: Knowledge-base ingestion root, and the path basis for document_id.

        Returns:
            A Document with a stable ID, the raw content Hash, and ordered Segments.
    """

    source_path, normalized_relative_path = _resolve_source_path(
        Path(path), Path(ingestion_root)
    )
    suffix = source_path.suffix.lower()
    try:
        file_type = SUPPORTED_FILE_TYPES[suffix]
    except KeyError as exc:
        display_suffix = suffix or "<no extension>"
        raise ValueError(f"Unsupported file type: {display_suffix}") from exc

    # content_hash must be based on the raw file bytes, never on PDF-extracted or decoded text.
    file_bytes = source_path.read_bytes()
    content_hash = build_content_hash(file_bytes)

    metadata: dict[str, int] = {"file_size": len(file_bytes)}
    if file_type == "pdf":
        segments, pdf_metadata = _load_pdf(source_path)
        metadata.update(pdf_metadata)
    else:
        # Markdown/TXT stay as the complete original text at the Loader stage; no heading parsing
        # and no cleaning.
        segments = _load_utf8_text(file_bytes)

    return Document(
        document_id=build_document_id(normalized_relative_path),
        source_path=normalized_relative_path,
        file_name=source_path.name,
        file_type=file_type,
        content_hash=content_hash,
        segments=segments,
        metadata=metadata,
    )
