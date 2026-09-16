"""将 PDF、Markdown 和 TXT 文件转换为项目内部 Document。"""

import unicodedata
from pathlib import Path
from typing import Literal

import pymupdf

from evidenceflow.domain.document import Document, DocumentSegment
from evidenceflow.domain.ids import build_content_hash, build_document_id

FileType = Literal["pdf", "md", "txt"]

# 扩展名匹配统一转为小写，因此 README.MD 等文件也能被正确识别。
SUPPORTED_FILE_TYPES: dict[str, FileType] = {
    ".pdf": "pdf",
    ".md": "md",
    ".txt": "txt",
}


def _resolve_source_path(path: Path, ingestion_root: Path) -> tuple[Path, str]:
    """校验文件边界，并返回真实文件路径及规范化相对路径。"""

    if not path.exists():
        raise FileNotFoundError(f"File does not exist: {path}")
    if not path.is_file():
        raise IsADirectoryError(f"Path is not a file: {path}")
    if not ingestion_root.exists():
        raise FileNotFoundError(f"Ingestion root does not exist: {ingestion_root}")
    if not ingestion_root.is_dir():
        raise NotADirectoryError(f"Ingestion root is not a directory: {ingestion_root}")

    # resolve() 会展开符号链接；随后再检查 relative_to()，可防止链接逃逸出
    # ingestion root 后仍被错误地用于生成 document_id。
    resolved_path = path.resolve()
    resolved_root = ingestion_root.resolve()
    try:
        relative_path = resolved_path.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"File is outside ingestion root: {path}") from exc

    # source_path 与 document_id 使用同一套 NFC + POSIX 表示，避免平台差异。
    normalized_relative_path = unicodedata.normalize("NFC", relative_path.as_posix())
    return resolved_path, normalized_relative_path


def _load_pdf(path: Path) -> tuple[list[DocumentSegment], dict[str, int]]:
    """按原始页面顺序提取 PDF 文本，并保留真实的 1-based 页码。"""

    segments: list[DocumentSegment] = []
    with pymupdf.open(path) as pdf:
        page_count = pdf.page_count
        for page_index, page in enumerate(pdf):
            content = page.get_text()
            # 空页不生成 Segment，但 page_index 不会重排，因此后续有文本页面
            # 仍保留其在原 PDF 中的真实页码。
            if not content.strip():
                continue
            segments.append(
                DocumentSegment(
                    content=content,
                    page_number=page_index + 1,
                    metadata={},
                )
            )

    # 当前阶段不实现 OCR；整份 PDF 无文本时应明确失败，而不是返回空文档。
    if not segments:
        raise ValueError("PDF contains no extractable text")
    return segments, {"page_count": page_count}


def _load_utf8_text(file_bytes: bytes) -> list[DocumentSegment]:
    """将完整 UTF-8 文本作为一个无页码 Segment 返回。"""

    # 直接解码首次读取的原始 bytes，既避免重复 I/O，也保证 content_hash 与
    # 实际解析的是同一个文件快照；解码失败时保留 Python 清晰的 UTF-8 异常。
    content = file_bytes.decode("utf-8")
    return [DocumentSegment(content=content, page_number=None, metadata={})]


def load_document(path: Path, ingestion_root: Path) -> Document:
    """加载一个受支持文件并转换为统一 Document Domain Model。

    Args:
        path: 待加载文件路径，可以是相对路径或绝对路径。
        ingestion_root: 知识库摄取根目录，也是 document_id 的路径基准。

    Returns:
        包含稳定 ID、原始内容 Hash 和有序 Segment 的 Document。
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

    # content_hash 必须基于原始文件 bytes，不能使用 PDF 提取文本或解码后文本。
    file_bytes = source_path.read_bytes()
    content_hash = build_content_hash(file_bytes)

    metadata: dict[str, int] = {"file_size": len(file_bytes)}
    if file_type == "pdf":
        segments, pdf_metadata = _load_pdf(source_path)
        metadata.update(pdf_metadata)
    else:
        # Markdown/TXT 在 Loader 阶段保持完整原文，不解析标题，也不执行清洗。
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
