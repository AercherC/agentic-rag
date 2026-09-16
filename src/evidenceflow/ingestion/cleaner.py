"""对 DocumentSegment 执行保守、结构感知的文本规范化。"""

import re
import unicodedata

from evidenceflow.domain.document import Document, DocumentSegment


class DocumentCleaningError(ValueError):
    """表示整个 Document 清洗后不再包含可用文本。"""


def _normalize_common_text(text: str) -> str:
    """执行所有文件类型共享且不会改变文本结构的规范化。"""

    # 先统一 Unicode 表示，再统一跨平台换行，保证后续规则面对稳定输入。
    normalized = unicodedata.normalize("NFC", text)
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")

    # BOM 只有位于文本开头时才是编码标记；正文中的 U+FEFF 不应被误删。
    if normalized.startswith("\ufeff"):
        normalized = normalized[1:]
    return normalized


def _strip_edge_blank_lines(text: str) -> str:
    """删除首尾纯空白行，同时保留首末内容行中可能有意义的缩进。"""

    lines = text.split("\n")
    start = 0
    end = len(lines)
    while start < end and not lines[start].strip():
        start += 1
    while end > start and not lines[end - 1].strip():
        end -= 1
    return "\n".join(lines[start:end])


def _clean_plain_text(text: str) -> str:
    """保守清理 PDF/TXT 行尾空白和过量空行。"""

    normalized = _normalize_common_text(text)
    # 普通文本的行尾空格与 Tab 不承载结构，可安全删除；不触碰行内空格。
    normalized = re.sub(r"[ \t]+$", "", normalized, flags=re.MULTILINE)
    normalized = _strip_edge_blank_lines(normalized)
    # 三个换行代表至少两个连续空白行，统一为一个空白行即可保留段落边界。
    return re.sub(r"\n{3,}", "\n\n", normalized)


def _clean_markdown_text(text: str) -> str:
    """仅规范化 Markdown 的 Unicode、换行、BOM 与边缘空白行。"""

    normalized = _normalize_common_text(text)
    # Markdown 行尾空格、内部空行和缩进都可能具有语法意义，因此只处理
    # Segment 外围的纯空白行，不复用普通文本的空白压缩规则。
    return _strip_edge_blank_lines(normalized)


def _clean_segment(segment: DocumentSegment, file_type: str) -> DocumentSegment | None:
    """清洗单个 Segment；清洗后无内容时返回 None。"""

    if file_type == "md":
        cleaned_text = _clean_markdown_text(segment.content)
    else:
        cleaned_text = _clean_plain_text(segment.content)

    # 使用 strip() 仅判断是否还存在可用字符，不把 strip 后的结果写回文本。
    if not cleaned_text.strip():
        return None
    # 深复制可避免返回对象与输入对象共享可变 metadata。
    return segment.model_copy(update={"content": cleaned_text}, deep=True)


def clean_document(document: Document) -> Document:
    """返回清洗后的新 Document，并保持来源身份和页面边界不变。

    Args:
        document: Loader 产生的项目内部 Document。

    Returns:
        仅 Segment 文本可能变化的全新 Document。

    Raises:
        DocumentCleaningError: 所有 Segment 清洗后均没有可用文本。
    """

    cleaned_segments: list[DocumentSegment] = []
    for segment in document.segments:
        cleaned_segment = _clean_segment(segment, document.file_type)
        if cleaned_segment is not None:
            # 保持原遍历顺序，因此 PDF 页码不会被重排或重新编号。
            cleaned_segments.append(cleaned_segment)

    if not cleaned_segments:
        raise DocumentCleaningError("Document contains no usable text after cleaning")

    # document_id、content_hash 与其他来源字段均来自原文档；只替换 Segments，
    # 且采用深复制保证 Cleaner 不会原地修改 Loader 的输出。
    return document.model_copy(update={"segments": cleaned_segments}, deep=True)
