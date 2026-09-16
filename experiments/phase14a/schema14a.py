"""Phase 14A 统一 StructuredDocument Schema（方案 §4 / §5）。

设计原则（与 13C 的 PDF-only schema 不同）：

1. **common core + format-specific payload**，而不是强迫 XLSX 模拟 PDF 页码。
2. ``source_ref`` 必须按格式变化：
   - DOCX：``{paragraph_index, xml_path, style_id, table_index, cell}`` —— 不强求 page+bbox。
   - XLSX：``{sheet, range, cell}`` —— 不得有 fake page / fake bbox。
   - PDF ：``{page, bbox}`` —— 仅作已冻结参考 adapter 的对照。
3. 统一字段（方案 §4 原样）：``document_id, source_format, source_file, elements,
   relationships, document_metadata``；共同 Element 字段见 ``ELEMENT_FIELDS``。

本模块只依赖标准库，因此既可被主 harness 导入，也可被沙盒 venv 内的 runner 子进程导入。
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Iterable, Sequence

from . import sandbox14a as sandbox

# 支持的源格式。
SOURCE_FORMATS: tuple[str, ...] = ("pdf", "docx", "xlsx")

# 统一的 Element 字段（方案 §4）。source_ref 内容随 source_format 而变。
ELEMENT_FIELDS: tuple[str, ...] = (
    "document_id",
    "element_id",
    "type",
    "raw_text",
    "normalized_text",
    "parent_id",
    "children",
    "source_ref",
    "retrieval_role",
    "metadata",
    "source_parser",
)

# source_ref 的合法键集合（按格式）。
DOCX_SOURCE_REF_KEYS = frozenset(
    {"paragraph_index", "xml_path", "style_id", "table_index", "cell"}
)
XLSX_SOURCE_REF_KEYS = frozenset({"sheet", "range", "cell", "table_name", "inferred"})
PDF_SOURCE_REF_KEYS = frozenset({"page", "bbox"})

# 建议的 element type 取值（不强制穷举；未知 type 标记为 unknown 即可）。
DOCX_ELEMENT_TYPES = frozenset(
    {
        "document_title",
        "heading",
        "paragraph",
        "list_item",
        "table",
        "table_row",
        "table_cell",
        "caption",
        "footnote",
        "endnote",
        "header",
        "footer",
        "image",
        "unknown",
    }
)
XLSX_ELEMENT_TYPES = frozenset(
    {
        "workbook_metadata",
        "sheet",
        "table",
        "table_header",
        "row_group",
        "cell",
        "key_value",
        "note",
        "unknown",
    }
)
PDF_ELEMENT_TYPES = frozenset(
    {  # 与 13C schema 对齐，仅作参考 adapter 对照
        "document_title",
        "section_header",
        "paragraph",
        "list_item",
        "figure",
        "figure_caption",
        "table",
        "table_caption",
        "equation",
        "footnote",
        "reference_entry",
        "header",
        "footer",
        "page_number",
        "unknown",
    }
)

TYPE_SET_BY_FORMAT: dict[str, frozenset[str]] = {
    "docx": DOCX_ELEMENT_TYPES,
    "xlsx": XLSX_ELEMENT_TYPES,
    "pdf": PDF_ELEMENT_TYPES,
}

# 解析器无关的类型白名单（用于校验，但允许 unknown 透传）。
KNOWN_TYPES = DOCX_ELEMENT_TYPES | XLSX_ELEMENT_TYPES | PDF_ELEMENT_TYPES

_WHITESPACE = re.compile(r"\s+")


def normalize_unicode(text: str) -> str:
    """NFC 归一 + 统一换行 + 去掉开头 BOM。"""
    normalized = unicodedata.normalize("NFC", text)
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    if normalized.startswith("\ufeff"):
        normalized = normalized[1:]
    return normalized


def collapse_whitespace(text: str) -> str:
    """任意空白（含换行）折叠为单空格并 strip。"""
    return _WHITESPACE.sub(" ", text).strip()


def tokenize(text: str) -> list[str]:
    return [m.group(0).lower() for m in re.finditer(r"[A-Za-z0-9]+(?:[.\-/][A-Za-z0-9]+)*", text)]


def source_ref_keys_for(fmt: str) -> frozenset[str]:
    return {
        "docx": DOCX_SOURCE_REF_KEYS,
        "xlsx": XLSX_SOURCE_REF_KEYS,
        "pdf": PDF_SOURCE_REF_KEYS,
    }.get(fmt, frozenset())


def make_element(
    *,
    document_id: str,
    element_id: str,
    type: str,
    raw_text: str,
    source_format: str,
    source_ref: dict[str, Any],
    source_parser: str,
    parent_id: str | None = None,
    children: Sequence[str] | None = None,
    retrieval_role: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """构造一个统一 Element，并补齐派生字段。"""

    if source_format not in SOURCE_FORMATS:
        raise ValueError(f"Unknown source_format: {source_format}")
    if type not in KNOWN_TYPES:
        type = "unknown"
    normalized = collapse_whitespace(normalize_unicode(raw_text)) if raw_text else ""
    return {
        "document_id": document_id,
        "element_id": element_id,
        "type": type,
        "raw_text": normalize_unicode(raw_text),
        "normalized_text": normalized,
        "parent_id": parent_id,
        "children": list(children or []),
        "source_ref": dict(source_ref or {}),
        "retrieval_role": retrieval_role,
        "metadata": dict(metadata or {}),
        "source_parser": source_parser,
    }


def make_document(
    *,
    document_id: str,
    source_format: str,
    source_file: str,
    elements: Sequence[dict[str, Any]],
    relationships: Sequence[dict[str, Any]] | None = None,
    document_metadata: dict[str, Any] | None = None,
    source_sha256: str | None = None,
) -> dict[str, Any]:
    """装配统一 StructuredDocument（方案 §4）。"""

    if source_format not in SOURCE_FORMATS:
        raise ValueError(f"Unknown source_format: {source_format}")
    payload: dict[str, Any] = {
        "document_id": document_id,
        "source_format": source_format,
        "source_file": source_file,
        "source_sha256": source_sha256 or sandbox.sha256_text(source_file),
        "element_count": len(elements),
        "elements": [dict(e) for e in elements],
        "relationships": list(relationships or []),
        "document_metadata": dict(document_metadata or {}),
    }
    return payload


def validate_element(element: dict[str, Any]) -> list[str]:
    """校验单个 element；返回问题列表（空列表表示合法）。"""

    problems: list[str] = []
    for field in ELEMENT_FIELDS:
        if field not in element:
            problems.append(f"missing field: {field}")
    if problems:
        return problems

    fmt = element.get("source_format") or element.get("document_metadata", {}).get("source_format")
    # source_format 在 document 层；element 层通过 document 关联。这里若 element 自带则校验。
    if not isinstance(element["document_id"], str) or not element["document_id"]:
        problems.append("document_id must be a non-empty string")
    if not isinstance(element["element_id"], str) or not element["element_id"]:
        problems.append("element_id must be a non-empty string")
    if not isinstance(element["raw_text"], str):
        problems.append("raw_text must be a string")
    if not isinstance(element["normalized_text"], str):
        problems.append("normalized_text must be a string")
    if not isinstance(element["children"], list):
        problems.append("children must be a list")
    if not isinstance(element["source_ref"], dict):
        problems.append("source_ref must be a dict")
    if not isinstance(element["metadata"], dict):
        problems.append("metadata must be a dict")
    return problems


def validate_source_ref(source_ref: dict[str, Any], fmt: str) -> list[str]:
    """校验 source_ref 的键集合是否与格式匹配（方案 §5：禁止伪造页码/bbox）。"""

    problems: list[str] = []
    allowed = source_ref_keys_for(fmt)
    if not allowed:
        return [f"unknown format for source_ref check: {fmt}"]
    for key in source_ref:
        if key not in allowed:
            problems.append(f"source_ref key '{key}' not allowed for format '{fmt}'")
    # 禁止跨格式伪造：XLSX/DOCX 不得出现 PDF 专属键。
    if fmt in ("docx", "xlsx"):
        for forbidden in ("page", "bbox"):
            if forbidden in source_ref:
                problems.append(f"format '{fmt}' must not carry fake PDF ref '{forbidden}'")
    return problems


def validate_document(document: dict[str, Any]) -> dict[str, Any]:
    """校验整个 StructuredDocument。"""

    problems: list[dict[str, Any]] = []
    fmt = document.get("source_format")
    elements = document.get("elements")
    if not isinstance(elements, list):
        return {"valid": False, "problems": [{"scope": "document", "issue": "elements not a list"}]}

    seen: set[str] = set()
    for index, element in enumerate(elements):
        for issue in validate_element(element):
            problems.append({"scope": f"element[{index}]", "issue": issue})
        # source_ref 格式一致性：element 自带 source_format 时检查，否则由 document 层传递。
        sr = element.get("source_ref") or {}
        for issue in validate_source_ref(sr, fmt):
            problems.append({"scope": f"element[{index}]", "issue": issue})
        eid = element.get("element_id")
        if isinstance(eid, str):
            if eid in seen:
                problems.append({"scope": f"element[{index}]", "issue": f"duplicate element_id {eid}"})
            seen.add(eid)

    return {
        "valid": not problems,
        "problems": problems,
        "element_count": len(elements),
        "type_counts": type_counts(elements),
    }


def type_counts(elements: Iterable[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for element in elements:
        counts[element.get("type", "unknown")] = counts.get(element.get("type", "unknown"), 0) + 1
    return dict(sorted(counts.items()))


def digest_of(document: dict[str, Any]) -> str:
    """对规范化 JSON 取 SHA-256，用于确定性门（方案 §12 W-G8 / X-G8）。"""
    return sandbox.sha256_json(document)
