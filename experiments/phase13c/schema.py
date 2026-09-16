"""Phase 13C 统一 ScientificDocument Schema（方案 §10）与文本规范化契约（§16）。

本模块定义所有 parser 输出必须映射到的**同一**中间表示。它的唯一目的是让
P0（当前工程）/ P1（MinerU）/ P2（Docling）/ P3（Marker）/ P4（GROBID）的输出可以逐元素
比较——如果每个 parser 用自己的结构，后面的 Reading Order / Table / Equation 指标
就无从谈起。

三个关键设计决定：

1. **raw 与 normalized 分离**。``raw_text`` 保留 parser 原始输出（含换行、连字符断行），
   用于度量"行结构是否丢失"；``normalized_text`` 才是面向比对/embedding 的形态。
2. **line_preserved 形态显式保存**。Phase 13A/13B 已证 sentencepiece 会把 ``\\n`` 折叠为
   空格，所以"换行是否被 parser 保住"必须在 schema 层记录，而不能事后从 token 反推。
3. **reading_order 是全局单调序号**，不依赖 parser 自己的块编号，因为各 parser 的 page/
   block 编号体系完全不同。
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Iterable, Sequence

from . import sandbox

# --------------------------------------------------------------------------------------
# 受控类型白名单（方案 §10）
# --------------------------------------------------------------------------------------

ELEMENT_TYPES: tuple[str, ...] = (
    "document_title",
    "author_block",
    "abstract",
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
    "watermark",
    "sidebar",
    "unknown",
)

ELEMENT_TYPE_SET = frozenset(ELEMENT_TYPES)

# 页面装饰类型：默认从 embedding corpus 排除，但 provenance 中保留删除记录（§17）。
FURNITURE_TYPES = frozenset(
    {"header", "footer", "page_number", "watermark", "sidebar"}
)
# 与正文有知识关系的结构化类型。
STRUCTURED_TYPES = frozenset(
    {"figure", "figure_caption", "table", "table_caption", "equation", "figure", "abstract"}
)
# 正文型类型。
BODY_TYPES = frozenset({"paragraph", "list_item", "section_header", "document_title", "reference_entry"})

# Schema 字段（方案 §10 原样），``None`` 表示该字段对本元素不适用。
ELEMENT_FIELDS: tuple[str, ...] = (
    "document_id",
    "element_id",
    "type",
    "raw_text",
    "normalized_text",
    "line_preserved_text",
    "page",
    "bbox",
    "reading_order",
    "section_path",
    "parent_id",
    "caption_of",
    "table_id",
    "figure_id",
    "formula_id",
    "source_parser",
    "source_ref",
    "confidence",
)

# 类型 -> 该类型允许非空的目标引用字段。
_CAPTION_OWNER = {"figure_caption": "figure", "table_caption": "table"}


def element_id_for(parser: str, document_id: str, page: int | None, index: int) -> str:
    """生成稳定的 element_id。

    为什么不沿用 13B 的 ``el-<12hex>``：13C 要在同一文档上并列多个 parser，
    身份必须含 parser 名，否则跨 arm 的 element_id 会碰撞，破坏 A/B 归属。
    """

    payload = f"{parser}|{document_id}|{page}|{index}"
    return f"el-{parser}-{sandbox.sha256_text(payload)[:12]}"


def make_element(
    *,
    document_id: str,
    element_id: str,
    type: str,
    raw_text: str,
    source_parser: str,
    page: int | None = None,
    bbox: Sequence[float] | None = None,
    reading_order: int | None = None,
    section_path: Sequence[str] | None = None,
    parent_id: str | None = None,
    caption_of: str | None = None,
    table_id: str | None = None,
    figure_id: str | None = None,
    formula_id: str | None = None,
    source_ref: str | None = None,
    confidence: float | None = None,
) -> dict[str, Any]:
    """按 §10 schema 构造一个 element，并补齐派生字段。"""

    if type not in ELEMENT_TYPE_SET:
        raise ValueError(f"Unknown element type: {type}")
    line_preserved = normalize_line_preserved(raw_text)
    return {
        "document_id": document_id,
        "element_id": element_id,
        "type": type,
        "raw_text": raw_text,
        "normalized_text": collapse_whitespace(join_hyphenated_breaks(line_preserved)),
        "line_preserved_text": line_preserved,
        "page": page,
        "bbox": [round(float(v), 2) for v in bbox] if bbox else None,
        "reading_order": reading_order,
        "section_path": list(section_path or []),
        "parent_id": parent_id,
        "caption_of": caption_of,
        "table_id": table_id,
        "figure_id": figure_id,
        "formula_id": formula_id,
        "source_parser": source_parser,
        "source_ref": source_ref,
        "confidence": confidence,
    }


def validate_element(element: dict[str, Any]) -> list[str]:
    """校验单个 element 是否满足 schema；返回问题列表（空列表表示合法）。"""

    problems: list[str] = []
    for field in ELEMENT_FIELDS:
        if field not in element:
            problems.append(f"missing field: {field}")
    if problems:
        return problems

    if element["type"] not in ELEMENT_TYPE_SET:
        problems.append(f"illegal type: {element['type']}")
    if not isinstance(element["document_id"], str) or not element["document_id"]:
        problems.append("document_id must be a non-empty string")
    if not isinstance(element["element_id"], str) or not element["element_id"]:
        problems.append("element_id must be a non-empty string")
    if not isinstance(element["raw_text"], str):
        problems.append("raw_text must be a string")
    if not isinstance(element["normalized_text"], str):
        problems.append("normalized_text must be a string")
    if not isinstance(element["section_path"], list) or not all(
        isinstance(part, str) for part in element["section_path"]
    ):
        problems.append("section_path must be a list of strings")
    if element["page"] is not None and not isinstance(element["page"], int):
        problems.append("page must be int or null")
    if element["bbox"] is not None:
        bbox = element["bbox"]
        if not isinstance(bbox, list) or len(bbox) != 4 or not all(
            isinstance(v, (int, float)) for v in bbox
        ):
            problems.append("bbox must be [x0,y0,x1,y1] of numbers or null")
    if element["reading_order"] is not None and not isinstance(element["reading_order"], int):
        problems.append("reading_order must be int or null")
    return problems


def validate_document(document: dict[str, Any]) -> dict[str, Any]:
    """校验一个 ScientificDocument，返回结构化校验结果。"""

    problems: list[dict[str, Any]] = []
    elements = document.get("elements")
    if not isinstance(elements, list):
        return {"valid": False, "problems": [{"scope": "document", "issue": "elements not a list"}]}

    seen: set[str] = set()
    for index, element in enumerate(elements):
        for issue in validate_element(element):
            problems.append({"scope": f"element[{index}]", "issue": issue})
        eid = element.get("element_id")
        if isinstance(eid, str):
            if eid in seen:
                problems.append({"scope": f"element[{index}]", "issue": f"duplicate element_id {eid}"})
            seen.add(eid)

    orders = [e["reading_order"] for e in elements if e.get("reading_order") is not None]
    # reading_order 必须在文档内唯一且能排序；重复意味着 parser 的块编号未去重。
    if len(orders) != len(set(orders)):
        problems.append({"scope": "document", "issue": "reading_order values are not unique"})

    return {
        "valid": not problems,
        "problems": problems,
        "element_count": len(elements),
        "type_counts": type_counts(elements),
    }


def type_counts(elements: Iterable[dict[str, Any]]) -> dict[str, int]:
    """统计元素类型分布。"""

    counts: dict[str, int] = {}
    for element in elements:
        counts[element["type"]] = counts.get(element["type"], 0) + 1
    return dict(sorted(counts.items()))


def make_document(
    *,
    document_id: str,
    source_pdf: str,
    source_sha256: str,
    parser: str,
    page_count: int,
    elements: Sequence[dict[str, Any]],
    layout_class: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """装配 ScientificDocument（方案 §10 的文档级容器）。"""

    payload: dict[str, Any] = {
        "document_id": document_id,
        "source_pdf": source_pdf,
        "source_sha256": source_sha256,
        "parser": parser,
        "page_count": page_count,
        "layout_class": layout_class,
        "element_count": len(elements),
        "elements": list(elements),
    }
    if extra:
        payload.update(extra)
    return payload


# --------------------------------------------------------------------------------------
# 文本规范化（方案 §16：允许 / 禁止清单）
# --------------------------------------------------------------------------------------

# 断行连字符：``trans-\nformer`` -> ``transformer``。仅当左侧是字母、右侧是字母时才合并，
# 避免把真正的复合词 ``state-\nof-the-art`` 之外的"列表符号 -"误伤。
_HYPHEN_BREAK = re.compile(r"(?<=[A-Za-z])[-\u2010\u2011]\n[ \t]*(?=[A-Za-z])")
# 分词：保留内部连字符/小数点/斜杠，便于与数值、术语对齐。
_TOKEN = re.compile(r"[A-Za-z0-9]+(?:[.\-/][A-Za-z0-9]+)*")
_WHITESPACE = re.compile(r"\s+")


def normalize_unicode(text: str) -> str:
    """NFC 归一 + 统一换行 + 去掉文本开头的 BOM。"""

    normalized = unicodedata.normalize("NFC", text)
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    if normalized.startswith("\ufeff"):
        normalized = normalized[1:]
    return normalized


def join_hyphenated_breaks(text: str) -> str:
    """修复跨行断词：把 ``trans-\\nformer`` 还原为 ``transformer``。

    **不**替换换行为空格——段落内的视觉换行是否应合并属于 Cleaner 的决定（§16），
    Schema 层只负责消除连字符噪声。
    """

    return _HYPHEN_BREAK.sub("", text)


def normalize_line_preserved(raw_text: str) -> str:
    """raw_text 的保守规范化：只做 Unicode/换行/BOM，并删除行尾空白。"""

    text = normalize_unicode(raw_text)
    return re.sub(r"[ \t]+(?=\n)", "", text)


def collapse_whitespace(text: str) -> str:
    """把任意空白（含换行）折叠为单个空格并 strip。"""

    return _WHITESPACE.sub(" ", text).strip()


def tokenize(text: str) -> list[str]:
    """小写化分词，供文本保真度/对齐指标使用。"""

    return [match.group(0).lower() for match in _TOKEN.finditer(text)]


def normalized_8grams(text: str, n: int = 8) -> set[str]:
    """返回词级 n-gram 集合（Real corpus 的 ``normalized_8gram_coverage`` 指标用）。"""

    tokens = tokenize(text)
    if len(tokens) < n:
        return {" ".join(tokens)} if tokens else set()
    return {" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def line_count(text: str) -> int:
    """非空行数，用于度量"行结构是否保留"。"""

    return sum(1 for line in text.split("\n") if line.strip())


# --------------------------------------------------------------------------------------
# 从 Phase 13B 的 8 类元素映射到 13C 的 19 类 schema
# --------------------------------------------------------------------------------------

# 13B 的 ``furniture`` 是可再分的：13C 要求把装饰细分为 header/footer/page_number/
# watermark/sidebar，否则"该删哪一类装饰"的指标无法计算。
_FURNITURE_RULES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"^arXiv:\d{4}\.\d{4,5}v?\d*", re.I), "sidebar"),
    (re.compile(r"^(?:preprint|under review|submitted to|accepted at)\b", re.I), "sidebar"),
    (re.compile(r"^©|^copyright|all rights reserved", re.I), "watermark"),
    (re.compile(r"^(?:draft|confidential|do not distribute)", re.I), "watermark"),
    (re.compile(r"^\d{1,4}$"), "page_number"),
    (re.compile(r"^page\s+\d+", re.I), "footer"),
)


def classify_furniture(text: str, page: int | None, is_first_page: bool) -> str:
    """把 13B 的 ``furniture`` 文本细分为 header/footer/page_number/watermark/sidebar。

    判据优先级：arXiv 侧栏 / 水印（内容自证）> 纯数字（页码或页眉页脚）>
    首页与非首页的位置启发（首页顶部视为 header，其余视为 footer）。
    """

    stripped = collapse_whitespace(text)
    for pattern, kind in _FURNITURE_RULES:
        if pattern.search(stripped):
            # 纯数字无法从文本判断是 header 还是 footer：统一归为 page_number，
            # 因为对检索而言两者的处理完全一致，且避免了不可自证的猜测。
            return kind
    if len(stripped) <= 120:
        return "header" if is_first_page else "footer"
    return "unknown"


def map_13b_type(
    element_13b: dict[str, Any],
    *,
    is_document_title: bool,
    is_first_page: bool,
) -> tuple[str, str | None]:
    """把 13B 的 element_type 映射为 13C 的 type，并返回细分的 furniture 子类。

    Returns:
        ``(type, furniture_subtype)``；非 furniture 元素的第二个返回值为 ``None``。
    """

    legacy = element_13b["element_type"]
    text = element_13b["text"]
    if legacy == "title":
        return ("document_title" if is_document_title else "section_header"), None
    if legacy in {"paragraph", "figure_caption", "table_caption", "table", "equation", "reference_entry"}:
        return legacy, None
    if legacy == "furniture":
        subtype = classify_furniture(text, element_13b.get("page"), is_first_page)
        return subtype, subtype
    return "unknown", None
