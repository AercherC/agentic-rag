"""把各 parser 的**原生输出**映射到 13C 统一 ScientificDocument schema。

## 为什么映射放在项目侧、而不是各 arm 内部

各 arm 只负责"跑工具、把工具的原生输出写到 sandbox"。把原生输出 → 统一 schema 的转换放在
项目侧（纯 JSON 变换，零第三方依赖），有三个直接好处：

1. **原生输出被完整保留**，任何映射争议都可以回溯到原始文件；
2. 映射逻辑只有一份，跨 arm 的坐标/类型口径不可能漂移；
3. 主 harness 不需要 import docling/marker/mineru，因此"项目 venv 未被污染"是可验证事实。

## 坐标口径

统一为：PDF 点（1/72 inch）、**左上原点**、``[x0, y0, x1, y1]``。
各 arm 的原始坐标在此处归一：

- Docling：点为整数、原点可能是 ``BOTTOMLEFT`` -> 用页高翻转；
- MinerU ``content_list``：``bbox`` 是 **0–1000 归一化**坐标 -> 用页宽/页高换算；
- Marker：点、左上原点，直接使用。

每个映射器都会在输出里写 ``bbox_source``，避免"看起来一样但其实不同坐标系"的隐性错误。
"""

from __future__ import annotations

import itertools
import re
from typing import Any, Sequence

from . import schema

# --------------------------------------------------------------------------------------
# 通用工具
# --------------------------------------------------------------------------------------


def _points_from_top_left(
    bbox: Sequence[float] | None, page_height: float | None, origin: str | None
) -> list[float] | None:
    """按坐标原点归一为左上原点的 PDF 点坐标。"""

    if not bbox or len(bbox) != 4:
        return None
    x0, y0, x1, y1 = (float(v) for v in bbox)
    if origin and "BOTTOM" in origin.upper() and page_height:
        # BOTTOMLEFT：t 是距底部的上边，b 是距底部的下边。
        top = page_height - max(y0, y1)
        bottom = page_height - min(y0, y1)
        return [round(min(x0, x1), 2), round(top, 2), round(max(x0, x1), 2), round(bottom, 2)]
    return [round(min(x0, x1), 2), round(min(y0, y1), 2), round(max(x0, x1), 2), round(max(y0, y1), 2)]


def _norm_1000_bbox(
    bbox: Sequence[float] | None, page_width: float | None, page_height: float | None
) -> list[float] | None:
    """把 0–1000 归一化 bbox 换算为 PDF 点。"""

    if not bbox or len(bbox) != 4 or not page_width or not page_height:
        return None
    x0, y0, x1, y1 = (float(v) for v in bbox)
    return [
        round(min(x0, x1) / 1000.0 * page_width, 2),
        round(min(y0, y1) / 1000.0 * page_height, 2),
        round(max(x0, x1) / 1000.0 * page_width, 2),
        round(max(y0, y1) / 1000.0 * page_height, 2),
    ]


_REF = re.compile(r"^#/")


def _strip_html(text: str) -> str:
    """去掉 HTML 标签，把 ``<br>`` 转为换行，用于表格/公式的文本化。"""

    value = re.sub(r"<br\s*/?>", "\n", text or "", flags=re.I)
    value = re.sub(r"</(?:td|th|tr|p|div)>", " ", value, flags=re.I)
    return re.sub(r"<[^>]+>", "", value)


def _table_from_html(html: str) -> dict[str, Any] | None:
    """从 HTML 表格抽取 ``rows`` 与跨列信息（供行列关系指标使用）。"""

    if not html or "<tr" not in html.lower():
        return None
    rows: list[list[str]] = []
    has_span = bool(re.search(r"colspan\s*=|rowspan\s*=", html, flags=re.I))
    for row_html in re.findall(r"<tr[^>]*>(.*?)</tr>", html, flags=re.I | re.S):
        cells = re.findall(r"<t[hd][^>]*>(.*?)</t[hd]>", row_html, flags=re.I | re.S)
        if not cells:
            continue
        # 跨列单元格在 HTML 里只有一个 <td>：按 colspan 补空占位，使列索引可与 GT 对齐。
        expanded: list[str] = []
        spans = re.findall(r"<t[hd]([^>]*)>", row_html, flags=re.I)
        for index, cell in enumerate(cells):
            expanded.append(_strip_html(cell).strip())
            colspan_match = re.search(r"colspan\s*=\s*[\"']?(\d+)", spans[index], flags=re.I) if index < len(spans) else None
            if colspan_match:
                for _ in range(int(colspan_match.group(1)) - 1):
                    expanded.append("")
        rows.append(expanded)
    if not rows:
        return None
    return {
        "rows": rows,
        "header_row": True,
        "has_spanning_cell": has_span,
        "spanning_rows": [
            index
            for index, row in enumerate(rows)
            if len(row) > 1 and all(not cell for cell in row[1:])
        ],
    }


def _table_from_docling(data: dict[str, Any]) -> dict[str, Any] | None:
    """把 Docling 的 ``table_cells`` 还原为二维网格。

    Docling 的每个 cell 自带 ``row_index``/``col_index``（0-based），跨行列由
    ``row_span``/``col_span`` 表达，因此可以直接定位，不需要猜测布局。
    """

    cells = data.get("table_cells") or []
    if not cells:
        return None
    num_rows = int(data.get("num_rows") or 0)
    num_cols = int(data.get("num_cols") or 0)
    if num_rows <= 0 or num_cols <= 0:
        num_rows = max(int(c.get("row_index", 0)) for c in cells) + 1
        num_cols = max(int(c.get("col_index", 0)) for c in cells) + 1
    grid = [["" for _ in range(num_cols)] for _ in range(num_rows)]
    has_span = False
    for cell in cells:
        row = int(cell.get("row_index", 0))
        col = int(cell.get("col_index", 0))
        if not (0 <= row < num_rows and 0 <= col < num_cols):
            continue
        grid[row][col] = _strip_html(str(cell.get("text", ""))).strip()
        if int(cell.get("row_span", 1) or 1) > 1 or int(cell.get("col_span", 1) or 1) > 1:
            has_span = True
    spanning_rows = [
        index for index, row in enumerate(grid) if len(row) > 1 and all(not c for c in row[1:])
    ]
    return {
        "rows": grid,
        "header_row": True,
        "has_spanning_cell": has_span,
        "spanning_rows": spanning_rows,
    }


# --------------------------------------------------------------------------------------
# Docling（DoclingDocument JSON）
# --------------------------------------------------------------------------------------

# Docling label -> 13C type。未列出的 label 一律落到 ``unknown``（显式暴露，便于报告）。
_DOCLING_LABEL_MAP = {
    "title": "document_title",
    "section_header": "section_header",
    "page_header": "header",
    "page_footer": "footer",
    "text": "paragraph",
    "paragraph": "paragraph",
    "list_item": "list_item",
    "caption": "figure_caption",
    "footnote": "footnote",
    "reference": "reference_entry",
    "formula": "equation",
    "code": "paragraph",
    "picture": "figure",
    "table": "table",
    "checkbox_selected": "list_item",
    "checkbox_unselected": "list_item",
    "empty_value": "unknown",
    "key_value_region": "unknown",
    "document_index": "unknown",
    "page_break": "unknown",
    "field_region": "unknown",
    "field_heading": "unknown",
    "field_item": "unknown",
    "field_key": "unknown",
    "field_value": "unknown",
    "marker": "unknown",
    "handwritten_text": "paragraph",
    "grading_scale": "unknown",
    "form": "unknown",
}


def _docling_ref_target(ref: str) -> tuple[str, int] | None:
    """解析 ``#/texts/12`` 形式的引用为 ``(collection, index)``。"""

    if not isinstance(ref, str) or not _REF.match(ref):
        return None
    parts = ref.split("/")
    if len(parts) < 3:
        return None
    try:
        return parts[1], int(parts[2])
    except ValueError:
        return None


def docling_json_to_schema(
    doc: dict[str, Any],
    *,
    document_id: str,
    source_pdf: str,
    source_sha256: str,
    layout_class: str | None = None,
) -> dict[str, Any]:
    """把 DoclingDocument 的 JSON 导出映射为 13C schema。

    reading order 取自 ``body.children`` 的**树序遍历**（groups 递归展开），这正是
    Docling 官方的文档顺序；不按 ``texts`` 数组下标排序——那只是存储顺序。
    """

    pages = doc.get("pages") or {}
    page_heights: dict[int, float] = {}
    page_widths: dict[int, float] = {}
    for key, meta in pages.items():
        size = (meta or {}).get("size") or {}
        try:
            page_no = int(meta.get("page_no", key))
        except (TypeError, ValueError):
            continue
        page_heights[page_no] = float(size.get("height") or 0) or None  # type: ignore[assignment]
        page_widths[page_no] = float(size.get("width") or 0) or None  # type: ignore[assignment]

    texts = doc.get("texts") or []
    pictures = doc.get("pictures") or []
    tables = doc.get("tables") or []
    groups = doc.get("groups") or []
    furniture = doc.get("furniture") or {}

    elements: list[dict[str, Any]] = []
    # 先为每个可引用对象分配 element_id，供 caption / parent 引用解析。
    id_by_ref: dict[str, str] = {}
    for collection_name, collection in (
        ("texts", texts),
        ("pictures", pictures),
        ("tables", tables),
        ("groups", groups),
    ):
        for index, item in enumerate(collection):
            ref = item.get("self_ref") or f"#/{collection_name}/{index}"
            id_by_ref[ref] = schema.element_id_for("docling", document_id, None, len(id_by_ref))

    # caption 的 self_ref -> 被描述的 figure/table 的 self_ref。
    caption_target: dict[str, str] = {}
    for item in tables + pictures:
        for caption in item.get("captions") or []:
            target = caption.get("$ref") if isinstance(caption, dict) else None
            if target:
                caption_target[target] = item.get("self_ref") or ""

    def emit(item: dict[str, Any], section_path: list[str], container: str) -> None:
        """把一个 Docling 条目转成 schema 元素（groups 递归展开）。"""

        ref = item.get("self_ref") or ""
        label = str(item.get("label") or "")
        if label in {"group", "inline_group", "ordered_list", "unordered_list"} or ref in {
            g.get("self_ref") for g in groups
        }:
            group_label = str(item.get("label") or "group")
            for child in item.get("children") or []:
                child_ref = child.get("$ref") if isinstance(child, dict) else None
                target = _docling_ref_target(child_ref or "")
                if target is None:
                    continue
                child_item = _lookup(doc, target)
                if child_item is None:
                    continue
                emit(
                    child_item,
                    section_path + [group_label] if group_label == "section" else section_path,
                    container,
                )
            return

        target = _docling_ref_target(ref)
        if target is None:
            return
        element_type = _DOCLING_LABEL_MAP.get(label, "unknown")
        text = str(item.get("text") or item.get("orig") or "")
        bbox = None
        page_no = None
        prov = item.get("prov") or []
        if prov:
            first = prov[0]
            page_no = first.get("page_no")
            raw_bbox = first.get("bbox")
            if isinstance(raw_bbox, dict):
                bbox = _points_from_top_left(
                    [raw_bbox.get("l", 0), raw_bbox.get("t", 0), raw_bbox.get("r", 0), raw_bbox.get("b", 0)],
                    page_heights.get(int(page_no)) if page_no else None,
                    str(raw_bbox.get("coord_origin") or ""),
                )
        table_structure = None
        if element_type == "table":
            table_structure = _table_from_docling(item.get("data") or {})
            if not text and table_structure:
                text = " ".join(cell for row in table_structure["rows"] for cell in row if cell)
        if element_type == "figure" and not text:
            text = ""

        caption_of = None
        if label == "caption":
            caption_of = caption_target.get(ref)
        # gallery / picture 的替代文本（如 ``picture_class``）不进入正文。

        element = schema.make_element(
            document_id=document_id,
            element_id=id_by_ref.get(ref) or schema.element_id_for("docling", document_id, page_no, len(elements)),
            type=element_type,
            raw_text=text,
            source_parser="P2_docling",
            page=int(page_no) if page_no else None,
            bbox=bbox,
            reading_order=len(elements),
            section_path=section_path,
            parent_id=None,
            caption_of=id_by_ref.get(caption_of) if caption_of else None,
            table_id=id_by_ref.get(ref) if element_type in {"table", "table_caption"} else None,
            figure_id=id_by_ref.get(ref) if element_type in {"figure", "figure_caption"} else None,
            formula_id=id_by_ref.get(ref) if element_type == "equation" else None,
            source_ref=f"docling:{container}:{ref}",
            confidence=None,
        )
        if table_structure:
            element["table_structure"] = table_structure
        elements.append(element)

    for child in (doc.get("body") or {}).get("children") or []:
        child_ref = child.get("$ref") if isinstance(child, dict) else None
        target = _docling_ref_target(child_ref or "")
        if target is None:
            continue
        child_item = _lookup(doc, target)
        if child_item is not None:
            emit(child_item, [], "body")

    # furniture（页眉/页脚）单独处理，附在文档顺序最后——它们不参与正文阅读顺序。
    for child in (furniture or {}).get("children") or []:
        child_ref = child.get("$ref") if isinstance(child, dict) else None
        target = _docling_ref_target(child_ref or "")
        if target is None:
            continue
        child_item = _lookup(doc, target)
        if child_item is not None:
            emit(child_item, [], "furniture")

    # 章节路径：按顺序维护标题栈，body 内元素继承当前路径。
    _assign_section_paths(elements)

    return schema.make_document(
        document_id=document_id,
        source_pdf=source_pdf,
        source_sha256=source_sha256,
        parser="P2_docling",
        page_count=len(pages),
        elements=elements,
        layout_class=layout_class,
        extra={"bbox_source": "pdf_points_top_left(from docling coord_origin)", "native_format": "DoclingDocument JSON"},
    )


def _lookup(doc: dict[str, Any], target: tuple[str, int]) -> dict[str, Any] | None:
    """按 ``(collection, index)`` 取 Docling 条目。"""

    collection_name, index = target
    collection = doc.get(collection_name)
    if not isinstance(collection, list) or not (0 <= index < len(collection)):
        return None
    return collection[index]


# --------------------------------------------------------------------------------------
# MinerU（content_list.json）
# --------------------------------------------------------------------------------------


def mineru_content_list_to_schema(
    content_list: Sequence[dict[str, Any]],
    *,
    page_sizes: dict[int, tuple[float, float]],
    document_id: str,
    source_pdf: str,
    source_sha256: str,
    parser: str = "P1_mineru",
    layout_class: str | None = None,
) -> dict[str, Any]:
    """把 MinerU 的 ``*_content_list.json`` 映射为 13C schema。

    MinerU 的 ``page_idx`` 是 0-based，转 1-based 以对齐 schema 约定；
    bbox 是 0–1000 归一化坐标，用对应页的宽高换算。
    """

    elements: list[dict[str, Any]] = []
    for item in content_list:
        page_idx = int(item.get("page_idx") or 0)
        page_no = page_idx + 1
        width, height = page_sizes.get(page_no, (None, None))  # type: ignore[assignment]
        bbox = _norm_1000_bbox(item.get("bbox"), width, height)
        raw_type = str(item.get("type") or "")
        text = ""
        element_type = "unknown"
        table_structure = None

        if raw_type == "text":
            text = str(item.get("text") or "")
            level = item.get("text_level")
            element_type = "section_header" if level else "paragraph"
        elif raw_type == "equation":
            text = str(item.get("text") or "")
            element_type = "equation"
        elif raw_type == "table":
            html = str(item.get("table_body") or "")
            table_structure = _table_from_html(html)
            text = " ".join(cell for row in (table_structure or {}).get("rows", []) for cell in row if cell)
            if not text:
                text = _strip_html(html)
            element_type = "table"
        elif raw_type == "image":
            element_type = "figure"
        elif raw_type in {"figure", "picture"}:
            element_type = "figure"
        else:
            text = str(item.get("text") or "")
            element_type = "unknown"

        # caption 字段（MinerU 把它们与 image/table 绑在同一块里）。
        caption_text = ""
        for key in ("img_caption", "table_caption", "image_caption"):
            values = item.get(key)
            if isinstance(values, list) and values:
                caption_text = " ".join(str(v) for v in values)
                break
            if isinstance(values, str) and values:
                caption_text = values
                break

        element = schema.make_element(
            document_id=document_id,
            element_id=schema.element_id_for(parser, document_id, page_no, len(elements)),
            type=element_type,
            raw_text=text,
            source_parser=parser,
            page=page_no,
            bbox=bbox,
            reading_order=len(elements),
            section_path=[],
            source_ref=f"mineru:content_list:{len(elements)}",
        )
        if table_structure:
            element["table_structure"] = table_structure
        elements.append(element)

        if caption_text:
            caption_element = schema.make_element(
                document_id=document_id,
                element_id=schema.element_id_for(parser, document_id, page_no, len(elements)),
                type="table_caption" if raw_type == "table" else "figure_caption",
                raw_text=caption_text,
                source_parser=parser,
                page=page_no,
                bbox=bbox,
                reading_order=len(elements),
                section_path=[],
                caption_of=element["element_id"],
                source_ref=f"mineru:content_list_caption:{len(elements)}",
            )
            elements.append(caption_element)

    _assign_section_paths(elements)
    return schema.make_document(
        document_id=document_id,
        source_pdf=source_pdf,
        source_sha256=source_sha256,
        parser=parser,
        page_count=len(page_sizes),
        elements=elements,
        layout_class=layout_class,
        extra={"bbox_source": "pdf_points_top_left(from mineru 0-1000 normalized)", "native_format": "content_list.json"},
    )


# --------------------------------------------------------------------------------------
# Marker（JSONRenderer 输出）
# --------------------------------------------------------------------------------------

_MARKER_BLOCK_MAP = {
    "SectionHeader": "section_header",
    "Text": "paragraph",
    "ListItem": "list_item",
    "Table": "table",
    "Figure": "figure",
    "Caption": "figure_caption",
    "Equation": "equation",
    "PageHeader": "header",
    "PageFooter": "footer",
    "Footnote": "footnote",
    "Reference": "reference_entry",
    "Code": "paragraph",
    "Form": "unknown",
    "Handwriting": "paragraph",
    "TextInlineMath": "paragraph",
    "TableOfContents": "unknown",
}


def marker_json_to_schema(
    rendered: dict[str, Any],
    *,
    page_sizes: dict[int, tuple[float, float]],
    document_id: str,
    source_pdf: str,
    source_sha256: str,
    layout_class: str | None = None,
) -> dict[str, Any]:
    """把 Marker 的 JSON 输出映射为 13C schema。

    Marker 的 block 树已按阅读顺序排列，因此直接前序遍历即可得到 ``reading_order``；
    标题栈用于生成 ``section_path``。Marker 的 ``bbox`` 是 PDF 点、左上原点。

    **页码必须由 ``Page`` 节点的嵌套序号推导。** Marker 的 JSONRenderer 在节点上
    **不写** ``page_id``（实测 Document > Page > blocks 三层，所有节点 page_id 均为缺失），
    页码信息完全由 ``Document > Page`` 的嵌套顺序隐含表达。早期实现直接读
    ``node.get("page_id")``，导致 Marker 全部元素 page=None，进而打坏三件事：
    page provenance（§14 门 3）、页约束 anchor 召回、以及依赖同页判定的几何匹配。
    """

    elements: list[dict[str, Any]] = []
    section_stack: list[str] = []
    # 文档内单调递增的元素序号，作为 element_id 身份的一部分（见 _marker_element 的说明）。
    counter = itertools.count()
    # Page 兄弟节点的出现顺序即 0-based 页码。
    page_counter = itertools.count()

    def walk(node: dict[str, Any], inherited_page: int | None) -> None:
        block_type = str(node.get("block_type") or "")
        children = node.get("children") or []

        # 显式 page_id 优先（某些 Marker 版本会写），否则沿用祖先 Page 推导出的页码。
        explicit_page = node.get("page_id")
        page = explicit_page if isinstance(explicit_page, int) else inherited_page

        if block_type == "Page":
            current = next(page_counter)
            for child in children:
                if isinstance(child, dict):
                    walk(child, current)
            return

        if block_type in {"Document", "FigureGroup", "TableGroup", "PictureGroup", "Line", "Span"}:
            for child in children:
                if isinstance(child, dict):
                    walk(child, page)
            return

        level = node.get("section_hierarchy_depth") or node.get("level")
        if block_type == "SectionHeader":
            heading_text = _node_text(node)
            depth = int(level) if isinstance(level, int) and level >= 0 else len(section_stack)
            section_stack[:] = section_stack[:depth]
            section_stack.append(heading_text)
            elements.append(
                _marker_element(
                    node, "section_header", heading_text, list(section_stack), document_id,
                    next(counter), page,
                )
            )
            for child in children:
                if isinstance(child, dict):
                    walk(child, page)
            return

        element_type = _MARKER_BLOCK_MAP.get(block_type)
        if element_type is None:
            for child in children:
                if isinstance(child, dict):
                    walk(child, page)
            return

        text = _node_text(node)
        element = _marker_element(
            node, element_type, text, list(section_stack), document_id, next(counter), page
        )
        if element_type == "table":
            html = str(node.get("html") or "")
            structure = _table_from_html(html)
            if structure:
                element["table_structure"] = structure
        elements.append(element)
        for child in children:
            if isinstance(child, dict):
                walk(child, page)

    for child in rendered.get("children") or []:
        if isinstance(child, dict):
            walk(child, None)

    _assign_section_paths(elements, preserve_existing=True)
    return schema.make_document(
        document_id=document_id,
        source_pdf=source_pdf,
        source_sha256=source_sha256,
        parser="P3_marker",
        page_count=len(page_sizes),
        elements=elements,
        layout_class=layout_class,
        extra={"bbox_source": "pdf_points_top_left(native)", "native_format": "marker JSONRenderer"},
    )


def _node_text(node: dict[str, Any]) -> str:
    """取 Marker block 的文本：优先 ``html`` 去标签，其次拼接 Span 子节点。"""

    spans = [
        str(child.get("text") or "")
        for child in node.get("children") or []
        if isinstance(child, dict) and str(child.get("block_type") or "") == "Span"
    ]
    if spans:
        return " ".join(part for part in spans if part).strip()
    text = node.get("text")
    if isinstance(text, str) and text.strip():
        return text.strip()
    html = node.get("html")
    if isinstance(html, str) and html.strip():
        return _strip_html(html).strip()
    return ""


def _marker_element(
    node: dict[str, Any],
    element_type: str,
    text: str,
    section_path: list[str],
    document_id: str,
    index: int,
    page: int | None,
) -> dict[str, Any]:
    """构造一个 Marker 元素的 schema 记录。

    ``index`` 必须是**文档内的遍历序号**，不能拿 ``len(text)`` 之类的派生量充当：
    ``element_id_for`` 的身份 payload 是 ``parser|document|page|index``，若 index 不是
    单调整数，同一页上字符长度相同的两个块会算出同一个 id。曾经的症状是
    ``1 Introduction`` 与 ``2 Related Work``（同为 14 字符、同页）id 冲突，
    导致 47 个元素只有 36 个唯一 id、``validate_document`` 报 duplicate element_id，
    并让按 id 集合计数的指标（如 section_header_precision）算出 > 1 的不可能值。

    ``page`` 是 **0-based** 页码（由 ``Page`` 节点序号推导），schema 里统一转成 1-based。
    """

    bbox = node.get("bbox")
    normalized = None
    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        normalized = [round(float(v), 2) for v in bbox]
    return schema.make_element(
        document_id=document_id,
        element_id=schema.element_id_for("marker", document_id, page, index),
        type=element_type,
        raw_text=text,
        source_parser="P3_marker",
        page=page + 1 if isinstance(page, int) else None,
        bbox=normalized,
        reading_order=0,  # 遍历结束后统一重排
        section_path=section_path,
        source_ref=f"marker:{node.get('id') or ''}",
    )


# --------------------------------------------------------------------------------------
# 章节路径与后处理
# --------------------------------------------------------------------------------------


def _assign_section_paths(
    elements: list[dict[str, Any]], *, preserve_existing: bool = False
) -> None:
    """按元素顺序维护标题栈，为正文元素赋予 ``section_path``，并重排 reading_order。"""

    stack: list[str] = []
    for index, element in enumerate(elements):
        element["reading_order"] = index
        element_type = element["type"]
        if element_type in {"section_header", "document_title"}:
            heading = element.get("normalized_text") or ""
            if element_type == "document_title":
                stack = []
                if not preserve_existing:
                    element["section_path"] = [heading]
                continue
            # 用标题编号推断层级："3.1 Encoder" -> depth 2；"3 Method" -> depth 1。
            depth = 1
            number_match = re.match(r"^\s*(\d+(?:\.\d+)*)", heading)
            if number_match:
                depth = number_match.group(1).count(".") + 1
            stack = stack[: depth - 1]
            stack.append(heading)
            if not preserve_existing or not element.get("section_path"):
                element["section_path"] = list(stack)
            continue
        if not preserve_existing or not element.get("section_path"):
            element["section_path"] = list(stack)
