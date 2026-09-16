"""Phase 14A-B 清洗器（方案 §13 DOCX / §14 XLSX）——纯函数，可独立测试。

输入是 runner 抽取的 **native 流**（与具体 python 库对象解耦），输出是 14B
StructuredDocument。所有删除/重分类/合并/拆分/推断都写入 cleaning_log（§15），
禁止任何 LLM 改写，全部 deterministic。

DOCX native 流约定（runner 负责 OOXML/python-docx 抽取）::

    {"doc_id", "body": [
        {"kind":"paragraph", "para_index", "text", "style", "bold", "size_pt",
         "numbering", "has_pic", "is_textbox"(可选)},
        {"kind":"table", "table_index", "rows", "cols", "cells", "style",
         "merged_regions":[{"anchor":"R2C1","row_span","col_span"}], "nested":[...]},
     ], "headers":[...], "footers":[...], "footnotes":[...],
     "page_breaks":[para_index, ...]}

XLSX native 流约定::

    {"doc_id", "sheets": [{"name","order","state","merged_ranges","anchor_cells",
     "auto_filter_ref","cells":[{coord,row,col,raw_value,formula,cached,data_type,
     number_format,hidden_row,hidden_col}], "named_tables":[{"name","ref"}],
     "charts":[...], "images":[...]}]}
"""

from __future__ import annotations

import re
from typing import Any

from . import schema14b

DOCX_PARSER_B0 = "DOCX-B0_python_docx_minimal"
DOCX_PARSER_B1 = "DOCX-B1_python_docx_semantic"
XLSX_PARSER_B0 = "XLSX-B0_openpyxl_cell_truth"
XLSX_PARSER_B1 = "XLSX-B1_openpyxl_region_structured"

# 显式 style → semantic type（方案 §13 Style → Semantic role 优先级第 1 位）。
EXPLICIT_STYLE_MAP: dict[str, tuple[str, int | None]] = {
    "Title": ("document_title", 0),
    "Caption": ("caption", None),
    "FootnoteText": ("footnote", None),
}
_LIST_LEVEL_RE = re.compile(r"List (?:Bullet|Number) (\d)")
_HEADING_RE = re.compile(r"^Heading (\d)$")
_NUMBERING_PREFIX_RE = re.compile(r"^\d+(?:\.\d+)+\s")  # 1.1 / 1.2.3 形式
# W9 的 bad heading 生成文本是 "1.1 1.1 Data Sources"（编号前缀 + 原 text），
# 也必须被 numbering 模式命中；宽松到 ^\d+(\.\d+)*\s 但排除普通句子开头的年份等。
_INFER_NUMBERING_RE = re.compile(r"^\d+(?:\.\d+){0,3}\s+\S")
# fallback 推断的字号阈值（pt）；仅对无显式语义样式的段落生效。
_INFER_MIN_SIZE_PT = 14.0


def _col_letter(n: int) -> str:
    s = ""
    while n > 0:
        n, r = divmod(n - 1, 26)
        s = chr(65 + r) + s
    return s


def _rc_to_ref(anchor: str, row_span: int, col_span: int) -> str:
    """R2C1 + span → "A2:A3" 形式的范围串（merged_cell_accuracy 的对齐空间）。"""

    r = int(anchor[1:anchor.index("C")])
    c = int(anchor[anchor.index("C") + 1:])
    r2, c2 = r + row_span - 1, c + col_span - 1
    a = f"{_col_letter(c)}{r}"
    b = f"{_col_letter(c2)}{r2}"
    return a if a == b else f"{a}:{b}"


def _classify_explicit(style: str) -> tuple[str, int | None]:
    """显式样式 → (semantic type, heading level)。未命中返回 paragraph。"""

    if style in EXPLICIT_STYLE_MAP:
        return EXPLICIT_STYLE_MAP[style]
    m = _HEADING_RE.match(style)
    if m:
        return "heading", int(m.group(1))
    if style.startswith("List"):
        return "list_item", None
    return "paragraph", None


# --------------------------------------------------------------------------------------
# DOCX 清洗
# --------------------------------------------------------------------------------------


def _docx_element(element_id: str, etype: str, text: str, sr: dict, parser: str,
                  role: str | None = None, **kw: Any) -> dict[str, Any]:
    role = role or schema14b.DOCX_ROLE_BY_TYPE.get(etype, "body")
    return schema14b.make_element14b(
        document_id=kw.pop("document_id"), element_id=element_id, type=etype, raw_text=text,
        source_format="docx", source_ref=sr, source_parser=parser,
        retrieval_role=role, **kw,
    )


def _docx_table_metadata(block: dict, *, serialize: bool) -> dict[str, Any]:
    """表格 metadata：raw cell matrix + merged + （B1）markdown 序列化。"""

    meta: dict[str, Any] = {
        "rows": block["rows"], "cols": block["cols"], "cells": block["cells"],
        "merged_regions": block.get("merged_regions", []),
        "table_style_id": block.get("style"),
    }
    if serialize:
        # serialized_markdown：表头行 + 数据行；合并/嵌套另存，不丢失信息（§13）。
        lines = ["| " + " | ".join(block["cells"][0]) + " |",
                 "| " + " | ".join(["---"] * block["cols"]) + " |"]
        for row in block["cells"][1:]:
            lines.append("| " + " | ".join(row) + " |")
        meta["serialized_markdown"] = "\n".join(lines)
    return meta


def clean_docx(native: dict[str, Any], *, semantic: bool) -> dict[str, Any]:
    """DOCX native 流 → 14B StructuredDocument。semantic=False 即 B0 最小归一。"""

    doc_id = native["doc_id"]
    parser = DOCX_PARSER_B1 if semantic else DOCX_PARSER_B0
    elements: list[dict[str, Any]] = []
    relationships: list[dict[str, Any]] = []
    cleaning_log: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []

    def _sr_para(idx: int, style: str | None) -> dict[str, Any]:
        return {"paragraph_index": idx, "xml_path": f"/w:document/w:body/w:p[{idx + 1}]",
                "style_id": style, "table_index": None, "cell": None}

    def _sr_table(idx: int, style: str | None = None) -> dict[str, Any]:
        return {"paragraph_index": None, "xml_path": f"table[{idx + 1}]",
                "style_id": style, "table_index": idx, "cell": None}

    # ---- 阶段 1：显式样式 → semantic type（B0/B1 共用；B1 留痕） ----
    pending_list_group: dict[str, int] = {}
    list_counter = 0
    for item in native["body"]:
        if item["kind"] == "paragraph":
            style = item.get("style") or "Normal"
            etype, level = _classify_explicit(style)
            explicit = etype != "paragraph" or style in EXPLICIT_STYLE_MAP
            semantic_inferred = False
            metadata: dict[str, Any] = {}
            if etype in ("heading", "document_title"):
                # document_title（Title 样式）也带 level=0，供 heading_level_accuracy 对位
                metadata["level"] = level if etype == "heading" else 0
            # ---- 阶段 2（仅 B1）：badly-styled fallback 推断，必须显式标记（§13） ----
            if semantic and not explicit and item.get("text", "").strip():
                features: dict[str, Any] = {}
                numbered = bool(_INFER_NUMBERING_RE.match(item["text"])) or bool(item.get("numbering"))
                big_bold = bool(item.get("bold")) and (item.get("size_pt") or 0) >= _INFER_MIN_SIZE_PT
                if item.get("is_textbox"):
                    pass  # textbox 不参与标题推断
                elif numbered or big_bold:
                    if numbered:
                        m = _NUMBERING_PREFIX_RE.match(item["text"])
                        depth = (m.group(0).count(".") + 1) if m else 1
                        features["numbering"] = True
                        features["numbering_depth"] = depth
                    if big_bold:
                        features["bold"] = True
                        features["size_pt"] = item.get("size_pt")
                    etype = "heading"
                    # 层级推断：编号深度优先，其次大字号→1 级
                    metadata["level"] = features.get("numbering_depth",
                                                     1 if (item.get("size_pt") or 0) >= 16 else 2)
                    metadata["inference_features"] = features
                    metadata["inference_confidence"] = 0.9 if (numbered and big_bold) else 0.75
                    semantic_inferred = True
            element_id = f"el-docx-p-{item['para_index']}"
            sr = _sr_para(item["para_index"], None if semantic_inferred else style)
            elements.append(_docx_element(
                element_id, etype, item.get("text", ""), sr, parser,
                document_id=doc_id, semantic_inferred=semantic_inferred, metadata=metadata))
            if explicit and etype != "paragraph":
                # 显式样式重分类留痕（§15：reclassify 必须 reason 可追溯；B0/B1 都记）
                schema14b.log_action(
                    cleaning_log, action="reclassify", target=element_id,
                    from_type="paragraph", to_type=etype,
                    reason=f"explicit_style: {style}", source_element_id=element_id)
            elif semantic and semantic_inferred:
                schema14b.log_action(
                    cleaning_log, action="promote_heading", target=element_id,
                    from_type="paragraph", to_type="heading",
                    reason="no_explicit_style; fallback inference (numbering/bold/size)",
                    source_element_id=element_id, detail=metadata.get("inference_features"))
            # 图片元素（顺序保持：与所属段落相邻；B0/B1 都记 presence）
            if item.get("has_pic"):
                img_id = f"el-docx-img-{item['para_index']}"
                elements.append(_docx_element(
                    img_id, "image", "", _sr_para(item["para_index"], style), parser,
                    document_id=doc_id, metadata={"near_paragraph_index": item["para_index"]}))
                if semantic:
                    schema14b.log_action(cleaning_log, action="record_image", target=img_id,
                                         reason="inline picture (no OCR/VLM in 14A-B)")
        else:  # table
            idx = item["table_index"]
            tbl_id = f"el-docx-tbl-{idx}"
            elements.append(_docx_element(
                tbl_id, "table", "", _sr_table(idx, item.get("style")), parser,
                document_id=doc_id,
                metadata=_docx_table_metadata(item, serialize=semantic)))
            # 嵌套子表：B1 作为独立 element（parent 链接 + relationship），B0 记 presence
            for ni, nest in enumerate(item.get("nested", [])):
                nest_meta = {"rows": nest["rows"], "cols": nest["cols"], "cells": nest["cells"]}
                if semantic:
                    nest_id = f"el-docx-tbl{idx}-nested-{ni}"
                    elements.append(_docx_element(
                        nest_id, "table", "", _sr_table(idx), parser, document_id=doc_id,
                        parent_id=tbl_id, metadata={**nest_meta, "nested_in_cell": True}))
                    relationships.append({"type": "contains", "from": tbl_id, "to": nest_id})
                    schema14b.log_action(cleaning_log, action="extract_nested_table",
                                         target=nest_id, reason="nested table inside cell",
                                         source_element_id=tbl_id)
                else:
                    item.setdefault("nested_presence", []).append(nest_meta)
            if semantic:
                schema14b.log_action(cleaning_log, action="serialize_table", target=tbl_id,
                                     reason="table normalization: matrix + markdown + merges")
                if item.get("merged_regions"):
                    schema14b.log_action(cleaning_log, action="preserve_merged_cells", target=tbl_id,
                                         reason="merged cell regions kept traceable",
                                         detail={"regions": item["merged_regions"]})
            else:
                # B0 也必须保留嵌套事实（不得静默丢弃）
                if item.get("nested_presence"):
                    elements[-1]["metadata"]["nested_presence"] = item["nested_presence"]

        # B1 list 归一化：连续同类 list_item 归组（被非 list 元素打断则另起新组）
        if semantic and elements and elements[-1]["type"] == "list_item":
            el = elements[-1]
            style = el["source_ref"].get("style_id") or "List"
            key = style
            prev = elements[-2] if len(elements) >= 2 else None
            prev_gid = prev.get("metadata", {}).get("list_id") if (prev and prev["type"] == "list_item") else None
            if key not in pending_list_group or prev_gid != f"list-{pending_list_group[key]}":
                list_counter += 1
                pending_list_group[key] = list_counter
            gid = pending_list_group[key]
            ordered = "Number" in style
            m = _LIST_LEVEL_RE.match(style)
            level = int(m.group(1)) if m else 1
            siblings = [e for e in elements if e["type"] == "list_item"
                        and e.get("metadata", {}).get("list_id") == f"list-{gid}"]
            el["metadata"].update({"list_id": f"list-{gid}", "ordered": ordered,
                                   "level": level, "item_index": len(siblings) + 1})
            if el["metadata"]["item_index"] == 1:
                schema14b.log_action(cleaning_log, action="normalize_list",
                                     target=el["element_id"], to_type="list_group",
                                     reason=f"group consecutive items of style {style}",
                                     source_element_id=el["element_id"],
                                     detail={"list_id": f"list-{gid}", "ordered": ordered})

    # ---- 阶段 4：caption / image 关联（仅 B1） ----
    if semantic:
        for i, el in enumerate(elements):
            if el["type"] == "image":
                # 向后找最近的 caption 元素（不超过 2 个元素距离）
                for j in range(i + 1, min(i + 3, len(elements))):
                    if elements[j]["type"] == "caption":
                        relationships.append({"type": "caption_of", "from": el["element_id"],
                                              "to": elements[j]["element_id"]})
                        schema14b.log_action(cleaning_log, action="associate_caption",
                                             target=elements[j]["element_id"],
                                             reason="caption follows image in document order",
                                             source_element_id=el["element_id"])
                        break
                    if elements[j]["type"] in ("heading", "table"):
                        break

    # ---- 阶段 5：header/footer 分离（furniture） ----
    for si, text in enumerate(native.get("headers", [])):
        elements.append(_docx_element(
            f"el-docx-h-{si}", "header", text,
            {"paragraph_index": None, "xml_path": "header", "style_id": None,
             "table_index": None, "cell": None}, parser, document_id=doc_id))
    for si, text in enumerate(native.get("footers", [])):
        elements.append(_docx_element(
            f"el-docx-f-{si}", "footer", text,
            {"paragraph_index": None, "xml_path": "footer", "style_id": None,
             "table_index": None, "cell": None}, parser, document_id=doc_id))
    if semantic and (native.get("headers") or native.get("footers")):
        schema14b.log_action(cleaning_log, action="separate_furniture", target="header/footer",
                             reason="header/footer assigned retrieval_role=furniture, excluded from body")

    # ---- 阶段 6：footnotes（reference） ----
    for fi, text in enumerate(native.get("footnotes", [])):
        elements.append(_docx_element(
            f"el-docx-fn-{fi}", "footnote", text,
            {"paragraph_index": None, "xml_path": "footnotes", "style_id": None,
             "table_index": None, "cell": None}, parser, document_id=doc_id))

    # ---- page break：只记 rendered hint 事实，不伪造 page provenance（§12） ----
    doc_metadata = {
        "parser": parser,
        "page_breaks": native.get("page_breaks", []),
        "page_break_provenance_kind": "rendered_hint" if native.get("page_breaks") else None,
    }
    return schema14b.make_document14b(
        document_id=doc_id, source_format="docx", source_file=native.get("source_file", f"{doc_id}.docx"),
        elements=elements, relationships=relationships, cleaning_log=cleaning_log,
        unresolved=unresolved, document_metadata=doc_metadata,
        source_sha256=native.get("source_sha256"),
    )


# --------------------------------------------------------------------------------------
# XLSX 清洗
# --------------------------------------------------------------------------------------


def _cell_element(doc_id: str, sheet: dict, c: dict, *, parser: str, in_table: dict | None) -> dict[str, Any]:
    """cell truth element：raw/formula/cached/number_format/hidden 全部保存（§14）。"""

    role = "hidden" if (c.get("hidden_row") or c.get("hidden_col")) else "body"
    return schema14b.make_element14b(
        document_id=doc_id, element_id=f"el-xlsx-cell-{sheet['order']}-{c['coord']}",
        type="cell", raw_text=str(c["raw_value"]), source_format="xlsx",
        source_ref={"sheet": sheet["name"], "range": (in_table or {}).get("range"),
                    "cell": c["coord"], "table_name": (in_table or {}).get("table_name"),
                    "inferred": (in_table or {}).get("inferred")},
        source_parser=parser, retrieval_role=role,
        metadata={
            "raw_value": c["raw_value"], "formula": c.get("formula"),
            "cached_value": c.get("cached"), "data_type": c.get("data_type"),
            "number_format": c.get("number_format"),
            "is_formula": c.get("formula") is not None,
            "is_hidden_row": bool(c.get("hidden_row")), "is_hidden_col": bool(c.get("hidden_col")),
            "in_table": in_table is not None,
        },
    )


def _parse_ref(ref: str) -> tuple[int, int, int, int]:
    """'A3:B8' → (rmin, rmax, cmin, cmax)。"""

    def part(p: str) -> tuple[int, int]:
        letters = "".join(ch for ch in p if ch.isalpha()).upper()
        digits = "".join(ch for ch in p if ch.isdigit())
        c = 0
        for ch in letters:
            c = c * 26 + (ord(ch) - 64)
        return int(digits), c

    a, b = ref.split(":") if ":" in ref else (ref, ref)
    r1, c1 = part(a)
    r2, c2 = part(b)
    return min(r1, r2), max(r1, r2), min(c1, c2), max(c1, c2)


def _detect_inferred_regions(cells: list[dict], claimed: set[tuple[int, int]]) -> tuple[list[dict], list[dict]]:
    """连续非空 cell 的 4-连通分量 → table_candidate；过 gate 才成为 table（§14）。

    返回 (accepted_regions, rejected_candidates)。每项含 range/rows/cols/density/
    header_plausible/reject_reason/trimmed_rows。连续非空 ≠ 必然是表（X15）。
    """

    grid = {(c["row"], c["col"]): c for c in cells
            if (c["row"], c["col"]) not in claimed and c["raw_value"] is not None}
    seen: set[tuple[int, int]] = set()
    accepted: list[dict] = []
    rejected: list[dict] = []
    for start in sorted(grid):
        if start in seen:
            continue
        # flood fill（4-邻接）
        stack, comp = [start], []
        seen.add(start)
        while stack:
            r, c0 = stack.pop()
            comp.append((r, c0))
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nxt = (r + dr, c0 + dc)
                if nxt in grid and nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        rows = sorted({r for r, _ in comp})
        cols = sorted({c0 for _, c0 in comp})
        # 边缘单 cell 行修剪（如表格下方紧贴的备注行），保留主体行
        body = sorted(comp)
        row_counts = {r: sum(1 for rr, _ in comp if rr == r) for r in rows}
        max_row_count = max(row_counts.values())
        trimmed: list[int] = []
        while len(rows) > 1 and row_counts[rows[0]] <= 1 < max_row_count:
            trimmed.append(("leading", rows[0]))
            rows.pop(0)
        while len(rows) > 1 and row_counts[rows[-1]] <= 1 < max_row_count:
            trimmed.append(("trailing", rows[-1]))
            rows.pop()
        comp = [(r, c0) for r, c0 in comp if r in rows]
        cols = sorted({c0 for _, c0 in comp})
        rmin, rmax, cmin, cmax = rows[0], rows[-1], cols[0], cols[-1]
        n_rows, n_cols = rmax - rmin + 1, cmax - cmin + 1
        density = len(comp) / (n_rows * n_cols)
        # header plausibility：候选首行（第一个非空行）应全为字符串
        first_row = min(r for r, _ in comp)
        first_vals = [grid[(r, c0)]["raw_value"] for r, c0 in comp if r == first_row]
        header_plausible = all(isinstance(v, str) for v in first_vals) and len(first_vals) >= 2
        rng = f"{_col_letter(cmin)}{rmin}:{_col_letter(cmax)}{rmax}"
        candidate = {"range": rng, "rows": n_rows, "cols": n_cols,
                     "density": round(density, 4), "header_plausible": header_plausible,
                     "trimmed_rows": trimmed}
        reasons = []
        if n_rows < 2:
            reasons.append("rows<2")
        if n_cols < 2:
            reasons.append("cols<2")
        if density < 0.6:
            reasons.append(f"density={density:.2f}<0.6")
        if not header_plausible:
            reasons.append("header_row_not_all_strings")
        if reasons:
            candidate["reject_reason"] = "; ".join(reasons)
            rejected.append(candidate)
        else:
            accepted.append(candidate)
    return accepted, rejected


def _header_hierarchy(cells_by_coord: dict[tuple[int, int], dict], merged: list[str],
                      rmin: int, rmax: int, cmin: int, cmax: int) -> tuple[list[dict], int]:
    """表头层级：首行存在跨列 merge → 两级表头；返回 (per-column headers, header_rows)。

    每列输出 {column_index, hierarchy:[...], merged_range?, anchor_cell?}，
    merged anchor 的值扩展到整个 merge 跨度（X12：2025/2026 → Q1/Q2）。
    """

    def val(r: int, c0: int) -> Any:
        c = cells_by_coord.get((r, c0))
        return c["raw_value"] if c else None

    top_merges = []
    for m in merged:
        a, b = m.split(":")
        mr1, mr2, mc1, mc2 = _parse_ref(m)
        if mr1 == rmin and mr2 == rmin and mc2 > mc1:
            top_merges.append((mc1, mc2, a))
    headers: list[dict] = []
    if top_merges and rmin < rmax:
        # 两级表头：row rmin 为组标签（merge 展开），row rmin+1 为叶标签
        header_rows = 2
        expand: dict[int, Any] = {}
        anchor_of: dict[int, str] = {}
        merge_of: dict[int, str] = {}
        for mc1, mc2, anchor in top_merges:
            v = cells_by_coord.get((rmin, mc1), {}).get("raw_value")
            anchor_of[mc1] = anchor
            merge_of[mc1] = f"{_col_letter(mc1)}{rmin}:{_col_letter(mc2)}{rmin}"
            for c0 in range(mc1, mc2 + 1):
                expand[c0] = v
        for c0 in range(cmin, cmax + 1):
            top = expand.get(c0)
            leaf = val(rmin + 1, c0)
            hier = [t for t in ([top] if top is not None else []) + [leaf] if t is not None]
            entry: dict[str, Any] = {"column_index": c0, "hierarchy": hier}
            if c0 in merge_of:
                entry["merged_range"] = merge_of[c0]
                entry["anchor_cell"] = anchor_of[c0]
            headers.append(entry)
    else:
        header_rows = 1
        for c0 in range(cmin, cmax + 1):
            leaf = val(rmin, c0)
            if leaf is not None:
                headers.append({"column_index": c0, "hierarchy": [leaf]})
    return headers, header_rows


def clean_xlsx(native: dict[str, Any], *, semantic: bool) -> dict[str, Any]:
    """XLSX native 流 → 14B StructuredDocument。semantic=False 即 B0 cell truth baseline。"""

    doc_id = native["doc_id"]
    parser = XLSX_PARSER_B1 if semantic else XLSX_PARSER_B0
    elements: list[dict[str, Any]] = []
    cleaning_log: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []

    def _sr(sheet: str | None, rng: str | None = None, cell: str | None = None,
            tname: str | None = None, inferred: bool | None = None) -> dict[str, Any]:
        return {"sheet": sheet, "range": rng, "cell": cell, "table_name": tname, "inferred": inferred}

    elements.append(schema14b.make_element14b(
        document_id=doc_id, element_id="el-xlsx-wb", type="workbook_metadata",
        raw_text=f"workbook:{doc_id}", source_format="xlsx", source_ref=_sr(None),
        source_parser=parser, retrieval_role="metadata",
        metadata={"sheet_count": len(native["sheets"]),
                  "sheet_names": [s["name"] for s in native["sheets"]]}))

    for sheet in native["sheets"]:
        name = sheet["name"]
        cells = sheet["cells"]
        named = sheet.get("named_tables", [])
        named_ranges: dict[str, str] = {t["name"]: t["ref"] for t in named}
        claimed_coords: set[tuple[int, int]] = set()
        # cell 真值坐标索引：header hierarchy 展开用（merged anchor 值回查）
        cells_by_coord: dict[tuple[int, int], dict] = {(c["row"], c["col"]): c for c in cells}

        elements.append(schema14b.make_element14b(
            document_id=doc_id, element_id=f"el-xlsx-sheet-{sheet['order']}", type="sheet",
            raw_text=f"sheet:{name}", source_format="xlsx", source_ref=_sr(name),
            source_parser=parser, retrieval_role="structure",
            metadata={"name": name, "order": sheet["order"], "state": sheet["state"],
                      "hidden": sheet["state"] == "hidden",
                      "merged_ranges": sheet.get("merged_ranges", []),
                      "anchor_cells": sheet.get("anchor_cells", {}),
                      "hidden_rows": sheet.get("hidden_rows", []),
                      "hidden_cols": sheet.get("hidden_cols", []),
                      "named_tables": [dict(t) for t in named],
                      "named_table_names": sorted(named_ranges),
                      "auto_filter_ref": sheet.get("auto_filter_ref"),
                      "charts": sheet.get("charts", []),
                      "visual_content_unparsed": bool(sheet.get("charts") or sheet.get("images")),
                      "images": sheet.get("images", [])}))

        # ---- native named table（优先级最高，§14） ----
        for t in sorted(named, key=lambda x: x["name"]):
            rmin, rmax, cmin, cmax = _parse_ref(t["ref"])
            tbl_id = f"el-xlsx-tbl-{sheet['order']}-{t['name']}"
            elements.append(schema14b.make_element14b(
                document_id=doc_id, element_id=tbl_id, type="table",
                raw_text=f"table:{t['ref']}", source_format="xlsx",
                source_ref=_sr(name, t["ref"], None, t["name"], False),
                source_parser=parser, retrieval_role="structure",
                metadata={"rows": rmax - rmin + 1, "cols": cmax - cmin + 1,
                          "named": True, "inferred": False, "table_name": t["name"]}))
            if semantic:
                schema14b.log_action(cleaning_log, action="detect_named_table", target=tbl_id,
                                     reason="Excel Named Table object (native, precedence over inference)",
                                     detail={"ref": t["ref"]})
                headers, _ = _header_hierarchy(cells_by_coord, sheet.get("merged_ranges", []),
                                               rmin, rmax, cmin, cmax)
                for h in headers:
                    elements.append(schema14b.make_element14b(
                        document_id=doc_id, element_id=f"{tbl_id}-h{h['column_index']}",
                        type="table_header", raw_text=str(h["hierarchy"][-1]),
                        source_format="xlsx",
                        source_ref=_sr(name, t["ref"], h.get("anchor_cell"), t["name"], False),
                        source_parser=parser, retrieval_role="structure",
                        metadata={"hierarchy": h["hierarchy"], "column_index": h["column_index"],
                                  "merged_range": h.get("merged_range")}))
            for c in cells:
                pos = (c["row"], c["col"])
                if rmin <= pos[0] <= rmax and cmin <= pos[1] <= cmax:
                    claimed_coords.add(pos)

        # ---- inferred region（仅 B1） ----
        if semantic:
            accepted, rejected = _detect_inferred_regions(cells, claimed_coords)
            for cand in accepted:
                rmin, rmax, cmin, cmax = _parse_ref(cand["range"])
                tbl_id = f"el-xlsx-tbl-{sheet['order']}-inf-{cand['range']}"
                elements.append(schema14b.make_element14b(
                    document_id=doc_id, element_id=tbl_id, type="table",
                    raw_text=f"table:{cand['range']}", source_format="xlsx",
                    source_ref=_sr(name, cand["range"], None, None, True),
                    source_parser=parser, retrieval_role="structure",
                    metadata={"rows": cand["rows"], "cols": cand["cols"],
                              "named": False, "inferred": True,
                              "inference_reason": "contiguous region passed gates",
                              "inference_gates": {"density": cand["density"],
                                                  "header_plausible": cand["header_plausible"],
                                                  "trimmed_rows": cand["trimmed_rows"]}}))
                schema14b.log_action(cleaning_log, action="infer_table_region", target=tbl_id,
                                     from_type="cells", to_type="table",
                                     reason="contiguous non-empty region passed min/density/header gates",
                                     detail={"range": cand["range"], **cand})
                headers, _ = _header_hierarchy(cells_by_coord, sheet.get("merged_ranges", []),
                                               rmin, rmax, cmin, cmax)
                for h in headers:
                    elements.append(schema14b.make_element14b(
                        document_id=doc_id, element_id=f"{tbl_id}-h{h['column_index']}",
                        type="table_header", raw_text=str(h["hierarchy"][-1]),
                        source_format="xlsx",
                        source_ref=_sr(name, cand["range"], h.get("anchor_cell"), None, True),
                        source_parser=parser, retrieval_role="structure",
                        metadata={"hierarchy": h["hierarchy"], "column_index": h["column_index"],
                                  "merged_range": h.get("merged_range"),
                                  "anchor_cell": h.get("anchor_cell"), "inferred": True}))
            for cand in rejected:
                # 拒绝的候选：若是 2 列且左列全字符串的键值块 → key_value，否则保持 cell
                kv_coords = None
                if cand["cols"] == 2 and cand["rows"] >= 2:
                    rmin, rmax, cmin, cmax = _parse_ref(cand["range"])
                    left_vals = [c["raw_value"] for c in cells
                                 if cmin <= c["col"] <= cmax and rmin <= c["row"] <= rmax
                                 and c["col"] == cmin]
                    if left_vals and all(isinstance(v, str) for v in left_vals):
                        kv_coords = (rmin, rmax, cmin, cmax)
                if kv_coords:
                    rmin, rmax, cmin, cmax = kv_coords
                    for c in cells:
                        if cmin <= c["col"] <= cmax and rmin <= c["row"] <= rmax:
                            c["kv_group"] = cand["range"]
                    schema14b.log_action(cleaning_log, action="reclassify_key_value",
                                         target=f"{name}!{cand['range']}",
                                         from_type="table_candidate", to_type="key_value",
                                         reason="candidate failed table gates; shape matches key-value block",
                                         detail={"reject_reason": cand["reject_reason"]})
                else:
                    schema14b.log_action(cleaning_log, action="reject_table_candidate",
                                         target=f"{name}!{cand['range']}",
                                         reason=cand["reject_reason"], detail=cand)

        # ---- cell truth（B0/B1 共用，引用 table 归属） ----
        cell_els: dict[str, dict] = {}
        for c in cells:
            pos = (c["row"], c["col"])
            in_table: dict | None = None
            for t in named:
                rmin, rmax, cmin, cmax = _parse_ref(t["ref"])
                if rmin <= pos[0] <= rmax and cmin <= pos[1] <= cmax:
                    in_table = {"range": t["ref"], "table_name": t["name"], "inferred": False}
                    break
            if in_table is None and semantic:
                for el in elements:
                    if (el["type"] == "table" and el["source_ref"]["sheet"] == name
                            and el["source_ref"]["inferred"] is True):
                        rmin, rmax, cmin, cmax = _parse_ref(el["source_ref"]["range"])
                        if rmin <= pos[0] <= rmax and cmin <= pos[1] <= cmax:
                            in_table = {"range": el["source_ref"]["range"], "table_name": None,
                                        "inferred": True}
                            break
            el = _cell_element(doc_id, sheet, c, parser=parser, in_table=in_table)
            if semantic and c.get("kv_group"):
                el["type"] = "key_value"
                el["metadata"]["kv_group"] = c["kv_group"]
            if semantic and (c.get("hidden_row") or c.get("hidden_col")):
                schema14b.log_action(cleaning_log, action="assign_hidden_role",
                                     target=f"{name}!{c['coord']}",
                                     from_type="body", to_type="hidden",
                                     reason="hidden row/col: preserve but exclude from body retrieval")
            cell_els[c["coord"]] = el
            elements.append(el)

        # ---- cached value 缺失 → unresolved（不得冒充、不得覆盖） ----
        for c in cells:
            if c.get("formula") is not None and c.get("cached") is None:
                schema14b.add_unresolved(unresolved, issue="cached_value_missing",
                                         target=f"{name}!{c['coord']}",
                                         detail={"formula": c["formula"]})

    doc_metadata = {"parser": parser, "charts_policy": "presence-only, visual_content_unparsed=true"}
    return schema14b.make_document14b(
        document_id=doc_id, source_format="xlsx", source_file=native.get("source_file", f"{doc_id}.xlsx"),
        elements=elements, cleaning_log=cleaning_log, unresolved=unresolved,
        document_metadata=doc_metadata, source_sha256=native.get("source_sha256"),
    )
