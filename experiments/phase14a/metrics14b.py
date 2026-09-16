"""Phase 14A-B 度量（方案 §16 DOCX / §17 XLSX）。

纪律（与 13C/14A-A 一致）：
- 比例指标分子分母同处一个计数空间，任何 >1 / <0 由编排器守卫抓取；
- 未采集 / 为 0 / 非 0 三态显式区分（None 表示该 arm 语义上不产生该指标，如 B0 的区域推断）；
- XLSX Gold 坐标可从 spec 精确算出，因此 cell 级比较用 (sheet, coordinate) 精确匹配；
- DOCX 文本比较用归一化包含 / token Jaccard（≥0.85），与 14A-A 相同。
"""

from __future__ import annotations

from typing import Any

from . import schema14a

_NORM = lambda t: schema14a.collapse_whitespace(schema14a.normalize_unicode(t or ""))  # noqa: E731


def _norm(text: str) -> str:
    return _NORM(text)


def _tokens(text: str) -> set[str]:
    return set(schema14a.tokenize(text or ""))


def _jaccard(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _matches(gold_text: str, out_texts: list[str], jac: float = 0.85) -> bool:
    gn = _norm(gold_text)
    if not gn:
        return False
    for ot in out_texts:
        on = _norm(ot)
        if gn in on or on in gn or _jaccard(gold_text, ot) >= jac:
            return True
    return False


def _col_letter(n: int) -> str:
    s = ""
    while n > 0:
        n, r = divmod(n - 1, 26)
        s = chr(65 + r) + s
    return s


def _col_number(letters: str) -> int:
    n = 0
    for ch in letters.upper():
        n = n * 26 + (ord(ch) - 64)
    return n


def _values_equal(gold: Any, out: Any) -> bool:
    """数值容差相等（0.816 vs "0.816"），否则字符串归一比较。"""

    try:
        return abs(float(gold) - float(out)) < 1e-9
    except (TypeError, ValueError):
        return _norm(str(gold)) == _norm(str(out))


# --------------------------------------------------------------------------------------
# DOCX Gold 收集
# --------------------------------------------------------------------------------------


def collect_gold_docx_b(case: dict) -> dict[str, Any]:
    g: dict[str, Any] = {
        "headings": [], "paragraphs": [], "list_items": [], "tables": [],
        "captions": [], "footnotes": [], "headers": [], "footers": [],
        "textboxes": [], "bad_headings": [], "image_count": 0, "order": [],
    }
    for b in case["blocks"]:
        k = b["kind"]
        if k == "title":
            g["headings"].append({"text": b["text"], "level": 0})
        elif k == "heading":
            g["headings"].append({"text": b["text"], "level": b.get("level", 1)})
        elif k == "paragraph":
            g["paragraphs"].append(b["text"])
        elif k == "list_item":
            g["list_items"].append({"text": b["text"], "ordered": "Number" in b.get("style", "")})
        elif k in ("table", "merged_table"):
            n_rows = len(b.get("headers", [])) + len(b.get("rows", []))
            n_cols = len(b["headers"][0]) if b.get("headers") else 1
            g["tables"].append({"rows": n_rows, "cols": n_cols,
                                "merges": [f"{m[0]}:{m[1]}" for m in b.get("merges", [])],
                                "nested": bool(b.get("nested"))})
        elif k == "caption":
            g["captions"].append(b["text"])
        elif k == "footnote":
            g["footnotes"].append(b["text"])
        elif k == "header":
            g["headers"].append(b["text"])
        elif k == "footer":
            g["footers"].append(b["text"])
        elif k == "image":
            g["image_count"] += 1
        elif k == "textbox":
            g["textboxes"].append(b["text"])
        elif k == "bad_heading":
            # 生成文本 = number 前缀 + text；两侧都用"包含"匹配可对齐
            text = (b.get("number", "") + (" " if b.get("number") else "") + b["text"]).strip()
            g["bad_headings"].append(text)
        # order 序列：paragraph 类（含 bad_heading/list）与 table，跳过装饰类
        if k in ("paragraph", "bad_heading", "list_item"):
            text = b["text"]
            if k == "bad_heading":
                text = (b.get("number", "") + (" " if b.get("number") else "") + text).strip()
            g["order"].append({"kind": "p", "text": text})
        elif k in ("table", "merged_table"):
            n_rows = len(b.get("headers", [])) + len(b.get("rows", []))
            n_cols = len(b["headers"][0]) if b.get("headers") else 1
            g["order"].append({"kind": "t", "rows": n_rows, "cols": n_cols})
    return g


# --------------------------------------------------------------------------------------
# DOCX 度量
# --------------------------------------------------------------------------------------


def compute_docx_metrics14b(case: dict, doc: dict) -> dict[str, Any]:
    gold = collect_gold_docx_b(case)
    els = doc["elements"]
    by_type: dict[str, list[dict]] = {}
    for e in els:
        by_type.setdefault(e["type"], []).append(e)

    def texts_of(types: list[str]) -> list[str]:
        return [e["normalized_text"] for t in types for e in by_type.get(t, [])
                if e["normalized_text"]]

    out_all = texts_of([t for t in by_type])

    def recall(texts: list[str]) -> float:
        if not texts:
            return 1.0
        return round(sum(1 for t in texts if _matches(t, out_all)) / len(texts), 4)

    heading_els = by_type.get("heading", []) + by_type.get("document_title", [])
    heading_texts = [e["normalized_text"] for e in heading_els]

    # 分类召回（D-G1 逐类）
    heading_recall = recall([h["text"] for h in gold["headings"]])
    bad_heading_recall = recall(gold["bad_headings"])
    paragraph_recall = recall(gold["paragraphs"])
    list_recall = recall([li["text"] for li in gold["list_items"]])
    caption_recall = recall(gold["captions"])
    footnote_recall = recall(gold["footnotes"])
    hf_recall = recall(gold["headers"] + gold["footers"])
    textbox_recall = recall(gold["textboxes"])

    # 总召回 + 未预期丢失（受控 Gold 必须零丢失，D-G1）
    gold_texts = ([h["text"] for h in gold["headings"]] + gold["bad_headings"] + gold["paragraphs"]
                  + [li["text"] for li in gold["list_items"]] + gold["captions"] + gold["footnotes"]
                  + gold["headers"] + gold["footers"] + gold["textboxes"])
    unmatched = [t for t in gold_texts if not _matches(t, out_all)]

    # element_text_precision：body/structure/reference 文本元素中能对上 Gold 的比例
    prec_types = ("paragraph", "heading", "document_title", "list_item", "caption", "footnote")
    prec_els = [e for e in els if e["type"] in prec_types
                and e.get("retrieval_role") in ("body", "structure", "reference")
                and e["normalized_text"]]
    precision = (round(sum(1 for e in prec_els
                           if _matches(e["normalized_text"], gold_texts)) / len(prec_els), 4)
                 if prec_els else None)

    # document_order_accuracy：gold 顺序序列在输出 body 序列中的贪心对齐（D-G2）
    out_seq = [e for e in els if e["type"] in ("paragraph", "heading", "document_title",
                                               "list_item", "caption", "table")]
    j = 0
    for e in out_seq:
        if j >= len(gold["order"]):
            break
        want = gold["order"][j]
        if want["kind"] == "t":
            if e["type"] == "table" and e["metadata"].get("rows") == want["rows"] \
                    and e["metadata"].get("cols") == want["cols"]:
                j += 1
        else:
            if e["type"] != "table" and _matches(want["text"], [e["normalized_text"]]):
                j += 1
    order_accuracy = round(j / len(gold["order"]), 4) if gold["order"] else 1.0

    # 显式 heading 层级（D-G3a）；inferred 单列（D-G3b）
    explicit_ok = 0
    for h in gold["headings"]:
        for e in heading_els:
            if _matches(h["text"], [e["normalized_text"]]):
                if not e.get("semantic_inferred") and e.get("metadata", {}).get("level") == h["level"]:
                    explicit_ok += 1
                    break
    heading_level_accuracy = round(explicit_ok / len(gold["headings"]), 4) if gold["headings"] else 1.0

    inferred_els = [e for e in heading_els if e.get("semantic_inferred")]
    inferred_marked_rate = 1.0  # 进入该列表的元素必然带 semantic_inferred=True
    style_coverage = (round(sum(1 for e in heading_els if e["source_ref"].get("style_id"))
                            / len(heading_els), 4) if heading_els else None)

    # list 结构
    list_ok = 0
    for li in gold["list_items"]:
        for e in by_type.get("list_item", []):
            md = e.get("metadata", {})
            if _matches(li["text"], [e["normalized_text"]]) and "list_id" in md \
                    and "item_index" in md and md.get("ordered") == li["ordered"]:
                list_ok += 1
                break
    list_structure_accuracy = round(list_ok / len(gold["list_items"]), 4) if gold["list_items"] else None

    # 表格
    table_accs = []
    for gt in gold["tables"]:
        hit = any(e["metadata"].get("rows") == gt["rows"] and e["metadata"].get("cols") == gt["cols"]
                  for e in by_type.get("table", []))
        table_accs.append(1.0 if hit else 0.0)
    table_structure_accuracy = (round(sum(table_accs) / len(table_accs), 4)
                                if table_accs else None)

    gold_merges = {m for t in gold["tables"] for m in t["merges"]}
    out_regions = {_region_ref(r) for e in by_type.get("table", [])
                   for r in e["metadata"].get("merged_regions", [])}
    merged_cell_accuracy = (round(sum(1 for m in gold_merges if m in out_regions) / len(gold_merges), 4)
                            if gold_merges else None)

    gold_nested = any(t["nested"] for t in gold["tables"])
    nested_traceable = None
    if gold_nested:
        # B1：独立 element + parent 链接；B0：table metadata 内 nested_presence 事实
        nested_traceable = bool(
            any(e.get("parent_id") for e in by_type.get("table", []))
            or any("nested_presence" in e["metadata"] for e in by_type.get("table", [])))

    # furniture 分离（D-G5）
    hf_els = by_type.get("header", []) + by_type.get("footer", [])
    if gold["headers"] or gold["footers"]:
        furniture_ok = all(e.get("retrieval_role") == "furniture" for e in hf_els) and len(hf_els) > 0
        body_texts = texts_of(["paragraph", "heading", "document_title", "list_item", "caption"])
        polluted = any(_matches(t, body_texts) for t in gold["headers"] + gold["footers"])
        header_footer_separation = bool(furniture_ok and not polluted)
    else:
        header_footer_separation = None

    # caption 保存 + image/caption 关系
    caption_preservation = caption_recall if gold["captions"] else None
    caption_linked = None
    if gold["captions"] and gold["image_count"] > 0:
        caption_linked = any(r.get("type") == "caption_of" for r in doc.get("relationships", []))

    # source_ref 覆盖 / fake provenance / duplicate
    src_eligible = [e for e in els if e["type"] != "workbook_metadata"]
    source_ref_coverage = (round(sum(1 for e in src_eligible if any(v is not None for v in (e["source_ref"] or {}).values()))
                                 / len(src_eligible), 4) if src_eligible else None)
    fake = sum(1 for e in els if set(e.get("source_ref", {}).keys()) & {"page", "bbox"})
    body_norm = [e["normalized_text"] for e in prec_els]
    dup = (round(sum(cnt - 1 for cnt in _dup_counts(body_norm)) / len(body_norm), 4)
           if body_norm else 0.0)

    return {
        "element_text_recall": round(1 - len(unmatched) / len(gold_texts), 4) if gold_texts else None,
        "element_text_precision": precision,
        "heading_recall": heading_recall,
        "bad_heading_recall": bad_heading_recall,
        "paragraph_recall": paragraph_recall,
        "list_recall": list_recall,
        "caption_recall": caption_recall,
        "footnote_recall": footnote_recall,
        "header_footer_recall": hf_recall,
        "textbox_recall": textbox_recall,
        "document_order_accuracy": order_accuracy,
        "heading_level_accuracy": heading_level_accuracy,
        "inferred_heading_count": len(inferred_els),
        "inferred_marked_rate": inferred_marked_rate,
        "style_provenance_coverage": style_coverage,
        "list_structure_accuracy": list_structure_accuracy,
        "table_structure_accuracy": table_structure_accuracy,
        "merged_cell_accuracy": merged_cell_accuracy,
        "nested_table_traceability": nested_traceable,
        "header_footer_separation": header_footer_separation,
        "caption_preservation": caption_preservation,
        "caption_linked": caption_linked,
        "source_ref_coverage": source_ref_coverage,
        "fake_provenance_count": fake,
        "duplicate_content_rate": dup,
        "unintended_loss": len(unmatched),
        "unmatched_texts": [t[:60] for t in unmatched],
    }


def _region_ref(region: dict) -> str:
    """{"anchor":"R2C1",spans} → "A2:A3"（与 gold merge 对齐的表示空间）。"""

    anchor = region["anchor"]
    r = int(anchor[1:anchor.index("C")])
    c = int(anchor[anchor.index("C") + 1:])

    def letter(n: int) -> str:
        s = ""
        while n > 0:
            n, rr = divmod(n - 1, 26)
            s = chr(65 + rr) + s
        return s

    a = f"{letter(c)}{r}"
    b = f"{letter(c + region['col_span'] - 1)}{r + region['row_span'] - 1}"
    return a if a == b else f"{a}:{b}"


def _dup_counts(items: list[str]) -> list[int]:
    from collections import Counter
    return [v for v in Counter(items).values() if v > 1]


# --------------------------------------------------------------------------------------
# XLSX Gold 坐标计算
# --------------------------------------------------------------------------------------


def gold_cells_xlsx(case: dict) -> list[dict[str, Any]]:
    """从 spec 精确计算每个非空 cell 的 (sheet, coordinate, value, kind)。

    合并区内的非锚点坐标被排除：Excel 语义下 merge 覆盖格不能持有独立值，
    其信息由 merge relation + anchor 扩展承载（X4/X12 表头）。
    """

    out: list[dict[str, Any]] = []

    def _in_merge_nonanchor(sheet_name: str, coord: str, merges_by_sheet: dict) -> bool:
        r = int("".join(ch for ch in coord if ch.isdigit()))
        c = _col_number("".join(ch for ch in coord if ch.isalpha()))
        for anchor, (r1, r2, c1, c2) in merges_by_sheet.get(sheet_name, []):
            if r1 <= r <= r2 and c1 <= c <= c2 and (r, c) != (r1, c1):
                return True
        return False

    merges_by_sheet: dict[str, list] = {}
    for sheet in case["sheets"]:
        entries = []
        for table in sheet.get("tables", []):
            for m in table.get("merges", []):
                a, b = f"{m[0]}:{m[1]}".split(":")
                r1 = int("".join(ch for ch in a if ch.isdigit()))
                r2 = int("".join(ch for ch in b if ch.isdigit()))
                c1 = _col_number("".join(ch for ch in a if ch.isalpha()))
                c2 = _col_number("".join(ch for ch in b if ch.isalpha()))
                entries.append((m[0], (min(r1, r2), max(r1, r2), min(c1, c2), max(c1, c2))))
        merges_by_sheet[sheet["name"]] = entries

    for sheet in case["sheets"]:
        name = sheet["name"]
        for note in sheet.get("notes", []) + sheet.get("side_notes", []):
            out.append({"sheet": name, "coord": note["cell"], "value": str(note["text"]), "kind": "note"})
        for kv in sheet.get("kv_pairs", []):
            out.append({"sheet": name, "coord": kv["cell"], "value": kv["text"], "kind": "kv"})
        for table in sheet.get("tables", []):
            anchor = table["anchor"]
            ar = int("".join(ch for ch in anchor if ch.isdigit()))
            ac = _col_number("".join(ch for ch in anchor if ch.isalpha()))
            if table.get("multi_level_header"):
                for i, h in enumerate(table.get("header_top", [])):
                    if h:
                        out.append({"sheet": name, "coord": f"{_col_letter(ac + i)}{ar}",
                                    "value": h, "kind": "header"})
                for i, h in enumerate(table.get("header_bottom", [])):
                    out.append({"sheet": name, "coord": f"{_col_letter(ac + i)}{ar + 1}",
                                "value": str(h), "kind": "header"})
                data_start = ar + 2
            else:
                for i, h in enumerate(table.get("headers", [])):
                    out.append({"sheet": name, "coord": f"{_col_letter(ac + i)}{ar}",
                                "value": str(h), "kind": "header"})
                data_start = ar + 1
            for ri, row in enumerate(table.get("rows", [])):
                for ci, v in enumerate(row):
                    if v is None:
                        continue
                    out.append({"sheet": name, "coord": f"{_col_letter(ac + ci)}{data_start + ri}",
                                "value": v, "kind": "data"})
            for cell_ref, (formula, _cached) in (table.get("formula_cells") or {}).items():
                out.append({"sheet": name, "coord": cell_ref, "value": formula, "kind": "formula"})
    return [g for g in out if not _in_merge_nonanchor(g["sheet"], g["coord"], merges_by_sheet)]


def gold_table_ranges(case: dict) -> tuple[list[dict[str, str]], list[str]]:
    """返回 (named_tables:[{name, ref}], unnamed_ranges:[ref])，与生成器同一套几何。"""

    named, unnamed = [], []
    for sheet in case["sheets"]:
        for table in sheet.get("tables", []):
            anchor = table["anchor"]
            ar = int("".join(ch for ch in anchor if ch.isdigit()))
            ac = _col_number("".join(ch for ch in anchor if ch.isalpha()))
            if table.get("multi_level_header"):
                n_cols = len(table["header_bottom"])
                n_rows = 2 + len(table.get("rows", []))
            else:
                n_cols = len(table.get("headers", []))
                n_rows = 1 + len(table.get("rows", []))
            ref = f"{anchor}:{_col_letter(ac + n_cols - 1)}{ar + n_rows - 1}"
            if table.get("is_named"):
                named.append({"name": table["table_name"], "ref": ref})
            else:
                unnamed.append(ref)
    return named, unnamed


def compute_xlsx_metrics14b(case: dict, doc: dict) -> dict[str, Any]:
    els = doc["elements"]
    # cell truth 空间包含被重分类为 key_value 的格子（重分类 ≠ 丢失，X-G1 语义）
    cell_els = [e for e in els if e["type"] in ("cell", "key_value")]
    sheet_els = [e for e in els if e["type"] == "sheet"]
    table_els = [e for e in els if e["type"] == "table"]
    header_els = [e for e in els if e["type"] == "table_header"]

    # ---- cell 精确匹配（(sheet, coord) 空间，X-G1） ----
    out_by_key: dict[tuple[str, str], dict] = {}
    for e in cell_els:
        key = (e["source_ref"].get("sheet"), e["source_ref"].get("cell"))
        if key not in out_by_key:
            out_by_key[key] = e  # 重复 cell 由 duplicate_cell_rate 另行计
    gold_cells = gold_cells_xlsx(case)
    matched = 0
    type_ok = 0
    coord_ok = 0
    for g in gold_cells:
        e = out_by_key.get((g["sheet"], g["coord"]))
        if e is None:
            continue
        if _values_equal(g["value"], e["metadata"].get("raw_value")):
            matched += 1
            coord_ok += 1
            kind = g["kind"]
            dt = e["metadata"].get("data_type")
            if kind == "formula":
                type_ok += 1 if dt == "f" else 0
            elif isinstance(_num(g["value"]), float):
                type_ok += 1 if dt in ("n", "f") else 0
            else:
                type_ok += 1 if dt in ("s", "str") else 0
    cell_recall = round(matched / len(gold_cells), 4) if gold_cells else None
    cell_coordinate_accuracy = round(coord_ok / len(gold_cells), 4) if gold_cells else None
    cell_type_accuracy = round(type_ok / len(gold_cells), 4) if gold_cells else None

    dup_cells: dict[tuple[str, str], int] = {}
    for e in cell_els:
        key = (e["source_ref"].get("sheet"), e["source_ref"].get("cell"))
        dup_cells[key] = dup_cells.get(key, 0) + 1
    duplicate_cell_rate = (round(sum(v - 1 for v in dup_cells.values()) / len(cell_els), 4)
                           if cell_els else 0.0)

    # ---- formula 双表示（X-G2） ----
    gold_formulas = []
    for sheet in case["sheets"]:
        for table in sheet.get("tables", []):
            for cell_ref, (formula, cached) in (table.get("formula_cells") or {}).items():
                gold_formulas.append({"sheet": sheet["name"], "cell": cell_ref,
                                      "formula": formula, "cached": cached})
    formula_ok = cached_ok = overwrite = 0
    for f in gold_formulas:
        e = out_by_key.get((f["sheet"], f["cell"]))
        if e is None:
            continue
        md = e["metadata"]
        formula_found = md.get("formula") == f["formula"]
        cached_found = md.get("cached_value") is not None and _values_equal(f["cached"], md.get("cached_value"))
        formula_ok += 1 if formula_found else 0
        cached_ok += 1 if cached_found else 0
        # silent overwrite：公式丢了或 raw_value 变成了缓存值
        if not formula_found or _norm(str(md.get("raw_value"))) == _norm(str(f["cached"])):
            overwrite += 1
    formula_recall = round(formula_ok / len(gold_formulas), 4) if gold_formulas else None
    cached_value_preservation = round(cached_ok / len(gold_formulas), 4) if gold_formulas else None
    formula_silent_overwrite = overwrite if gold_formulas else 0

    # ---- merge（X-G3） ----
    gold_merges = []
    for sheet in case["sheets"]:
        for table in sheet.get("tables", []):
            for m in table.get("merges", []):
                gold_merges.append({"sheet": sheet["name"], "ref": f"{m[0]}:{m[1]}", "anchor": m[0]})
    merged_seen: dict[str, dict] = {}
    for s in sheet_els:
        nm = s["metadata"].get("name")
        for r in s["metadata"].get("merged_ranges", []):
            merged_seen[(nm, r)] = True
        for r, a in (s["metadata"].get("anchor_cells") or {}).items():
            merged_seen.setdefault((nm, r), True)
    merge_ok = sum(1 for g in gold_merges if merged_seen.get((g["sheet"], g["ref"])))
    merged_range_accuracy = round(merge_ok / len(gold_merges), 4) if gold_merges else None
    anchors_by_sheet = {s["metadata"].get("name"): set((s["metadata"].get("anchor_cells") or {}).values())
                        for s in sheet_els}
    anchor_ok = sum(1 for g in gold_merges if g["anchor"] in anchors_by_sheet.get(g["sheet"], set()))
    merge_anchor_coverage = round(anchor_ok / len(gold_merges), 4) if gold_merges else None

    # ---- named / inferred table（X-G5 / X-G6） ----
    gold_named, gold_unnamed = gold_table_ranges(case)
    out_named = {}
    for t in table_els:
        tn = t["source_ref"].get("table_name")
        if tn:
            out_named[tn] = t["source_ref"].get("range")
    if not out_named:  # B0：named tables 记录在 sheet metadata 事实层
        for s in sheet_els:
            for t in s["metadata"].get("named_tables", []):
                out_named.setdefault(t["name"], t["ref"])
    named_ok = sum(1 for g in gold_named if out_named.get(g["name"]) == g["ref"])
    named_table_accuracy = round(named_ok / len(gold_named), 4) if gold_named else None

    out_inferred = [t["source_ref"].get("range") for t in table_els
                    if t["source_ref"].get("inferred") is True]
    inferred_table_precision: float | None = None
    inferred_table_recall: float | None = None
    if out_inferred:
        hits = sum(1 for r in out_inferred if r in set(gold_unnamed))
        inferred_table_precision = round(hits / len(out_inferred), 4)
    if gold_unnamed:
        inferred_table_recall = (round(sum(1 for r in gold_unnamed if r in set(out_inferred))
                                       / len(gold_unnamed), 4) if out_inferred else 0.0)

    # ---- header hierarchy（X12） ----
    hierarchy_accuracy: float | None = None
    multi_tables = [t for s in case["sheets"] for t in s.get("tables", []) if t.get("multi_level_header")]
    if multi_tables:
        _named, _unnamed = gold_table_ranges(case)
        # 期望层级：merge 展开 top + leaf
        expected: set[tuple] = set()
        for t in multi_tables:
            top = list(t.get("header_top", []))
            bottom = list(t.get("header_bottom", []))
            expand = {}
            for m in t.get("merges", []):
                c1 = _col_number("".join(ch for ch in m[0] if ch.isalpha()))
                c2 = _col_number("".join(ch for ch in m[1] if ch.isalpha()))
                v = top[c1 - _col_number("".join(ch for ch in t["anchor"] if ch.isalpha()))]
                for ci in range(c1, c2 + 1):
                    expand[ci] = v
            base = _col_number("".join(ch for ch in t["anchor"] if ch.isalpha()))
            for i, leaf in enumerate(bottom):
                top_v = expand.get(base + i)
                hier = ([top_v] if top_v else []) + [leaf]
                expected.add(tuple(str(x) for x in hier))
        got: set[tuple] = set()
        for h in header_els:
            got.add(tuple(str(x) for x in h["metadata"].get("hierarchy", [])))
        hierarchy_accuracy = round(len(expected & got) / len(expected), 4) if expected else None

    # ---- sheet 身份（X-G4） ----
    gold_sheets = [(s["name"], s.get("state", "visible")) for s in case["sheets"]]
    out_sheets = [(s["metadata"].get("name"), s["metadata"].get("state")) for s in sheet_els]
    sheet_order_accuracy = round(sum(1 for g in gold_sheets if g in out_sheets) / len(gold_sheets), 4) \
        if gold_sheets else None

    # ---- hidden（X-G8）：sheet 级事实（hidden_rows/hidden_cols/sheet state）优先，
    # cell 级 flag 作为补充证据（空列的隐藏无法用 cell 证明） ----
    hidden_facts = 0
    hidden_ok = 0
    meta_by_sheet = {s["metadata"].get("name"): s["metadata"] for s in sheet_els}
    for s in case["sheets"]:
        md = meta_by_sheet.get(s["name"], {})
        if s.get("state") == "hidden":
            hidden_facts += 1
            hidden_ok += any(e["metadata"].get("name") == s["name"] and e["metadata"].get("hidden")
                             for e in sheet_els)
        for r in s.get("hidden_rows", []):
            hidden_facts += 1
            hidden_ok += (r in (md.get("hidden_rows") or [])
                          or any((e["source_ref"].get("sheet") == s["name"])
                                 and e["metadata"].get("is_hidden_row")
                                 and _row_of(e["source_ref"].get("cell")) == r
                                 for e in cell_els))
        for cl in s.get("hidden_cols", []):
            hidden_facts += 1
            hidden_ok += (cl in (md.get("hidden_cols") or [])
                          or any((e["source_ref"].get("sheet") == s["name"])
                                 and e["metadata"].get("is_hidden_col")
                                 and _col_of(e["source_ref"].get("cell")) == cl
                                 for e in cell_els))
    hidden_state_accuracy = round(hidden_ok / hidden_facts, 4) if hidden_facts else None
    hidden_role_ok = all(e.get("retrieval_role") == "hidden" for e in cell_els
                         if e["metadata"].get("is_hidden_row") or e["metadata"].get("is_hidden_col"))

    # ---- number format / display（X-G7 概念；14B 冻结为 raw 不被覆盖） ----
    gold_fmts = []
    for sheet in case["sheets"]:
        for ref, fmt in (sheet.get("number_formats") or {}).items():
            gold_fmts.append({"sheet": sheet["name"], "cell": ref, "fmt": fmt})
    gold_keys = {(g["sheet"], g["coord"]) for g in gold_cells}
    fmt_ok = sum(1 for g in gold_fmts
                 if (e := out_by_key.get((g["sheet"], g["cell"])))
                 and e["metadata"].get("number_format") == g["fmt"])
    number_format_preserved = round(fmt_ok / len(gold_fmts), 4) if gold_fmts else None
    # display 不覆盖 raw：只对 Gold 声明过 raw 值的 cell 判定（未声明 ≠ 被覆盖）
    display_overwrite = sum(1 for g in gold_fmts
                            if (g["sheet"], g["cell"]) in gold_keys
                            and (e := out_by_key.get((g["sheet"], g["cell"])))
                            and not _values_equal(e["metadata"].get("raw_value"),
                                                  _gold_raw_value(case, g["sheet"], g["cell"])))

    # ---- auto filter（事实层，报告用） ----
    autofilter_ok = None
    gold_af = [(s["name"], s["auto_filter"]["ref"]) for s in case["sheets"] if s.get("auto_filter")]
    if gold_af:
        out_af = {s["metadata"].get("name"): s["metadata"].get("auto_filter_ref") for s in sheet_els}
        autofilter_ok = all(out_af.get(nm) == ref for nm, ref in gold_af)

    # ---- sheet order（X-G4 已含）；source_ref / fake / misc ----
    src_eligible = [e for e in els if e["type"] not in ("workbook_metadata",)]
    source_ref_coverage = (round(sum(1 for e in src_eligible
                                     if any(v is not None for v in (e["source_ref"] or {}).values()))
                                 / len(src_eligible), 4) if src_eligible else None)
    fake = sum(1 for e in els if set(e.get("source_ref", {}).keys()) & {"page", "bbox"})

    return {
        "non_empty_cell_recall": cell_recall,
        "cell_coordinate_accuracy": cell_coordinate_accuracy,
        "cell_type_accuracy": cell_type_accuracy,
        "formula_recall": formula_recall,
        "cached_value_preservation": cached_value_preservation,
        "formula_silent_overwrite": formula_silent_overwrite,
        "merged_range_accuracy": merged_range_accuracy,
        "merge_anchor_coverage": merge_anchor_coverage,
        "named_table_accuracy": named_table_accuracy,
        "inferred_table_precision": inferred_table_precision,
        "inferred_table_recall": inferred_table_recall,
        "header_hierarchy_accuracy": hierarchy_accuracy,
        "hidden_state_accuracy": hidden_state_accuracy,
        "hidden_role_policy_ok": hidden_role_ok if hidden_facts else None,
        "sheet_order_accuracy": sheet_order_accuracy,
        "number_format_preserved": number_format_preserved,
        "display_overwrite_count": display_overwrite,
        "auto_filter_preserved": autofilter_ok,
        "source_ref_coverage": source_ref_coverage,
        "duplicate_cell_rate": duplicate_cell_rate,
        "fake_provenance_count": fake,
        "cell_elements": len(cell_els),
        "table_elements": len(table_els),
        "cleaning_log_count": len(doc.get("cleaning_log", [])),
        "unresolved_count": len(doc.get("unresolved", [])),
    }


def _num(v: Any) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _row_of(coord: str | None) -> int | None:
    if not coord:
        return None
    digits = "".join(ch for ch in coord if ch.isdigit())
    return int(digits) if digits else None


def _col_of(coord: str | None) -> str | None:
    if not coord:
        return None
    return "".join(ch for ch in coord if ch.isalpha()).upper() or None


def _gold_raw_value(case: dict, sheet_name: str, coord: str) -> Any:
    for g in gold_cells_xlsx(case):
        if g["sheet"] == sheet_name and g["coord"] == coord:
            return g["value"]
    return None
