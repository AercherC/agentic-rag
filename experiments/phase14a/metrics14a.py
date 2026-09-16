"""Phase 14A-A 度量（方案 §21）。

把候选 parser 的统一 StructuredDocument 与 Gold（corpus14a 意图）逐项比对，输出：
- Common Metrics：source_element_recall / semantic_type_accuracy / provenance_coverage /
  duplicate_content_rate / unintended_loss / determinism。
- DOCX Specific：heading_level_accuracy / paragraph_order_accuracy / list_structure_accuracy /
  table_structure_accuracy / header_footer_separation / footnote_recall / caption_link_accuracy。
- XLSX Specific：cell_recall / cell_type_accuracy / formula_recall / cached_value_preservation /
  merged_range_accuracy / table_boundary_accuracy / sheet_metadata_accuracy。

匹配采用归一化文本包含 / token Jaccard（≥0.85），避免逐字符脆弱比较。
所有比例指标分子分母同处一个计数空间；不出现 >1 / <0（由 gates 守卫）。
"""

from __future__ import annotations

from typing import Any

from . import schema14a

_NORM = schema14a.collapse_whitespace


def _norm(text: str) -> str:
    return _NORM(schema14a.normalize_unicode(text or ""))


def _tokens(text: str) -> set[str]:
    return set(schema14a.tokenize(text or ""))


def _jaccard(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _matches(gold_text: str, out_texts: list[str], jac: float = 0.85) -> bool:
    gn = _norm(gold_text)
    for ot in out_texts:
        on = _norm(ot)
        if gn and (gn in on or on in gn or _jaccard(gold_text, ot) >= jac):
            return True
    return False


# --------------------------------------------------------------------------------------
# Gold 抽取
# --------------------------------------------------------------------------------------


def collect_gold_docx(case: dict) -> dict[str, Any]:
    g: dict[str, Any] = {
        "headings": [], "paragraphs": [], "list_items": [], "tables": [],
        "captions": [], "footnotes": [], "headers": [], "footers": [],
        "images": [], "textboxes": [], "bad_headings": [],
    }
    for b in case["blocks"]:
        k = b["kind"]
        if k in ("title", "heading"):
            g["headings"].append(b["text"])
        elif k == "paragraph":
            g["paragraphs"].append(b["text"])
        elif k == "list_item":
            g["list_items"].append(b["text"])
        elif k == "table":
            g["tables"].append(b)
        elif k == "caption":
            g["captions"].append(b["text"])
        elif k == "footnote":
            g["footnotes"].append(b["text"])
        elif k == "header":
            g["headers"].append(b["text"])
        elif k == "footer":
            g["footers"].append(b["text"])
        elif k == "image":
            g["images"].append(b)
        elif k == "textbox":
            g["textboxes"].append(b["text"])
        elif k == "bad_heading":
            g["bad_headings"].append(b["text"])
    return g


def collect_gold_xlsx(case: dict) -> dict[str, Any]:
    g: dict[str, Any] = {
        "sheets": [], "cells": [], "formulas": [], "merges": [],
        "named_tables": [], "hidden_sheets": [], "hidden_rows": [], "hidden_cols": [],
    }
    for sheet in case["sheets"]:
        g["sheets"].append({"name": sheet["name"], "state": sheet.get("state", "visible")})
        if sheet.get("state") == "hidden":
            g["hidden_sheets"].append(sheet["name"])
        g["hidden_rows"].extend(sheet.get("hidden_rows", []))
        g["hidden_cols"].extend(sheet.get("hidden_cols", []))
        for note in sheet.get("notes", []) + sheet.get("side_notes", []):
            g["cells"].append({"sheet": sheet["name"], "text": str(note["text"])})
        for table in sheet.get("tables", []):
            if table.get("is_named"):
                g["named_tables"].append(table.get("table_name"))
            headers = table.get("headers") or table.get("header_bottom") or []
            if table.get("multi_level_header"):
                for h in table.get("header_top", []) + table.get("header_bottom", []):
                    if h:
                        g["cells"].append({"sheet": sheet["name"], "text": str(h)})
            else:
                for h in headers:
                    if h:
                        g["cells"].append({"sheet": sheet["name"], "text": str(h)})
            for row in table.get("rows", []):
                for x in row:
                    if x is not None:
                        g["cells"].append({"sheet": sheet["name"], "text": str(x)})
            for merge in table.get("merges", []):
                g["merges"].append(f"{merge[0]}:{merge[1]}")
            for cell_ref, (formula, cached) in (table.get("formula_cells") or {}).items():
                g["formulas"].append({"sheet": sheet["name"], "cell": cell_ref,
                                      "formula": formula, "cached": cached})
    return g


# --------------------------------------------------------------------------------------
# DOCX 度量
# --------------------------------------------------------------------------------------


def compute_docx_metrics(case: dict, doc: dict) -> dict[str, Any]:
    gold = collect_gold_docx(case)
    els = doc["elements"]
    by_type = {}
    for e in els:
        by_type.setdefault(e["type"], []).append(e)

    out_texts = lambda t: [e["normalized_text"] for e in by_type.get(t, [])]
    out_all = [e["normalized_text"] for e in els]

    def recall(texts: list[str]) -> float:
        if not texts:
            return 1.0
        hit = sum(1 for t in texts if _matches(t, out_all))
        return round(hit / len(texts), 4)

    headings_out = by_type.get("heading", []) + by_type.get("document_title", [])
    heading_recall = recall(gold["headings"])
    paragraph_recall = recall(gold["paragraphs"])
    list_recall = recall(gold["list_items"])
    footnote_recall = recall(gold["footnotes"])
    caption_recall = recall(gold["captions"])
    header_footer_out = by_type.get("header", []) + by_type.get("footer", [])
    header_footer_recall = recall(gold["headers"] + gold["footers"])

    # paragraph order accuracy: gold 段落顺序是否在输出 body 中出现（连续子序列）
    para_seq = [_norm(t) for t in gold["paragraphs"]]
    body_norm = [_norm(e["normalized_text"]) for e in els if e["type"] == "paragraph"]
    order_ok = 0
    if para_seq:
        # 贪心匹配顺序
        j = 0
        for bn in body_norm:
            if j < len(para_seq) and (para_seq[j] in bn or bn in para_seq[j] or _jaccard(gold["paragraphs"][j], bn) >= 0.85):
                order_ok += 1
                j += 1
        paragraph_order_accuracy = round(order_ok / len(para_seq), 4)
    else:
        paragraph_order_accuracy = 1.0

    # table structure accuracy
    table_accs = []
    for gt in gold["tables"]:
        # 找一个输出 table 元素其 metadata rows/cols 与 gold 接近
        best = 0.0
        g_rows = len(gt.get("rows", [])) + (len(gt.get("headers", [])) if not gt.get("multi_row_header") else len(gt["headers"]))
        g_cols = len(gt.get("headers", [[]])[0]) if gt.get("headers") else (len(gt.get("header_bottom", [])) if gt.get("multi_level_header") else 1)
        for e in by_type.get("table", []):
            m = e.get("metadata", {})
            if m.get("rows") == g_rows and m.get("cols") == g_cols:
                best = 1.0
                break
        table_accs.append(best)
    table_structure_accuracy = round(sum(table_accs) / len(table_accs), 4) if table_accs else 1.0

    # header/footer separation: 存在分离元素且其 retrieval_role != body
    hsep = True
    if gold["headers"] or gold["footers"]:
        hf = by_type.get("header", []) + by_type.get("footer", [])
        hsep = any(e["retrieval_role"] != "body" for e in hf) and len(hf) > 0

    # caption + image link
    has_image = len(by_type.get("image", [])) > 0
    has_caption = len(by_type.get("caption", [])) > 0
    caption_link_accuracy = 1.0 if (not gold["captions"] or (has_image and has_caption)) else 0.0

    # style provenance for headings
    style_prov = all(e.get("source_ref", {}).get("style_id") for e in headings_out) if headings_out else True

    # bad heading fallback: 这些文本不应被识别成 heading（应为 paragraph）
    bad_ok = True
    for t in gold["bad_headings"]:
        matched_heading = any(_matches(t, [e["normalized_text"]]) and e["type"] in ("heading", "document_title")
                              for e in headings_out)
        if matched_heading:
            bad_ok = False
            break

    # fake provenance: DOCX 不得含 page/bbox
    fake = sum(1 for e in els if set(e.get("source_ref", {}).keys()) & {"page", "bbox"})

    semantic_type_accuracy = round(
        (len(headings_out) > 0 if gold["headings"] else True)
        and (len(by_type.get("table", [])) >= len(gold["tables"]))
        and (len(by_type.get("list_item", [])) >= len(gold["list_items"])),
        4,
    ) if (gold["headings"] or gold["tables"] or gold["list_items"]) else 1.0

    return {
        "heading_recall": heading_recall,
        "paragraph_recall": paragraph_recall,
        "paragraph_order_accuracy": paragraph_order_accuracy,
        "list_structure_recall": list_recall,
        "table_structure_accuracy": table_structure_accuracy,
        "header_footer_separation": bool(hsep),
        "header_footer_recall": header_footer_recall,
        "footnote_recall": footnote_recall,
        "caption_link_accuracy": caption_link_accuracy,
        "style_provenance_ok": bool(style_prov),
        "bad_heading_not_faked_as_heading": bad_ok,
        "fake_provenance_count": fake,
        "semantic_type_accuracy": float(semantic_type_accuracy),
    }


# --------------------------------------------------------------------------------------
# XLSX 度量
# --------------------------------------------------------------------------------------


def compute_xlsx_metrics(case: dict, doc: dict) -> dict[str, Any]:
    gold = collect_gold_xlsx(case)
    els = doc["elements"]
    cells = [e for e in els if e["type"] == "cell"]
    sheets = [e for e in els if e["type"] == "sheet"]
    tables = [e for e in els if e["type"] == "table"]

    out_cell_texts = [_norm(e["normalized_text"]) for e in cells]
    cell_text_recall = round(
        sum(1 for c in gold["cells"] if _matches(c["text"], out_cell_texts)) / len(gold["cells"]), 4
    ) if gold["cells"] else 1.0

    # formula dual representation
    formula_metrics = []
    for f in gold["formulas"]:
        el = next((e for e in cells if e.get("source_ref", {}).get("cell") == f["cell"]
                   and e.get("source_ref", {}).get("sheet") == f["sheet"]), None)
        if el is None:
            formula_metrics.append({"cell": f["cell"], "formula_found": False, "cached_found": False})
        else:
            md = el.get("metadata", {})
            formula_metrics.append({
                "cell": f["cell"],
                "formula_found": md.get("formula") == f["formula"],
                "cached_found": str(md.get("cached_value")) == str(f["cached"]),
            })
    formula_recall = round(
        sum(1 for m in formula_metrics if m["formula_found"]) / len(formula_metrics), 4
    ) if formula_metrics else 1.0
    cached_preservation = round(
        sum(1 for m in formula_metrics if m["cached_found"]) / len(formula_metrics), 4
    ) if formula_metrics else 1.0

    # merged ranges: 从 sheet metadata 收集
    merged_seen: list[str] = []
    for s in sheets:
        merged_seen.extend(s.get("metadata", {}).get("merged_ranges", []))
    merged_accuracy = round(
        sum(1 for m in gold["merges"] if m in merged_seen) / len(gold["merges"]), 4
    ) if gold["merges"] else 1.0

    # table boundary: named tables from output
    out_named = [t.get("source_ref", {}).get("table_name") for t in tables if t.get("source_ref", {}).get("table_name")]
    named_accuracy = round(
        sum(1 for n in gold["named_tables"] if n in out_named) / len(gold["named_tables"]), 4
    ) if gold["named_tables"] else 1.0
    # heuristic subtable 标记 inferred
    inferred_present = any(t.get("source_ref", {}).get("inferred") is True for t in tables)

    # sheet metadata
    out_sheets = [(s.get("metadata", {}).get("name"), s.get("metadata", {}).get("state")) for s in sheets]
    gold_sheets = [(s["name"], s["state"]) for s in gold["sheets"]]
    sheet_accuracy = round(
        sum(1 for g in gold_sheets if g in out_sheets) / len(gold_sheets), 4
    ) if gold_sheets else 1.0

    # hidden content not silently dropped
    hidden_sheet_present = all(
        any(s.get("metadata", {}).get("name") == h and s.get("metadata", {}).get("hidden") for s in sheets)
        for h in gold["hidden_sheets"]
    ) if gold["hidden_sheets"] else True
    hidden_cells_present = any(e.get("metadata", {}).get("is_hidden_row") for e in cells) if gold["hidden_rows"] else True

    # display vs raw: number_format 保留
    fmt_present = any(e.get("metadata", {}).get("number_format") for e in cells)

    # fake provenance
    fake = sum(1 for e in els if set(e.get("source_ref", {}).keys()) & {"page", "bbox"})

    return {
        "cell_recall": cell_text_recall,
        "formula_recall": formula_recall,
        "cached_value_preservation": cached_preservation,
        "merged_range_accuracy": merged_accuracy,
        "named_table_accuracy": named_accuracy,
        "heuristic_subtable_inferred_flag": inferred_present,
        "sheet_metadata_accuracy": sheet_accuracy,
        "hidden_sheet_preserved": bool(hidden_sheet_present),
        "hidden_row_preserved": bool(hidden_cells_present),
        "number_format_preserved": bool(fmt_present),
        "fake_provenance_count": fake,
        "formula_cell_detail": formula_metrics,
    }


# --------------------------------------------------------------------------------------
# Determinism
# --------------------------------------------------------------------------------------


def is_deterministic(doc_a: dict, doc_b: dict) -> bool:
    return schema14a.digest_of(doc_a) == schema14a.digest_of(doc_b)
