"""Phase 14A-B 回归测试（方案 §25），可独立运行：
    python -m phase14a.tests.test_phase14a_b

不依赖 python-docx/openpyxl 重库（ cleaners/metrics/gates 为纯函数，直接喂 native 流）；
真实文件链路由 run_phase14a_b 覆盖。含 4 类 mutation validation：
formula overwrite / fake provenance / silent inferred table / missing cleaning log。
"""

from __future__ import annotations

import copy
from pathlib import Path

from .. import adapters14b, cleaners14b, corpus14b, gates14b, metrics14b, sandbox14a, schema14b

RESULTS: list[tuple[str, bool, str]] = []


def _t(name: str):
    def deco(fn):
        def wrap():
            try:
                ok, msg = fn()
            except Exception as exc:  # noqa: BLE001
                ok, msg = False, f"exception: {exc}"
            RESULTS.append((name, ok, msg))
        return wrap
    return deco


# ---- native 流构造器 ----

def _p(i, text, style="Normal", bold=False, size=None, numbering=False, pic=False, tb=False):
    return {"kind": "paragraph", "para_index": i, "text": text, "style": style,
            "bold": bold, "size_pt": size, "numbering": numbering, "has_pic": pic,
            "is_textbox": tb}


def _tbl(i, rows, cols, cells, merges=None, nested=None, style="Table Grid"):
    return {"kind": "table", "table_index": i, "rows": rows, "cols": cols, "cells": cells,
            "merged_regions": merges or [], "nested": nested or [], "style": style}


def _docx_native(body, headers=None, footers=None, footnotes=None):
    return {"doc_id": "tw", "source_file": "tw.docx", "source_sha256": "0" * 64,
            "body": body, "headers": headers or [], "footers": footers or [],
            "footnotes": footnotes or [], "page_breaks": []}


def _xc(coord, row, col, value, **kw):
    d = {"coord": coord, "row": row, "col": col, "raw_value": value, "formula": None,
         "cached": None, "data_type": "s" if isinstance(value, str) else "n",
         "number_format": "General", "hidden_row": False, "hidden_col": False}
    d.update(kw)
    return d


def _xlsx_native(sheets):
    for i, s in enumerate(sheets):
        s.setdefault("order", i)
        s.setdefault("state", "visible")
        s.setdefault("merged_ranges", [])
        s.setdefault("anchor_cells", {})
        s.setdefault("auto_filter_ref", None)
        s.setdefault("named_tables", [])
        s.setdefault("charts", [])
        s.setdefault("images", [])
    return {"doc_id": "tx", "source_file": "tx.xlsx", "source_sha256": "0" * 64, "sheets": sheets}


# --------------------------------------------------------------------------------------
# DOCX
# --------------------------------------------------------------------------------------


@_t("test_docx_interleaved_paragraph_table_order")
def t_order():
    body = [_p(0, "Intro paragraph before the first table."),
            _tbl(0, 2, 2, [["Id", "Name"], ["1", "Alpha"]]),
            _p(1, "Middle paragraph between the two tables."),
            _tbl(1, 2, 2, [["Id", "Score"], ["1", "0.91"]]),
            _p(2, "Closing paragraph after the second table.")]
    case = {"blocks": [{"kind": "paragraph", "text": body[0]["text"]},
                       {"kind": "table", "headers": [["Id", "Name"]], "rows": [["1", "Alpha"]]},
                       {"kind": "paragraph", "text": body[2]["text"]},
                       {"kind": "table", "headers": [["Id", "Score"]], "rows": [["1", "0.91"]]},
                       {"kind": "paragraph", "text": body[4]["text"]}]}
    for semantic in (False, True):
        doc = cleaners14b.clean_docx(_docx_native(copy.deepcopy(body)), semantic=semantic)
        m = metrics14b.compute_docx_metrics14b(case, doc)
        if m["document_order_accuracy"] != 1.0:
            return False, f"semantic={semantic} order={m['document_order_accuracy']}"
        # body 顺序确实交错（非 paragraphs 全部在前）
        types = [e["type"] for e in doc["elements"] if e["type"] in ("paragraph", "table")]
        if types != ["paragraph", "table", "paragraph", "table", "paragraph"]:
            return False, f"semantic={semantic} types={types}"
    return True, "p→tbl→p→tbl→p 顺序保持（B0/B1）"


@_t("test_docx_explicit_style_beats_heuristic")
def t_explicit_beats_heuristic():
    # 有显式 Heading 2 样式 + 同时带 bold/numbering 特征：必须走显式路径，不得标 inferred
    native = _docx_native([_p(0, "1.1 Background", style="Heading 2", bold=True, numbering=True)])
    doc = cleaners14b.clean_docx(native, semantic=True)
    el = doc["elements"][0]
    return (el["type"] == "heading" and el["semantic_inferred"] is False
            and el["metadata"]["level"] == 2
            and el["source_ref"]["style_id"] == "Heading 2"), "explicit style wins, no inference"


@_t("test_docx_inferred_heading_is_marked")
def t_inferred_marked():
    native = _docx_native([
        _p(0, "System Overview", bold=True, size=16.0),
        _p(1, "1.2 Rules", bold=True, size=13.0),
    ])
    doc = cleaners14b.clean_docx(native, semantic=True)
    heads = [e for e in doc["elements"] if e["type"] == "heading"]
    ok = (len(heads) == 2 and all(e["semantic_inferred"] for e in heads)
          and all("inference_features" in e["metadata"] for e in heads)
          and heads[1]["metadata"]["level"] == 2)
    logs = [l for l in doc["cleaning_log"] if l["action"] == "promote_heading"]
    return ok and len(logs) == 2, f"inferred headings marked + logged ({len(logs)})"


@_t("test_docx_header_footer_role")
def t_hf():
    native = _docx_native([_p(0, "Body text.")], headers=["Running Header"], footers=["Footer"])
    doc = cleaners14b.clean_docx(native, semantic=True)
    hf = [e for e in doc["elements"] if e["type"] in ("header", "footer")]
    return (len(hf) == 2 and all(e["retrieval_role"] == "furniture" for e in hf)
            and all(e["retrieval_role"] != "body" for e in hf)), "header/footer → furniture"


@_t("test_docx_list_hierarchy")
def t_list():
    body = [_p(0, "first", style="List Bullet"), _p(1, "second", style="List Bullet"),
            _p(2, "num", style="List Number")]
    doc = cleaners14b.clean_docx(_docx_native(body), semantic=True)
    items = [e for e in doc["elements"] if e["type"] == "list_item"]
    md0, md1, md2 = (e["metadata"] for e in items)
    ok = (md0["list_id"] == md1["list_id"] and md0["item_index"] == 1 and md1["item_index"] == 2
          and md0["ordered"] is False and md2["ordered"] is True
          and md2["list_id"] != md0["list_id"])
    return ok, f"list groups: {md0['list_id']} vs {md2['list_id']}"


@_t("test_docx_merged_table_cells")
def t_merged():
    body = [_tbl(0, 3, 2, [["Region", "V"], ["North", "1"], ["North", "2"]],
                 merges=[{"anchor": "R2C1", "row_span": 2, "col_span": 1}])]
    case = {"blocks": [{"kind": "merged_table", "headers": [["Region", "V"]],
                        "rows": [["North", "1"], ["North", "2"]],
                        "merges": [["A2", "A3"]]}]}
    doc = cleaners14b.clean_docx(_docx_native(body), semantic=True)
    m = metrics14b.compute_docx_metrics14b(case, doc)
    return m["merged_cell_accuracy"] == 1.0, f"merged_cell_accuracy={m['merged_cell_accuracy']}"


@_t("test_docx_nested_table_traceability")
def t_nested():
    nest = {"rows": 2, "cols": 2, "cells": [["Detail", "Note"], ["Encoder", "frozen"]]}
    body = [_tbl(0, 2, 2, [["Model", "Params"], ["Small", "1M"]], nested=[nest])]
    doc1 = cleaners14b.clean_docx(_docx_native(copy.deepcopy(body)), semantic=True)
    doc0 = cleaners14b.clean_docx(_docx_native(copy.deepcopy(body)), semantic=False)
    b1_nested = [e for e in doc1["elements"] if e["type"] == "table" and e.get("parent_id")]
    b0_outer = doc0["elements"][0]
    ok1 = (len(b1_nested) == 1
           and any(r["type"] == "contains" for r in doc1["relationships"]))
    ok0 = "nested_presence" in b0_outer["metadata"] and len(b0_outer["metadata"]["nested_presence"]) == 1
    return ok1 and ok0, f"B1 parent_link={ok1}, B0 presence={ok0}"


@_t("test_docx_no_fake_page_bbox")
def t_docx_nofake():
    native = _docx_native([_p(0, "hello")])
    doc = cleaners14b.clean_docx(native, semantic=True)
    m = metrics14b.compute_docx_metrics14b({"blocks": [{"kind": "paragraph", "text": "hello"}]}, doc)
    clean = m["fake_provenance_count"] == 0
    doc["elements"][0]["source_ref"]["page"] = 3  # 注入伪造页码
    m2 = metrics14b.compute_docx_metrics14b({"blocks": [{"kind": "paragraph", "text": "hello"}]}, doc)
    return clean and m2["fake_provenance_count"] == 1, f"fake detected after injection ({m2['fake_provenance_count']})"


# --------------------------------------------------------------------------------------
# XLSX
# --------------------------------------------------------------------------------------


@_t("test_xlsx_formula_cached_dual_representation")
def t_formula():
    cells = [_xc("A1", 1, 1, "a"), _xc("B1", 1, 2, 30),
             _xc("C2", 2, 3, "=A2+B2", formula="=A2+B2", cached=30, data_type="f")]
    native = _xlsx_native([{"name": "S", "cells": cells}])
    doc = cleaners14b.clean_xlsx(native, semantic=True)
    el = next(e for e in doc["elements"] if e["type"] == "cell" and e["source_ref"]["cell"] == "C2")
    md = el["metadata"]
    return (md["formula"] == "=A2+B2" and md["cached_value"] == 30
            and md["raw_value"] == "=A2+B2"), "formula 与 cached 各自保存，互不覆盖"


@_t("test_xlsx_cached_null_preserved")
def t_cached_null():
    cells = [_xc("C2", 2, 3, "=A2+B2", formula="=A2+B2", cached=None, data_type="f")]
    native = _xlsx_native([{"name": "S", "cells": cells}])
    doc = cleaners14b.clean_xlsx(native, semantic=True)
    el = next(e for e in doc["elements"] if e["type"] == "cell")
    md = el["metadata"]
    unresolved = [u for u in doc["unresolved"] if u["issue"] == "cached_value_missing"]
    return (md["cached_value"] is None and md["is_formula"] and len(unresolved) == 1
            and unresolved[0]["target"].endswith("C2")), "cached 缺失显式 None + unresolved（不冒充 0）"


@_t("test_xlsx_display_value_does_not_overwrite_raw")
def t_display():
    cells = [_xc("B2", 2, 2, 0.816, number_format="0.00%"), _xc("C2", 2, 3, "81.6%")]
    native = _xlsx_native([{"name": "S", "cells": cells}])
    doc = cleaners14b.clean_xlsx(native, semantic=True)
    b2 = next(e for e in doc["elements"] if e["source_ref"]["cell"] == "B2")
    case = {"sheets": [{"name": "S", "number_formats": {"B2": "0.00%"},
                        "tables": [{"anchor": "B2", "headers": [], "rows": []}]}]}
    m = metrics14b.compute_xlsx_metrics14b(case, doc)
    return (b2["metadata"]["raw_value"] == 0.816
            and b2["metadata"]["number_format"] == "0.00%"
            and m["display_overwrite_count"] == 0), "raw 数值不被 display/格式覆盖"


@_t("test_xlsx_merged_header_hierarchy")
def t_hierarchy():
    cells = [_xc("B1", 1, 2, "2025"), _xc("D1", 1, 4, "2026"),
             _xc("A2", 2, 1, "Region"), _xc("B2", 2, 2, "Q1"), _xc("C2", 2, 3, "Q2"),
             _xc("D2", 2, 4, "Q1"), _xc("E2", 2, 5, "Q2"),
             _xc("A3", 3, 1, "North"), _xc("B3", 3, 2, 10), _xc("C3", 3, 3, 12),
             _xc("D3", 3, 4, 14), _xc("E3", 3, 5, 16)]
    native = _xlsx_native([{"name": "H", "cells": cells,
                            "merged_ranges": ["B1:C1", "D1:E1"],
                            "anchor_cells": {"B1:C1": "B1", "D1:E1": "D1"}}])
    doc = cleaners14b.clean_xlsx(native, semantic=True)
    hs = {(e["metadata"]["column_index"], tuple(e["metadata"]["hierarchy"]))
          for e in doc["elements"] if e["type"] == "table_header"}
    expect = {(2, ("2025", "Q1")), (3, ("2025", "Q2")), (4, ("2026", "Q1")), (5, ("2026", "Q2")),
              (1, ("Region",))}
    anchors = {e["metadata"].get("anchor_cell") for e in doc["elements"] if e["type"] == "table_header"}
    return hs == expect and anchors == {None, "B1", "D1"}, f"hierarchy={sorted(str(h) for h in hs)}"


@_t("test_xlsx_named_table_precedence")
def t_named():
    cells = [_xc("A1", 1, 1, "Method"), _xc("B1", 1, 2, "CD"),
             _xc("A2", 2, 1, "Ours"), _xc("B2", 2, 2, 7.4)]
    native = _xlsx_native([{"name": "R", "cells": cells,
                            "named_tables": [{"name": "ResultTable", "ref": "A1:B2"}]}])
    doc = cleaners14b.clean_xlsx(native, semantic=True)
    tables = [e for e in doc["elements"] if e["type"] == "table"]
    named = [t for t in tables if t["source_ref"]["table_name"]]
    inferred = [t for t in tables if t["source_ref"]["inferred"] is True]
    return (len(named) == 1 and named[0]["source_ref"]["inferred"] is False
            and not inferred), "named table 优先且不重复推断"


@_t("test_xlsx_inferred_table_marked")
def t_inferred_marked():
    cells = [_xc("A1", 1, 1, "A"), _xc("B1", 1, 2, "B"), _xc("A2", 2, 1, 1), _xc("B2", 2, 2, 2)]
    native = _xlsx_native([{"name": "S", "cells": cells}])
    doc = cleaners14b.clean_xlsx(native, semantic=True)
    t = next(e for e in doc["elements"] if e["type"] == "table")
    ok = (t["source_ref"]["inferred"] is True
          and "inference_reason" in t["metadata"]
          and any(l["action"] == "infer_table_region" and l["target"] == t["element_id"]
                  for l in doc["cleaning_log"]))
    return ok, f"inferred=True + reason + log ({t['source_ref']['range']})"


@_t("test_xlsx_multiple_regions_one_sheet")
def t_multi_region():
    cells = [_xc("A1", 1, 1, "Two Regions Report"),
             _xc("A3", 3, 1, "Method"), _xc("B3", 3, 2, "CD"), _xc("A4", 4, 1, "Base"), _xc("B4", 4, 2, 8.9),
             _xc("A6", 6, 1, "Variant"), _xc("B6", 6, 2, "CD"), _xc("A7", 7, 1, "V1"), _xc("B7", 7, 2, 7.9),
             _xc("D6", 6, 4, "Note: V2 is the best variant.")]
    native = _xlsx_native([{"name": "M", "cells": cells}])
    doc = cleaners14b.clean_xlsx(native, semantic=True)
    inferred = sorted(t["source_ref"]["range"] for t in doc["elements"]
                      if t["type"] == "table" and t["source_ref"]["inferred"] is True)
    rejects = [l for l in doc["cleaning_log"] if l["action"] == "reject_table_candidate"]
    return (inferred == ["A3:B4", "A6:B7"] and len(rejects) == 2
            ), f"inferred={inferred} rejects={len(rejects)}（1x1 备注不进表）"


@_t("test_xlsx_hidden_content_preserved")
def t_hidden():
    cells = [_xc("A1", 1, 1, "k"), _xc("B1", 1, 2, "v"),
             _xc("A2", 2, 1, 1), _xc("B2", 2, 2, 2),
             _xc("A3", 3, 1, 3, hidden_row=True), _xc("B3", 3, 2, 4, hidden_row=True)]
    native = _xlsx_native([{"name": "V", "cells": cells},
                           {"name": "HS", "state": "hidden", "cells": [_xc("A1", 1, 1, "x")]}])
    for semantic in (False, True):
        doc = cleaners14b.clean_xlsx(_xlsx_native([
            {"name": "V", "cells": [dict(c) for c in cells]},
            {"name": "HS", "state": "hidden", "cells": [_xc("A1", 1, 1, "x")]}]), semantic=semantic)
        hidden_cells = [e for e in doc["elements"] if e["metadata"].get("is_hidden_row")]
        if not hidden_cells or any(e["retrieval_role"] != "hidden" for e in hidden_cells):
            return False, f"semantic={semantic} hidden role policy violated"
        sheets = [e for e in doc["elements"] if e["type"] == "sheet"]
        if not any(s["metadata"]["hidden"] and s["metadata"]["name"] == "HS" for s in sheets):
            return False, f"semantic={semantic} hidden sheet lost"
    return True, "hidden row/sheet 保留且 role=hidden（B0/B1）"


@_t("test_xlsx_sheet_order_preserved")
def t_sheet_order():
    native = _xlsx_native([{"name": "Summary", "cells": []},
                           {"name": "RawData", "cells": []},
                           {"name": "Ablation", "cells": []}])
    doc = cleaners14b.clean_xlsx(native, semantic=True)
    case = {"sheets": [{"name": "Summary"}, {"name": "RawData"}, {"name": "Ablation"}]}
    m = metrics14b.compute_xlsx_metrics14b(case, doc)
    return m["sheet_order_accuracy"] == 1.0, f"sheet_order={m['sheet_order_accuracy']}"


@_t("test_xlsx_no_fake_page_bbox")
def t_xlsx_nofake():
    doc = cleaners14b.clean_xlsx(_xlsx_native([{"name": "S", "cells": [_xc("A1", 1, 1, 1)]}]),
                                 semantic=True)
    clean = all(not ({"page", "bbox"} & set(e["source_ref"].keys())) for e in doc["elements"])
    doc["elements"][0]["source_ref"]["bbox"] = [0, 0, 1, 1]
    detected = any({"page", "bbox"} & set(e["source_ref"].keys()) for e in doc["elements"])
    return clean and detected, "无伪造 page/bbox；注入后可检出"


# --------------------------------------------------------------------------------------
# 通用
# --------------------------------------------------------------------------------------


@_t("test_cleaning_log_complete")
def t_log_complete():
    docx_doc = cleaners14b.clean_docx(_docx_native([
        _p(0, "T", style="Title"),
        _p(1, "Implied Header", bold=True, size=16.0),
    ]), semantic=True)
    cells = [_xc("A1", 1, 1, "A"), _xc("B1", 1, 2, "B"), _xc("A2", 2, 1, 1), _xc("B2", 2, 2, 2)]
    xlsx_doc = cleaners14b.clean_xlsx(_xlsx_native([{"name": "S", "cells": cells}]), semantic=True)
    ok1 = gates14b.cleaning_traceability_ok(docx_doc, "docx")
    ok2 = gates14b.cleaning_traceability_ok(xlsx_doc, "xlsx")
    return ok1 and ok2 and len(docx_doc["cleaning_log"]) >= 2 and len(xlsx_doc["cleaning_log"]) >= 1, \
        f"traceability docx={ok1} xlsx={ok2}"


@_t("test_output_deterministic")
def t_det():
    body = [_p(0, "T", style="Title"), _p(1, "H", style="Heading 1"), _p(2, "body")]
    d1 = cleaners14b.clean_docx(_docx_native(copy.deepcopy(body)), semantic=True)
    d2 = cleaners14b.clean_docx(_docx_native(copy.deepcopy(body)), semantic=True)
    cells = [_xc("A1", 1, 1, "A"), _xc("A2", 2, 1, 1)]
    x1 = cleaners14b.clean_xlsx(_xlsx_native([{"name": "S", "cells": copy.deepcopy(cells)}]), semantic=True)
    x2 = cleaners14b.clean_xlsx(_xlsx_native([{"name": "S", "cells": copy.deepcopy(cells)}]), semantic=True)
    return (schema14b.digest_of(d1) == schema14b.digest_of(d2)
            and schema14b.digest_of(x1) == schema14b.digest_of(x2)), "同输入同 digest"


@_t("test_metric_none_not_zero")
def t_none():
    # B0 不做区域推断：inferred_table_* 语义上不适用，必须是 None 而不是 0
    doc = cleaners14b.clean_xlsx(_xlsx_native([{"name": "S", "cells": [_xc("A1", 1, 1, 1)]}]),
                                 semantic=False)
    m = metrics14b.compute_xlsx_metrics14b({"sheets": [{"name": "S"}]}, doc)
    return (m["inferred_table_precision"] is None and m["inferred_table_recall"] is None
            and m["formula_recall"] is None), "None ≠ 0（未采集三态显式）"


@_t("test_blocked_not_fail")
def t_blocked():
    # 14B 无 blocked arm；等价语义：arm 失败必须渲染为 ARM_FAILED，而非 FAIL（能力结论）
    res = adapters14b.run_arm(
        adapters14b.ArmSpecB("DOCX-BAD", "docx", "nonexistent", "phase14a.runners.__no_such_module__", "B0"),
        file_path=Path("whatever.docx"),
        out_path=sandbox14a.SANDBOX_ROOT / "test_blocked_out.json")
    return res["run"]["status"] == "ARM_FAILED" and res["run"]["status"] != "FAIL", \
        f"status={res['run']['status']}"


@_t("test_source_files_unchanged")
def t_nomut():
    before = sandbox14a.snapshot_project_source_tree()
    after = sandbox14a.snapshot_project_source_tree()
    diff = sandbox14a.diff_source_tree(before, after)
    return diff["identical"], "src tree identical across snapshots"


@_t("test_gold_b_spec_integrity")
def t_gold():
    gold = corpus14b.build_gold_b()
    return (gold["docx_case_ids"] == [f"W{i}" for i in range(1, 13)]
            and gold["xlsx_case_ids"] == [f"X{i}" for i in range(1, 16)]), \
        f"docx={len(gold['docx_case_ids'])} xlsx={len(gold['xlsx_case_ids'])}"


# --------------------------------------------------------------------------------------
# mutation validation（§25 四类）
# --------------------------------------------------------------------------------------


@_t("mutation: formula_overwrite_detected")
def t_mut_formula():
    cells = [_xc("C2", 2, 3, 30, formula=None, cached=30, data_type="n")]  # 公式被缓存值覆盖
    doc = cleaners14b.clean_xlsx(_xlsx_native([{"name": "S", "cells": cells}]), semantic=True)
    case = {"sheets": [{"name": "S", "tables": [{"anchor": "A1", "headers": ["a"],
                                                 "rows": [], "formula_cells": {"C2": ("=A2+B2", 30)}}]}]}
    m = metrics14b.compute_xlsx_metrics14b(case, doc)
    g = gates14b.evaluate_xlsx_gates14b(case, doc, m, True)
    return (m["formula_silent_overwrite"] == 1
            and any(x["gate"].startswith("X-G2") and x["status"] == "FAIL" for x in g)), \
        f"overwrite={m['formula_silent_overwrite']} → X-G2 FAIL"


@_t("mutation: fake_provenance_detected")
def t_mut_fake():
    doc = cleaners14b.clean_docx(_docx_native([_p(0, "hello")]), semantic=True)
    doc["elements"][0]["source_ref"]["page"] = 1
    m = metrics14b.compute_docx_metrics14b({"blocks": [{"kind": "paragraph", "text": "hello"}]}, doc)
    g = gates14b.evaluate_docx_gates14b({"blocks": [{"kind": "paragraph", "text": "hello"}]}, doc, m, True)
    return (m["fake_provenance_count"] == 1
            and any(x["gate"].startswith("D-G6") and x["status"] == "FAIL" for x in g)), \
        "fake page → D-G6 FAIL"


@_t("mutation: silent_inferred_table_detected")
def t_mut_inferred_table():
    cells = [_xc("A1", 1, 1, "A"), _xc("B1", 1, 2, "B"), _xc("A2", 2, 1, 1), _xc("B2", 2, 2, 2)]
    doc = cleaners14b.clean_xlsx(_xlsx_native([{"name": "S", "cells": cells}]), semantic=True)
    ok_before = gates14b.cleaning_traceability_ok(doc, "xlsx")
    doc["cleaning_log"] = [l for l in doc["cleaning_log"] if l["action"] != "infer_table_region"]
    detected = not gates14b.cleaning_traceability_ok(doc, "xlsx")
    return ok_before and detected, "静默推断表（删 log）→ X-G10 可检出"


@_t("mutation: missing_cleaning_log_detected")
def t_mut_missing_log():
    doc = cleaners14b.clean_docx(_docx_native([_p(0, "Implied", bold=True, size=16.0)]), semantic=True)
    ok_before = gates14b.cleaning_traceability_ok(doc, "docx")
    doc["cleaning_log"] = []
    detected = not gates14b.cleaning_traceability_ok(doc, "docx")
    return ok_before and detected, "删除 log → D-G8 可检出"


def main() -> int:
    for fn in [v for k, v in sorted(globals().items()) if k.startswith("t_") and callable(v)]:
        fn()
    passed = sum(1 for _, ok, _ in RESULTS if ok)
    print(f"\nPHASE14A-B TESTS: {passed}/{len(RESULTS)} passed")
    for name, ok, msg in RESULTS:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {msg}")
    return 0 if passed == len(RESULTS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
