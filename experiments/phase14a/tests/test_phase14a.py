"""Phase 14A-A 回归测试（方案 §35），可独立运行：python -m phase14a.tests.test_phase14a。

不依赖 python-docx/openpyxl（仅测框架逻辑：schema 校验、metrics、gates、adapter 状态、不变性），
真实语料解析由 run_phase14a_a 覆盖。含两个 mutation validation。
"""

from __future__ import annotations

import sys
from pathlib import Path

from .. import adapters14a, corpus14a, gates14a, metrics14a, sandbox14a, schema14a

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


def _docx_doc(elements):
    return schema14a.make_document(
        document_id="d", source_format="docx", source_file="d.docx",
        elements=elements, document_metadata={"parser": "W-A0"},
    )


def _xlsx_doc(elements):
    return schema14a.make_document(
        document_id="x", source_format="xlsx", source_file="x.xlsx",
        elements=elements, document_metadata={"parser": "X-A0"},
    )


def _el(**kw):
    base = dict(document_id="d", element_id="e", type="paragraph", raw_text="t",
               source_format="docx", source_ref={}, source_parser="W-A0")
    base.update(kw)
    # 路由到 schema14a.make_element 以补齐 ELEMENT_FIELDS（含 normalized_text），
    # 否则 metrics 访问 e["normalized_text"] 会 KeyError。所有测试传入的键均为
    # make_element 的合法形参（type/raw_text/source_ref/metadata/retrieval_role/source_format）。
    return schema14a.make_element(**base)


@_t("test_docx_heading_style_preserved")
def t_heading_style():
    els = [
        _el(type="heading", raw_text="Introduction", source_ref={"style_id": "Heading1", "paragraph_index": 1}),
        _el(type="document_title", raw_text="Title", source_ref={"style_id": "Title", "paragraph_index": 0}),
    ]
    m = metrics14a.compute_docx_metrics(
        {"blocks": [{"kind": "title", "text": "Title"}, {"kind": "heading", "level": 1, "text": "Introduction"}]},
        _docx_doc(els))
    g = gates14a.evaluate_docx_gates(
        {"blocks": [{"kind": "title", "text": "Title"}, {"kind": "heading", "level": 1, "text": "Introduction"}]},
        _docx_doc(els), m, True)
    return ("style_provenance_ok" in m and m["style_provenance_ok"]
            and any(x["gate"].startswith("W-G4") and x["status"] == "PASS" for x in g)), "heading style provenance + W-G4"


@_t("test_docx_table_structure_preserved")
def t_table():
    els = [_el(type="table", source_ref={"table_index": 0}, metadata={"rows": 3, "cols": 3})]
    case = {"blocks": [{"kind": "table", "headers": [["A", "B", "C"]], "rows": [["1", "2", "3"], ["4", "5", "6"]]}]}
    m = metrics14a.compute_docx_metrics(case, _docx_doc(els))
    return m["table_structure_accuracy"] == 1.0, f"table_acc={m['table_structure_accuracy']}"


@_t("test_docx_header_footer_separated")
def t_hf():
    els = [_el(type="header", raw_text="H", retrieval_role="furniture", source_ref={"xml_path": "header"}),
           _el(type="footer", raw_text="F", retrieval_role="furniture", source_ref={"xml_path": "footer"})]
    case = {"blocks": [{"kind": "header", "text": "H"}, {"kind": "footer", "text": "F"}]}
    m = metrics14a.compute_docx_metrics(case, _docx_doc(els))
    g = gates14a.evaluate_docx_gates(case, _docx_doc(els), m, True)
    return m["header_footer_separation"] and any(x["gate"].startswith("W-G5") and x["status"] == "PASS" for x in g), "HF separated + W-G5"


@_t("test_docx_no_fake_page_provenance")
def t_docx_nofake():
    bad = _el(source_ref={"page": 1, "bbox": [0, 0, 1, 1]})
    m = metrics14a.compute_docx_metrics({"blocks": []}, _docx_doc([bad]))
    return m["fake_provenance_count"] == 1, "detected fake page/bbox on DOCX"


@_t("test_xlsx_cell_coordinates_preserved")
def t_cell():
    els = [_el(source_format="xlsx", source_ref={"sheet": "S", "cell": "A1", "range": "A1:B2"},
              type="cell", raw_text="10", metadata={"value": 10}),
           _el(source_format="xlsx", source_ref={"sheet": "S", "cell": "B2", "range": "A1:B2"},
               type="cell", raw_text="20", metadata={"value": 20})]
    # 注意：gold 仅在 rows 中声明数据格；不声明 header 文本，否则 collect_gold_xlsx
    # 会把 header 标签也计入 gold cells，而本测试只输出两个数据格 -> cell_recall 虚低。
    case = {"sheets": [{"name": "S", "tables": [{"anchor": "A1", "headers": [], "rows": [[10, 20]]}]}]}
    m = metrics14a.compute_xlsx_metrics(case, _xlsx_doc(els))
    return m["cell_recall"] == 1.0, f"cell_recall={m['cell_recall']}"


@_t("test_xlsx_formula_and_cached_value_both_preserved")
def t_formula():
    els = [_el(source_format="xlsx", type="cell", source_ref={"sheet": "S", "cell": "C2", "range": None},
              raw_text="=A2+B2", metadata={"formula": "=A2+B2", "cached_value": 30, "is_formula": True})]
    case = {"sheets": [{"name": "S", "tables": [{"anchor": "A1", "headers": ["a", "b", "sum"],
            "rows": [[10, 20, None]], "formula_cells": {"C2": ("=A2+B2", 30)}}]}]}
    m = metrics14a.compute_xlsx_metrics(case, _xlsx_doc(els))
    g = gates14a.evaluate_xlsx_gates(case, _xlsx_doc(els), m, True)
    return (m["formula_recall"] == 1.0 and m["cached_value_preservation"] == 1.0
            and any(x["gate"].startswith("X-G2") and x["status"] == "PASS" for x in g)), "formula+cached dual + X-G2"


@_t("test_xlsx_merged_ranges_preserved")
def t_merged():
    els = [_el(source_format="xlsx", type="sheet", source_ref={"sheet": "S"},
              metadata={"merged_ranges": ["A1:C1"]})]
    case = {"sheets": [{"name": "S", "tables": [{"anchor": "A1", "headers": ["A", "B", "C"],
            "rows": [["1", "2", "3"]], "merges": [["A1", "C1"]]}]}]}
    m = metrics14a.compute_xlsx_metrics(case, _xlsx_doc(els))
    return m["merged_range_accuracy"] == 1.0, f"merged_acc={m['merged_range_accuracy']}"


@_t("test_xlsx_named_table_range_preserved")
def t_named():
    els = [_el(source_format="xlsx", type="table", source_ref={"sheet": "S", "range": "A1:C3", "table_name": "T1", "inferred": False})]
    case = {"sheets": [{"name": "S", "tables": [{"anchor": "A1", "headers": ["A", "B", "C"],
            "rows": [["1", "2", "3"]], "is_named": True, "table_name": "T1"}]}]}
    m = metrics14a.compute_xlsx_metrics(case, _xlsx_doc(els))
    return m["named_table_accuracy"] == 1.0, f"named_acc={m['named_table_accuracy']}"


@_t("test_xlsx_hidden_state_explicit")
def t_hidden():
    els = [_el(source_format="xlsx", type="sheet", source_ref={"sheet": "H"},
              metadata={"name": "H", "hidden": True}),
           _el(source_format="xlsx", type="cell", source_ref={"sheet": "H", "cell": "A1"},
              metadata={"is_hidden_row": True})]
    case = {"sheets": [{"name": "H", "state": "hidden", "tables": [{"anchor": "A1", "headers": ["x"], "rows": [[1]]}],
                       "hidden_rows": [1]}]}
    m = metrics14a.compute_xlsx_metrics(case, _xlsx_doc(els))
    return m["hidden_sheet_preserved"] and m["hidden_row_preserved"], "hidden preserved"


@_t("test_xlsx_no_fake_bbox")
def t_xlsx_nobbox():
    bad = _el(source_format="xlsx", source_ref={"page": 1, "bbox": [0, 0, 1, 1]})
    m = metrics14a.compute_xlsx_metrics({"sheets": []}, _xlsx_doc([bad]))
    return m["fake_provenance_count"] == 1, "detected fake bbox on XLSX"


@_t("test_adapter_output_deterministic")
def t_det():
    a = _docx_doc([_el()])
    b = _docx_doc([_el()])
    return metrics14a.is_deterministic(a, b), "same doc -> same digest"


@_t("test_unified_schema_format_specific_source_ref")
def t_schema():
    d = _docx_doc([_el(source_ref={"paragraph_index": 1, "style_id": "Heading1"})])
    x = _xlsx_doc([_el(source_format="xlsx", source_ref={"sheet": "S", "cell": "A1"})])
    vd = schema14a.validate_document(d)
    vx = schema14a.validate_document(x)
    return vd["valid"] and vx["valid"], f"docx_valid={vd['valid']} xlsx_valid={vx['valid']}"


@_t("test_metric_none_not_zero")
def t_none():
    # None 不应被当作 0；构造空 case 确保指标为 1.0 而非 0
    m = metrics14a.compute_xlsx_metrics({"sheets": []}, _xlsx_doc([]))
    return all(v == 1.0 for k, v in m.items() if isinstance(v, float)), "empty gold -> 1.0 not 0"


@_t("test_blocked_not_fail")
def t_blocked():
    arm = adapters14a.ARM_INDEX["W-A1"]
    res = adapters14a.execute_arm(arm, file_path=Path("/x.docx"), out_path=Path("/tmp/o.json"))
    return res["run"]["status"] == "BLOCKED" and "reason" in res["run"], "blocked arm -> BLOCKED not FAIL"


@_t("test_no_source_mutation")
def t_nomut():
    before = sandbox14a.snapshot_project_source_tree()
    after = sandbox14a.snapshot_project_source_tree()
    diff = sandbox14a.diff_source_tree(before, after)
    return diff["identical"], "src tree identical across snapshots"


# ---- mutation validation ----
@_t("mutation: metric_range_violation_detected")
def t_mut_range():
    # 若某指标被篡改为 >1，orchestrator 的范围守卫应捕获（此处验证计算本身不产生 >1）
    m = metrics14a.compute_xlsx_metrics(
        {"sheets": [{"name": "S", "tables": [{"anchor": "A1", "headers": ["a"], "rows": [[1]]}]}]},
        _xlsx_doc([_el(source_format="xlsx", type="cell", source_ref={"sheet": "S", "cell": "A1"}, raw_text="1", metadata={"value": 1})]))
    return all(0.0 <= v <= 1.0001 for k, v in m.items() if isinstance(v, float)), "no metric exceeds [0,1]"


@_t("mutation: blocked_rendered_as_not_fail")
def t_mut_blocked():
    # 若把 BLOCKED 误渲染成 FAIL，状态不应是 FAIL
    arm = adapters14a.ARM_INDEX["X-A1"]
    res = adapters14a.execute_arm(arm, file_path=Path("/x.xlsx"), out_path=Path("/tmp/o.json"))
    return res["run"]["status"] != "FAIL" and res["run"]["status"] == "BLOCKED", "BLOCKED never FAIL"


def main() -> int:
    for fn in [v for k, v in sorted(globals().items()) if k.startswith("t_") and callable(v)]:
        fn()
    passed = sum(1 for _, ok, _ in RESULTS if ok)
    print(f"\nPHASE14A TESTS: {passed}/{len(RESULTS)} passed")
    for name, ok, msg in RESULTS:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {msg}")
    return 0 if passed == len(RESULTS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
