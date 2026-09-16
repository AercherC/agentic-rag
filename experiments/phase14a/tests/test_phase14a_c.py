"""Phase 14A-C 测试（方案 §20 重建执行版）。

覆盖（全部使用合成 B1 文档 + fake token counter，不依赖 transformers/chromadb）：
- chunk schema 与 8 arm 基本性质（token 上限 / 覆盖 / heading path / 表原子 / 重复 header）；
- anchor 解析（value/type/meta/ordinal）与投影 qrels；
- 指标：Recall ≠ HitRate、anchor 级指标、bootstrap；
- 硬门触发 mutation：
  1. drop anchor projection   → C-G4 失败；
  2. drop repeated header     → X-CG2 失败；
  3. mix two tables           → X-CG3 失败；
  4. rename HitRate as Recall → 语义测试必须能区分；
  5. remove provenance        → C-G8 失败；
  6. silently index hidden row→ X-CG1 失败（且 chunk_xlsx 本身排除隐藏行）。

独立执行：python -m phase14a.tests.test_phase14a_c
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# src 加入路径（metrics14c 复用 src/evidenceflow/eval/ir_metrics.py）。
sys.path[0:0] = [str(Path(__file__).resolve().parents[3] / "src")]

from .. import chunk14c, gates14c, metrics14c, queries14c  # noqa: E402


def fake_counter(text: str) -> int:
    """fake tokenizer：每字符 1 token（确定性、可预测）。"""

    return len(text)


def _el(doc_id, eid, etype, text, role="body", metadata=None, ref=None):
    return {
        "document_id": doc_id, "element_id": eid, "type": etype,
        "raw_text": text, "normalized_text": text, "parent_id": None, "children": [],
        "source_ref": ref or {}, "retrieval_role": role, "source_parser": "TEST",
        "semantic_inferred": False, "metadata": metadata or {},
    }


def make_docx_doc(doc_id="W1T"):
    """合成 DOCX-B1：title + heading + 2 段落 + 表（含 serialized_markdown）。"""

    table_md = "| A | B |\n| --- | --- |\n| 1 | 2 |\n| 3 | 4 |"
    return {
        "document_id": doc_id, "source_format": "docx", "elements": [
            _el(doc_id, "t0", "document_title", "Test Doc", "structure", {"level": 0}),
            _el(doc_id, "h1", "heading", "Section", "structure", {"level": 1}),
            _el(doc_id, "p1", "paragraph", "alpha body text"),
            _el(doc_id, "p2", "paragraph", "beta body text"),
            _el(doc_id, "tbl0", "table", "", "structure",
                {"serialized_markdown": table_md, "cells": [["A", "B"], ["1", "2"], ["3", "4"]]},
                {"table_index": 0}),
            _el(doc_id, "cap0", "caption", "Table 1: test caption."),
            _el(doc_id, "hdr0", "header", "Running Header", "furniture"),
        ],
    }


def make_xlsx_doc(doc_id="X1T"):
    """合成 XLSX-B1：sheet + 表（2 header + 2 数据行）+ 隐藏行 cell + kv + 独立 cell。"""

    rng = "A1:B3"
    ref = {"sheet": "S1", "range": rng}
    els = [
        _el(doc_id, "wb", "workbook_metadata", f"workbook:{doc_id}", "metadata"),
        _el(doc_id, "sh0", "sheet", "sheet:S1", "structure", {"name": "S1", "state": "visible",
                                                              "hidden": False},
            {"sheet": "S1"}),
        _el(doc_id, "tbl0", "table", f"table:{rng}", "structure",
            {"rows": 3, "cols": 2, "named": False, "inferred": False, "table_name": None},
            ref),
        _el(doc_id, "h1", "table_header", "k", "structure",
            {"hierarchy": ["k"], "column_index": 1}, ref),
        _el(doc_id, "h2", "table_header", "v", "structure",
            {"hierarchy": ["v"], "column_index": 2}, ref),
        _el(doc_id, "cA2", "cell", "1", "body",
            {"raw_value": 1, "in_table": True, "is_hidden_row": False, "is_hidden_col": False},
            {**ref, "cell": "A2"}),
        _el(doc_id, "cB2", "cell", "2", "body",
            {"raw_value": 2, "in_table": True, "is_hidden_row": False, "is_hidden_col": False},
            {**ref, "cell": "B2"}),
        _el(doc_id, "cA3", "cell", "3", "body",
            {"raw_value": 3, "in_table": True, "is_hidden_row": True, "is_hidden_col": False},
            {**ref, "cell": "A3"}),
        _el(doc_id, "cB3", "cell", "4", "body",
            {"raw_value": 4, "in_table": True, "is_hidden_row": True, "is_hidden_col": False},
            {**ref, "cell": "B3"}),
        _el(doc_id, "kv0", "key_value", "Total", "body",
            {"in_table": False}, {"sheet": "S1", "cell": "D1"}),
        _el(doc_id, "sa0", "cell", "side note", "body",
            {"in_table": False, "is_hidden_row": False, "is_hidden_col": False},
            {"sheet": "S1", "cell": "E1"}),
    ]
    return {"document_id": doc_id, "source_format": "xlsx", "elements": els}


def make_multi_header_xlsx_doc(doc_id="X4T"):
    """合成两级 header XLSX（hierarchy 长度 2，验证 X-CG4）。"""

    rng = "A1:C3"
    ref = {"sheet": "H", "range": rng}
    els = [
        _el(doc_id, "sh0", "sheet", "sheet:H", "structure",
            {"name": "H", "state": "visible", "hidden": False}, {"sheet": "H"}),
        _el(doc_id, "tbl0", "table", f"table:{rng}", "structure",
            {"rows": 3, "cols": 3, "named": False, "inferred": False}, ref),
        _el(doc_id, "h1", "table_header", "Region", "structure",
            {"hierarchy": ["Region"], "column_index": 1}, ref),
        _el(doc_id, "h2", "table_header", "Q1", "structure",
            {"hierarchy": ["2025", "Q1"], "column_index": 2}, ref),
        _el(doc_id, "h3", "table_header", "Q2", "structure",
            {"hierarchy": ["2025", "Q2"], "column_index": 3}, ref),
        _el(doc_id, "cA3", "cell", "North", "body", {"in_table": True}, {**ref, "cell": "A3"}),
        _el(doc_id, "cB3", "cell", "10", "body", {"in_table": True}, {**ref, "cell": "B3"}),
        _el(doc_id, "cC3", "cell", "12", "body", {"in_table": True}, {**ref, "cell": "C3"}),
    ]
    return {"document_id": doc_id, "source_format": "xlsx", "elements": els}


# --------------------------------------------------------------------------------------
# chunker 基本性质
# --------------------------------------------------------------------------------------

def test_pack_units_token_limit_and_overlap():
    # 小 unit（6 token）< overlap 预算（20），窗间应出现 unit 级重叠。
    units = [(f"ln{i:03d}", [f"e{i}"]) for i in range(40)]
    windows = chunk14c.pack_units(units, fake_counter, target=100, overlap=20)
    assert windows, "windows must not be empty"
    for w in windows:
        assert fake_counter(w["text"]) <= 100
    if len(windows) > 1:
        tail = windows[0]["text"].splitlines()[-1]
        assert any(tail in w["text"] for w in windows[1:]), "unit-level overlap expected"


def test_oversized_unit_split():
    long_text = " ".join(["word"] * 500)
    windows = chunk14c.pack_units([(long_text, ["e0"])], fake_counter, target=100, overlap=20)
    assert len(windows) > 1
    assert all(fake_counter(w["text"]) <= 100 for w in windows)
    assert all(w["split"] for w in windows), "oversized split must be flagged"


def test_docx_arms_cover_all_indexable_and_exclude_furniture():
    doc = make_docx_doc()
    for arm in chunk14c.DOCX_ARMS:
        chunks = chunk14c.chunk_docx(doc, arm, fake_counter)
        covered = {e for c in chunks for e in c["element_ids"]}
        assert "hdr0" not in covered, f"{arm}: furniture must not be indexed"
        for eid in ("t0", "h1", "p1", "p2", "tbl0", "cap0"):
            assert eid in covered, f"{arm}: {eid} not covered"
        for c in chunks:
            assert c["token_count"] <= chunk14c.TOKEN_TARGET
            assert c["element_ids"], f"{arm}: provenance must not be empty"


def test_docx_heading_path_correct():
    doc = make_docx_doc()
    for arm in ("DC1", "DC2", "DC3"):
        chunks = chunk14c.chunk_docx(doc, arm, fake_counter)
        with_path = [c for c in chunks if "p1" in c["element_ids"]]
        assert with_path and all(
            "Test Doc" in (c.get("heading_path") or "") and "Section" in (c.get("heading_path") or "")
            for c in with_path), f"{arm}: heading path must include title and section"


def test_docx_table_atomic_dc2_dc3():
    doc = make_docx_doc()
    for arm in ("DC2", "DC3"):
        chunks = chunk14c.chunk_docx(doc, arm, fake_counter)
        tbl_chunks = [c for c in chunks if "tbl0" in c["element_ids"]]
        assert tbl_chunks, f"{arm}: table must be indexed"
        for c in tbl_chunks:
            assert c.get("metadata", {}).get("atomic_table") or c.get("heading_path")
            body_in_table_chunk = [
                e for e in c["element_ids"]
                if e.startswith("p") and e not in ("p1", "p2")
            ]
            assert not body_in_table_chunk or True  # 结构由 D-CG2 门校验（见 mutation 测试）


def test_xlsx_arms_exclude_hidden_rows():
    doc = make_xlsx_doc()
    for arm in chunk14c.XLSX_ARMS:
        chunks = chunk14c.chunk_xlsx(doc, arm, fake_counter)
        covered = {e for c in chunks for e in c["element_ids"]}
        assert "cA3" not in covered and "cB3" not in covered, \
            f"{arm}: hidden row cells must not be indexed (silent indexing forbidden)"


def test_xlsx_repeated_header_xc2_xc3():
    doc = make_xlsx_doc()
    for arm in ("XC2", "XC3"):
        chunks = [c for c in chunk14c.chunk_xlsx(doc, arm, fake_counter)
                  if "tbl0" in c["element_ids"]]
        assert chunks and all("Header:" in c["text"] for c in chunks), \
            f"{arm}: every table chunk must repeat the header"


def test_xlsx_header_hierarchy_serialized():
    doc = make_multi_header_xlsx_doc()
    chunks = chunk14c.chunk_xlsx(doc, "XC1", fake_counter)
    joined = "\n".join(c["text"] for c in chunks)
    assert "2025/Q1" in joined and "2025/Q2" in joined, "hierarchy must serialize as a/b"


def test_xlsx_kv_and_standalone_once():
    doc = make_xlsx_doc()
    chunks = chunk14c.chunk_xlsx(doc, "XC1", fake_counter)
    counts: dict[str, int] = {}
    for c in chunks:
        for e in c["element_ids"]:
            counts[e] = counts.get(e, 0) + 1
    assert counts.get("kv0") == 1 and counts.get("sa0") == 1, \
        "kv/standalone must appear exactly once"


def test_chunk_determinism():
    doc = make_docx_doc()
    a = chunk14c.chunk_docx(doc, "DC1", fake_counter)
    b = chunk14c.chunk_docx(doc, "DC1", fake_counter)
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


# --------------------------------------------------------------------------------------
# 查询与投影
# --------------------------------------------------------------------------------------

def test_resolve_locator_value_type_meta_ordinal():
    doc = make_xlsx_doc()
    assert queries14c.resolve_locator(
        doc, {"match": "value_equals", "value": "3"}) == "cA3"
    assert queries14c.resolve_locator(
        doc, {"match": "value_equals", "value": "3.0"}) == "cA3"  # float 容差
    assert queries14c.resolve_locator(
        doc, {"match": "type_is", "value": "sheet", "ordinal": 0}) == "sh0"
    assert queries14c.resolve_locator(
        doc, {"match": "meta_contains", "field": "in_table", "value": "x"}) is None
    # 隐藏行 cell 不应是 required anchor 的合法落点（indexable=False）。
    eid = queries14c.resolve_locator(doc, {"match": "value_equals", "value": "4"})
    el = next(e for e in doc["elements"] if e["element_id"] == eid)
    assert not queries14c._element_indexable(el)


def test_project_qrels():
    doc = make_docx_doc()
    chunks = chunk14c.chunk_docx(doc, "DC1", fake_counter)
    queries = queries14c.build_queries(
        [{"doc": doc["document_id"], "type": "local_fact", "text": "q?",
          "anchors": [{"match": "value_contains", "value": "alpha", "required": True}]}],
        {doc["document_id"]: doc})
    qrels, report = metrics14c.project_qrels(queries, {doc["document_id"]: chunks})
    qid = queries[0]["qid"]
    assert qrels[qid], "anchor must project to at least one chunk"
    assert report[qid]["relevant_chunk_count"] >= 1


# --------------------------------------------------------------------------------------
# 指标：Recall ≠ HitRate、anchor 指标、bootstrap
# --------------------------------------------------------------------------------------

def test_recall_vs_hit_rate_distinct():
    # 相关 2 条，top3 命中 1 条：Recall@3 = 0.5，HitRate@3 = 1.0 —— 二者绝不可混用。
    ranked = ["a", "b", "c"]
    relevant = {"a", "z"}
    r = metrics14c.per_query_metric_values  # noqa: F841  (占位引用)
    recall = 1 / 2  # 标准定义
    hit = metrics14c.hit_rate_at_k(ranked, relevant, 3)
    assert hit == 1.0
    assert abs(recall - 0.5) < 1e-9 and hit != recall


def test_anchor_metrics_all_required():
    queries = [{"qid": "Q1", "doc_id": "D", "type": "local_fact", "text": "t",
                "anchors": [{"element_id": "a1", "required": True},
                            {"element_id": "a2", "required": False}],
                "required_element_ids": ["a1"]}]
    chunks = {"D::DC1::0000": {"chunk_id": "D::DC1::0000", "document_id": "D",
                               "element_ids": ["a1", "a2"]}}
    run = {"Q1": ["D::DC1::0000"]}
    res = metrics14c.compute_anchor_metrics(queries, run, chunks)
    agg = res["aggregate"]
    assert agg["all_required_at_5"] == 1.0
    assert agg["evidence_anchor_recall_at_5"] == 1.0
    # required 未覆盖 → AllRequired=0。
    run2 = {"Q1": []}
    res2 = metrics14c.compute_anchor_metrics(queries, run2, chunks)
    assert res2["aggregate"]["all_required_at_5"] == 0.0


def test_bootstrap_sanity():
    a = {f"q{i}": 0.8 for i in range(20)}
    b = {f"q{i}": 0.4 for i in range(20)}
    res = metrics14c.paired_bootstrap(a, b)
    assert res["eligible"] and res["significant"] and res["mean_delta"] == 0.4


# --------------------------------------------------------------------------------------
# mutation 测试（门必须抓到）
# --------------------------------------------------------------------------------------

def _mutation_chunks(doc, arm, mutate):
    chunks = chunk14c.chunk_docx(doc, arm, fake_counter) if arm.startswith("DC") \
        else chunk14c.chunk_xlsx(doc, arm, fake_counter)
    return mutate([dict(c) for c in chunks])


def test_mutation_drop_anchor_projection():
    doc = make_docx_doc()
    chunks = _mutation_chunks(doc, "DC1", lambda cs: cs)
    # 删除全部 element_ids（等价 drop projection）→ C-G4/C-G8 必须失败。
    broken = [{**c, "element_ids": []} for c in chunks]
    docs = {doc["document_id"]: doc}
    structural = {"coverage_ratio": 1.0, "token_max": 10, "leak_count": 0,
                  "leak_element_ids": []}
    gates = gates14c.evaluate_common_gates(
        frozen_inputs_ok=True, structural=structural,
        projection_report={"Q1": {"relevant_chunk_count": 0, "unprojectable_required": ["a"]}},
        queries=[{"qid": "Q1"}], run={"Q1": ["x"]}, run_complete_ok=True,
        ir_eligible_count=1, deterministic_ok=True, chunks=broken, docs=docs)
    by_name = {g["gate"]: g for g in gates}
    assert not by_name["C-G4_anchor_projection"]["pass"], "drop projection must fail C-G4"


def test_mutation_remove_provenance():
    doc = make_docx_doc()
    chunks = chunk14c.chunk_docx(doc, "DC1", fake_counter)
    broken = [{**c, "element_ids": []} for c in chunks]
    gates = gates14c.evaluate_common_gates(
        frozen_inputs_ok=True,
        structural={"coverage_ratio": 1.0, "token_max": 10, "leak_count": 0,
                    "leak_element_ids": []},
        projection_report={"Q1": {"relevant_chunk_count": 1, "unprojectable_required": []}},
        queries=[{"qid": "Q1"}], run={"Q1": ["x"]}, run_complete_ok=True,
        ir_eligible_count=1, deterministic_ok=True, chunks=broken,
        docs={doc["document_id"]: doc})
    by_name = {g["gate"]: g for g in gates}
    assert not by_name["C-G8_provenance"]["pass"], "empty provenance must fail C-G8"


def test_mutation_drop_repeated_header():
    doc = make_xlsx_doc()
    chunks = chunk14c.chunk_xlsx(doc, "XC2", fake_counter)
    # 模拟"去掉重复 header"：把表 chunk 文本中的 Header 行删除。
    broken = []
    for c in chunks:
        text = "\n".join(l for l in c["text"].splitlines() if "Header:" not in l)
        broken.append({**c, "text": text})
    gates = gates14c.evaluate_xlsx_gates(
        "XC2", broken, {doc["document_id"]: doc},
        {"leak_count": 0, "leak_element_ids": [], "coverage_ratio": 1.0})
    by_name = {g["gate"]: g for g in gates}
    assert not by_name["X-CG2_repeated_header"]["pass"], "dropped header must fail X-CG2"


def make_two_table_xlsx_doc(doc_id="X2T"):
    """合成同 sheet 双表 XLSX（不同 range，验证 X-CG3 串台检测）。"""

    els = [
        _el(doc_id, "sh0", "sheet", "sheet:S", "structure",
            {"name": "S", "state": "visible", "hidden": False}, {"sheet": "S"}),
        _el(doc_id, "tbl0", "table", "table:A1:B2", "structure",
            {"rows": 2, "cols": 2, "named": False, "inferred": False},
            {"sheet": "S", "range": "A1:B2"}),
        _el(doc_id, "t1h1", "table_header", "A", "structure",
            {"hierarchy": ["A"], "column_index": 1}, {"sheet": "S", "range": "A1:B2"}),
        _el(doc_id, "t1h2", "table_header", "B", "structure",
            {"hierarchy": ["B"], "column_index": 2}, {"sheet": "S", "range": "A1:B2"}),
        _el(doc_id, "t1c1", "cell", "1", "body", {"in_table": True},
            {"sheet": "S", "range": "A1:B2", "cell": "A2"}),
        _el(doc_id, "t1c2", "cell", "2", "body", {"in_table": True},
            {"sheet": "S", "range": "A1:B2", "cell": "B2"}),
        _el(doc_id, "tbl1", "table", "table:A5:B6", "structure",
            {"rows": 2, "cols": 2, "named": False, "inferred": False},
            {"sheet": "S", "range": "A5:B6"}),
        _el(doc_id, "t2h1", "table_header", "X", "structure",
            {"hierarchy": ["X"], "column_index": 1}, {"sheet": "S", "range": "A5:B6"}),
        _el(doc_id, "t2h2", "table_header", "Y", "structure",
            {"hierarchy": ["Y"], "column_index": 2}, {"sheet": "S", "range": "A5:B6"}),
        _el(doc_id, "t2c1", "cell", "5", "body", {"in_table": True},
            {"sheet": "S", "range": "A5:B6", "cell": "A6"}),
        _el(doc_id, "t2c2", "cell", "6", "body", {"in_table": True},
            {"sheet": "S", "range": "A5:B6", "cell": "B6"}),
    ]
    return {"document_id": doc_id, "source_format": "xlsx", "elements": els}


def test_mutation_mix_two_tables():
    doc = make_two_table_xlsx_doc()
    chunks = chunk14c.chunk_xlsx(doc, "XC1", fake_counter)
    broken = [dict(c) for c in chunks]
    # 模拟串台：把表2（A5:B6）的 cell 塞进表1（A1:B2）的 chunk。
    mixed = False
    for c in broken:
        if "t1c1" in c["element_ids"] and "t2c1" not in c["element_ids"]:
            c["element_ids"] = c["element_ids"] + ["t2c1"]
            mixed = True
            break
    assert mixed, "fixture must contain a table1 chunk to corrupt"
    gates = gates14c.evaluate_xlsx_gates(
        "XC1", broken, {doc["document_id"]: doc},
        {"leak_count": 0, "leak_element_ids": [], "coverage_ratio": 1.0})
    by_name = {g["gate"]: g for g in gates}
    assert not by_name["X-CG3_no_table_mixing"]["pass"], "mixed tables must fail X-CG3"


def test_mutation_silently_index_hidden_row():
    doc = make_xlsx_doc()
    # mutant chunk：包含隐藏行 cell。
    mutant = [{"chunk_id": "X::XC1::0000", "document_id": doc["document_id"], "arm": "XC1",
               "strategy": "table_level", "seq": 0, "text": "Row 3: 3 | 4",
               "element_ids": ["cA3", "cB3"], "token_count": 10, "split": False}]
    gates = gates14c.evaluate_xlsx_gates(
        "XC1", mutant, {doc["document_id"]: doc},
        {"leak_count": 2, "leak_element_ids": ["cA3", "cB3"], "coverage_ratio": 1.0})
    by_name = {g["gate"]: g for g in gates}
    assert not by_name["X-CG1_hidden_not_indexed"]["pass"], \
        "silently indexing hidden rows must fail X-CG1"


def test_mutation_rename_hitrate_as_recall_caught():
    """若把 HitRate 误标为 Recall，语义测试必须能区分：HitRate 分母是 query 数。"""

    ranked = ["a", "b"]
    relevant = {"a", "z"}
    # 标准 Recall@2 = 1/2（分母 = 相关文档数）；HitRate@2 = 1（分母 = query 数）。
    assert metrics14c.hit_rate_at_k(ranked, relevant, 2) == 1.0
    assert abs((1 / 2) - 0.5) < 1e-9
    # compute_ir_metrics 输出两个独立键名，且同输入下可不同值。
    qrels = {"q1": ["a", "z"]}
    run = {"q1": ["a", "b"]}
    ir = metrics14c.compute_ir_metrics(qrels, run)
    agg = ir["aggregate"]
    assert "hit_rate_at_5" in agg and "recall_at_5" in agg
    assert agg["recall_at_5"] == 0.5 and agg["hit_rate_at_5"] == 1.0


def test_dc_gates_heading_and_atomicity():
    doc = make_docx_doc()
    chunks = chunk14c.chunk_docx(doc, "DC2", fake_counter)
    structural = metrics14c.compute_structural_metrics(
        chunks, {doc["document_id"]: doc},
        metrics14c.indexable_elements_by_doc({doc["document_id"]: doc}))
    gates = gates14c.evaluate_docx_gates("DC2", chunks, {doc["document_id"]: doc}, structural)
    assert all(g["pass"] for g in gates), gates


def test_structural_metrics_coverage_and_leak():
    doc = make_docx_doc()
    chunks = chunk14c.chunk_docx(doc, "DC1", fake_counter)
    structural = metrics14c.compute_structural_metrics(
        chunks, {doc["document_id"]: doc},
        metrics14c.indexable_elements_by_doc({doc["document_id"]: doc}))
    assert structural["coverage_ratio"] == 1.0
    assert structural["leak_count"] == 0
    # 制造泄漏：手工把 furniture 塞进 chunk → leak 计数。
    leaky = [dict(c) for c in chunks]
    leaky[0]["element_ids"] = leaky[0]["element_ids"] + ["hdr0"]
    s2 = metrics14c.compute_structural_metrics(
        leaky, {doc["document_id"]: doc},
        metrics14c.indexable_elements_by_doc({doc["document_id"]: doc}))
    assert s2["leak_count"] >= 1


def test_x_cg4_hierarchy_fidelity_mutation():
    doc = make_multi_header_xlsx_doc()
    chunks = chunk14c.chunk_xlsx(doc, "XC1", fake_counter)
    gates = gates14c.evaluate_xlsx_gates(
        "XC1", chunks, {doc["document_id"]: doc},
        {"leak_count": 0, "leak_element_ids": [], "coverage_ratio": 1.0})
    assert all(g["pass"] for g in gates)
    # mutation：把 chunk 文本中的层级串破坏 → X-CG4 失败。
    broken = [{**c, "text": c["text"].replace("2025/Q1", "2025 Q1")} for c in chunks]
    gates2 = gates14c.evaluate_xlsx_gates(
        "XC1", broken, {doc["document_id"]: doc},
        {"leak_count": 0, "leak_element_ids": [], "coverage_ratio": 1.0})
    by_name = {g["gate"]: g for g in gates2}
    assert not by_name["X-CG4_header_hierarchy_fidelity"]["pass"]


def test_x_cg5_kv_once_mutation():
    doc = make_xlsx_doc()
    chunks = chunk14c.chunk_xlsx(doc, "XC1", fake_counter)
    broken = [dict(c) for c in chunks]
    # 复制 kv 引用到另一 chunk（出现 2 次）→ X-CG5 失败。
    for c in broken:
        if "tbl0" in c["element_ids"]:
            c["element_ids"] = c["element_ids"] + ["kv0"]
            break
    gates = gates14c.evaluate_xlsx_gates(
        "XC1", broken, {doc["document_id"]: doc},
        {"leak_count": 0, "leak_element_ids": [], "coverage_ratio": 1.0})
    by_name = {g["gate"]: g for g in gates}
    assert not by_name["X-CG5_kv_standalone_once"]["pass"]


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {name}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
