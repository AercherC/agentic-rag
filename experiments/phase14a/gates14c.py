"""Phase 14A-C 硬门（方案 §15-§17 重建执行版）。

门定义重建原则：与 14A-B 一致，门只做**测量完整性/结构保真**判定，不设主观质量阈值；
测量语义修订必须升 gate_definitions 版本。凡因方案压缩而重建的门在
``GATE_DEFINITIONS_VERSION`` 与报告中标注 reconstructed。

公共门 C-G1..8（每 arm 全部通过才算过门）：
- C-G1  冻结输入一致     ：输入 B1 文档 sha256 与 14A-B 清洗产物一致（runner 提供）；
- C-G2  序列化覆盖完整   ：indexable 元素 100% 出现在 ≥1 chunk（coverage_ratio == 1.0）；
- C-G3  token 上限       ：所有 chunk ≤ 512 token（bge-m3 tokenizer 口径）；
- C-G4  anchor 投影有效  ：每 query ≥1 相关 chunk，且无 unprojectable required anchor；
- C-G5  检索完整         ：每 query 的 run 恰好返回 min(20, 语料量) 条且无缺失 qid；
- C-G6  qrels 非空       ：IR 聚合 eligible_count == query 总数（无空 qrels 稀释）；
- C-G7  确定性           ：同输入两次切分 chunk 集合 digest 一致（runner 提供）；
- C-G8  provenance 完整  ：每 chunk element_ids 非空且均存在于对应 B1 文档。

DOCX 门 D-CG1..4：
- D-CG1 heading path 正确（DC1/DC2/DC3）：含 heading 元素的 chunk，heading_path 必须
  含该 heading 文本且以 document_title 开头；DC0 = N/A（无结构感知设计）；
- D-CG2 表原子性（DC2/DC3）：含 table 元素的 chunk 不得混入其他 body 元素；
  DC0/DC1 = N/A（表格内联设计）；
- D-CG3 furniture/hidden 不入索引：leak_count == 0（header/footer/隐藏元素）；
- D-CG4 基线公平性：各 arm 的入索引元素集合与 token 上限一致（覆盖集合相同）。

XLSX 门 X-CG1..5：
- X-CG1 hidden 不入索引：leak_count == 0，且隐藏行/列/隐藏 sheet 元素确不在任何 chunk；
- X-CG2 repeated header（XC2/XC3）：每个表 chunk 均含 "Header:" 行；XC0/XC1 = N/A；
- X-CG3 表不串台：每 chunk 至多 1 个 table 元素，cell 元素同源（同一 sheet+range）；
- X-CG4 header 层级保真：多级 header（hierarchy 长度 >1）的层级串 "a/b" 必须出现在 chunk 文本中；
- X-CG5 kv/standalone 归属：key_value 与独立 cell 元素在所有 chunk 中恰好出现 1 次。
"""

from __future__ import annotations

from typing import Any

from .chunk14c import TOKEN_TARGET

GATE_DEFINITIONS_VERSION = "phase14a-c-gates-v1.0-reconstructed"


def _check(name: str, ok: bool, detail: dict[str, Any] | None = None) -> dict[str, Any]:
    """单门判定结果（三态语义：ok bool 明确，不引入 None）。"""

    entry: dict[str, Any] = {"gate": name, "pass": bool(ok)}
    if detail:
        entry["detail"] = detail
    return entry


def evaluate_common_gates(
    *,
    frozen_inputs_ok: bool,
    structural: dict[str, Any],
    projection_report: dict[str, dict[str, Any]],
    queries: list[dict[str, Any]],
    run: dict[str, list[str]],
    run_complete_ok: bool,
    ir_eligible_count: int,
    deterministic_ok: bool,
    chunks: list[dict[str, Any]],
    docs: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """C-G1..8（公共门）。"""

    results: list[dict[str, Any]] = []
    results.append(_check("C-G1_frozen_inputs", frozen_inputs_ok))

    cov = structural["coverage_ratio"]
    results.append(_check("C-G2_serialization_coverage", cov == 1.0, {"coverage_ratio": cov}))

    max_tok = structural["token_max"]
    results.append(_check("C-G3_token_limit", max_tok <= TOKEN_TARGET,
                          {"token_max": max_tok, "limit": TOKEN_TARGET}))

    bad = {qid: r for qid, r in projection_report.items()
           if r["relevant_chunk_count"] == 0 or r["unprojectable_required"]}
    results.append(_check("C-G4_anchor_projection", not bad,
                          {"failed_qids": sorted(bad)}))

    results.append(_check("C-G5_run_complete", run_complete_ok))

    expected = len(queries)
    results.append(_check("C-G6_qrels_nonempty", ir_eligible_count == expected,
                          {"eligible": ir_eligible_count, "expected": expected}))

    results.append(_check("C-G7_determinism", deterministic_ok))

    # C-G8 provenance：element_ids 非空且全部存在于**本**文档
    # （element_id 只在文档内唯一，禁止跨文档集合合并判定）。
    all_ids = {
        d: {el["element_id"] for el in doc["elements"]} for d, doc in docs.items()
    }
    bad_chunks = [
        c["chunk_id"] for c in chunks
        if not c["element_ids"] or not set(c["element_ids"]) <= all_ids.get(c["document_id"], set())
    ]
    results.append(_check("C-G8_provenance", not bad_chunks, {"bad_chunks": bad_chunks[:20]}))
    return results


def evaluate_docx_gates(
    arm: str,
    chunks: list[dict[str, Any]],
    docs: dict[str, dict[str, Any]],
    structural: dict[str, Any],
) -> list[dict[str, Any]]:
    """D-CG1..4（DOCX 门；N/A 视为 pass 并记录 scope）。"""

    results: list[dict[str, Any]] = []
    els_by_doc = {d: {el["element_id"]: el for el in doc["elements"]} for d, doc in docs.items()}

    # D-CG1 heading path 正确（DC1/DC2/DC3）。
    if arm == "DC0":
        results.append(_check("D-CG1_heading_path", True, {"scope": "N/A_fixed_token"}))
    else:
        bad: list[str] = []
        for c in chunks:
            for eid in c["element_ids"]:
                el = els_by_doc[c["document_id"]].get(eid)
                if el is None or el["type"] not in ("heading", "document_title"):
                    continue
                hp = c.get("heading_path") or ""
                if el.get("normalized_text") not in hp:
                    bad.append(c["chunk_id"])
                if el["type"] == "document_title" and not hp.startswith(el.get("normalized_text") or ""):
                    bad.append(c["chunk_id"])
        results.append(_check("D-CG1_heading_path", not bad, {"bad_chunks": sorted(set(bad))[:20]}))

    # D-CG2 表原子性（DC2/DC3）：table chunk 不得混入其他 body 元素。
    if arm in ("DC2", "DC3"):
        bad_t: list[str] = []
        for c in chunks:
            table_ids = [e for e in c["element_ids"]
                         if els_by_doc[c["document_id"]].get(e, {}).get("type") == "table"]
            if not table_ids:
                continue
            for eid in c["element_ids"]:
                el = els_by_doc[c["document_id"]].get(eid)
                if el is None:
                    continue
                if el["type"] not in ("table", "heading", "document_title", "caption") \
                        and el["retrieval_role"] == "body":
                    bad_t.append(c["chunk_id"])
        results.append(_check("D-CG2_table_atomicity", not bad_t, {"bad_chunks": sorted(set(bad_t))[:20]}))
    else:
        results.append(_check("D-CG2_table_atomicity", True, {"scope": "N/A_inline_table_design"}))

    # D-CG3 furniture/hidden 不入索引。
    results.append(_check("D-CG3_no_furniture_hidden", structural["leak_count"] == 0,
                          {"leak_element_ids": structural["leak_element_ids"][:20]}))

    # D-CG4 基线公平性：覆盖集合完整（与 C-G2 同口径，但强调 arm 间同元素集合）。
    cov = structural["coverage_ratio"]
    results.append(_check("D-CG4_baseline_fairness", cov == 1.0,
                          {"coverage_ratio": cov, "note": "all arms share same indexable element set"}))
    return results


def evaluate_xlsx_gates(
    arm: str,
    chunks: list[dict[str, Any]],
    docs: dict[str, dict[str, Any]],
    structural: dict[str, Any],
) -> list[dict[str, Any]]:
    """X-CG1..5（XLSX 门）。"""

    results: list[dict[str, Any]] = []
    els_by_doc = {d: {el["element_id"]: el for el in doc["elements"]} for d, doc in docs.items()}

    # X-CG1 hidden 不入索引：leak==0 且隐藏元素不在 covered 中（双重断言；
    # covered 用 (doc_id, element_id) 文档作用域，避免跨文档同名 id 污染）。
    covered: set[tuple[str, str]] = {
        (c["document_id"], eid) for c in chunks for eid in c["element_ids"]
    }
    hidden_indexed: list[str] = []
    for _d, doc in docs.items():
        for el in doc["elements"]:
            md = el.get("metadata") or {}
            if md.get("is_hidden_row") or md.get("is_hidden_col") or \
                    el.get("retrieval_role") == "hidden":
                if (_d, el["element_id"]) in covered:
                    hidden_indexed.append(f"{_d}:{el['element_id']}")
    ok = structural["leak_count"] == 0 and not hidden_indexed
    results.append(_check("X-CG1_hidden_not_indexed", ok,
                          {"leak_count": structural["leak_count"],
                           "hidden_indexed": hidden_indexed[:20]}))

    # X-CG2 repeated header（XC2/XC3）：只校验**表** chunk（standalone/kv chunk 无表头属正常）。
    if arm in ("XC2", "XC3"):
        bad_h = [
            c["chunk_id"] for c in chunks
            if "table" in (c.get("metadata") or {}) and "Header:" not in c["text"]
        ]
        results.append(_check("X-CG2_repeated_header", not bad_h, {"bad_chunks": bad_h[:20]}))
    else:
        results.append(_check("X-CG2_repeated_header", True, {"scope": "N/A"}))

    # X-CG3 表不串台：chunk 内 cell 同源（同一 sheet+range）且 table 元素 ≤1。
    # XC0 = whole-sheet 设计（单 sheet 一 chunk 天然含多表）→ N/A。
    if arm == "XC0":
        results.append(_check("X-CG3_no_table_mixing", True,
                              {"scope": "N/A_whole_sheet_design"}))
    else:
        bad_m: list[str] = []
        for c in chunks:
            sources: set[tuple] = set()
            table_count = 0
            for eid in c["element_ids"]:
                el = els_by_doc[c["document_id"]].get(eid)
                if el is None:
                    continue
                if el["type"] == "table":
                    table_count += 1
                ref = el.get("source_ref") or {}
                if el["type"] in ("cell", "key_value", "note") and ref.get("range"):
                    sources.add((ref.get("sheet"), ref.get("range")))
            if table_count > 1 or len(sources) > 1:
                bad_m.append(c["chunk_id"])
        results.append(_check("X-CG3_no_table_mixing", not bad_m, {"bad_chunks": bad_m[:20]}))

    # X-CG4 header 层级保真：hierarchy 长度 >1 的层级串必须出现。
    missing: list[str] = []
    for _d, doc in docs.items():
        doc_chunks = [c for c in chunks if c["document_id"] == _d]
        joined_text = "\n".join(c["text"] for c in doc_chunks)
        for el in doc["elements"]:
            if el["type"] != "table_header":
                continue
            hier = (el.get("metadata") or {}).get("hierarchy") or []
            if len(hier) > 1:
                want = "/".join(str(x) for x in hier)
                if want not in joined_text:
                    missing.append(f"{_d}:{el['element_id']}:{want}")
    results.append(_check("X-CG4_header_hierarchy_fidelity", not missing, {"missing": missing[:20]}))

    # X-CG5 kv/standalone 恰好出现 1 次（按文档分桶；隐藏 sheet 元素按设计不入索引，
    # 与 X-CG1 一致，豁免不计）。
    from .metrics14c import _hidden_sheets

    kv_ids: dict[str, set[str]] = {}
    for _d, doc in docs.items():
        hidden_sheets = _hidden_sheets(doc)
        ids = set()
        for el in doc["elements"]:
            md = el.get("metadata") or {}
            if el["type"] == "key_value" or (
                el["type"] == "cell" and not md.get("in_table")
                and el.get("retrieval_role") in ("body", "structure")
                and not md.get("is_hidden_row") and not md.get("is_hidden_col")
            ):
                if (el.get("source_ref") or {}).get("sheet") in hidden_sheets:
                    continue
                ids.add(el["element_id"])
        kv_ids[_d] = ids
    counts: dict[tuple[str, str], int] = {}
    for c in chunks:
        for eid in c["element_ids"]:
            if eid in kv_ids.get(c["document_id"], set()):
                counts[(c["document_id"], eid)] = counts.get((c["document_id"], eid), 0) + 1
    bad_kv = sorted(
        f"{d}:{i}" for d, ids in kv_ids.items() for i in ids
        if counts.get((d, i), 0) != 1
    )
    results.append(_check("X-CG5_kv_standalone_once", not bad_kv, {"bad_ids": bad_kv[:20]}))
    return results
