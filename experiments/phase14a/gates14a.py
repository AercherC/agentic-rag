"""Phase 14A-A 硬门（方案 §12 DOCX / §13 XLSX）。

每个门返回 {gate, status: PASS|FAIL, detail}。失败必须归因于具体指标，不笼统写 FAIL。
阈值固定（与 13C 同一纪律：阈值一个不动，改口径必须升版本）。
"""

from __future__ import annotations

from typing import Any

# 阈值（与 13C 一致，冻结）
RECALL_MIN = 0.9
CELL_RECALL_MIN = 0.95
TABLE_ACC_MIN = 0.9


def _gate(gate: str, ok: bool, detail: str) -> dict[str, Any]:
    return {"gate": gate, "status": "PASS" if ok else "FAIL", "detail": detail}


def evaluate_docx_gates(case: dict, doc: dict, m: dict, determinism_ok: bool) -> list[dict[str, Any]]:
    g = []
    g.append(_gate("W-G1 Structural Preservation",
                   m["heading_recall"] >= RECALL_MIN and m["paragraph_recall"] >= RECALL_MIN
                   and m["list_structure_recall"] >= RECALL_MIN and m["table_structure_accuracy"] >= TABLE_ACC_MIN,
                   f"heading={m['heading_recall']} para={m['paragraph_recall']} "
                   f"list={m['list_structure_recall']} table={m['table_structure_accuracy']}"))
    g.append(_gate("W-G2 No Semantic Flattening",
                   bool(m["semantic_type_accuracy"]) and m["bad_heading_not_faked_as_heading"],
                   f"semantic_type_acc={m['semantic_type_accuracy']} "
                   f"bad_heading_faked={not m['bad_heading_not_faked_as_heading']}"))
    g.append(_gate("W-G3 Table Fidelity",
                   m["table_structure_accuracy"] >= TABLE_ACC_MIN,
                   f"table_structure_accuracy={m['table_structure_accuracy']}"))
    g.append(_gate("W-G4 Style Provenance",
                   bool(m["style_provenance_ok"]),
                   f"heading elements carry style_id={m['style_provenance_ok']}"))
    # W-G5：仅当 gold 含 header/footer 时校验分离
    expects_hf = any(b["kind"] in ("header", "footer") for b in case["blocks"])
    g.append(_gate("W-G5 Header/Footer Separation",
                   (m["header_footer_separation"] if expects_hf else True),
                   f"separated={m['header_footer_separation']} expects={expects_hf}"))
    # W-G6：仅当 gold 含 caption 时校验 image+caption link
    expects_cap = any(b["kind"] == "caption" for b in case["blocks"])
    g.append(_gate("W-G6 Image/Caption Relationship",
                   (m["caption_link_accuracy"] == 1.0 if expects_cap else True),
                   f"caption_link_accuracy={m['caption_link_accuracy']} expects={expects_cap}"))
    g.append(_gate("W-G7 Source Traceability",
                   m["fake_provenance_count"] == 0,
                   f"fake_provenance_count={m['fake_provenance_count']} (must be 0; no page/bbox)"))
    g.append(_gate("W-G8 Reproducibility",
                   determinism_ok,
                   "same-file normalized digest consistent across runs" if determinism_ok else "digest mismatch"))
    return g


def evaluate_xlsx_gates(case: dict, doc: dict, m: dict, determinism_ok: bool) -> list[dict[str, Any]]:
    g = []
    g.append(_gate("X-G1 Cell Fidelity",
                   m["cell_recall"] >= CELL_RECALL_MIN,
                   f"cell_recall={m['cell_recall']}"))
    g.append(_gate("X-G2 Formula Dual Representation",
                   m["formula_recall"] == 1.0 and m["cached_value_preservation"] == 1.0,
                   f"formula_recall={m['formula_recall']} cached_preservation={m['cached_value_preservation']}"))
    expects_merge = any(t.get("merges") for s in case["sheets"] for t in s.get("tables", []))
    g.append(_gate("X-G3 Merged Range",
                   (m["merged_range_accuracy"] == 1.0 if expects_merge else True),
                   f"merged_range_accuracy={m['merged_range_accuracy']} expects={expects_merge}"))
    expects_named = any(t.get("is_named") for s in case["sheets"] for t in s.get("tables", []))
    # 启发式子表标志仅在存在多个（非命名）子表时才有意义；命名表本身 inferred=False，不应被要求。
    expects_multi = sum(len(s.get("tables", [])) for s in case["sheets"]) > 1
    g.append(_gate("X-G4 Table/Subtable Boundary",
                   (m["named_table_accuracy"] == 1.0 if expects_named else True)
                   and (m["heuristic_subtable_inferred_flag"] if expects_multi else True),
                   f"named_acc={m['named_table_accuracy']} inferred_flag={m['heuristic_subtable_inferred_flag']}"))
    g.append(_gate("X-G5 Sheet Semantics",
                   m["sheet_metadata_accuracy"] == 1.0,
                   f"sheet_metadata_accuracy={m['sheet_metadata_accuracy']}"))
    expects_hidden = any(s.get("state") == "hidden" or s.get("hidden_rows") or s.get("hidden_cols")
                        for s in case["sheets"])
    g.append(_gate("X-G6 Hidden Content Policy",
                   (m["hidden_sheet_preserved"] and m["hidden_row_preserved"] if expects_hidden else True),
                   f"hidden_sheet_preserved={m['hidden_sheet_preserved']} "
                   f"hidden_row_preserved={m['hidden_row_preserved']} expects={expects_hidden}"))
    g.append(_gate("X-G7 Display vs Raw Value",
                   m["number_format_preserved"],
                   f"number_format_preserved={m['number_format_preserved']}"))
    g.append(_gate("X-G8 Source Traceability",
                   m["fake_provenance_count"] == 0,
                   f"fake_provenance_count={m['fake_provenance_count']} (must be 0; no page/bbox)"))
    g.append(_gate("X-Reproducibility",
                   determinism_ok,
                   "same-file normalized digest consistent" if determinism_ok else "digest mismatch"))
    return g


def all_pass(gates: list[dict[str, Any]]) -> bool:
    return all(g["status"] == "PASS" for g in gates)
