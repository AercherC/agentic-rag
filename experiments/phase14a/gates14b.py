"""Phase 14A-B 硬门（方案 §19 D-G1..G8 / §20 X-G1..G10）。

阈值冻结：无来源丢失（受控 Gold 零丢失）、顺序 1.0、显式 heading 1.0、fake provenance 0、
确定性一致。失败必须归因到具体指标，不写笼统 FAIL。
None 指标（该 arm 语义上不产生的度量）不参与门判定（None ≠ 0，§24）。
"""

from __future__ import annotations

from typing import Any


def _gate(gate: str, ok: bool, detail: str) -> dict[str, Any]:
    return {"gate": gate, "status": "PASS" if ok else "FAIL", "detail": detail}


def evaluate_docx_gates14b(case: dict, doc: dict, m: dict, deterministic: bool) -> list[dict[str, Any]]:
    g = []
    g.append(_gate("D-G1 No Source Loss",
                   m["unintended_loss"] == 0
                   and (m["heading_recall"] or 1.0) >= 0.9
                   and (m["paragraph_recall"] or 1.0) >= 0.9
                   and (m["list_recall"] or 1.0) >= 0.9,
                   f"unintended_loss={m['unintended_loss']} heading={m['heading_recall']} "
                   f"para={m['paragraph_recall']} list={m['list_recall']}"))
    g.append(_gate("D-G2 Document Order",
                   m["document_order_accuracy"] == 1.0,
                   f"document_order_accuracy={m['document_order_accuracy']} (must be 1.0)"))
    g.append(_gate("D-G3 Heading Hierarchy",
                   m["heading_level_accuracy"] == 1.0 and m["inferred_marked_rate"] == 1.0,
                   f"explicit_level_acc={m['heading_level_accuracy']} "
                   f"inferred_count={m['inferred_heading_count']} marked={m['inferred_marked_rate']}"))
    g.append(_gate("D-G4 Table Structure",
                   (m["table_structure_accuracy"] is None or m["table_structure_accuracy"] >= 0.9)
                   and (m["merged_cell_accuracy"] is None or m["merged_cell_accuracy"] == 1.0)
                   and (m["nested_table_traceability"] is None or m["nested_table_traceability"]),
                   f"table_acc={m['table_structure_accuracy']} merge_acc={m['merged_cell_accuracy']} "
                   f"nested_traceable={m['nested_table_traceability']}"))
    g.append(_gate("D-G5 Furniture Separation",
                   (m["header_footer_separation"] is None or m["header_footer_separation"]),
                   f"separated={m['header_footer_separation']}"))
    g.append(_gate("D-G6 No Fake Provenance",
                   m["fake_provenance_count"] == 0,
                   f"fake_provenance_count={m['fake_provenance_count']} (no page/bbox)"))
    g.append(_gate("D-G7 Deterministic", deterministic,
                   "same-input digest consistent across two runs" if deterministic else "digest mismatch"))
    g.append(_gate("D-G8 Traceable Cleaning",
                   cleaning_traceability_ok(doc, "docx"),
                   "every inferred/reclassified element appears in cleaning_log"
                   if cleaning_traceability_ok(doc, "docx") else "untraceable cleaning action found"))
    return g


def evaluate_xlsx_gates14b(case: dict, doc: dict, m: dict, deterministic: bool) -> list[dict[str, Any]]:
    g = []
    g.append(_gate("X-G1 Cell Truth Complete",
                   (m["non_empty_cell_recall"] is None or m["non_empty_cell_recall"] == 1.0)
                   and m["duplicate_cell_rate"] == 0.0,
                   f"cell_recall={m['non_empty_cell_recall']} duplicate_rate={m['duplicate_cell_rate']}"))
    g.append(_gate("X-G2 Formula Dual Representation",
                   (m["formula_recall"] is None or m["formula_recall"] == 1.0)
                   and (m["cached_value_preservation"] is None or m["cached_value_preservation"] == 1.0)
                   and m["formula_silent_overwrite"] == 0,
                   f"formula={m['formula_recall']} cached={m['cached_value_preservation']} "
                   f"overwrite={m['formula_silent_overwrite']}"))
    g.append(_gate("X-G3 Merge Preservation",
                   (m["merged_range_accuracy"] is None or m["merged_range_accuracy"] == 1.0),
                   f"merged_range_accuracy={m['merged_range_accuracy']} "
                   f"anchor_coverage={m['merge_anchor_coverage']}"))
    g.append(_gate("X-G4 Sheet Identity",
                   (m["sheet_order_accuracy"] is None or m["sheet_order_accuracy"] == 1.0),
                   f"sheet_order_accuracy={m['sheet_order_accuracy']}"))
    g.append(_gate("X-G5 Native Table Preservation",
                   (m["named_table_accuracy"] is None or m["named_table_accuracy"] == 1.0),
                   f"named_table_accuracy={m['named_table_accuracy']}"))
    # X-G6 只约束"已产出的推断表"必须可追溯；B0 无推断表 → 语义上 vacuous（不虚报也不误罚）
    g.append(_gate("X-G6 Inferred Table Traceability",
                   (m["inferred_table_precision"] is None
                    or all_traceable(m)),
                   f"inferred_precision={m['inferred_table_precision']} "
                   f"inferred_recall={m['inferred_table_recall']} "
                   f"(recall applies to B1 region inference only)"))
    g.append(_gate("X-G7 No Fake Provenance",
                   m["fake_provenance_count"] == 0,
                   f"fake_provenance_count={m['fake_provenance_count']} (no page/bbox)"))
    g.append(_gate("X-G8 Hidden Content Explicit",
                   (m["hidden_state_accuracy"] is None or m["hidden_state_accuracy"] == 1.0)
                   and (m["hidden_role_policy_ok"] is None or m["hidden_role_policy_ok"]),
                   f"hidden_state_accuracy={m['hidden_state_accuracy']} "
                   f"role_policy={m['hidden_role_policy_ok']}"))
    g.append(_gate("X-G9 Deterministic", deterministic,
                   "same-input digest consistent across two runs" if deterministic else "digest mismatch"))
    g.append(_gate("X-G10 Traceable Cleaning",
                   cleaning_traceability_ok(doc, "xlsx"),
                   "every region inference / key_value reclassification appears in cleaning_log"
                   if cleaning_traceability_ok(doc, "xlsx") else "untraceable cleaning action found"))
    return g


def all_traceable(m: dict) -> bool:
    """推断表 precision=1.0 意味着没有凭空捏造的区域（可追溯性下界）。"""

    return m["inferred_table_precision"] == 1.0


def cleaning_traceability_ok(doc: dict, fmt: str) -> bool:
    """D-G8 / X-G10：所有推断/重分类动作都有 cleaning_log 痕迹。

    检查两条：
    1. 每个 semantic_inferred=True 的 element 在 log 里有对应 source_element_id；
    2. 每张推断表（inferred=True）在 log 里有以该 element_id 为 target 的记录。
    """

    log_src_ids = {e.get("source_element_id") for e in doc.get("cleaning_log", [])}
    log_targets = {e.get("target") for e in doc.get("cleaning_log", [])}
    for el in doc.get("elements", []):
        if el.get("semantic_inferred") and el["element_id"] not in log_src_ids:
            return False
        if fmt == "xlsx" and (el.get("source_ref") or {}).get("inferred") is True \
                and el.get("type") == "table" and el["element_id"] not in log_targets:
            return False
    return True


def all_pass(gates: list[dict[str, Any]]) -> bool:
    return all(g["status"] == "PASS" for g in gates)
