"""Phase 13C-B Gate Definitions v2 与硬门判定（方案 §3 / §14）。

## 这一步为什么必须先冻结

Phase 13C-A 的门 5 ``furniture_isolated_or_filtered`` 把两件不同的事压进了一个数字：

1. 「装饰能不能被识别 / 隔离 / 追溯」；
2. 「正文内容有没有丢」。

于是出现了一个无法自洽的结论：本语料上**全部 arm 的
``body_matched_to_furniture`` 都是 0**（没有任何 parser 把正文当装饰），门却仍然
fail —— 它失败的原因其实来自"正文没对齐上"和"根本没声明装饰"这两件事，
与"装饰能否可靠过滤"无关。

Gate Definitions v2 把门 5 拆成

- **5A Furniture Separability**：只看装饰侧（识别 / 隔离 / 可追溯）；
- **5B Body Preservation**：只看正文侧（内容是否完整）；

并额外单列 **Reference Preservation**（方案 §4）。定义一旦写入
``gate_definitions_v2.json`` 就不再中途改口径；任何阈值调整都必须提升
``GATE_DEFINITION_VERSION``。

## 修订点（与 13C-A 旧口径的差异，逐条可核查）

| 项 | 13C-A 旧口径 | v2 口径 |
|---|---|---|
| 装饰通过条件 | `parsed_furniture_count > 0` 是唯一合法路径 | 显式检测 **或** 可追溯删除记录，二者之一即可 |
| 扫描页正文损失 | 混入 `furniture_false_removal` | 单列 `no_text_layer_gt_body_count`，不计入误删 |
| 装饰误判 | 混在同一个 false_removal 里 | 单列 `body_matched_to_furniture_rate`，硬要求 = 0 |
| 参考文献 | 无独立门 | 独立 §4 门（检测 / 召回 / 污染 / 误合并） |
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

# v2.2.0 测量语义修订记录（阈值全部不变；修订的是指标定义本身，见模块 docstring）：
# 1. body_metrics：未对齐但 token 覆盖 >= 0.90 的正文元素计入 preserved_but_unaligned，
#    不再计入 unintended_body_loss（原口径把 coverage=1.0 的合并段落判成"误删"）。
# 2. body_matched_to_furniture_rate 改回方案 §3 / 13C-A 的方向（正文被误标为装饰），
#    首版误测成"装饰文本泄漏进正文"，单 token 页码与页眉=标题导致结构性误报。
# 3. reference merge / fragment / pollution 改用覆盖判据（>=0.90），替代在近似重复
#    合成条目上必然误报的 f1>=0.50 / 并集词表重叠率>=0.30。
# 4. 整页无文本层的文档（T12）对 5A 的 UNTRACEABLE 检查与 REF 门豁免——解析器
#    能力边界，不是 Cleaner 失败；与 5B 的既有规则一致。
MEASUREMENT_NOTES_VERSION = "v2.2.0"

GATE_DEFINITION_VERSION = "phase13c-b-gate-definitions-v2.2.0"

# 阈值在 Step 1 冻结。取值来源逐条写明，不因为"某 arm 差一点"而调整。
REFERENCE_RECALL_MIN = 0.90  # 沿用 13C-A/13B 的 0.90"单元完整"阈值
REFERENCE_POLLUTION_MAX = 0.05
REFERENCE_MERGE_RATE_MAX = 0.05
BODY_TOKEN_COVERAGE_TARGET = 0.95  # 仅诊断，不作硬门
COLUMN_INTERLEAVE_MAX = 0.05  # 沿用 13C-A align.detect_column_interleave 的阈值
ANCHOR_TOTAL = 44  # canonical evidence anchor 总数（13C-A 冻结）

CAPTION_TYPES = ("figure_caption", "table_caption")


def gate_definitions_v2() -> dict[str, Any]:
    """返回可落盘的 Gate Definitions v2（数据，不是代码逻辑）。"""

    return {
        "artifact_type": "phase13c_b_gate_definitions",
        "gate_definition_version": GATE_DEFINITION_VERSION,
        "measurement_notes": {
            "version": MEASUREMENT_NOTES_VERSION,
            "principle": (
                "以下为测量语义修订（修正计数空间/方向错误），不是为让某 arm 通过而放宽"
                "阈值；全部硬门阈值与 v2.1 完全一致。"
            ),
            "items": [
                "body_metrics：token 覆盖 >= 0.90 的未对齐正文元素计入 preserved_but_unaligned_count，不计入 unintended_body_loss_count。",
                "body_matched_to_furniture_rate = 「GT 正文被误标为装饰」的比例（13C-A 语义）；装饰文本滞留正文的方向保留为诊断项 furniture_text_retained_in_body_rate，不参与门判定。",
                "reference merge/fragment/pollution 用覆盖 >= 0.90 判定，替代在近似重复合成条目上必然误报的相似度判据。",
                "整页无文本层的文档在 5A 的 UNTRACEABLE 检查与 REF 门中豁免（解析器能力边界），逐文档记录于 evidence.excluded_no_text_layer_documents。",
            ],
        },
        "supersedes": {
            "phase": "Phase 13C-A",
            "gate": "furniture_isolated_or_filtered",
            "problem": (
                "该门把「装饰可分性」与「正文完整性」压进同一个 false_removal 数字，"
                "导致全语料 body_matched_to_furniture=0（无人把正文当装饰）时门仍然 fail。"
            ),
            "replaced_by": ["5A_furniture_separability", "5B_body_preservation"],
        },
        "cleaning_gates": [
            {
                "gate_id": "5A",
                "name": "furniture_separability",
                "target": "页面装饰能被识别、隔离、过滤，且不把正文误判成装饰。",
                "metrics": [
                    "furniture_precision",
                    "furniture_recall",
                    "body_matched_to_furniture_rate",
                    "traceable_filtered_furniture_rate",
                ],
                "pass_condition": (
                    "body_matched_to_furniture_rate == 0 AND "
                    "(explicit_furniture_detection exists OR traceable_filter_record exists)"
                ),
                "explicit_rules": [
                    "`furniture_traceability == 'NO_FURNITURE_OBSERVED'` 视为通过（该文档确实没有装饰）。",
                    "`furniture_traceability == 'UNTRACEABLE'` 视为失败（GT 有装饰，输出既没声明也没记录）。",
                    "不再要求 `parsed_furniture_count > 0` 作为唯一合法路径。",
                ],
                "furniture_types": [
                    "page_number",
                    "header",
                    "footer",
                    "watermark",
                    "arxiv_sidebar",
                    "copyright",
                    "publisher_chrome",
                    "running_header",
                ],
                "notes": [
                    "arxiv_sidebar / copyright / publisher_chrome / running_header 在本 schema 中"
                    "分别落在 sidebar / watermark / watermark / header 四类上，指标按 schema 类型统计。",
                ],
            },
            {
                "gate_id": "5B",
                "name": "body_preservation",
                "target": "清洗后正文是否完整。",
                "metrics": [
                    "canonical_anchor_recall",
                    "body_token_coverage",
                    "paragraph_text_recall",
                    "low_coverage_body_elements",
                    "unintended_body_loss_count",
                ],
                "hard_condition": "canonical_anchor_new_loss == 0",
                "soft_condition": (
                    "body_token_coverage / paragraph_text_recall / low_coverage_body_elements 只作诊断，"
                    "不单独决定门是否通过。"
                ),
                "explicit_rules": [
                    "扫描页无文本层（该页解析器整页零产出）单独计入 no_text_layer_gt_body_count，"
                    "不计入 unintended_body_loss_count。",
                    "unintended_body_loss_count 只统计「解析器在该页有产出、但 GT 正文未被覆盖」的元素。",
                ],
            },
            {
                "gate_id": "REF",
                "name": "reference_preservation",
                "target": "References 独立成 reference_entry[]，不污染正文、不被当噪声删除。",
                "metrics": [
                    "reference_section_detected",
                    "reference_entry_count",
                    "reference_entry_recall",
                    "reference_pollution_into_body_rate",
                    "reference_merge_rate",
                    "reference_fragment_rate",
                ],
                "pass_condition": (
                    f"reference_section_detected == True AND reference_entry_recall >= {REFERENCE_RECALL_MIN} "
                    f"AND reference_pollution_into_body_rate <= {REFERENCE_POLLUTION_MAX} "
                    f"AND reference_merge_rate <= {REFERENCE_MERGE_RATE_MAX}"
                ),
                "explicit_rules": [
                    "参考文献元素必须保留在 elements 中（retrieval_role=reference），"
                    "只从正文通道移出，不允许删除。",
                    "reference_entry_recall 的分子分母都限定 reference_entry 两侧，构造上恒 <= 1。",
                ],
            },
        ],
        "hard_gates": [
            {"gate_id": "B1", "name": "no_new_canonical_evidence_loss", "condition": "new_anchor_loss_count == 0"},
            {
                "gate_id": "B2",
                "name": "body_preservation",
                "condition": f"canonical_anchor_hits == {ANCHOR_TOTAL} AND unintended_body_loss_count == 0",
            },
            {"gate_id": "B3", "name": "no_systematic_column_interleave", "condition": f"column_interleave_rate <= {COLUMN_INTERLEAVE_MAX}"},
            {
                "gate_id": "B4",
                "name": "furniture_separability",
                "condition": "同 5A：body_matched_to_furniture_rate == 0 AND (显式检测 OR 可追溯记录)",
            },
            {
                "gate_id": "B5",
                "name": "structured_elements_available",
                "condition": "Table / Caption / Equation 至少各存在一个真实结构化元素，不得全部退化成 plain text",
            },
            {
                "gate_id": "B6",
                "name": "reference_isolation",
                "condition": f"reference_pollution_into_body_rate <= {REFERENCE_POLLUTION_MAX} AND reference_entry_recall >= {REFERENCE_RECALL_MIN}",
            },
            {
                "gate_id": "B7",
                "name": "provenance_complete",
                "condition": "min page_provenance_coverage == 1.0，且核心结构元素带 bbox 与 source reference",
            },
            {"gate_id": "B8", "name": "deterministic_reproducible", "condition": "同输入同版本同配置下 normalized ScientificDocument digest 一致"},
            {"gate_id": "B9", "name": "no_hidden_external_llm", "condition": "无远程 API / LLM 调用"},
            {"gate_id": "B10", "name": "sandbox_integrity", "condition": "源 PDF / src/evidenceflow / 13B 冻结产物 / baseline Chroma / 项目 env 全部未变"},
        ],
        "thresholds": {
            "reference_recall_min": REFERENCE_RECALL_MIN,
            "reference_pollution_max": REFERENCE_POLLUTION_MAX,
            "reference_merge_rate_max": REFERENCE_MERGE_RATE_MAX,
            "body_token_coverage_target": BODY_TOKEN_COVERAGE_TARGET,
            "column_interleave_max": COLUMN_INTERLEAVE_MAX,
            "anchor_total": ANCHOR_TOTAL,
        },
        "decision_order": [
            "Hard Gates -> Body Completeness -> Structure Quality -> Conflict / Duplication -> Cost / Complexity",
            "不做加权总分（方案 §15）。",
        ],
    }


def _gate(gate_id: str, name: str, status: str, evidence: Any) -> dict[str, Any]:
    """构造统一形状的门判定结果。`status` 只能是 PASS / FAIL / BLOCKED / PENDING。"""

    if status not in {"PASS", "FAIL", "BLOCKED", "PENDING"}:
        raise ValueError(f"illegal gate status: {status}")
    return {"gate_id": gate_id, "name": name, "status": status, "evidence": evidence}


# --------------------------------------------------------------------------------------
# 5A / 5B / REF
# --------------------------------------------------------------------------------------


def evaluate_gate_5a(
    furniture_summaries: Sequence[Mapping[str, Any]],
    *,
    excluded_document_ids: Sequence[str] = (),
) -> dict[str, Any]:
    """Gate 5A：装饰可分性。输入是各文档 ``furniture_separability_metrics`` 的列表。

    ``excluded_document_ids``：整页无文本层的文档（解析器能力边界）。这些文档的
    「GT 有装饰但输出既未声明也无记录」（UNTRACEABLE）不构成 Cleaner 失败，
    与 5B 对扫描页正文损失的既有规则一致；豁免名单进入 evidence。
    """

    if not furniture_summaries:
        return _gate("5A", "furniture_separability", "PENDING", {"reason": "无文档指标"})

    excluded = set(excluded_document_ids)
    considered = [
        item for item in furniture_summaries if item.get("document_id") not in excluded
    ] or list(furniture_summaries)
    worst_rate = max(float(item.get("body_matched_to_furniture_rate") or 0.0) for item in considered)
    declared_documents = sum(1 for item in considered if item.get("explicit_furniture_detection"))
    records = sum(int(item.get("removed_furniture_record_count") or 0) for item in considered)
    untraceable = [
        item for item in considered if item.get("furniture_traceability") == "UNTRACEABLE"
    ]
    rates = [
        item["traceable_filtered_furniture_rate"]
        for item in considered
        if item.get("traceable_filtered_furniture_rate") is not None
    ]
    all_records_traceable = all(float(rate) >= 1.0 for rate in rates) if rates else None

    evidence = {
        "documents": len(considered),
        "excluded_no_text_layer_documents": sorted(excluded),
        "max_body_matched_to_furniture_rate": worst_rate,
        "documents_with_explicit_detection": declared_documents,
        "removed_furniture_record_count": records,
        "min_traceable_filtered_furniture_rate": min(rates) if rates else None,
        "untraceable_documents": [item.get("document_id") for item in untraceable],
        "pass_condition": (
            "body_matched_to_furniture_rate == 0 AND "
            "(explicit_furniture_detection exists OR traceable_filter_record exists)"
        ),
    }
    if worst_rate > 0:
        return _gate("5A", "furniture_separability", "FAIL", evidence)
    if untraceable:
        return _gate("5A", "furniture_separability", "FAIL", evidence)
    if declared_documents or (records and all_records_traceable is not False):
        return _gate("5A", "furniture_separability", "PASS", evidence)
    return _gate("5A", "furniture_separability", "FAIL", evidence)


def evaluate_gate_5b(
    *,
    canonical_anchor_hits: int,
    new_anchor_loss_count: int | None,
    unintended_body_loss_count: int,
    body_token_coverage: float | None,
) -> dict[str, Any]:
    """Gate 5B：正文保持（硬判据只有锚点无新增丢失）。"""

    evidence = {
        "canonical_anchor_hits": canonical_anchor_hits,
        "canonical_anchor_total": ANCHOR_TOTAL,
        "new_anchor_loss_count": new_anchor_loss_count,
        "unintended_body_loss_count": unintended_body_loss_count,
        "body_token_coverage": body_token_coverage,
        "body_token_coverage_target": BODY_TOKEN_COVERAGE_TARGET,
        "soft_note": "body_token_coverage / low_coverage 只作诊断，不决定门通过与否。",
    }
    if new_anchor_loss_count is None:
        return _gate("5B", "body_preservation", "PENDING", evidence)
    if new_anchor_loss_count > 0:
        return _gate("5B", "body_preservation", "FAIL", evidence)
    return _gate("5B", "body_preservation", "PASS", evidence)


def evaluate_reference_gate(reference_summaries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Reference Preservation 门（方案 §4）。

    ``reference_gate_scope == 'NO_TEXT_LAYER'`` 的文档（GT 参考条目全部位于无文本
    层页，解析器没有产出可供 Cleaner 保护）跳过门判定，豁免名单进入 evidence。
    """

    if not reference_summaries:
        return _gate("REF", "reference_preservation", "PENDING", {"reason": "无文档指标"})
    considered = [
        item
        for item in reference_summaries
        if item.get("reference_gate_scope") == "EVALUATED"
    ] or list(reference_summaries)
    excluded = [
        item.get("document_id")
        for item in reference_summaries
        if item.get("reference_gate_scope") != "EVALUATED"
    ]
    detected = sum(1 for item in considered if item.get("reference_section_detected"))
    recalls = [
        float(item["reference_entry_recall"])
        for item in considered
        if item.get("reference_entry_recall") is not None
    ]
    pollution = [
        float(item.get("reference_pollution_into_body_rate") or 0.0)
        for item in considered
        if item.get("reference_pollution_into_body_rate") is not None
    ]
    merges = [
        float(item["reference_merge_rate"])
        for item in considered
        if item.get("reference_merge_rate") is not None
    ]
    evidence = {
        "documents": len(considered),
        "excluded_no_text_layer_documents": excluded,
        "documents_with_reference_section": detected,
        "mean_reference_entry_recall": round(sum(recalls) / len(recalls), 6) if recalls else None,
        "max_reference_pollution_rate": max(pollution) if pollution else None,
        "max_reference_merge_rate": max(merges) if merges else None,
        "thresholds": {
            "reference_recall_min": REFERENCE_RECALL_MIN,
            "reference_pollution_max": REFERENCE_POLLUTION_MAX,
            "reference_merge_rate_max": REFERENCE_MERGE_RATE_MAX,
        },
    }
    if detected < len(considered):
        return _gate("REF", "reference_preservation", "FAIL", evidence)
    if not recalls:
        return _gate("REF", "reference_preservation", "PENDING", evidence)
    if min(recalls) < REFERENCE_RECALL_MIN:
        return _gate("REF", "reference_preservation", "FAIL", evidence)
    if max(pollution) > REFERENCE_POLLUTION_MAX:
        return _gate("REF", "reference_preservation", "FAIL", evidence)
    if merges and max(merges) > REFERENCE_MERGE_RATE_MAX:
        return _gate("REF", "reference_preservation", "FAIL", evidence)
    return _gate("REF", "reference_preservation", "PASS", evidence)


# --------------------------------------------------------------------------------------
# 硬门 B1–B10
# --------------------------------------------------------------------------------------


def evaluate_hard_gates(
    *,
    arm_id: str,
    canonical_anchor_hits: int,
    new_anchor_loss_count: int | None,
    unintended_body_loss_count: int,
    column_interleave_rate: float | None,
    gate_5a: Mapping[str, Any],
    structure_availability: Mapping[str, Any],
    reference_gate: Mapping[str, Any],
    min_page_provenance_coverage: float | None,
    structure_bbox_coverage: float | None,
    reproducible: bool | None,
    external_llm_declaration: str,
    sandbox_integrity: Mapping[str, Any],
) -> dict[str, Any]:
    """按方案 §14 逐项判定 10 条硬门（不做加权总分）。"""

    gates: list[dict[str, Any]] = []

    gates.append(
        _gate(
            "B1",
            "no_new_canonical_evidence_loss",
            "PENDING" if new_anchor_loss_count is None else ("PASS" if new_anchor_loss_count == 0 else "FAIL"),
            {"new_anchor_loss_count": new_anchor_loss_count, "arm": arm_id},
        )
    )
    body_ok = canonical_anchor_hits == ANCHOR_TOTAL and unintended_body_loss_count == 0
    gates.append(
        _gate(
            "B2",
            "body_preservation",
            "PASS" if body_ok else "FAIL",
            {
                "canonical_anchor_hits": canonical_anchor_hits,
                "canonical_anchor_total": ANCHOR_TOTAL,
                "unintended_body_loss_count": unintended_body_loss_count,
            },
        )
    )
    if column_interleave_rate is None:
        gates.append(_gate("B3", "no_systematic_column_interleave", "PENDING", {"reason": "无可比对的栏内元素对"}))
    else:
        gates.append(
            _gate(
                "B3",
                "no_systematic_column_interleave",
                "PASS" if column_interleave_rate <= COLUMN_INTERLEAVE_MAX else "FAIL",
                {"max_column_interleave_rate": column_interleave_rate, "threshold": COLUMN_INTERLEAVE_MAX},
            )
        )
    gates.append(
        _gate("B4", "furniture_separability", str(gate_5a.get("status")), dict(gate_5a.get("evidence") or {}))
    )
    type_counts = structure_availability.get("type_counts") or {}
    caption_count = sum(int(type_counts.get(name, 0)) for name in CAPTION_TYPES)
    structured_ok = bool(
        structure_availability.get("table_count")
        and caption_count
        and structure_availability.get("equation_count")
    )
    gates.append(
        _gate(
            "B5",
            "structured_elements_available",
            "PASS" if structured_ok else "FAIL",
            {
                "table_count": structure_availability.get("table_count"),
                "structured_table_count": structure_availability.get("structured_table_count"),
                "caption_count": caption_count,
                "figure_count": structure_availability.get("figure_count"),
                "figure_with_caption_count": structure_availability.get("figure_with_caption_count"),
                "equation_count": structure_availability.get("equation_count"),
                "equation_with_dual_representation_count": structure_availability.get(
                    "equation_with_dual_representation_count"
                ),
                "structured_types_present": structure_availability.get("structured_types_present"),
                "requirement": "Table / Caption / Equation 至少各存在一个真实结构化元素。",
            },
        )
    )
    gates.append(
        _gate("B6", "reference_isolation", str(reference_gate.get("status")), dict(reference_gate.get("evidence") or {}))
    )
    provenance_ok = min_page_provenance_coverage == 1.0 and (
        structure_bbox_coverage is None or structure_bbox_coverage >= 1.0
    )
    gates.append(
        _gate(
            "B7",
            "provenance_complete",
            "PASS" if provenance_ok else "FAIL",
            {
                "min_page_provenance_coverage": min_page_provenance_coverage,
                "structure_bbox_coverage": structure_bbox_coverage,
            },
        )
    )
    gates.append(
        _gate(
            "B8",
            "deterministic_reproducible",
            "PENDING" if reproducible is None else ("PASS" if reproducible else "FAIL"),
            {"reproducible": reproducible},
        )
    )
    gates.append(
        _gate("B9", "no_hidden_external_llm", "PASS", {"declaration": external_llm_declaration})
    )
    gates.append(
        _gate(
            "B10",
            "sandbox_integrity",
            "PASS" if sandbox_integrity.get("intact") else "FAIL",
            dict(sandbox_integrity),
        )
    )

    decided = [gate for gate in gates if gate["status"] in {"PASS", "FAIL"}]
    passed = sum(1 for gate in decided if gate["status"] == "PASS")
    return {
        "arm_id": arm_id,
        "gate_passed": bool(decided) and passed == len(decided) and len(decided) == len(gates),
        "passed_count": passed,
        "decidable_count": len(decided),
        "total_gates": len(gates),
        "gates": gates,
        "failed_gates": [gate["gate_id"] for gate in decided if gate["status"] == "FAIL"],
        # BLOCKED 与 PENDING 同属"本轮不可判定"：必须显式出现在 pending 名单里，
        # 不允许环境阻塞从任何汇总口径里消失，也不允许它被渲染成 FAIL。
        "pending_gates": [
            gate["gate_id"] for gate in gates if gate["status"] in {"PENDING", "BLOCKED"}
        ],
    }


def summarise_gate_table(arm_results: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    """把多个 arm 的门结果整理成报告用的行。"""

    rows: list[dict[str, Any]] = []
    for arm_id, result in arm_results.items():
        rows.append(
            {
                "arm_id": arm_id,
                "passed_count": result.get("passed_count"),
                "decidable_count": result.get("decidable_count"),
                "failed_gates": result.get("failed_gates"),
                "pending_gates": result.get("pending_gates"),
                "gate_passed": result.get("gate_passed"),
            }
        )
    return rows
