"""Phase 13C-B Cleaning & Fusion Benchmark 的纯逻辑测试（方案 §18）。

覆盖范围（全部离线、确定性、无网络、不加载模型、不写项目目录）：

1. Gate 5A / 5B / REF 的门语义（含「扫描页豁免」「装饰不吞正文」两条 v2.2 规则）；
2. Cleaner 契约：参考文献不被当装饰删除、公式双表示、装饰删除记录可追溯；
3. Fusion 契约：P0 锚点保留、对齐文本去重、P1 表格结构保留、冲突不静默覆盖；
4. Conditional OCR 触发规则只对坏页触发；
5. 产物合同：BLOCKED 不得渲染成 FAIL、比例型指标越界必须被机械抓到。

mutation validation（缺陷重新注入验证）：每个关键判据都配一个"把缺陷注回去就该
失败"的反向用例——coverage=1.0 的合并段落不得计为丢失、precision>1 必须报越界、
正文被误标为装饰必须让 5A 失败。
"""

from __future__ import annotations

import json

import pytest

from experiments.phase13c import cleaning, cleaning_metrics, fusion, gates_v2, schema
from experiments.run_phase13c_cleaning_benchmark import conditional_ocr_trigger

# --------------------------------------------------------------------------------------
# 公共构造 helper
# --------------------------------------------------------------------------------------

PAGE = 1
BBOX = [10.0, 10.0, 200.0, 30.0]


def parsed(
    element_id: str,
    type_: str,
    text: str,
    *,
    page: int = PAGE,
    role: str | None = None,
    order: int | None = None,
    bbox=None,
) -> dict:
    """构造一个"清洗输入"风格的元素（带 source_parser，可选 retrieval_role）。"""

    element = schema.make_element(
        document_id="doc-test",
        element_id=element_id,
        type=type_,
        raw_text=text,
        source_parser="test-parser",
        page=page,
        bbox=list(bbox) if bbox else list(BBOX),
        reading_order=order,
        section_path=[],
    )
    if role:
        element["retrieval_role"] = role
    return element


def mini_doc(elements: list[dict], *, document_id: str = "doc-test") -> dict:
    """构造 clean_document / fuse_documents 可接受的最小文档。"""

    return {
        "document_id": document_id,
        "elements": elements,
        "metadata": {"source_parser": "test-parser"},
        "page_count": 3,
    }


def fused_retrieval_text(fused: dict) -> str:
    """拼接融合文档的全部检索文本（用于断言内容在/不在）。"""

    ids = set(fused.get("retrieval_element_ids") or [])
    return "\n".join(
        (element.get("normalized_text") or "")
        for element in fused.get("elements", [])
        if element.get("element_id") in ids
    )


# --------------------------------------------------------------------------------------
# Gate 5A —— 装饰可分性
# --------------------------------------------------------------------------------------


def test_gate5a_furniture_separability():
    """5A 三态：过 / 正文被吞 / UNTRACEABLE；豁免名单必须生效。"""

    passing = {
        "document_id": "A",
        "body_matched_to_furniture_rate": 0.0,
        "explicit_furniture_detection": True,
        "removed_furniture_record_count": 4,
        "traceable_filtered_furniture_rate": 1.0,
        "furniture_traceability": "EXPLICIT_DETECTION",
    }
    assert gates_v2.evaluate_gate_5a([passing]).get("status") == "PASS"

    # mutation：正文被误标为装饰（rate > 0）→ 必须失败。
    body_eaten = dict(passing, document_id="B", body_matched_to_furniture_rate=0.25)
    assert gates_v2.evaluate_gate_5a([passing, body_eaten]).get("status") == "FAIL"

    # GT 有装饰但输出既没声明也没记录 → 失败。
    untraceable = {
        "document_id": "C",
        "body_matched_to_furniture_rate": 0.0,
        "explicit_furniture_detection": False,
        "removed_furniture_record_count": 0,
        "traceable_filtered_furniture_rate": None,
        "furniture_traceability": "UNTRACEABLE",
    }
    assert gates_v2.evaluate_gate_5a([passing, untraceable]).get("status") == "FAIL"

    # 整页无文本层的文档（解析器能力边界）可被豁免；豁免名单进入 evidence。
    exempt = gates_v2.evaluate_gate_5a(
        [passing, untraceable], excluded_document_ids=["C"]
    )
    assert exempt.get("status") == "PASS"
    assert exempt["evidence"]["excluded_no_text_layer_documents"] == ["C"]


# --------------------------------------------------------------------------------------
# Gate 5B —— 正文保持
# --------------------------------------------------------------------------------------


def test_gate5b_body_preservation():
    """5B 硬判据只有锚点无新增丢失；coverage 只是诊断，不能翻转门。"""

    ok = gates_v2.evaluate_gate_5b(
        canonical_anchor_hits=44,
        new_anchor_loss_count=0,
        unintended_body_loss_count=3,
        body_token_coverage=0.80,
    )
    assert ok.get("status") == "PASS"

    lost = gates_v2.evaluate_gate_5b(
        canonical_anchor_hits=44,
        new_anchor_loss_count=1,
        unintended_body_loss_count=0,
        body_token_coverage=1.0,
    )
    assert lost.get("status") == "FAIL"

    unknown = gates_v2.evaluate_gate_5b(
        canonical_anchor_hits=44,
        new_anchor_loss_count=None,
        unintended_body_loss_count=0,
        body_token_coverage=None,
    )
    assert unknown.get("status") == "PENDING"


# --------------------------------------------------------------------------------------
# Reference Preservation —— 参考文献不得被当装饰/噪声删除
# --------------------------------------------------------------------------------------


def test_reference_not_counted_as_furniture():
    """References 区条目必须落进 reference 通道，任何一步都不得进 removed_furniture。"""

    document = mini_doc(
        [
            parsed("h-1", "section_header", "References"),
            parsed("r-1", "paragraph", "[1] A. Author, On document understanding, 2020.", order=1),
            parsed("r-2", "paragraph", "[2] B. Author, On ingestion pipelines, 2021.", order=2),
        ]
    )
    cleaned = cleaning.clean_document(document)

    assert len(cleaned["reference_element_ids"]) == 2
    by_id = {element["element_id"]: element for element in cleaned["elements"]}
    for ref_id in cleaned["reference_element_ids"]:
        assert by_id[ref_id]["type"] == "reference_entry"
        assert by_id[ref_id]["retrieval_role"] == "reference"
    # 条目文本不得出现在任何装饰删除记录里。
    removed_text = "\n".join(str(record.get("text") or "") for record in cleaned["removed_furniture"])
    assert "A. Author" not in removed_text
    assert "B. Author" not in removed_text
    assert cleaned["cleaning_report"]["references"]["reference_section_detected"] is True


def test_scanned_page_not_body_loss_for_native_parser():
    """扫描页（整页无文本层）上的 GT 正文记入 no_text_layer，不计入误删。

    mutation 对照：同一元素若在有文本层的页上，必须被计为 unintended loss。
    """

    gt = [{"element_id": "gt-1", "type": "paragraph", "text": "unique body sentence content", "page": 3}]
    # 清洗侧在该页有产出（结构元素），但 GT 文本不在其中。
    cleaned = {
        "retrieval_element_ids": ["s-1"],
        "elements": [parsed("s-1", "section_header", "3 Method", page=3)],
    }
    alignment = cleaning_metrics.align.align_elements(gt, cleaned["elements"])

    scanned = cleaning_metrics.body_metrics(gt, cleaned, alignment, no_text_layer_pages=[3])
    assert scanned["unintended_body_loss_count"] == 0
    assert scanned["no_text_layer_gt_body_count"] == 1

    text_layer = cleaning_metrics.body_metrics(gt, cleaned, alignment)
    assert text_layer["unintended_body_loss_count"] == 1
    assert text_layer["no_text_layer_gt_body_count"] == 0


def test_body_merge_is_not_counted_as_loss():
    """mutation validation：coverage=1.0 的未对齐元素（被合并）不得计为误删。

    这是首版实现的缺陷：把"合并导致 1-1 对齐失败"的正文判成丢失。
    """

    gt = [{"element_id": "gt-1", "type": "paragraph", "text": "alpha beta gamma", "page": 1}]
    # 清洗侧把该段与大量其他文本合并成一个元素（相似度 < 0.5，对不上），但 token 全在。
    merged_text = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu"
    cleaned = {
        "retrieval_element_ids": ["p-1"],
        "elements": [parsed("p-1", "paragraph", merged_text)],
    }
    alignment = cleaning_metrics.align.align_elements(gt, cleaned["elements"])
    body = cleaning_metrics.body_metrics(gt, cleaned, alignment)
    assert body["unintended_body_loss_count"] == 0
    assert body["preserved_but_unaligned_count"] == 1


# --------------------------------------------------------------------------------------
# Cleaner 契约
# --------------------------------------------------------------------------------------


def test_removed_furniture_has_provenance():
    """装饰删除记录必须四项齐全（page / bbox / reason / source_parser），可追溯率 1.0。"""

    document = mini_doc(
        [
            parsed("p-1", "paragraph", "Body text about encoders and decoders.", order=0),
            parsed("h-1", "header", "Running Header", order=1, page=2),
            parsed("n-1", "page_number", "2", order=2, page=2),
        ]
    )
    cleaned = cleaning.clean_document(document)
    records = cleaned["removed_furniture"]
    assert len(records) == 2
    for record in records:
        assert record["page"] is not None
        assert record["bbox"]
        assert record["reason"]
        assert record["source_parser"]
    # 指标层的可追溯判定必须给出满分。
    gt: list[dict] = []
    fur = cleaning_metrics.furniture_separability_metrics(gt, cleaned)
    assert fur["traceable_filtered_furniture_rate"] == 1.0
    assert fur["furniture_traceability"] == "EXPLICIT_DETECTION"


def test_formula_dual_representation():
    """公式必须保留 raw 与 normalized 双表示，谁也不覆盖谁（走完整清洗链）。"""

    document = mini_doc(
        [
            parsed("eq-1", "equation", "$N _ { p } = 1 2 8$", order=0),
        ]
    )
    cleaned = cleaning.clean_document(document)
    element = next(e for e in cleaned["elements"] if e["element_id"] == "eq-1")
    dual = element["equation"]
    assert dual["raw_text"].startswith("$")
    assert dual["normalized_latex"] == "Np = 128"
    assert dual["raw_text"] != dual["normalized_latex"]


def test_heuristic_furniture_rules_are_text_only():
    """裸页码与跨页重复短文本被识别为装饰；长正文与首尾不同的短文本不受影响。"""

    document = mini_doc(
        [
            parsed("p-1", "paragraph", "Encoder text one.", page=1, order=0),
            parsed("n-1", "paragraph", "2", page=2, order=1),
            parsed("h-1", "paragraph", "Same Running Header", page=2, order=2),
            parsed("h-2", "paragraph", "Same Running Header", page=3, order=3),
            parsed("b-1", "paragraph", "Same Running Header", page=3, order=4, bbox=[10.0, 100.0, 200.0, 120.0]),
        ]
    )
    cleaned = cleaning.clean_document(document)
    roles = {element["element_id"]: element.get("retrieval_role") for element in cleaned["elements"]}
    assert roles["n-1"] == "furniture"
    # 跨页重复 → 全部判为 running header（第三处同文也一样）。
    assert roles["h-1"] == roles["h-2"] == roles["b-1"] == "furniture"
    assert roles["p-1"] == "body"
    reasons = {record["reason"] for record in cleaned["removed_furniture"]}
    assert cleaning.REASON_REPEATED_TEXT in reasons


# --------------------------------------------------------------------------------------
# Fusion 契约
# --------------------------------------------------------------------------------------


def _p0_doc() -> dict:
    return mini_doc(
        [
            parsed(
                "p0-anchor",
                "paragraph",
                "The anchor sentence preserves evidence flow across the whole pipeline.",
                role="body",
                order=0,
            ),
            parsed("p0-title", "document_title", "Structure-Aware Ingestion", role="body", order=1),
        ],
        document_id="p0",
    )


def test_fusion_preserves_p0_anchor():
    """P1 缺了 P0 的正文（anchor）时，融合文档必须原样保留它。"""

    p1 = mini_doc(
        [
            parsed("p1-title", "document_title", "Structure-Aware Ingestion", role="body", order=0),
        ],
        document_id="p1",
    )
    fused = fusion.fuse_documents(dict(_p0_doc()), dict(p1))
    text = fused_retrieval_text(fused)
    assert "The anchor sentence preserves evidence flow" in text
    assert fused["fusion_metrics"]["primary_only_count"] >= 1


def test_fusion_deduplicates_aligned_text():
    """P0/P1 同文同页对齐后只保留一份，不得重复输出同一段。"""

    p1 = mini_doc(
        [
            parsed(
                "p1-anchor",
                "paragraph",
                "The anchor sentence preserves evidence flow across the whole pipeline.",
                role="body",
                order=0,
            ),
            parsed("p1-title", "document_title", "Structure-Aware Ingestion", role="body", order=1),
        ],
        document_id="p1",
    )
    fused = fusion.fuse_documents(dict(_p0_doc()), dict(p1))
    hits = [
        element
        for element in fused["elements"]
        if "The anchor sentence preserves" in (element.get("normalized_text") or "")
    ]
    assert len(hits) == 1
    assert fused["fusion_metrics"]["duplicate_content_rate"] == 0.0


def test_fusion_keeps_p1_table_structure():
    """P0 只有表格扁平文本时，融合结果必须采信 P1 的行列结构（§7.3）。"""

    rows = [["Method", "CD", "F1"], ["Ours", "7.05", "0.901"]]
    p0_table_text = "Method CD F1 Ours 7.05 0.901"
    p0 = mini_doc(
        [
            parsed("p0-tbl", "paragraph", p0_table_text, role="body", order=0),
        ],
        document_id="p0",
    )
    p1_element = parsed(
        "p1-tbl",
        "table",
        p0_table_text,
        role="structure",
        order=0,
    )
    p1_element["table"] = {
        "caption": None,
        "headers": ["Method", "CD", "F1"],
        "rows": rows,
        "raw_structure": {"rows": rows, "header_row": True},
        "serialized_markdown": "| Method | CD | F1 |\n| --- | --- | --- |\n| Ours | 7.05 | 0.901 |",
        "page": PAGE,
        "bbox": list(BBOX),
        "section_path": [],
        "structure_available": True,
        "serialization": "markdown",
    }
    p1 = mini_doc([p1_element], document_id="p1")

    fused = fusion.fuse_documents(dict(p0), dict(p1))
    tables = [element for element in fused["elements"] if element.get("type") == "table"]
    assert len(tables) == 1
    table = tables[0]["table"]
    assert table["rows"] == rows
    assert table["headers"] == ["Method", "CD", "F1"]
    assert table["primary_fallback_text"] == p0_table_text
    assert table["structure_available"] is True


def test_conflict_is_not_silently_overwritten():
    """P0/P1 的 section_path 冲突必须留痕：采信 P1 但记录 P0，不得静默覆盖（§7.2）。"""

    p0_element = parsed(
        "p0-1",
        "paragraph",
        "Encoders convert document pages into structured tokens.",
        role="body",
        order=0,
    )
    p0_element["section_path"] = ["1 Introduction"]
    p1_element = parsed(
        "p1-1",
        "paragraph",
        "Encoders convert document pages into structured tokens.",
        role="body",
        order=0,
    )
    p1_element["section_path"] = ["3 Method"]
    fused = fusion.fuse_documents(mini_doc([p0_element]), mini_doc([p1_element]))

    kinds = {conflict["kind"] for conflict in fused["conflicts"]}
    assert "section_conflict" in kinds or "text_divergence" in kinds
    merged = [element for element in fused["elements"] if element.get("type") == "paragraph"]
    assert merged, "融合后正文元素不能消失"
    # 无论采用谁的 path，两侧 path 都必须可追溯。
    element = merged[0]
    recorded = json.dumps(element, ensure_ascii=False) + json.dumps(fused["conflicts"], ensure_ascii=False)
    assert "1 Introduction" in recorded
    # 完全静默（既无冲突记录也无 P0 path 痕迹）才算失败——上面已断言。


def test_unresolved_reference_alignment_is_recorded():
    """P1 的参考文献无法与 P0 对齐时，必须进入未决冲突，而不是静默合并（§7.6）。"""

    # 放在不同页 + 不同 bbox，确保它不会与 P0 元素发生几何对齐。
    p1_ref = parsed(
        "p1-ref",
        "reference_entry",
        "[9] Z. Author, Extra entry, 2024.",
        role="reference",
        page=3,
        bbox=[40.0, 500.0, 300.0, 520.0],
    )
    fused = fusion.fuse_documents(dict(_p0_doc()), mini_doc([p1_ref], document_id="p1"))
    kinds = {conflict["kind"] for conflict in fused["unresolved_conflicts"]}
    assert "REFERENCE_ALIGNMENT_UNCERTAIN" in kinds
    assert fused["reference_sidecar"]
    # P1 的条目不得混进正文检索集。
    assert "p1-ref" not in set(fused["retrieval_element_ids"])


# --------------------------------------------------------------------------------------
# Conditional OCR（§10）
# --------------------------------------------------------------------------------------


def test_conditional_ocr_only_on_bad_pages():
    """好页永不触发；空页 / 乱码页才触发。"""

    good = conditional_ocr_trigger(
        "This page has plenty of healthy born-digital text with normal token lengths.",
        has_visual_region=True,
        page_width_ratio=0.9,
    )
    assert good["trigger"] is False

    empty = conditional_ocr_trigger("", has_visual_region=True, page_width_ratio=0.9)
    assert empty["trigger"] is True
    assert "page_has_no_text_layer" in empty["reasons"]

    garbled = conditional_ocr_trigger(
        "\ufffd\ufffd\ufffd \ufffd\ufffd \ufffd\ufffd\ufffd\ufffd native text is broken",
        has_visual_region=False,
        page_width_ratio=0.0,
    )
    assert garbled["trigger"] is True
    assert "native_text_is_garbled" in garbled["reasons"]


# --------------------------------------------------------------------------------------
# 产物合同：BLOCKED ≠ FAIL；比例越界必须被抓
# --------------------------------------------------------------------------------------


def test_report_blocked_arm_status():
    """环境阻塞（BLOCKED）必须原样传递为 BLOCKED / pending，不得变成 FAIL。"""

    blocked_5a = {"gate_id": "5A", "name": "furniture_separability", "status": "BLOCKED", "evidence": {}}
    result = gates_v2.evaluate_hard_gates(
        arm_id="arm-x",
        canonical_anchor_hits=44,
        new_anchor_loss_count=0,
        unintended_body_loss_count=0,
        column_interleave_rate=0.0,
        gate_5a=blocked_5a,
        structure_availability={
            "type_counts": {"table_caption": 1},
            "table_count": 1,
            "caption_count": 1,
            "equation_count": 1,
        },
        reference_gate={"gate_id": "REF", "name": "reference_preservation", "status": "PASS", "evidence": {}},
        min_page_provenance_coverage=1.0,
        structure_bbox_coverage=1.0,
        reproducible=True,
        external_llm_declaration="none",
        sandbox_integrity={"intact": True},
    )
    by_id = {gate["gate_id"]: gate for gate in result["gates"]}
    assert by_id["B4"]["status"] == "BLOCKED"
    assert "B4" in result["pending_gates"]
    assert "B4" not in result["failed_gates"]
    # 非法状态字符串必须被拒绝（防止渲染层把 BLOCKED 写成 FAIL 之类的静默变形）。
    with pytest.raises(ValueError):
        gates_v2._gate("X", "n", "Blocked", {})


def test_report_metric_range():
    """比例型指标越界（如 precision > 1）必须被机械抓到——mutation validation。"""

    bad_payload = {
        "references": {"reference_entry_precision": 1.5, "reference_entry_recall": 0.9},
        "furniture": {"body_matched_to_furniture_rate": 2.0},
        "body": {"body_token_coverage": 1.0},
    }
    violations = cleaning_metrics.metric_range_violations(bad_payload)
    flagged = {violation["metric"] for violation in violations}
    assert "references.reference_entry_precision" in flagged
    assert "furniture.body_matched_to_furniture_rate" in flagged

    clean_payload = {
        "references": {"reference_entry_precision": 1.0, "reference_entry_recall": 0.0},
        "body": {"body_token_coverage": 0.5},
    }
    assert cleaning_metrics.metric_range_violations(clean_payload) == []


# --------------------------------------------------------------------------------------
# 确定性：清洗是纯函数（同输入同 digest），这是硬门 B8 的前提
# --------------------------------------------------------------------------------------


def test_clean_document_is_deterministic():
    """同输入两次清洗产生逐字段一致的文档（B8 可复现门的基础）。"""

    document = mini_doc(
        [
            parsed("p-1", "paragraph", "Deterministic cleaning keeps digests stable.", order=0),
            parsed("e-1", "equation", "$a ^ { 2 } + b ^ { 2 } = c ^ { 2 }$", order=1),
        ]
    )
    first = cleaning.clean_document(mini_doc(list(document["elements"])))
    second = cleaning.clean_document(mini_doc(list(document["elements"])))
    assert first["elements"] == second["elements"]
    assert first["retrieval_element_ids"] == second["retrieval_element_ids"]
