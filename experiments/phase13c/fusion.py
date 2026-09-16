"""Phase 13C-B B2 Fusion：基于 provenance 把 P0 的正文与 P1 的结构合成一个 ScientificDocument。

## 为什么是 provenance 对齐，而不是文本猜测

方案 §6 的硬要求：融合键必须是 ``document_id / page / bbox / reading_order /
normalized text overlap / section context / element type``，优先级依次为

    same page -> bbox overlap/containment -> normalized text overlap -> section proximity

原因很实际：受控语料里图、表、公式的 **GT 文本为空**（图是矢量绘制的，表格内容在结构化
字段里），对这类元素做文本匹配在构造上恒为 0（13C-A 的教训）。因此几何（同页 bbox）
必须排在文本之前，否则结构元素完全无法对齐。

## 不重复内容的硬规则

融合最容易犯的错误是"把同一段正文放两次"。本模块用三条规则从构造上堵住它：

1. **P1-only 的正文型元素一律不加入**（``suppressed``），只记录在案。正文由 P0 独家提供。
2. **P1-only 的参考文献一律不加入**（只进 sidecar）：P0 与 P1 的条目切分方式不同，
   强行合只会让同一条文献出现两次。方案 §7.6 要求"如无法可靠对齐 ->
   ``REFERENCE_ALIGNMENT_UNCERTAIN``，不静默合并"，这里就是它的实现。
3. **加入前先查重**：P1-only 元素若其文本被已入列元素覆盖 ≥ :data:`DUPLICATE_COVERAGE`，
   则丢弃并记 ``DUPLICATE_SUPPRESSED``。

## 冲突不静默覆盖

任何"两侧都对同一处内容给出了不同答案"的情形都会写入 ``conflicts``，并带
``severity``：``INFO``（差异已被确定性规则解决且不影响证据）/ ``RESOLVED``（有明确
优先级规则）/ ``UNRESOLVED``（无法可靠裁决，必须暴露给下游）。§35 明确禁止"P0/P1
冲突被静默覆盖"。
"""

from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

from . import align, cleaning, metrics, schema

FUSION_VERSION = "phase13c-b-fusion-1.0.0"

# 几何对齐阈值：与 13C-A 的图/表检出判据保持一致（可比较）。
ALIGN_IOU_THRESHOLD = 0.5
# 文本对齐阈值：多重集合 F1。
ALIGN_TEXT_THRESHOLD = 0.60
# 几何弱对齐 + 文本中等对齐也可接受（表/图被 P0 拆成多块时的常见情形）。
ALIGN_WEAK_IOU = 0.20
ALIGN_WEAK_TEXT = 0.40
# A 的文本被"已入列元素"覆盖到该比例以上即视为重复内容。
DUPLICATE_COVERAGE = 0.90
# 落盘记录上限（计数不受影响）。
CONFLICT_RECORD_LIMIT = 60
SUPPRESSED_RECORD_LIMIT = 200

# 冲突严重度。
SEVERITY_INFO = "INFO"
SEVERITY_RESOLVED = "RESOLVED"
SEVERITY_UNRESOLVED = "UNRESOLVED"

# 结构型类型（融合时优先采信 P1）。
_STRUCTURE_PREFER_P1 = frozenset(
    {"table", "table_caption", "figure", "figure_caption", "equation", "section_header", "abstract"}
)
# 正文型类型（融合时优先采信 P0，方案 §7.1）。
_BODY_PREFER_P0 = frozenset({"paragraph", "list_item", "document_title"})


def element_text(element: Mapping[str, Any]) -> str:
    """取元素的比对文本（优先 ``normalized_text``），与 13C-A 的 ``align._text_of`` 同语义。

    本地实现一份而不是调私有函数：对齐/查重是本模块的核心判据，依赖一个下划线
    前缀的内部函数会让语义在重构时静默变化。
    """

    for key in ("normalized_text", "text", "raw_text"):
        value = element.get(key)
        if isinstance(value, str) and value.strip():
            return schema.collapse_whitespace(value)
    return ""


def element_tokens(element: Mapping[str, Any]) -> Any:
    """元素的 token 多重集合。"""

    return align.token_counter(element_text(element))


# --------------------------------------------------------------------------------------
# 对齐
# --------------------------------------------------------------------------------------


def align_documents(
    primary: Sequence[dict[str, Any]],
    secondary: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """provenance 对齐：返回一对一对齐记录（贪心、确定性、可复跑一致）。

    评分优先级严格按方案 §6：同页 + 几何重合优先，文本相似度只作补充。
    同分时用两侧 ``element_id`` 字典序打破平局，保证跨运行完全一致。
    """

    candidates: list[tuple[float, int, int, float, float, float]] = []
    for left_index, left in enumerate(primary):
        left_page = left.get("page")
        left_box = metrics.element_box(left)
        left_tokens = element_tokens(left)
        for right_index, right in enumerate(secondary):
            if left_page != right.get("page"):
                continue
            right_box = metrics.element_box(right)
            iou = metrics.bbox_iou(left_box, right_box)
            overlap = metrics.bbox_overlap_coefficient(left_box, right_box)
            text = align.multiset_f1(left_tokens, element_tokens(right))
            # 公式用重叠系数（GT/解析器框约定不同，IoU 会被系统性压低，13C-A 教训）。
            geometric = max(iou, overlap if _is_equation_pair(left, right) else 0.0)
            if not (
                iou >= ALIGN_IOU_THRESHOLD
                or text >= ALIGN_TEXT_THRESHOLD
                or (geometric >= ALIGN_WEAK_IOU and text >= ALIGN_WEAK_TEXT)
            ):
                continue
            score = round(0.5 * geometric + 0.5 * text, 6)
            candidates.append((score, left_index, right_index, iou, overlap, text))

    candidates.sort(
        key=lambda item: (
            -item[0],
            str(primary[item[1]].get("element_id")),
            str(secondary[item[2]].get("element_id")),
        )
    )
    left_taken: set[int] = set()
    right_taken: set[int] = set()
    records: list[dict[str, Any]] = []
    for score, left_index, right_index, iou, overlap, text in candidates:
        if left_index in left_taken or right_index in right_taken:
            continue
        left_taken.add(left_index)
        right_taken.add(right_index)
        left = primary[left_index]
        right = secondary[right_index]
        records.append(
            {
                "primary_index": left_index,
                "secondary_index": right_index,
                "primary_element_id": left.get("element_id"),
                "secondary_element_id": right.get("element_id"),
                "primary_type": left.get("type"),
                "secondary_type": right.get("type"),
                "page": left.get("page"),
                "score": score,
                "bbox_iou": round(iou, 6),
                "bbox_overlap_coefficient": round(overlap, 6),
                "text_f1": round(text, 6),
                "alignment_key": "same_page+geometry" if iou >= ALIGN_IOU_THRESHOLD else "same_page+text",
            }
        )
    records.sort(key=lambda record: record["primary_index"])
    return records


def _is_equation_pair(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    """两侧至少一侧是公式时启用"重叠系数"判据。"""

    return left.get("type") == "equation" or right.get("type") == "equation"


# --------------------------------------------------------------------------------------
# 融合
# --------------------------------------------------------------------------------------


def fuse_documents(
    primary: dict[str, Any],
    secondary: dict[str, Any],
    *,
    primary_arm: str = "P0_current",
    secondary_arm: str = "P1_mineru",
) -> dict[str, Any]:
    """把 P0 清洗文档与 P1 清洗文档融合为一个 ScientificDocument。

    ``primary`` 必须是 P0（正文权威来源），``secondary`` 是 P1（结构来源）。
    """

    primary_elements = list(primary.get("elements", []))
    secondary_elements = list(secondary.get("elements", []))
    records = align_documents(primary_elements, secondary_elements)

    by_primary = {record["primary_index"]: record for record in records}
    secondary_matched = {record["secondary_index"] for record in records}

    conflicts: list[dict[str, Any]] = []
    reference_sidecar: list[dict[str, Any]] = []
    suppressed: list[dict[str, Any]] = []

    fused: list[dict[str, Any]] = []
    for index, element in enumerate(primary_elements):
        record = by_primary.get(index)
        if record is None:
            fused.append(_stamp(dict(element), [primary_arm], "P0_ONLY", None, conflicts, reference_sidecar))
            continue
        other = secondary_elements[record["secondary_index"]]
        fused.append(
            _merge_pair(
                element, other, record, primary_arm, secondary_arm, conflicts, reference_sidecar
            )
        )

    # 已入列文本的 token 多重集合（按页维护），用于 P1-only 的查重。
    page_tokens: dict[int | None, Any] = {}
    for element in fused:
        page_tokens.setdefault(element.get("page"), {})
    _reindex_page_tokens(fused, page_tokens)

    covered_tokens: dict[int | None, Any] = {page: tokens.copy() for page, tokens in page_tokens.items()}
    for right_index, element in enumerate(secondary_elements):
        if right_index in secondary_matched:
            continue
        element_type = element.get("type")
        if element_type in {"reference_entry"} or element.get("retrieval_role") == "reference":
            reference_sidecar.append(
                {
                    "action": "REFERENCE_ALIGNMENT_UNCERTAIN",
                    "secondary_element_id": element.get("element_id"),
                    "page": element.get("page"),
                    "text_head": (element.get("normalized_text") or "")[:120],
                    "reason": "P1 的参考文献条目边界与 P0 不同，无法可靠一对一对齐；保留在 sidecar，不合并进正文。",
                }
            )
            continue
        if element_type in {"header", "footer", "page_number", "watermark", "sidebar"}:
            suppressed.append(
                {
                    "secondary_element_id": element.get("element_id"),
                    "type": element_type,
                    "reason": "装饰元素不进入检索正文（P0 侧已单独记录 removed_furniture）。",
                }
            )
            continue
        tokens = element_tokens(element)
        if not tokens:
            if element_type == "figure":
                # §5：figure 由 P1 提供。图元素天然无文本（P0 侧也没有对应元素可对齐），
                # 按"无文本即跳过"处理会让 B2 丢掉全部图形结构，因此作为 P1_ONLY
                # 结构元素保留（normalize_figure 的 figure 字典随元素走，provenance 完整）。
                added = _stamp(dict(element), [secondary_arm], "P1_ONLY", None, conflicts, reference_sidecar)
                fused.append(added)
                continue
            suppressed.append(
                {
                    "secondary_element_id": element.get("element_id"),
                    "type": element_type,
                    "reason": "无文本（空块）且未与 P0 元素对齐：无内容可校验，跳过以避免空元素。",
                }
            )
            continue
        if element_type in _BODY_PREFER_P0:
            # §7.1：正文由 P0 独家提供。P1 多出来的段落只登记，不加入。
            suppressed.append(
                {
                    "secondary_element_id": element.get("element_id"),
                    "type": element_type,
                    "reason": "P0 正文权威（方案 §7.1），P1-only 正文不加入以避免重复内容。",
                    "text_head": (element.get("normalized_text") or "")[:120],
                }
            )
            continue
        page = element.get("page")
        coverage = _coverage_against(tokens, covered_tokens.get(page, {}))
        if coverage >= DUPLICATE_COVERAGE:
            suppressed.append(
                {
                    "secondary_element_id": element.get("element_id"),
                    "type": element_type,
                    "reason": "DUPLICATE_SUPPRESSED：文本已被同页已入列元素覆盖 "
                    f"{coverage:.3f} ≥ {DUPLICATE_COVERAGE}。",
                    "text_head": (element.get("normalized_text") or "")[:120],
                }
            )
            continue
        added = _stamp(dict(element), [secondary_arm], "P1_ONLY", None, conflicts, reference_sidecar)
        fused.append(added)
        _accumulate_tokens(covered_tokens.setdefault(page, {}), tokens)

    _assign_reading_order(fused, primary_elements)
    fused = _renumber(fused)

    # 参考文献无法可靠对齐是**真正未决**的冲突（方案 §7.6）：必须出现在
    # unresolved_conflicts 里，而不是只躺在 sidecar 中。
    if reference_sidecar:
        conflicts.append(
            {
                "kind": "REFERENCE_ALIGNMENT_UNCERTAIN",
                "severity": SEVERITY_UNRESOLVED,
                "count": len(reference_sidecar),
                "resolution": "不合并：P0 的参考文献条目保持原样，P1 的条目进入 reference_sidecar 仅供后续元数据补测。",
            }
        )

    unresolved = [conflict for conflict in conflicts if conflict["severity"] == SEVERITY_UNRESOLVED]
    duplicates = _duplicate_pairs(fused)
    # 真实论文上 section_conflict 可达数百条（P0/P1 的标题集不同，几乎每个元素都会命中）。
    # 逐条落盘会让产物膨胀且难以阅读，因此**只截断"落盘记录"**，计数仍然是精确的。
    stored_conflicts = _cap_records_by_kind(conflicts, CONFLICT_RECORD_LIMIT)

    retrieval_ids = [
        element["element_id"]
        for element in fused
        if element.get("retrieval_role") in {"body", "structure", "reference"}
    ]
    return {
        "document_id": primary.get("document_id"),
        "title": primary.get("title") or secondary.get("title"),
        "metadata": {
            "fusion_version": FUSION_VERSION,
            "primary_arm": primary_arm,
            "secondary_arm": secondary_arm,
            "primary_cleaner": primary.get("metadata", {}).get("cleaner_version"),
            "secondary_cleaner": secondary.get("metadata", {}).get("cleaner_version"),
            "source_pdf": primary.get("metadata", {}).get("source_pdf"),
            "source_sha256": primary.get("metadata", {}).get("source_sha256"),
            "page_count": primary.get("page_count"),
            "title_source": "primary" if primary.get("title") else "secondary",
        },
        "page_count": primary.get("page_count"),
        "element_count": len(fused),
        "elements": fused,
        "retrieval_element_ids": retrieval_ids,
        "reference_element_ids": [
            element["element_id"] for element in fused if element.get("retrieval_role") == "reference"
        ],
        "removed_furniture": list(primary.get("removed_furniture", [])),
        "removed_elements": list(primary.get("removed_elements", [])),
        # 文档级清洗报告（如参考文献区检测结果）继承自 primary：融合层不重跑清洗，
        # 丢失它会让下游把 "reference_section_detected" 误读成 False（未采集 ≠ 检测失败）。
        "cleaning_report": dict(primary.get("cleaning_report") or {}),
        "conflicts": stored_conflicts,
        "conflict_counts": _count_by_kind(conflicts),
        "conflict_records": {
            "total": len(conflicts),
            "stored": len(stored_conflicts),
            "record_limit_per_kind": CONFLICT_RECORD_LIMIT,
            "note": "conflicts[] 只保留每类前 N 条（按 element 顺序，确定性）；精确计数见 conflict_counts / fusion_metrics.conflict_kind_counts。",
        },
        "unresolved_conflicts": unresolved,
        "reference_sidecar": _cap_records(reference_sidecar, SUPPRESSED_RECORD_LIMIT),
        "suppressed_secondary_elements": _cap_records(suppressed, SUPPRESSED_RECORD_LIMIT),
        "fusion_metrics": fusion_metrics(fused, records, conflicts, duplicates, suppressed),
    }


def _merge_pair(
    primary: Mapping[str, Any],
    secondary: Mapping[str, Any],
    record: Mapping[str, Any],
    primary_arm: str,
    secondary_arm: str,
    conflicts: list[dict[str, Any]],
    reference_sidecar: list[dict[str, Any]],
) -> dict[str, Any]:
    """融合一对已对齐元素（方案 §7.1–§7.5）。"""

    merged = dict(primary)
    primary_type = primary.get("type")
    secondary_type = secondary.get("type")

    primary_text = (primary.get("normalized_text") or "").strip()
    secondary_text = (secondary.get("normalized_text") or "").strip()

    if primary_text and secondary_text and primary_text != secondary_text:
        conflicts.append(
            {
                "kind": "text_divergence",
                "severity": SEVERITY_INFO if record["text_f1"] >= ALIGN_TEXT_THRESHOLD else SEVERITY_RESOLVED,
                "page": primary.get("page"),
                "primary_element_id": primary.get("element_id"),
                "secondary_element_id": secondary.get("element_id"),
                "text_f1": record["text_f1"],
                "resolution": "P0 正文为权威文本源（方案 §7.1）；P1 原文保留在 source_refs。",
                "primary_text_head": primary_text[:160],
                "secondary_text_head": secondary_text[:160],
            }
        )

    if primary_type != secondary_type:
        # 类型不一致本身要暴露出来：它决定了"这条内容的语义标签是谁说了算"。
        decided_by_p1 = secondary_type in _STRUCTURE_PREFER_P1
        conflicts.append(
            {
                "kind": "type_conflict",
                "severity": SEVERITY_RESOLVED if decided_by_p1 else SEVERITY_INFO,
                "page": primary.get("page"),
                "primary_element_id": primary.get("element_id"),
                "secondary_element_id": secondary.get("element_id"),
                "primary_type": primary_type,
                "secondary_type": secondary_type,
                "resolution": (
                    f"结构型内容采信 {secondary_arm} 的 {secondary_type}（方案 §7.3/§7.5）。"
                    if decided_by_p1
                    else f"保留 {primary_arm} 的 {primary_type}；P1 的 {secondary_type} 记为差异。"
                ),
            }
        )

    # 公式：双表示并存，禁止一个覆盖另一个（方案 §7.5）。
    if primary_type == "equation" or secondary_type == "equation":
        merged["equation"] = {
            "raw_text": (primary.get("equation") or {}).get("raw_text")
            or secondary.get("raw_text"),
            "normalized_latex": (secondary.get("equation") or {}).get("normalized_latex")
            or secondary.get("normalized_text"),
            "primary_normalized_text": primary_text,
            "equation_number": (primary.get("equation") or {}).get("equation_number")
            or (secondary.get("equation") or {}).get("equation_number"),
            "page": primary.get("page"),
            "bbox": primary.get("bbox") or secondary.get("bbox"),
            "section_path": list(primary.get("section_path") or secondary.get("section_path") or []),
            "nearby_paragraph_ids": [],
            "source_parsers": [primary_arm, secondary_arm],
            "dual_representation_preserved": True,
        }
        merged["type"] = "equation"
        merged["retrieval_role"] = "structure"

    # 表格：采信 P1 的结构，同时保留 P0 的原始文本用于 fallback 与校验（方案 §7.3）。
    if secondary_type == "table" or primary_type == "table":
        secondary_table = secondary.get("table") or {}
        primary_table = primary.get("table") or {}
        if secondary_table.get("structure_available"):
            merged["type"] = "table"
            merged["retrieval_role"] = "structure"
            merged["table"] = {
                **secondary_table,
                "primary_fallback_text": primary_text,
                "primary_structure": primary_table.get("raw_structure"),
                "source_parsers": [primary_arm, secondary_arm],
            }
            _accumulate_caption(merged, secondary.get("caption_text") or primary.get("caption_text"))

    # 图：元数据优先 P1（方案 §7.4）。
    if secondary_type == "figure":
        merged["type"] = "figure"
        merged["retrieval_role"] = "structure"
        merged["figure"] = {
            **(secondary.get("figure") or {}),
            "source_parsers": [primary_arm, secondary_arm],
        }
        _accumulate_caption(merged, secondary.get("caption_text"))

    # caption 文本：取"隔离度更高"的那一侧（方案 §7.4：择优保留 canonical text fidelity）。
    if primary_type in {"figure_caption", "table_caption"} or secondary_type in {
        "figure_caption",
        "table_caption",
    }:
        chosen, choice = _choose_caption(primary_text, secondary_text, primary_arm, secondary_arm)
        merged["caption_text"] = chosen
        merged["caption_choice"] = choice
        if secondary_text and primary_text and secondary_text != primary_text:
            conflicts.append(
                {
                    "kind": "caption_choice",
                    "severity": SEVERITY_RESOLVED,
                    "page": primary.get("page"),
                    "primary_element_id": primary.get("element_id"),
                    "secondary_element_id": secondary.get("element_id"),
                    "choice": choice,
                    "primary_text_head": primary_text[:160],
                    "secondary_text_head": secondary_text[:160],
                }
            )

    # 章节路径：优先 P1，但与 P0 不一致时必须显式记录（方案 §7.2）。
    primary_path = list(primary.get("section_path") or [])
    secondary_path = list(secondary.get("section_path") or [])
    if secondary_path:
        merged["section_path"] = secondary_path
    if primary_path != secondary_path:
        conflicts.append(
            {
                "kind": "section_conflict",
                "severity": SEVERITY_RESOLVED,
                "page": primary.get("page"),
                "primary_element_id": primary.get("element_id"),
                "secondary_element_id": secondary.get("element_id"),
                "primary_section_path": primary_path,
                "secondary_section_path": secondary_path,
                "resolution": "采信 P1 的 section path（方案 §7.2），两侧路径均已记录，未静默覆盖。",
            }
        )

    if not primary_text and secondary_text:
        # §7.1 的例外：P0 缺失且 P1 有高置信文本时才允许 P1 补充正文。
        merged.update(cleaning.normalize_body_text(secondary.get("raw_text") or secondary_text))
        conflicts.append(
            {
                "kind": "p0_missing_text_filled_by_p1",
                "severity": SEVERITY_RESOLVED,
                "page": primary.get("page"),
                "primary_element_id": primary.get("element_id"),
                "secondary_element_id": secondary.get("element_id"),
                "resolution": "P0 该处无文本，按方案 §7.1 例外由 P1 补齐。",
            }
        )

    merged = _stamp(
        merged,
        [primary_arm, secondary_arm],
        "ALIGNED",
        {
            "primary_element_id": primary.get("element_id"),
            "secondary_element_id": secondary.get("element_id"),
            "bbox_iou": record["bbox_iou"],
            "text_f1": record["text_f1"],
            "alignment_key": record["alignment_key"],
        },
        conflicts,
        reference_sidecar,
    )
    merged["source_refs"] = [
        {"arm": primary_arm, "element_id": primary.get("element_id"), "source_ref": primary.get("source_ref")},
        {
            "arm": secondary_arm,
            "element_id": secondary.get("element_id"),
            "source_ref": secondary.get("source_ref"),
        },
    ]
    return merged


def _accumulate_caption(element: dict[str, Any], caption: str | None) -> None:
    """把 caption 文本挂到宿主元素上（已存在时不覆盖，改为追加到 caption_candidates）。"""

    if not caption:
        return
    if not element.get("caption_text"):
        element["caption_text"] = caption
        return
    if element["caption_text"] != caption:
        element.setdefault("caption_candidates", [])
        if caption not in element["caption_candidates"]:
            element["caption_candidates"].append(caption)


def _caption_isolation_score(text: str) -> float:
    """caption 的"隔离度"：越接近单句，越像一条干净的 caption。

    13C-A 观测到的 caption 失败模式是：解析器把 caption 与紧随其后的正文段落
    合并（``T08`` 的 MinerU 输出把两句正文追加在 Figure 2 的 caption 后面）。
    因此这里用"超出第一句的句数"作为惩罚项，而不是用长度——长度会把
    "一条很长但完整的 caption"误判成被稀释。
    """

    stripped = schema.collapse_whitespace(text)
    if not stripped:
        return 0.0
    sentences = [part for part in re.split(r"(?<=[.!?])\s+", stripped) if part.strip()]
    return 1.0 / max(1, len(sentences))


def _choose_caption(
    primary_text: str, secondary_text: str, primary_arm: str, secondary_arm: str
) -> tuple[str, str]:
    """选择 caption 文本；返回 ``(选定文本, 决策说明)``。"""

    if not primary_text:
        return secondary_text, f"P0 无 caption，采用 {secondary_arm}"
    if not secondary_text:
        return primary_text, f"P1 无 caption，采用 {primary_arm}"
    primary_score = _caption_isolation_score(primary_text)
    secondary_score = _caption_isolation_score(secondary_text)
    if secondary_score > primary_score:
        return secondary_text, f"隔离度更高者胜出（{secondary_arm} {secondary_score:.3f} > {primary_arm} {primary_score:.3f}）"
    return primary_text, f"隔离度更高者胜出（{primary_arm} {primary_score:.3f} >= {secondary_arm} {secondary_score:.3f}）"


def _stamp(
    element: dict[str, Any],
    parsers: Sequence[str],
    status: str,
    alignment: dict[str, Any] | None,
    conflicts: list[dict[str, Any]],
    reference_sidecar: list[dict[str, Any]],
) -> dict[str, Any]:
    """补齐 §8 schema 的融合字段（``source_parsers`` / ``fusion_status`` / ``confidence``）。"""

    element["source_parsers"] = list(parsers)
    element["fusion_status"] = status
    element["confidence"] = None
    if alignment:
        element["alignment"] = alignment
    element.setdefault("parent_id", None)
    element.setdefault("children", [])
    element.setdefault("caption_of", element.get("caption_of"))
    element.setdefault("source_refs", [])
    if element.get("retrieval_role") is None:
        element["retrieval_role"] = cleaning.retrieval_role_for(element.get("type", "unknown"))
    return element


# --------------------------------------------------------------------------------------
# reading order / 查重
# --------------------------------------------------------------------------------------


def _assign_reading_order(
    fused: Sequence[dict[str, Any]], primary_elements: Sequence[dict[str, Any]]
) -> None:
    """以 P0 的阅读顺序为骨架插入 P1-only 元素，并在元素上写入 ``reading_order``。

    为什么不直接按 (page, y, x) 重排：P0 的双栏阅读顺序是 13C-A 已验证可靠的
    （双栏交叉率 0.0000）。按几何全局重排会把已验证的正确顺序换成一个未验证的启发式。
    因此这里只决定"P1-only 元素插在哪个 P0 元素之后"。
    """

    by_id = {element.get("element_id"): element for element in fused}
    backbone: dict[str | None, list[dict[str, Any]]] = {}
    assigned: set[str] = set()
    for element in fused:
        if element.get("fusion_status") != "P1_ONLY":
            continue
        backbone.setdefault(_nearest_primary(element, primary_elements), []).append(element)

    counter = 0
    for primary in primary_elements:
        element = by_id.get(primary.get("element_id"))
        if element is None:
            continue
        element["reading_order"] = counter
        assigned.add(element["element_id"])
        counter += 1
        for extra in sorted(
            backbone.pop(element["element_id"], []), key=lambda item: str(item.get("element_id"))
        ):
            extra["reading_order"] = counter
            assigned.add(extra["element_id"])
            counter += 1
    # 锚点不在 P0 序列里的 P1-only 元素（例如整页是图）：按 element_id 排序追加。
    for group in backbone.values():
        for extra in sorted(group, key=lambda item: str(item.get("element_id"))):
            extra["reading_order"] = counter
            assigned.add(extra["element_id"])
            counter += 1
    for element in fused:
        if element["element_id"] not in assigned:
            element["reading_order"] = counter
            counter += 1


def _nearest_primary(
    element: Mapping[str, Any], primary_elements: Sequence[Mapping[str, Any]]
) -> str | None:
    """为 P1-only 元素找最近的 P0 锚点（同页优先，其次按垂直距离）。"""

    page = element.get("page")
    box = metrics.element_box(element)
    best_id: str | None = None
    best_distance: float | None = None
    for candidate in primary_elements:
        if candidate.get("page") != page:
            continue
        candidate_box = metrics.element_box(candidate)
        if box is None or candidate_box is None:
            distance = abs((candidate.get("reading_order") or 0) - (element.get("reading_order") or 0))
        else:
            distance = abs(candidate_box[1] - box[1])
        if best_distance is None or distance < best_distance:
            best_id, best_distance = candidate.get("element_id"), distance
    if best_id is None:
        # 同页没有 P0 元素（例如整页是图）：挂到最后一个"页号更小"的 P0 元素上。
        earlier = [c for c in primary_elements if (c.get("page") or 0) <= (page or 0)]
        if earlier:
            best_id = earlier[-1].get("element_id")
    return best_id


def _renumber(fused: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """按 ``reading_order`` 稳定排序（同序号用 element_id 打破平局）。"""

    ordered = sorted(
        fused,
        key=lambda element: (
            element.get("reading_order") if element.get("reading_order") is not None else 10**9,
            str(element.get("element_id")),
        ),
    )
    for index, element in enumerate(ordered):
        element["reading_order"] = index
    return ordered


def _reindex_page_tokens(
    fused: Sequence[dict[str, Any]], page_tokens: dict[int | None, Any]
) -> None:
    """初始化"每页已入列 token 集合"。"""

    for element in fused:
        page = element.get("page")
        page_tokens.setdefault(page, {})
        _accumulate_tokens(page_tokens[page], element_tokens(element))


def _accumulate_tokens(target: Any, tokens: Any) -> None:
    """把 token 多重集合累加进目标（目标可能是普通 dict）。"""

    if hasattr(target, "update"):
        target.update(tokens)
    else:  # pragma: no cover - 防御性
        for key, value in tokens.items():
            target[key] = target.get(key, 0) + value


def _coverage_against(tokens: Any, corpus: Any) -> float:
    """``tokens`` 的 token 有多大比例已被 ``corpus`` 覆盖（含重复计数）。"""

    total = sum(tokens.values()) if hasattr(tokens, "values") else 0
    if total <= 0:
        return 1.0
    getter = corpus.get if hasattr(corpus, "get") else (lambda key, default=0: 0)
    hit = sum(min(count, getter(token, 0)) for token, count in tokens.items())
    return hit / total


def _count_by_kind(records: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """按 ``kind`` 统计记录数。"""

    counts: dict[str, int] = {}
    for record in records:
        key = str(record.get("kind"))
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _cap_records_by_kind(
    records: Sequence[dict[str, Any]], limit: int
) -> list[dict[str, Any]]:
    """每类最多保留 ``limit`` 条（按输入顺序，确定性）。"""

    kept: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    for record in records:
        key = str(record.get("kind"))
        counts[key] = counts.get(key, 0) + 1
        if counts[key] <= limit:
            kept.append(record)
    return kept


def _cap_records(records: Sequence[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    """整体截断到 ``limit`` 条。"""

    return list(records[:limit])


def _duplicate_pairs(fused: Sequence[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """找出融合结果里互相高度覆盖的元素对（同页、token 双向覆盖 ≥ 0.9）。

    必须区分两类，否则会把"解析器本来就重复输出"记成"融合引入了重复"：

    - ``fusion_introduced``：**至少一侧是 P1_ONLY**。这才是方案 §35 要拦的
      "fusion 出现 duplicate body 内容且无法去重"。P0 侧的元素全部来自解析器，
      它们之间的重复不是融合产生的。
    - ``pre_existing``：两侧都是 ALIGNED / P0_ONLY，即 P0 自己就输出了两份相同文本
      （实测 adapointr 上 P0 把表格单元格与表头都当作独立段落输出，出现
      ``Multi-Head Self-Attention`` ×2 这类重复）。这是**解析器层**的既存问题，
      记录但不作为融合的失败判据。

    只比较检索元素（body/structure/reference）：装饰元素的重复（如两页各有一个
    相同页眉文本）与内容重复无关。
    """

    by_page: dict[int | None, list[dict[str, Any]]] = {}
    for element in fused:
        if element.get("retrieval_role") not in {"body", "structure", "reference"}:
            continue
        if not element_text(element):
            continue
        by_page.setdefault(element.get("page"), []).append(element)

    result: dict[str, list[dict[str, Any]]] = {"fusion_introduced": [], "pre_existing": []}
    for page, elements in sorted(by_page.items(), key=lambda item: str(item[0])):
        counters = [element_tokens(element) for element in elements]
        for left in range(len(elements)):
            for right in range(left + 1, len(elements)):
                if not counters[left] or not counters[right]:
                    continue
                forward = align.multiset_recall(counters[left], counters[right])
                backward = align.multiset_recall(counters[right], counters[left])
                if forward < DUPLICATE_COVERAGE or backward < DUPLICATE_COVERAGE:
                    continue
                introduced = (
                    elements[left].get("fusion_status") == "P1_ONLY"
                    or elements[right].get("fusion_status") == "P1_ONLY"
                )
                result["fusion_introduced" if introduced else "pre_existing"].append(
                    {
                        "page": page,
                        "left_element_id": elements[left].get("element_id"),
                        "right_element_id": elements[right].get("element_id"),
                        "left_status": elements[left].get("fusion_status"),
                        "right_status": elements[right].get("fusion_status"),
                        "forward_recall": round(forward, 6),
                        "backward_recall": round(backward, 6),
                        "left_text_head": element_text(elements[left])[:120],
                    }
                )
    return result


def fusion_metrics(
    fused: Sequence[dict[str, Any]],
    records: Sequence[dict[str, Any]],
    conflicts: Sequence[dict[str, Any]],
    duplicates: Mapping[str, Sequence[dict[str, Any]]],
    suppressed: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """方案 §13 的融合特有指标。"""

    status_counts: dict[str, int] = {}
    for element in fused:
        key = str(element.get("fusion_status"))
        status_counts[key] = status_counts.get(key, 0) + 1
    total = max(1, len(fused))
    unresolved = [conflict for conflict in conflicts if conflict["severity"] == SEVERITY_UNRESOLVED]
    kind_counts: dict[str, int] = {}
    for conflict in conflicts:
        kind_counts[conflict["kind"]] = kind_counts.get(conflict["kind"], 0) + 1
    return {
        "fusion_version": FUSION_VERSION,
        "fused_element_count": len(fused),
        "aligned_element_count": status_counts.get("ALIGNED", 0),
        "primary_only_count": status_counts.get("P0_ONLY", 0),
        "secondary_only_count": status_counts.get("P1_ONLY", 0),
        "aligned_element_rate": round(status_counts.get("ALIGNED", 0) / total, 6),
        "suppressed_secondary_count": len(suppressed),
        "conflict_count": len(conflicts),
        "conflict_kind_counts": dict(sorted(kind_counts.items())),
        "unresolved_conflict_count": len(unresolved),
        # 融合引入的重复（§35 停止条件之一）。分子只数"至少一侧是 P1_ONLY"的元素对，
        # 因此与"P0 本来就重复输出"的计数处在不同的计数空间，两者不可相加。
        "duplicate_content_count": len(duplicates.get("fusion_introduced", [])),
        "duplicate_content_rate": round(len(duplicates.get("fusion_introduced", [])) / total, 6),
        "pre_existing_duplicate_pair_count": len(duplicates.get("pre_existing", [])),
        "duplicate_reading_note": (
            "duplicate_content_* 只统计融合引入的重复（至少一侧 fusion_status=P1_ONLY）；"
            "pre_existing_duplicate_pair_count 是 P0 自身输出的重复文本块对数，属解析器层既存问题，"
            "两者口径不同、不可相加。"
        ),
        "multi_source_element_count": sum(
            1 for element in fused if len(element.get("source_parsers") or []) > 1
        ),
    }
