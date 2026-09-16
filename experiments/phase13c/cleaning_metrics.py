"""Phase 13C-B Cleaning Benchmark 的指标层（方案 §12）。

本模块只做两件事：

1. **把旧门 5 的混合指标彻底拆开**（方案 §3）：
   - 装饰侧只度量"能否被识别 / 隔离 / 追溯"（:func:`furniture_separability_metrics`）；
   - 正文侧只度量"内容是否完整"（:func:`body_metrics`）。
   两者不再共用一个数字，因此不会再出现 13C-A 那种"门失败原因其实与装饰无关"的情形。

2. **补齐 13C-A 没有的度量**：正文 token 覆盖、无文本层页与真实正文损失的区分、
   跨章节误合并、参考文献保持、来源可追溯、融合可追溯。

## 两个必须显式区分的计数空间

- ``unintended_body_loss_count`` **只统计"解析器在该页确实产出了文本、但 GT 正文没被
  覆盖"的元素**。扫描件页（解析器整页零产出）单独记入 ``no_text_layer_gt_body_count``。
  这正是方案 §3 要求的修订：不得再把"扫描页无文本层"直接算作 native parser 的正文误删
  —— 那是"该 parser 没有 OCR 能力"，不是"它把正文删了"。
- ``reference_entry_recall`` 的分子分母都限定在 **reference_entry 类型**的两侧，
  避免 13C-A 出现过的 ``precision > 1``（分子按 GT 类型过滤、分母按 parsed 类型过滤，
  两侧不在同一计数空间）。
"""

from __future__ import annotations

import re
from typing import Any, Iterable, Mapping, Sequence

from . import align, metrics, schema

# 与 13C-A 对齐的阈值（同处一个计数空间才能比较）。
MATCH_THRESHOLD = align.MATCH_THRESHOLD
PARTIAL_THRESHOLD = align.PARTIAL_OVERLAP_THRESHOLD
# 跨章节误合并判定用的局部重叠阈值。
CROSS_SECTION_OVERLAP = 0.20
# "低覆盖"= 文档级 token 覆盖率低于它，内容真的缺失（与 metrics.BODY_LOSS_MEDIUM_COVERAGE 同值）。
LOW_COVERAGE = metrics.BODY_LOSS_MEDIUM_COVERAGE
# "保留"下限：覆盖达到它的未对齐元素视为内容仍在（与 metrics.BODY_LOSS_HIGH_COVERAGE 同值）。
PRESERVED_COVERAGE = metrics.BODY_LOSS_HIGH_COVERAGE

# 结构元素类型（用于 Gate B5）。
STRUCTURE_GATE_TYPES = ("table", "figure_caption", "table_caption", "figure", "equation")


def _looks_like_latex(text: str) -> bool:
    """判断文本是否含 LaTeX 命令（与 13C-A 同一判据，本地实现以免依赖私有函数）。"""

    return bool(re.search(r"\\[a-zA-Z]{2,}", text))


def _f1(precision: float | None, recall: float | None) -> float | None:
    """由 precision/recall 计算 F1；任一为 ``None`` 或两者和为 0 时返回 ``None``。"""

    if precision is None or recall is None:
        return None
    if precision + recall == 0:
        return 0.0
    return round(2 * precision * recall / (precision + recall), 6)


def _text(element: Mapping[str, Any]) -> str:
    for key in ("normalized_text", "text", "raw_text"):
        value = element.get(key)
        if isinstance(value, str) and value.strip():
            return schema.collapse_whitespace(value)
    return ""


def _tokens(text: str) -> Any:
    return align.token_counter(text)


def retrieval_elements(cleaned: Mapping[str, Any]) -> list[dict[str, Any]]:
    """取清洗后的检索元素（正文 + 结构 + 参考文献，不含装饰与丢弃项）。"""

    ids = set(cleaned.get("retrieval_element_ids") or [])
    return [element for element in cleaned.get("elements", []) if element.get("element_id") in ids]


def _coverage(tokens: Any, corpus: Any) -> float:
    """``tokens`` 被 ``corpus`` 覆盖的比例（含重复计数）。"""

    total = sum(tokens.values())
    if total <= 0:
        return 1.0
    return sum(min(count, corpus.get(token, 0)) for token, count in tokens.items()) / total


# --------------------------------------------------------------------------------------
# §12.1 正文 + §12.2 装饰
# --------------------------------------------------------------------------------------


def body_metrics(
    ground_truth: Sequence[dict[str, Any]],
    cleaned: Mapping[str, Any],
    alignment: Mapping[str, Any],
    *,
    no_text_layer_pages: Iterable[int] = (),
) -> dict[str, Any]:
    """正文保持：锚点相关以外的正文完整度（受控语料）。

    ``no_text_layer_pages`` 是**源 PDF 的页级属性**（整页无可抽取文本层），由调用方
    从 PDF 直接探测后传入。用它而不是"本 arm 在该页有没有产出"来判定，是因为后者会让
    「某 arm 在扫描页上只补了结构元素」被误判成「正文被删」。
    """

    body_gt = [element for element in ground_truth if element.get("type") in schema.BODY_TYPES]
    kept = retrieval_elements(cleaned)
    kept_tokens: Any = {}
    for element in kept:
        for token, count in _tokens(_text(element)).items():
            kept_tokens[token] = kept_tokens.get(token, 0) + count

    # 无文本层页（源 PDF 属性）；未提供时退回"本 arm 在该页有没有产出"。
    blank_pages = {int(page) for page in no_text_layer_pages}
    pages_with_output = {
        element.get("page") for element in kept if element.get("page") is not None
    }

    matched_ids = {
        match["gt_element_id"]
        for match in alignment["matches"]
        if match["similarity"] >= MATCH_THRESHOLD
    }
    # 判"丢失"的覆盖率下限：token 覆盖 >= PRESERVED_COVERAGE 的元素视为
    # 「内容仍在、只是被合并/拆分导致 1-1 对齐失败」，不计入误删——
    # 否则清洗层的段落合并会被误判成正文丢失（实测 coverage=1.0 也曾被计入）。
    preserved_unaligned: list[dict[str, Any]] = []
    uncovered: list[dict[str, Any]] = []
    no_text_layer: list[dict[str, Any]] = []
    low_coverage: list[dict[str, Any]] = []
    document_tokens = kept_tokens
    for element in body_gt:
        element_tokens = _tokens(_text(element))
        if not element_tokens:
            continue
        if element["element_id"] in matched_ids:
            continue
        coverage = _coverage(element_tokens, document_tokens)
        record = {
            "gt_element_id": element["element_id"],
            "type": element.get("type"),
            "page": element.get("page"),
            "coverage": round(coverage, 6),
        }
        if element.get("page") in blank_pages or (
            not blank_pages and element.get("page") not in pages_with_output
        ):
            no_text_layer.append(record)
            continue
        if coverage >= PRESERVED_COVERAGE:
            preserved_unaligned.append(record)
            continue
        uncovered.append(record)
        if coverage < LOW_COVERAGE:
            low_coverage.append(record)

    total_body_tokens = sum(sum(_tokens(_text(element)).values()) for element in body_gt)
    covered_body_tokens = sum(
        min(count, document_tokens.get(token, 0))
        for element in body_gt
        for token, count in _tokens(_text(element)).items()
    )
    return {
        "gt_body_element_count": len(body_gt),
        "cleaned_retrieval_element_count": len(kept),
        "paragraph_text_recall": alignment.get("match_rate"),
        # 正文 token 覆盖：与"未被对齐"分开，因为它能区分"被拆到多处"与"真的没了"。
        "body_token_coverage": (
            round(covered_body_tokens / total_body_tokens, 6) if total_body_tokens else None
        ),
        "body_gt_token_total": total_body_tokens,
        "body_gt_token_covered": covered_body_tokens,
        "unmatched_body_element_count": len(uncovered) + len(preserved_unaligned),
        # 硬判据：解析器有产出、内容也确实不在检索正文里的正文元素数（真丢失）。
        "unintended_body_loss_count": len(uncovered),
        "unintended_body_loss_examples": uncovered[:12],
        # 诊断项：内容保留但 1-1 对齐失败（合并/拆分），与误删不可相加。
        "preserved_but_unaligned_count": len(preserved_unaligned),
        "preserved_but_unaligned_examples": preserved_unaligned[:12],
        # 单独记账，不参与误删判定（方案 §3）。
        "no_text_layer_gt_body_count": len(no_text_layer),
        "low_coverage_body_element_count": len(low_coverage),
        "low_coverage_body_examples": low_coverage[:12],
        "no_text_layer_body_examples": no_text_layer[:6],
        "reading_note": (
            "unintended_body_loss_count 只统计「该页有文本层、且 token 覆盖 < "
            f"{PRESERVED_COVERAGE}」的正文元素；覆盖 >= {PRESERVED_COVERAGE} 的未对齐元素计入 "
            "preserved_but_unaligned_count（内容仍在，只是被合并/拆分），"
            "no_text_layer_gt_body_count 是源 PDF 整页无可抽取文本（扫描件）的 GT 正文数，"
            "三者口径不同，不可相加。"
        ),
    }


def furniture_separability_metrics(
    ground_truth: Sequence[dict[str, Any]],
    cleaned: Mapping[str, Any],
) -> dict[str, Any]:
    """Gate 5A —— 装饰可分性（与正文完整度完全解耦）。

    ## ``body_matched_to_furniture_rate`` 的语义（v2.2 修订）

    该指标度量的是**正文被误标为装饰**：GT 正文元素的 token 被「输出中装饰侧的
    文本池」（被删除的装饰记录 + 输出中装饰类型元素）覆盖 >= 0.90 时计 1。
    13C-A 与方案 §3 的通过条件（``body_matched_to_furniture_rate == 0``）都是这个
    方向——即 Cleaner 不得把正文当噪声删掉。

    首版实现误测成了反方向（GT 装饰文本是否泄漏进检索正文），且用全语料 token
    覆盖做判据：页码这类**单 token 装饰**必然在正文某处出现、页眉常与论文标题同文，
    导致已正确删除装饰的 arm 也被判 100% 泄漏（计数空间缺陷）。泄漏方向仍有诊断
    价值，保留为 ``furniture_text_retained_in_body_rate``，但**不参与门判定**，
    并附带解读注意事项。
    """

    gt_furniture = [
        element for element in ground_truth if element.get("type") in schema.FURNITURE_TYPES
    ]
    gt_body = [element for element in ground_truth if element.get("type") in schema.BODY_TYPES]
    parsed_furniture = [
        element for element in cleaned.get("elements", []) if element.get("type") in schema.FURNITURE_TYPES
    ]
    removed = list(cleaned.get("removed_furniture") or [])
    kept = retrieval_elements(cleaned)

    # 装饰侧文本池：被删除的装饰记录 + 输出中仍为装饰类型的元素。
    furniture_pool: Any = {}
    for record in removed:
        for token, count in _tokens(str(record.get("text") or "")).items():
            furniture_pool[token] = furniture_pool.get(token, 0) + count
    for element in parsed_furniture:
        for token, count in _tokens(_text(element)).items():
            furniture_pool[token] = furniture_pool.get(token, 0) + count

    # ① 硬判据方向：GT 正文被误标为装饰。
    misclassified: list[dict[str, Any]] = []
    for element in gt_body:
        element_tokens = _tokens(_text(element))
        if not element_tokens:
            continue
        coverage = _coverage(element_tokens, furniture_pool)
        if coverage >= 0.90:
            misclassified.append(
                {
                    "gt_element_id": element["element_id"],
                    "type": element.get("type"),
                    "page": element.get("page"),
                    "coverage_in_furniture_pool": round(coverage, 6),
                }
            )

    # ② 诊断方向（不参与门判定）：GT 装饰文本是否仍出现在检索正文里。
    kept_tokens: Any = {}
    for element in kept:
        for token, count in _tokens(_text(element)).items():
            kept_tokens[token] = kept_tokens.get(token, 0) + count
    retained: list[dict[str, Any]] = []
    for element in gt_furniture:
        element_tokens = _tokens(_text(element))
        if not element_tokens:
            continue
        coverage = _coverage(element_tokens, kept_tokens)
        if coverage >= 0.90:
            retained.append(
                {
                    "gt_element_id": element["element_id"],
                    "type": element.get("type"),
                    "page": element.get("page"),
                    "coverage_in_retrieval_body": round(coverage, 6),
                }
            )

    # 删除记录的完整性：page / bbox / reason / source_parser 四项齐全才算可追溯。
    traceable = 0
    for record in removed:
        if (
            record.get("page") is not None
            and record.get("bbox")
            and record.get("reason")
            and record.get("source_parser")
        ):
            traceable += 1

    declared = len(parsed_furniture)
    total_records = len(removed)
    if declared or total_records:
        traceability = "EXPLICIT_DETECTION" if declared else "TRACEABLE_FILTER_RECORD"
    elif gt_furniture:
        traceability = "UNTRACEABLE"
    else:
        traceability = "NO_FURNITURE_OBSERVED"
    return {
        "gt_furniture_count": len(gt_furniture),
        "gt_body_count": len(gt_body),
        "parsed_furniture_count": declared,
        "explicit_furniture_detection": declared > 0,
        "removed_furniture_record_count": total_records,
        "traceable_removed_furniture_count": traceable,
        "traceable_filtered_furniture_rate": (
            round(traceable / total_records, 6) if total_records else None
        ),
        "furniture_traceability": traceability,
        # 关键硬判据：正文被误标为装饰（应为 0）。
        "body_matched_to_furniture_count": len(misclassified),
        "body_matched_to_furniture_rate": (
            round(len(misclassified) / len(gt_body), 6) if gt_body else 0.0
        ),
        "body_matched_to_furniture_examples": misclassified[:12],
        # 诊断项：装饰文本仍留在检索正文里。单 token 装饰（页码）与"页眉=论文标题"
        # 会让它结构性偏高，只作观察，不参与门判定。
        "furniture_text_retained_in_body_count": len(retained),
        "furniture_text_retained_in_body_rate": (
            round(len(retained) / len(gt_furniture), 6) if gt_furniture else 0.0
        ),
        "furniture_text_retained_in_body_examples": retained[:12],
        "furniture_text_retained_note": (
            "该指标用全语料 token 覆盖度量，单 token 装饰（页码）与页眉=标题同文都会"
            "触发，不能区分「装饰未删除」与「同文出现在正文」。装饰是否被隔离请看"
            "parsed_furniture_count / removed_furniture_record_count / furniture_traceability。"
        ),
        "furniture_by_type": _count_types(gt_furniture),
        "removed_furniture_by_type": _count_types(removed),
    }


def _count_types(elements: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for element in elements:
        key = str(element.get("type"))
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


# --------------------------------------------------------------------------------------
# §12.3 章节
# --------------------------------------------------------------------------------------


def section_structure_metrics(
    ground_truth: Sequence[dict[str, Any]],
    cleaned: Mapping[str, Any],
    alignment: Mapping[str, Any],
) -> dict[str, Any]:
    """章节结构 + 跨章节误合并计数（后者在 13C-A 里是缺失的）。"""

    base = metrics.section_metrics(ground_truth, cleaned.get("elements", []), alignment)

    gt_indexed = {element["element_id"]: element for element in ground_truth}
    section_conflicts: list[dict[str, Any]] = []
    for match in alignment["matches"]:
        if match["similarity"] < MATCH_THRESHOLD or match["gt_type"] != "section_header":
            continue
        gt_path = list(gt_indexed[match["gt_element_id"]].get("section_path") or [])
        parsed_element = next(
            (
                element
                for element in cleaned.get("elements", [])
                if element.get("element_id") == match["parsed_element_id"]
            ),
            None,
        )
        parsed_path = list((parsed_element or {}).get("section_path") or [])
        if gt_path != parsed_path:
            section_conflicts.append(
                {
                    "gt_element_id": match["gt_element_id"],
                    "gt_path": gt_path,
                    "parsed_path": parsed_path,
                }
            )

    # 跨章节误合并：一个解析元素同时与 ≥2 个不同顶层章节的 GT 元素达到局部重叠阈值。
    kept = retrieval_elements(cleaned)
    gt_tokens = {element["element_id"]: _tokens(_text(element)) for element in ground_truth}
    cross_section: list[dict[str, Any]] = []
    for element in kept:
        element_tokens = _tokens(_text(element))
        if not element_tokens:
            continue
        sections: dict[str, set[str]] = {}
        for gt_element in ground_truth:
            if gt_element.get("type") not in schema.BODY_TYPES | set(STRUCTURE_GATE_TYPES):
                continue
            tokens = gt_tokens.get(gt_element["element_id"])
            if not tokens:
                continue
            if align.multiset_f1(element_tokens, tokens) < CROSS_SECTION_OVERLAP:
                continue
            path = list(gt_element.get("section_path") or [])
            top = path[0] if path else ""
            sections.setdefault(top, set()).add(gt_element["element_id"])
        if len(sections) >= 2:
            cross_section.append(
                {
                    "parsed_element_id": element.get("element_id"),
                    "page": element.get("page"),
                    "sections": sorted(sections),
                    "type": element.get("type"),
                }
            )

    return {
        **base,
        "section_conflict_count": len(section_conflicts),
        "section_conflicts": section_conflicts[:15],
        "cross_section_merge_count": len(cross_section),
        "cross_section_merge_examples": cross_section[:12],
    }


# --------------------------------------------------------------------------------------
# §12.4–§12.6 结构元素
# --------------------------------------------------------------------------------------


def structure_metrics(
    ground_truth: Sequence[dict[str, Any]],
    cleaned: Mapping[str, Any],
    alignment: Mapping[str, Any],
) -> dict[str, Any]:
    """表格 / 图注 / 公式的指标，并额外披露"原始 LaTeX 约定可比性"。

    13C-A 的 ``latex_convention.comparable`` 是在"当轮输出文本"上算的。本阶段 Cleaner
    会规范化数学跨度，若仍按规范化后的文本判定，可比性结论会被 Cleaner 自己改变。
    因此这里**额外**给出基于 ``raw_text`` 的 ``latex_convention_raw``，并规定：
    能力结论以 raw 为准，规范化后的文本指标只作为"清洗是否提升可比性"的观测。
    """

    elements = cleaned.get("elements", [])
    equation_payload = metrics.equation_metrics(ground_truth, elements, alignment)
    raw_equations = [
        element for element in elements if element.get("type") == "equation"
    ]
    gt_equations = [element for element in ground_truth if element.get("type") == "equation"]
    payload = {
        "table": metrics.table_metrics(ground_truth, elements, alignment),
        "figure_caption": metrics.figure_caption_metrics(ground_truth, elements, alignment),
        "equation": equation_payload,
        "latex_convention_raw": {
            "gt_latex_like_count": sum(
                1 for element in gt_equations if _looks_like_latex(_text(element))
            ),
            "parsed_latex_like_count": sum(
                1 for element in raw_equations if _looks_like_latex(element.get("raw_text") or "")
            ),
            "comparable": bool(
                sum(1 for element in gt_equations if _looks_like_latex(_text(element)))
            )
            == bool(
                sum(
                    1
                    for element in raw_equations
                    if _looks_like_latex(element.get("raw_text") or "")
                )
            ),
            "note": (
                "基于 raw_text 的可比性（Cleaner 之前的形态）。能力结论以本字段为准；"
                "equation.latex_convention 基于清洗后文本，只反映清洗是否让两侧约定接近。"
            ),
        },
    }
    return payload


def structure_availability(cleaned: Mapping[str, Any]) -> dict[str, Any]:
    """Gate B5：结构元素是否真的可用（而不是全部退化成纯文本）。"""

    counts = _count_types(retrieval_elements(cleaned))
    table_elements = [
        element
        for element in retrieval_elements(cleaned)
        if element.get("type") == "table"
    ]
    structured_tables = sum(
        1 for element in table_elements if (element.get("table") or {}).get("structure_available")
    )
    figure_elements = [
        element for element in retrieval_elements(cleaned) if element.get("type") == "figure"
    ]
    figures_with_caption = sum(
        1 for element in figure_elements if element.get("caption_text") or element.get("caption_host_id")
    )
    equations = [
        element for element in retrieval_elements(cleaned) if element.get("type") == "equation"
    ]
    equations_with_dual = sum(
        1 for element in equations if (element.get("equation") or {}).get("raw_text")
    )
    return {
        "type_counts": counts,
        "table_count": len(table_elements),
        "structured_table_count": structured_tables,
        "figure_count": len(figure_elements),
        "figure_with_caption_count": figures_with_caption,
        "equation_count": len(equations),
        "equation_with_dual_representation_count": equations_with_dual,
        "structured_types_present": [
            element_type for element_type in STRUCTURE_GATE_TYPES if counts.get(element_type)
        ],
        "not_all_plain_text": any(counts.get(element_type) for element_type in STRUCTURE_GATE_TYPES),
    }


# --------------------------------------------------------------------------------------
# §12.7 参考文献
# --------------------------------------------------------------------------------------


def reference_preservation_metrics(
    ground_truth: Sequence[dict[str, Any]],
    cleaned: Mapping[str, Any],
    alignment: Mapping[str, Any],
    *,
    no_text_layer_pages: Iterable[int] = (),
) -> dict[str, Any]:
    """参考文献保持：条目切分、污染、误合并与碎片化。

    v2.2 修订（测量语义，不是放宽阈值）：

    - **误合并 / 碎片** 改用 token 覆盖 >= 0.90 判定。原实现用 multiset_f1 >= 0.50，
      而合成语料的参考文献条目互为近似重复（"[13] ... part 13," vs "[53] ... part 53,"
      的 f1 = 0.826），导致 T11 有 60/72 条被误判为"合并"。
    - **污染** 改为「GT 参考文献条目的内容（覆盖 >= 0.90）出现在 parsed 非参考文献
      元素里」。原实现用"正文元素 token 与**全部** GT 参考 token 并集的重叠率 >= 0.30"，
      合成语料里正文与参考文献共享大量词汇（MinerU/Docling/retrieval/年份等），
      普通带引用的正文段落也被判成污染。
    - **扫描页范围**：GT 参考文献条目所在页全部无文本层时（如 T12 之于 native
      parser），解析器根本没有提取到参考文献，Cleaner 无从保护——这是解析器能力
      边界而非清洗失败。此时 recall 如实置 ``None`` 并标
      ``reference_gate_scope = NO_TEXT_LAYER``，门判定跳过该文档（与 5B 的规则一致）。
    """

    blank_pages = {int(page) for page in no_text_layer_pages}
    gt_refs_all = [element for element in ground_truth if element.get("type") == "reference_entry"]
    # 范围判定：GT 参考条目全部位于无文本层页 → NO_TEXT_LAYER（解析器能力边界，
    # Cleaner 无从保护）；部分在文本层页 → 只对文本层页上的条目判定。
    if not gt_refs_all:
        scope = "NO_GT_REFS"
        gt_refs: list[dict[str, Any]] = []
    elif blank_pages and all(element.get("page") in blank_pages for element in gt_refs_all):
        scope = "NO_TEXT_LAYER"
        gt_refs = []
    else:
        scope = "EVALUATED"
        gt_refs = [element for element in gt_refs_all if element.get("page") not in blank_pages]
    parsed_refs = [
        element for element in cleaned.get("elements", []) if element.get("type") == "reference_entry"
    ]
    gt_tokens = {element["element_id"]: _tokens(_text(element)) for element in gt_refs}

    tp = sum(
        1
        for match in alignment["matches"]
        if match["similarity"] >= MATCH_THRESHOLD and match["gt_type"] == "reference_entry"
    )

    # 误合并：一个 parsed ref 的 token 覆盖了 >=2 个 GT ref 的 >=90%（条目被粘在一起）。
    # 覆盖判据使近似重复的相邻条目不再互相触发（缺对方的编号 token，覆盖到不了 0.90）。
    merged = 0
    fragments = 0
    for element in parsed_refs:
        element_tokens = _tokens(_text(element))
        if not element_tokens:
            continue
        covered = [
            gt_id
            for gt_id, tokens in gt_tokens.items()
            if tokens and _coverage(tokens, element_tokens) >= PRESERVED_COVERAGE
        ]
        if len(covered) >= 2:
            merged += 1
    for gt_id, tokens in gt_tokens.items():
        if not tokens:
            continue
        pieces = sum(
            1
            for element in parsed_refs
            if _tokens(_text(element))
            and _coverage(tokens, _tokens(_text(element))) >= PRESERVED_COVERAGE
        )
        if pieces >= 2:
            fragments += 1

    # 污染：GT 参考条目的内容出现在 parsed 非参考文献检索元素里（条目被塞回正文）。
    body_like_elements = [
        element
        for element in cleaned.get("elements", [])
        if element.get("type") != "reference_entry"
        and element.get("retrieval_role") in {"body", "structure", None}
    ]
    polluted_gt: list[dict[str, Any]] = []
    for gt_id, tokens in gt_tokens.items():
        if not tokens:
            continue
        for element in body_like_elements:
            body_tokens = _tokens(_text(element))
            if not body_tokens:
                continue
            coverage = _coverage(tokens, body_tokens)
            if coverage >= PRESERVED_COVERAGE:
                polluted_gt.append(
                    {
                        "gt_element_id": gt_id,
                        "parsed_element_id": element.get("element_id"),
                        "reference_coverage_in_body_element": round(coverage, 6),
                    }
                )
                break

    report = cleaned.get("cleaning_report", {}).get("references", {}) or {}
    return {
        "reference_gate_scope": scope,
        "gt_reference_entry_count": len(gt_refs_all),
        "gt_reference_entry_on_text_layer_count": len(gt_refs),
        "parsed_reference_entry_count": len(parsed_refs),
        "reference_section_detected": bool(report.get("reference_section_detected")),
        "reference_detection_method": report.get("detection_method"),
        # 分子分母都限定 reference_entry 两侧，构造上恒 ≤ 1。
        "reference_entry_recall": (
            round(tp / len(gt_refs), 6) if gt_refs else None
        ),
        "reference_entry_precision": (
            round(tp / len(parsed_refs), 6) if parsed_refs else None
        ),
        "reference_entry_f1": _f1(
            round(tp / len(parsed_refs), 6) if parsed_refs else None,
            round(tp / len(gt_refs), 6) if gt_refs else None,
        ),
        "reference_merge_count": merged,
        "reference_merge_rate": round(merged / len(parsed_refs), 6) if parsed_refs else None,
        "reference_fragment_count": fragments,
        "reference_fragment_rate": round(fragments / len(gt_refs), 6) if gt_refs else None,
        "reference_pollution_into_body_count": len(polluted_gt),
        "reference_pollution_into_body_rate": (
            round(len(polluted_gt) / len(gt_refs), 6) if gt_refs else None
        ),
        "reference_pollution_examples": polluted_gt[:12],
        "citation_callout_preservation": base_callout_preservation(alignment, ground_truth, cleaned),
        "reference_element_share": (
            round(len(parsed_refs) / max(1, len(retrieval_elements(cleaned))), 6)
        ),
    }


def base_callout_preservation(
    alignment: Mapping[str, Any],
    ground_truth: Sequence[dict[str, Any]],
    cleaned: Mapping[str, Any],
) -> dict[str, Any]:
    """引用标注保留（citation callout）：GT 正文里的 [n] 标注是否在输出中保留。

    从 13C-A 的 ``metrics.reference_metrics`` 中拆出（该函数其余部分被本模块的
    v2.2 口径替代），保持同一计数方式：对齐上的 GT 正文元素逐个检查其 callout。
    """

    from phase13c.schema import collapse_whitespace as _cw

    callout_total = 0
    callout_kept = 0
    callout_missing: list[dict[str, Any]] = []
    parsed_by_id = {
        element.get("element_id"): element for element in cleaned.get("elements", [])
    }
    for match in alignment["matches"]:
        if match["similarity"] < MATCH_THRESHOLD:
            continue
        gt_element = next(
            (e for e in ground_truth if e.get("element_id") == match["gt_element_id"]), None
        )
        if gt_element is None:
            continue
        callouts = gt_element.get("citation_callouts") or []
        if not callouts:
            continue
        parsed_element = parsed_by_id.get(match.get("parsed_element_id"))
        parsed_text = _cw(_text(parsed_element or {}))
        for callout in callouts:
            callout_total += 1
            if callout in parsed_text:
                callout_kept += 1
            elif len(callout_missing) < 12:
                callout_missing.append({"gt_element_id": gt_element["element_id"], "callout": callout})
    return {
        "callout_total": callout_total,
        "callout_kept": callout_kept,
        "callout_missing_count": callout_total - callout_kept,
        "callout_missing_examples": callout_missing,
        "rate": round(callout_kept / callout_total, 6) if callout_total else None,
    }


# --------------------------------------------------------------------------------------
# §12.8 provenance
# --------------------------------------------------------------------------------------


def provenance_extended_metrics(cleaned: Mapping[str, Any]) -> dict[str, Any]:
    """在 13C-A 的 page/bbox 覆盖之外，增加"来源解析器"与"融合"两层的可追溯性。"""

    base = metrics.provenance_metrics(cleaned.get("elements", []))
    kept = retrieval_elements(cleaned)
    with_parser = [element for element in kept if element.get("source_parser")]
    with_ref = [element for element in kept if element.get("source_ref")]
    fused_ready = [
        element
        for element in kept
        if element.get("source_parsers")
        and element.get("fusion_status") in {"ALIGNED", "P0_ONLY", "P1_ONLY"}
    ]
    is_fusion = bool(cleaned.get("metadata", {}).get("fusion_version"))
    return {
        **base,
        "source_parser_traceability": (
            round(len(with_parser) / len(kept), 6) if kept else None
        ),
        "source_ref_traceability": round(len(with_ref) / len(kept), 6) if kept else None,
        # B0/B1 不是融合产物 -> None（"未采集"），而不是 0（"全都没有"）。
        "fusion_traceability": (
            round(len(fused_ready) / len(kept), 6) if is_fusion and kept else None
        ),
        "fusion_traceability_applicable": is_fusion,
    }


# --------------------------------------------------------------------------------------
# 组合入口
# --------------------------------------------------------------------------------------


def evaluate_cleaned_document(
    ground_truth: Sequence[dict[str, Any]],
    cleaned: Mapping[str, Any],
    *,
    no_text_layer_pages: Iterable[int] = (),
) -> dict[str, Any]:
    """对一个清洗后的受控语料文档计算方案 §12 的全部指标。"""

    alignment = align.align_elements(ground_truth, cleaned.get("elements", []))
    payload = {
        "element_count": len(cleaned.get("elements", [])),
        "retrieval_element_count": len(retrieval_elements(cleaned)),
        "alignment": {
            "match_rate": alignment["match_rate"],
            "complete_unit_rate": alignment["complete_unit_rate"],
            "fragment_count": len(alignment["fragments"]),
            "merged_count": len(alignment["merged"]),
            "missing_count": len(alignment["missing"]),
            "spurious_count": len(alignment["spurious"]),
        },
        "body": body_metrics(
            ground_truth, cleaned, alignment, no_text_layer_pages=no_text_layer_pages
        ),
        "furniture": furniture_separability_metrics(ground_truth, cleaned),
        "section": section_structure_metrics(ground_truth, cleaned, alignment),
        "structure": structure_metrics(ground_truth, cleaned, alignment),
        "structure_availability": structure_availability(cleaned),
        "references": reference_preservation_metrics(
            ground_truth, cleaned, alignment, no_text_layer_pages=no_text_layer_pages
        ),
        "provenance": provenance_extended_metrics(cleaned),
        "text_fidelity": metrics.text_fidelity_metrics(ground_truth, cleaned.get("elements", []), alignment),
        "paragraph_integrity": metrics.paragraph_integrity_metrics(
            ground_truth, cleaned.get("elements", []), alignment
        ),
        "reading_order": metrics.reading_order_metrics(
            ground_truth, cleaned.get("elements", []), alignment
        ),
        "validation": schema.validate_document(cleaned),
    }
    payload["metric_range_violations"] = metric_range_violations(payload)
    return payload


def evaluate_cleaned_real_document(
    cleaned: Mapping[str, Any],
    anchors: Sequence[dict[str, Any]],
    *,
    raw_document: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """对一篇真实论文计算 anchor / provenance / 正文保持（无 GT 元素级标注）。"""

    kept = retrieval_elements(cleaned)
    payload = {
        "element_count": len(cleaned.get("elements", [])),
        "retrieval_element_count": len(kept),
        "anchors": metrics.anchor_recall_metrics(anchors, kept),
        "provenance": provenance_extended_metrics(cleaned),
        "type_counts": _count_types(kept),
        "structure_availability": structure_availability(cleaned),
        "furniture": {
            "removed_furniture_record_count": len(cleaned.get("removed_furniture") or []),
            "removed_by_type": _count_types(cleaned.get("removed_furniture") or []),
            "traceable_filtered_furniture_rate": _traceable_rate(
                cleaned.get("removed_furniture") or []
            ),
        },
        "references": {
            "reference_entry_count": sum(
                1 for element in kept if element.get("type") == "reference_entry"
            ),
            "reference_section_detected": bool(
                (cleaned.get("cleaning_report", {}).get("references") or {}).get(
                    "reference_section_detected"
                )
            ),
        },
    }
    if raw_document is not None:
        raw_tokens = sum(
            sum(_tokens(_text(element)).values())
            for element in raw_document.get("elements", [])
        )
        kept_tokens = sum(sum(_tokens(_text(element)).values()) for element in kept)
        payload["body_token_retention"] = {
            "raw_token_count": raw_tokens,
            "retained_token_count": kept_tokens,
            # 无 GT 时，这是唯一可用的"正文没被清洗删掉"证据：清洗前后的 token 之比。
            "retention_rate": round(kept_tokens / raw_tokens, 6) if raw_tokens else None,
            "note": (
                "真实语料没有元素级 GT，故用「清洗后检索正文 token / 清洗前全量 token」"
                "作为正文保持的下界证据；装饰与丢弃项会被剔除，因此该值应略小于 1。"
            ),
        }
    return payload


def _traceable_rate(records: Sequence[Mapping[str, Any]]) -> float | None:
    if not records:
        return None
    ok = sum(
        1
        for record in records
        if record.get("page") is not None
        and record.get("bbox")
        and record.get("reason")
        and record.get("source_parser")
    )
    return round(ok / len(records), 6)


# --------------------------------------------------------------------------------------
# 不可能值守卫（方案 §35）
# --------------------------------------------------------------------------------------

# 这些键在语义上必须是比例；> 1 或 < 0 说明分子分母不在同一计数空间。
_RATIO_KEYS = (
    "body_matched_to_furniture_rate",
    "traceable_filtered_furniture_rate",
    "reference_entry_recall",
    "reference_entry_precision",
    "reference_entry_f1",
    "reference_merge_rate",
    "reference_fragment_rate",
    "section_header_precision",
    "section_header_recall",
    "section_header_f1",
    "section_path_accuracy",
    "body_token_coverage",
    "page_provenance_coverage",
    "bbox_provenance_coverage",
    "source_parser_traceability",
    "source_ref_traceability",
    "fusion_traceability",
    "paragraph_text_recall",
    "paragraph_text_precision",
    "canonical_anchor_recall",
    "table_detection_precision",
    "figure_detection_precision",
    "equation_detection_precision",
    "retention_rate",
)


def metric_range_violations(payload: Any, *, path: str = "") -> list[dict[str, Any]]:
    """递归找出"不可能值"：被明确定义为比例的字段落在 [0, 1] 之外。

    13C-A 曾出现 ``equation precision = 11.0``（分子按 GT 过滤、分母按 parser 计数）。
    本函数把这类越界变成机械可检的事实，而不是靠人眼在报告里发现。
    """

    violations: list[dict[str, Any]] = []
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            location = f"{path}.{key}" if path else str(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                if key in _RATIO_KEYS and not (0.0 <= float(value) <= 1.0):
                    violations.append({"metric": location, "value": value, "reason": "ratio out of [0,1]"})
            elif isinstance(value, (Mapping, list)):
                violations.extend(metric_range_violations(value, path=location))
    elif isinstance(payload, list):
        for index, item in enumerate(payload[:50]):
            violations.extend(metric_range_violations(item, path=f"{path}[{index}]"))
    return violations


# --------------------------------------------------------------------------------------
# arm 级聚合
# --------------------------------------------------------------------------------------


def aggregate(documents: Mapping[str, Mapping[str, Any]], keys: Sequence[str]) -> dict[str, Any]:
    """按文档对指定指标取均值；``None``（未采集）不参与均值，但仍单独报告出现次数。"""

    summary: dict[str, Any] = {}
    for key in keys:
        values = [
            document.get(key)
            for document in documents.values()
            if isinstance(document.get(key), (int, float)) and not isinstance(document.get(key), bool)
        ]
        missing = sum(1 for document in documents.values() if document.get(key) is None)
        summary[key] = round(sum(values) / len(values), 6) if values else None
        if missing:
            summary[f"{key}__not_collected_documents"] = missing
    return summary
