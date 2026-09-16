"""Phase 13C 评测指标库（方案 §11 / §25 / §30）。

全部函数都是**纯函数**：输入 GT、parser 输出、对齐结果或统计量，输出可 JSON 序列化的
dict；不读文件、不写文件、不调模型。因此它们可以被单元测试逐项固定。

指标组织严格对应方案章节，便于报告逐节引用：

- §11.1 Text Fidelity      -> :func:`text_fidelity_metrics` / :func:`anchor_recall_metrics`
- §11.2 Reading Order       -> :func:`reading_order_metrics`
- §11.3 Section Structure   -> :func:`section_metrics`
- §11.4 Paragraph Integrity -> :func:`paragraph_integrity_metrics`
- §11.5 Furniture Cleaning  -> :func:`furniture_metrics`
- §11.6 Figure / Caption    -> :func:`figure_caption_metrics`
- §11.7 Table               -> :func:`table_metrics`
- §11.8 Equation            -> :func:`equation_metrics`
- §11.9 References          -> :func:`reference_metrics`
- §11.10 Provenance         -> :func:`provenance_metrics`
- §30 Cost                  -> :func:`cost_metrics`

约定：无法计算的指标返回 ``None``（而不是 0）。理由是 0 会被误读成"质量为零"，
而 ``None`` 明确表示"该 parser 没有提供计算所需的信息"——这本身就是 Benchmark 结论。
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any, Iterable, Sequence

from .align import (
    MATCH_THRESHOLD,
    align_elements,
    count_gt_by_type,
    detect_column_interleave,
    kendall_tau,
    matched_pairs_for_reading_order,
    multiset_f1,
    multiset_recall,
    pair_order_accuracy,
    token_counter,
    type_confusion,
)
from .schema import (
    BODY_TYPES,
    FURNITURE_TYPES,
    collapse_whitespace,
    normalized_8grams,
    tokenize,
)

# 数值型 token，用于判断"header 是否与数据列对齐"这类结构关系。
_NUMBER = re.compile(r"^[+-]?(?:\d+(?:\.\d+)?|\.\d+)%?$")

# bbox IoU 检出阈值。IoU 是无量纲的比值，因此不依赖各 parser 的坐标尺度——
# 这正是在"坐标系各不相同"的前提下仍能公平比较的唯一几何判据。
#
# 取 0.5 的依据是原型实测的分布：真正对应的块（同一图/表的两个框）IoU 普遍在
# 0.80~0.98，而不对应的块大多为 0（少数为 0.09~0.14 的"同栏相邻"误配）。
# 0.5 落在两个簇之间，且远离双方，所以结果对阈值不敏感。
GEOMETRIC_IOU_THRESHOLD = 0.5

# 重叠系数判据（inter / min(area)）——**只用于公式**。
#
# 为什么公式不能用 IoU：GT 的公式框是**行级框**（占满整栏宽度，右侧还含公式编号
# "（3）"），而 parser 的框紧贴字形。同一处公式的两个框一个大一个小，
# 交并比会被"并集"分母系统性压低。实测（T01/T09/T10 × 4 arms，29 对）：
#
#   对应块（同页最近）      最低 0.840 / 中位 0.997   <- 见 bbox_overlap_coefficient
#   误配块（同页次近）      最高 0.412 / 中位 0.000
#   跨页误配                最高 0.253
#
# IoU 在同一批数据上只有 0.23~0.94（中位约 0.24），落在 0.5 阈值以下，
# 会把 Docling 这种真实检出了公式的 parser 误判为"没检出"。
# 重叠系数对"框大框小"不敏感（一个框完全含于另一个即为 1.0），
# 实测把对应侧与误配侧分成 0.412 / 0.840 两个相距 2 倍的簇。
# 取 0.6 落在两簇正中的空隙里，距两侧都最远（0.6/0.412 ≈ 1.46，0.840/0.6 = 1.40），
# 因此结果对阈值不敏感。
GEOMETRIC_OVERLAP_THRESHOLD = 0.6

# 「正文文本丢失」的成因分层阈值（用**文档级 token 覆盖率**）。
#
# 为什么需要这个分解：``furniture_false_removal`` 只给出"有多少 GT 正文元素没有对齐到
# 单个 parser 元素"这一个数，但它的成因完全不同，混在一起会误导归因。
# 实测（12 个受控模板 × 5 arm）：某 arm 的 236 个"丢失"里有 193 个（82%）其 token
# **全部出现在文档别处**——那是分块/合并差异，不是丢失；而另一个 arm 的 216 个里有
# 114 个（53%）覆盖率极低——那才是真的没抽出来。
#
# 判分层：
# - ``>= 0.9``：内容都在，只是没对齐到单个元素（拆分/合并/顺序差异）；
# - ``>= 0.5``：部分差异（截断、少量 token 变化）；
# - ``< 0.5``：内容真的缺失（未抽取、被丢弃、无文本层）。
BODY_LOSS_HIGH_COVERAGE = 0.9
BODY_LOSS_MEDIUM_COVERAGE = 0.5


# --------------------------------------------------------------------------------------
# 基础工具
# --------------------------------------------------------------------------------------


def _ratio(numerator: float, denominator: float) -> float | None:
    """安全除法：分母为 0 时返回 ``None`` 而不是抛异常或编造 0。"""

    if denominator <= 0:
        return None
    return round(numerator / denominator, 6)


def levenshtein_distance(left: str, right: str, *, max_length: int = 4000) -> tuple[int, bool]:
    """计算两串的编辑距离，返回 ``(distance, truncated)``。

    使用两行滚动 DP，空间 O(min(n,m))。超长输入按 ``max_length`` 截断并置位
    ``truncated``——这是显式的降级，而不是静默失真。
    """

    truncated = False
    if len(left) > max_length:
        left = left[:max_length]
        truncated = True
    if len(right) > max_length:
        right = right[:max_length]
        truncated = True
    if left == right:
        return 0, truncated
    if not left:
        return len(right), truncated
    if not right:
        return len(left), truncated

    previous = list(range(len(right) + 1))
    for i, left_char in enumerate(left, start=1):
        current = [i] + [0] * len(right)
        for j, right_char in enumerate(right, start=1):
            cost = 0 if left_char == right_char else 1
            current[j] = min(
                previous[j] + 1,        # 删除
                current[j - 1] + 1,     # 插入
                previous[j - 1] + cost, # 替换
            )
        previous = current
    return previous[-1], truncated


def character_error_rate(reference: str, hypothesis: str) -> dict[str, Any]:
    """字符错误率 = 编辑距离 / 参考长度。"""

    distance, truncated = levenshtein_distance(reference, hypothesis)
    return {
        "distance": distance,
        "reference_length": len(reference),
        "cer": _ratio(distance, len(reference)),
        "truncated": truncated,
    }


def token_prf(reference: str, hypothesis: str) -> dict[str, float]:
    """基于 token 多重集合的 precision / recall / f1。"""

    ref_counter = token_counter(reference)
    hyp_counter = token_counter(hypothesis)
    return {
        "precision": round(_multiset_precision(ref_counter, hyp_counter), 6),
        "recall": round(multiset_recall(ref_counter, hyp_counter), 6),
        "f1": round(multiset_f1(ref_counter, hyp_counter), 6),
    }


def _multiset_precision(reference: Counter[str], hypothesis: Counter[str]) -> float:
    """precision = 重叠 token / hypothesis token 数。

    等价于把两侧互换后的 recall，因此直接复用 :func:`multiset_recall`，
    避免同一套多重集合语义出现两份实现（一旦两份实现漂移，指标就不可信）。
    """

    return multiset_recall(hypothesis, reference)


def normalized_exact_match(reference: str, hypothesis: str) -> bool:
    """空白折叠后的严格相等判定（方案 §11.1 ``normalized_exact_match``）。"""

    return collapse_whitespace(reference) == collapse_whitespace(hypothesis)


# --------------------------------------------------------------------------------------
# 几何检出（figure / table 的正确判据）
# --------------------------------------------------------------------------------------


def element_box(element: dict[str, Any]) -> tuple[float, float, float, float] | None:
    """取出元素合法的 ``(x0, y0, x1, y1)``；缺失或退化时返回 ``None``。

    "退化"指宽或高非正（零面积）。这种框无法参与相交计算（IoU 恒为 0 或未定义），
    必须显式排除，而不是让它悄悄产生一个 0 分。
    """

    bbox = element.get("bbox")
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    try:
        x0, y0, x1, y1 = (float(value) for value in bbox)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in (x0, y0, x1, y1)):
        return None
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1, y1)


def bbox_iou(
    left: tuple[float, float, float, float] | None,
    right: tuple[float, float, float, float] | None,
) -> float:
    """两个矩形的交并比；任一缺失或不相交时返回 0.0。"""

    if left is None or right is None:
        return 0.0
    inter_x0 = max(left[0], right[0])
    inter_y0 = max(left[1], right[1])
    inter_x1 = min(left[2], right[2])
    inter_y1 = min(left[3], right[3])
    if inter_x1 <= inter_x0 or inter_y1 <= inter_y0:
        return 0.0
    intersection = (inter_x1 - inter_x0) * (inter_y1 - inter_y0)
    area_left = (left[2] - left[0]) * (left[3] - left[1])
    area_right = (right[2] - right[0]) * (right[3] - right[1])
    union = area_left + area_right - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def bbox_overlap_coefficient(
    left: tuple[float, float, float, float] | None,
    right: tuple[float, float, float, float] | None,
) -> float:
    """两个矩形的重叠系数 ``inter / min(area)``；任一缺失或不相交时返回 0.0。

    与 :func:`bbox_iou` 的区别在于分母：IoU 用并集，对"一个大框配一个小框"很敏感
    （同一处内容也会被压到 0.2 附近）；重叠系数用较小者，一个框完全含于另一个时
    即为 1.0。当两侧的框**约定不同**（如公式的行级框 vs 字形框）时，
    重叠系数才是可比的那个量。
    """

    if left is None or right is None:
        return 0.0
    inter_x0 = max(left[0], right[0])
    inter_y0 = max(left[1], right[1])
    inter_x1 = min(left[2], right[2])
    inter_y1 = min(left[3], right[3])
    if inter_x1 <= inter_x0 or inter_y1 <= inter_y0:
        return 0.0
    intersection = (inter_x1 - inter_x0) * (inter_y1 - inter_y0)
    area_left = (left[2] - left[0]) * (left[3] - left[1])
    area_right = (right[2] - right[0]) * (right[3] - right[1])
    smaller = min(area_left, area_right)
    if smaller <= 0:
        return 0.0
    return intersection / smaller


def _pages_comparable(
    gt_page: Any, parsed_page: Any
) -> tuple[bool, bool]:
    """判断一对元素的页码是否允许比较，返回 ``(允许比较, 页码已验证)``。

    两侧页码都是 ``int`` 时必须相等（否则是跨页误配，直接否决）。
    只要有一侧缺失页码，就无法验证——此时**允许**比较并把 ``page_verified`` 置为
    ``False``，由调用方统计。

    为什么缺失页码时不直接否决：那样会让"输出了几何信息但没输出页码"的 parser
    在检出指标上被结构性判死，而它其实真的找到了这个元素——那属于 provenance 缺陷，
    应当由页面覆盖指标与报告归因去体现，而不是伪装成"没检出"。
    """

    if isinstance(gt_page, int) and isinstance(parsed_page, int):
        return (gt_page == parsed_page, True)
    return (True, False)


def geometric_detection(
    ground_truth: Sequence[dict[str, Any]],
    parsed: Sequence[dict[str, Any]],
    *,
    gt_type: str,
    parsed_type: str | None = None,
    criterion: str = "iou",
    iou_threshold: float | None = None,
) -> dict[str, Any]:
    """按**同页 bbox 几何重叠**判定 figure / table / equation 是否被检出。

    为什么不能沿用文本相似度：受控语料里的图是矢量绘制的，其 GT ``text`` 是空字符串；
    表格的 GT 内容则在结构化的 ``table_structure`` 中、扁平文本为空或只有片段；
    公式的 GT 是 asciiMath 纯文本而 MinerU 输出原始 LaTeX。对这三类元素做 token 匹配，
    相似度要么**在构造上**恒为 0，要么测的是"选了哪种输出约定"——都测不出"有没有检出"。
    实测该做法让所有 arm 的 ``figure_detection`` 都是 0.0（含 baseline），
    并让 MinerU 的 ``equation_detection_recall`` 被压到 0.071（而它其实检出了全部 11 个公式）。

    判据 ``criterion`` 有两种，因为两类元素的**框大小约定**不同：

    - ``"iou"``（默认，用于 figure / table）：两侧框大小相近，交并比可直接用；
    - ``"overlap"``（用于 equation）：GT 是行级框、parser 是字形框，
      IoU 被并集分母压低，改用重叠系数 ``inter / min(area)``。
      两者的实测分布与阈值选取理由见模块常量 ``GEOMETRIC_OVERLAP_THRESHOLD``。

    匹配规则：

    - 候选两侧都必须有**合法的** bbox（缺失/零面积不参与）；
    - 贪心一对一，得分降序，同分用 ``element_id`` 字典序破平局（跨运行确定性）；
    - 同页约束见 :func:`_pages_comparable`；
    - 阈值按 ``criterion`` 取默认值（``iou_threshold`` 显式给出时优先，保持向后兼容）。

    返回的 ``detection`` 以"有合法 bbox 的 GT 元素数"为分母——没有 bbox 的 GT 本来
    就无法做几何判定，把它计入分母等于编造一个失败。无 bbox 的数量单独报出，
    由调用方决定是否需要用文本判据补足。
    """

    if criterion not in {"iou", "overlap"}:
        raise ValueError(f"未知的几何判据：{criterion}")
    if iou_threshold is not None:
        threshold = iou_threshold
    else:
        threshold = (
            GEOMETRIC_IOU_THRESHOLD if criterion == "iou" else GEOMETRIC_OVERLAP_THRESHOLD
        )
    score_of = bbox_iou if criterion == "iou" else bbox_overlap_coefficient

    target_type = parsed_type or gt_type
    gt_candidates: list[tuple[dict[str, Any], tuple[float, float, float, float]]] = []
    for element in ground_truth:
        if element.get("type") != gt_type:
            continue
        box = element_box(element)
        if box is not None:
            gt_candidates.append((element, box))

    parsed_candidates: list[tuple[dict[str, Any], tuple[float, float, float, float]]] = []
    for element in parsed:
        if element.get("type") != target_type:
            continue
        box = element_box(element)
        if box is not None:
            parsed_candidates.append((element, box))

    pairs: list[tuple[float, int, int]] = []
    for gi, (_, gt_box) in enumerate(gt_candidates):
        gt_page = gt_candidates[gi][0].get("page")
        for pi, (parsed_element, parsed_box) in enumerate(parsed_candidates):
            comparable, _ = _pages_comparable(gt_page, parsed_element.get("page"))
            if not comparable:
                continue
            score = score_of(gt_box, parsed_box)
            if score >= threshold:
                pairs.append((score, gi, pi))

    # 同分时用 element_id 字典序破平局，保证同一输入永远给出同一组匹配。
    pairs.sort(
        key=lambda item: (
            -item[0],
            str(gt_candidates[item[1]][0].get("element_id")),
            str(parsed_candidates[item[2]][0].get("element_id")),
        )
    )
    gt_taken: set[int] = set()
    parsed_taken: set[int] = set()
    matches: list[dict[str, Any]] = []
    for score, gi, pi in pairs:
        if gi in gt_taken or pi in parsed_taken:
            continue
        gt_taken.add(gi)
        parsed_taken.add(pi)
        gt_element = gt_candidates[gi][0]
        gt_box = gt_candidates[gi][1]
        parsed_element = parsed_candidates[pi][0]
        parsed_box = parsed_candidates[pi][1]
        _, page_verified = _pages_comparable(gt_element.get("page"), parsed_element.get("page"))
        matches.append(
            {
                "gt_element_id": gt_element.get("element_id"),
                "parsed_element_id": parsed_element.get("element_id"),
                # 命中得分与判据名一起记录：公式用的是重叠系数、图/表用的是 IoU，
                # 都记成 "iou" 会让复核者按错的口径去理解数值。
                "score": round(score, 6),
                "criterion": criterion,
                # 同时保留原始 IoU，便于解释"为什么公式的 IoU 那么低却仍算命中"。
                "iou": round(bbox_iou(gt_box, parsed_box), 6),
                "page": gt_element.get("page"),
                "parsed_page": parsed_element.get("page"),
                "page_verified": page_verified,
            }
        )

    unverified = [m for m in matches if not m["page_verified"]]
    return {
        "gt_type": gt_type,
        "parsed_type": target_type,
        # 判据名与阈值必须一起报出：复核者据此判断该 arm 的分数是"交并比"还是
        # "重叠系数"口径，两者数值不可直接跨类型比较（公式 0.6 / 图·表 0.5）。
        "criterion": criterion,
        "threshold": threshold,
        "gt_count": sum(1 for e in ground_truth if e.get("type") == gt_type),
        "gt_with_bbox_count": len(gt_candidates),
        "parsed_with_bbox_count": len(parsed_candidates),
        "match_count": len(matches),
        "matched_gt_element_ids": [m["gt_element_id"] for m in matches],
        "detection": _ratio(len(matches), len(gt_candidates)),
        # 精确率（parser 视角）：匹配是严格一对一的，所以命中对数既是"命中 GT 数"，
        # 也是"命中 parser 元素数"。没有这一项，一个把图统统标成表的 parser 会在
        # detection（召回）上拿到满分而看不出问题——实测 Marker 在 T10（GT 2 表 5 图）
        # 输出了 6 个 Table 块，其中 4 个实际是图，召回 1.0 而精确率仅 0.333。
        "precision": _ratio(len(matches), len(parsed_candidates)),
        "page_unverified_match_count": len(unverified),
        "matches": matches,
    }


# --------------------------------------------------------------------------------------
# §11.1 Text Fidelity
# --------------------------------------------------------------------------------------


def text_fidelity_metrics(
    ground_truth: Sequence[dict[str, Any]],
    parsed: Sequence[dict[str, Any]],
    alignment: dict[str, Any],
    *,
    body_types: Iterable[str] = BODY_TYPES,
) -> dict[str, Any]:
    """受控语料的文本保真度：段落级 recall / precision / EM / CER。

    只在"正文型" GT 元素上计算（段落、列表项、标题、参考文献条目），因为表格/公式的
    文本表示取决于 parser 的序列化方式，放进同一个 CER 会掩盖真正的正文损失。
    段落级 CER 只统计**对齐成功**的配对，未对齐的段落由 recall 负责体现缺失。
    """

    body_type_set = set(body_types)
    gt_indexed = {i: element for i, element in enumerate(ground_truth)}
    gt_tokens = {i: token_counter(_text(element)) for i, element in enumerate(ground_truth)}

    total_gt_tokens = 0
    recovered_gt_tokens = 0
    total_parsed_tokens = 0
    overlapping_tokens = 0
    exact_matches = 0
    matched_body = 0
    cer_values: list[float] = []
    cer_detail: list[dict[str, Any]] = []
    type_confusion_body: dict[str, int] = {}

    aligned_parsed_indices = {m["parsed_index"] for m in alignment["matches"]}

    for match in alignment["matches"]:
        gi = match["gt_index"]
        pi = match["parsed_index"]
        if match["similarity"] < MATCH_THRESHOLD:
            continue
        gt_element = gt_indexed[gi]
        if gt_element.get("type") not in body_type_set:
            continue
        parsed_tokens = token_counter(_text(parsed[pi]))
        gt_counter = gt_tokens[gi]
        if not gt_counter:
            continue
        matched_body += 1
        total_gt_tokens += sum(gt_counter.values())
        total_parsed_tokens += sum(parsed_tokens.values())
        overlapping_tokens += sum((gt_counter & parsed_tokens).values())
        recovered_gt_tokens += sum((gt_counter & parsed_tokens).values())
        if normalized_exact_match(_text(gt_element), _text(parsed[pi])):
            exact_matches += 1
        cer = character_error_rate(
            collapse_whitespace(_text(gt_element)), collapse_whitespace(_text(parsed[pi]))
        )
        if cer["cer"] is not None:
            cer_values.append(cer["cer"])
            cer_detail.append(
                {
                    "gt_element_id": match["gt_element_id"],
                    "gt_type": gt_element.get("type"),
                    "parsed_type": parsed[pi].get("type"),
                    "cer": cer["cer"],
                    "truncated": cer["truncated"],
                }
            )
        gt_type = str(gt_element.get("type"))
        if parsed[pi].get("type") != gt_type:
            key = f"{gt_type}->{parsed[pi].get('type')}"
            type_confusion_body[key] = type_confusion_body.get(key, 0) + 1

    # 未对齐的正文 GT 是 recall 的损失来源。
    missing_body_tokens = 0
    for gi, element in gt_indexed.items():
        if element.get("type") not in body_type_set:
            continue
        if gi in {m["gt_index"] for m in alignment["matches"] if m["similarity"] >= MATCH_THRESHOLD}:
            continue
        missing_body_tokens += sum(gt_tokens[gi].values())

    # spurious：parser 产出的正文型元素在 GT 中没有对应。
    spurious_body_tokens = 0
    for pi, element in enumerate(parsed):
        if pi in aligned_parsed_indices:
            continue
        if element.get("type") in body_type_set or element.get("type") in FURNITURE_TYPES:
            spurious_body_tokens += sum(token_counter(_text(element)).values())

    return {
        "body_gt_element_count": sum(
            1 for element in ground_truth if element.get("type") in body_type_set
        ),
        "matched_body_element_count": matched_body,
        "paragraph_text_recall": _ratio(recovered_gt_tokens, total_gt_tokens + missing_body_tokens),
        "paragraph_text_precision": _ratio(overlapping_tokens, total_parsed_tokens + spurious_body_tokens),
        "normalized_exact_match": _ratio(exact_matches, matched_body),
        "character_error_rate_mean": (
            round(sum(cer_values) / len(cer_values), 6) if cer_values else None
        ),
        "character_error_rate_p90": (
            round(sorted(cer_values)[int(0.9 * (len(cer_values) - 1))], 6) if cer_values else None
        ),
        "character_error_rate_max": round(max(cer_values), 6) if cer_values else None,
        "cer_detail_count": len(cer_detail),
        "body_type_confusion": dict(sorted(type_confusion_body.items())),
        "type_confusion_matrix": type_confusion(alignment),
        "gt_type_counts": count_gt_by_type(ground_truth),
    }


def anchor_recall_metrics(
    anchors: Sequence[dict[str, Any]],
    parsed: Sequence[dict[str, Any]],
    *,
    threshold: float = 0.60,
) -> dict[str, Any]:
    """Real corpus 的 canonical anchor 召回（不依赖 chunk id，方案 §26.2）。

    对每个 canonical evidence anchor，检查 parser 输出中是否存在一个元素"覆盖"了它：
    anchor 文本的 token 多重集合 recall ≥ ``threshold``。这里刻意只用 recall 而不用 F1，
    因为对一个短 anchor 而言，一个包含它的长段落应当算作命中（这正是 RAG 需要的行为）。
    """

    parsed_counters = [token_counter(_text(element)) for element in parsed]
    hits = 0
    detail: list[dict[str, Any]] = []
    for anchor in anchors:
        anchor_counter = token_counter(anchor.get("text", ""))
        best = 0.0
        best_id: str | None = None
        best_page: int | None = None
        if anchor_counter:
            for index, counter in enumerate(parsed_counters):
                if not counter:
                    continue
                score = multiset_recall(anchor_counter, counter)
                if score > best:
                    best = score
                    best_id = parsed[index].get("element_id")
                    best_page = parsed[index].get("page")
        hit = bool(anchor_counter) and best >= threshold
        hits += int(hit)
        detail.append(
            {
                "anchor_id": anchor.get("anchor_id"),
                "group": anchor.get("group"),
                "page": anchor.get("page"),
                "hit": hit,
                "best_recall": round(best, 6),
                "best_element_id": best_id,
                "best_element_page": best_page,
            }
        )
    return {
        "anchor_count": len(anchors),
        "anchor_hits": hits,
        "canonical_anchor_recall": _ratio(hits, len(anchors)),
        "threshold": threshold,
        "detail": detail,
    }


def ngram_coverage_metrics(
    ground_truth_text: str,
    parsed_text: str,
    *,
    n: int = 8,
) -> dict[str, Any]:
    """Real corpus 的 ``normalized_8gram_coverage``（方案 §11.1）。

    覆盖率 = GT 文本的 8-gram 集合中被 parser 输出覆盖的比例。它比 token recall 更严格，
    因为顺序被打乱的正常文本仍可能有高 token recall，但 8-gram 会掉。
    """

    gt_grams = normalized_8grams(ground_truth_text, n)
    parsed_grams = normalized_8grams(parsed_text, n)
    covered = len(gt_grams & parsed_grams)
    return {
        "n": n,
        "gt_gram_count": len(gt_grams),
        "covered_gram_count": covered,
        "normalized_ngram_coverage": _ratio(covered, len(gt_grams)),
    }


# --------------------------------------------------------------------------------------
# §11.2 Reading Order
# --------------------------------------------------------------------------------------


def reading_order_metrics(
    ground_truth: Sequence[dict[str, Any]],
    parsed: Sequence[dict[str, Any]],
    alignment: dict[str, Any],
) -> dict[str, Any]:
    """阅读顺序：块对准确率、Kendall tau、双栏交叉检测。"""

    pairs = matched_pairs_for_reading_order(alignment)
    interleave = detect_column_interleave(ground_truth, parsed, alignment)
    return {
        "matched_pair_count": len(pairs),
        "block_pair_order_accuracy": pair_order_accuracy(pairs),
        "kendall_tau": kendall_tau(pairs),
        "column_interleave": interleave,
        # 具体错误分类（方案 §11.2 要求的高风险页面人工标注对象）。
        "column_interleave_flag": "COLUMN_INTERLEAVE" if interleave["interleave_detected"] else None,
    }


# --------------------------------------------------------------------------------------
# §11.3 Section Structure
# --------------------------------------------------------------------------------------


def section_metrics(
    ground_truth: Sequence[dict[str, Any]],
    parsed: Sequence[dict[str, Any]],
    alignment: dict[str, Any],
) -> dict[str, Any]:
    """章节结构：标题识别 P/R/F1、section_path 正确率、子节层级正确率。"""

    gt_headers = [e for e in ground_truth if e.get("type") == "section_header"]
    gt_doc_titles = [e for e in ground_truth if e.get("type") == "document_title"]
    parsed_headers = [e for e in parsed if e.get("type") in {"section_header", "document_title"}]

    # 分母必须用**元素实例数**而不是 element_id 的集合大小。
    #
    # 为什么：分子 correctly_typed 统计的是"匹配条目数"（每个条目对应一个 parsed 元素实例），
    # 一旦 parser 输出的 element_id 出现重复（例如映射层算错了身份 payload），
    # set() 会把多个实例折叠成一个，分母变小，precision 就会算出 > 1 的不可能值
    # —— 实测曾出现 section_header_f1 = 1.0474。id 重复本身由 validate_document 负责报错，
    # 指标这里只需保证在退化输入下仍然输出数学上合法的数值。
    gt_header_count = len(gt_headers)
    parsed_header_count = len(parsed_headers)

    # 标题精度的分子必须是"既被对齐、又被正确分类为标题"的那部分。
    confusion = type_confusion(alignment)
    correctly_typed = 0
    for match in alignment["matches"]:
        if match["similarity"] < MATCH_THRESHOLD:
            continue
        if match["gt_type"] == "section_header" and match["parsed_type"] in {
            "section_header",
            "document_title",
        }:
            correctly_typed += 1

    # section_path 正确率：对齐成功的标题元素，其 section_path 与 GT 逐项相等。
    path_correct = 0
    path_total = 0
    level_correct = 0
    level_total = 0
    path_failures: list[dict[str, Any]] = []
    for match in alignment["matches"]:
        if match["similarity"] < MATCH_THRESHOLD or match["gt_type"] != "section_header":
            continue
        gt_element = ground_truth[match["gt_index"]]
        parsed_element = parsed[match["parsed_index"]]
        gt_path = list(gt_element.get("section_path") or [])
        parsed_path = list(parsed_element.get("section_path") or [])
        path_total += 1
        if gt_path == parsed_path:
            path_correct += 1
        elif len(path_failures) < 15:
            path_failures.append(
                {
                    "gt_element_id": match["gt_element_id"],
                    "gt_path": gt_path,
                    "parsed_path": parsed_path,
                }
            )
        # 子节层级：GT path 的深度是否被保守地恢复（parser 常把三级标题压成两级）。
        gt_depth = len(gt_path)
        parsed_depth = len(parsed_path)
        if gt_depth >= 2:
            level_total += 1
            if parsed_depth >= 2:
                level_correct += 1
            else:
                if len(path_failures) < 15:
                    path_failures.append(
                        {
                            "gt_element_id": match["gt_element_id"],
                            "issue": "UNDER_NESTED",
                            "gt_depth": gt_depth,
                            "parsed_depth": parsed_depth,
                        }
                    )

    return {
        "gt_section_header_count": gt_header_count,
        "gt_document_title_count": len(gt_doc_titles),
        "parsed_header_count": parsed_header_count,
        "section_header_true_positive": correctly_typed,
        "section_header_precision": _ratio(correctly_typed, parsed_header_count),
        "section_header_recall": _ratio(correctly_typed, gt_header_count),
        "section_header_f1": _f1(
            _ratio(correctly_typed, parsed_header_count),
            _ratio(correctly_typed, gt_header_count),
        ),
        "section_path_accuracy": _ratio(path_correct, path_total),
        "subsection_level_accuracy": _ratio(level_correct, level_total),
        "path_failures": path_failures,
    }


def _f1(precision: float | None, recall: float | None) -> float | None:
    """由 precision/recall 计算 F1；任一为 ``None`` 或两者和为 0 时返回 ``None``。"""

    if precision is None or recall is None:
        return None
    if precision + recall == 0:
        return 0.0
    return round(2 * precision * recall / (precision + recall), 6)


# --------------------------------------------------------------------------------------
# §11.4 Paragraph Integrity
# --------------------------------------------------------------------------------------


def paragraph_integrity_metrics(
    ground_truth: Sequence[dict[str, Any]],
    parsed: Sequence[dict[str, Any]],
    alignment: dict[str, Any],
) -> dict[str, Any]:
    """段落完整性：边界 P/R 与碎片化/误合并率。"""

    gt_paragraphs = [e for e in ground_truth if e.get("type") == "paragraph"]
    parsed_paragraphs = [e for e in parsed if e.get("type") in {"paragraph", "list_item", "abstract"}]

    tp = sum(
        1
        for match in alignment["matches"]
        if match["similarity"] >= MATCH_THRESHOLD
        and match["gt_type"] == "paragraph"
        and match["parsed_type"] in {"paragraph", "list_item", "abstract"}
    )

    # 碎片化只统计"正文段落 GT 被切成多块"的情况；表体/参考文献的切分另有规则。
    fragmented = [
        item
        for item in alignment["fragments"]
        if item["gt_type"] == "paragraph"
    ]
    merged = [
        item
        for item in alignment["merged"]
        if item["parsed_type"] in {"paragraph", "list_item", "abstract"}
    ]
    gt_paragraph_ids = {e["element_id"] for e in gt_paragraphs}

    return {
        "gt_paragraph_count": len(gt_paragraphs),
        "parsed_paragraph_count": len(parsed_paragraphs),
        "paragraph_true_positive": tp,
        "paragraph_boundary_precision": _ratio(tp, len(parsed_paragraphs)),
        "paragraph_boundary_recall": _ratio(tp, len(gt_paragraphs)),
        "fragmented_paragraph_count": len(fragmented),
        "fragmented_paragraph_rate": _ratio(len(fragmented), len(gt_paragraph_ids)),
        "fragmented_examples": fragmented[:12],
        "merged_unrelated_paragraph_count": len(merged),
        "merged_unrelated_paragraph_rate": _ratio(
            len(merged), len(parsed_paragraphs)
        ),
        "merged_examples": merged[:12],
    }


# --------------------------------------------------------------------------------------
# §11.5 Furniture Cleaning
# --------------------------------------------------------------------------------------


def _multiset_coverage(reference: Counter[str], corpus: Counter[str]) -> float:
    """``reference`` 的 token 有多大比例出现在 ``corpus`` 中（含重复计数）。

    与 ``multiset_f1`` 的区别：这里只问"在不在"，不惩罚 corpus 里的多余 token。
    正因如此它才能区分"内容都在、只是被拆成别的形状"与"内容真的不在"——
    前者覆盖率高（所有 token 都能在文档别处找到），后者覆盖率接近 0。
    """

    total = sum(reference.values())
    if total <= 0:
        return 1.0
    hit = sum(min(count, corpus.get(token, 0)) for token, count in reference.items())
    return hit / total


def furniture_metrics(
    ground_truth: Sequence[dict[str, Any]],
    parsed: Sequence[dict[str, Any]],
    alignment: dict[str, Any],
) -> dict[str, Any]:
    """装饰元素识别：precision / recall 与"误删正文"（false removal）。

    ``furniture_false_removal`` 的定义：GT 中属于正文（BODY_TYPES）的元素没有被任何
    parser 元素对齐到——即它被 parser 当作装饰丢弃或彻底漏掉。
    """

    gt_furniture = [
        e for e in ground_truth if e.get("type") in FURNITURE_TYPES or e.get("type") == "unknown" and e.get("on_furniture")
    ]
    parsed_furniture = [e for e in parsed if e.get("type") in FURNITURE_TYPES]

    gt_furniture_ids = {e["element_id"] for e in gt_furniture}
    matched_furniture = sum(
        1
        for match in alignment["matches"]
        if match["similarity"] >= MATCH_THRESHOLD
        and match["gt_element_id"] in gt_furniture_ids
        and match["parsed_type"] in FURNITURE_TYPES
    )

    body_gt = [e for e in ground_truth if e.get("type") in BODY_TYPES]
    matched_gt_ids = {
        m["gt_element_id"] for m in alignment["matches"] if m["similarity"] >= MATCH_THRESHOLD
    }
    false_removal = [e for e in body_gt if e["element_id"] not in matched_gt_ids]

    # 「正文被误判为装饰」的直接计数。这一项是 ``furniture_false_removal`` **测不到**的：
    # 后者只问"有没有对齐上"，不看对齐到的元素是什么类型；正文若被标成 header/footer，
    # 只要文本还对得上就**不算**丢失。所以必须单独数，否则"把正文当装饰"这类错误
    # 在 false_removal 上完全隐形（实测本语料 5 个 arm 均为 0，即无人犯此错）。
    body_matched_to_furniture = sum(
        1
        for match in alignment["matches"]
        if match["similarity"] >= MATCH_THRESHOLD
        and match["gt_type"] in BODY_TYPES
        and match["parsed_type"] in FURNITURE_TYPES
    )

    # 把 false_removal 按「文档级 token 覆盖率」分三层，区分"分块差异"与"真丢失"。
    document_tokens: Counter[str] = Counter()
    for element in parsed:
        document_tokens.update(token_counter(_text(element)))
    coverage_split = {"high": 0, "medium": 0, "low": 0}
    for element in false_removal:
        coverage = _multiset_coverage(token_counter(_text(element)), document_tokens)
        if coverage >= BODY_LOSS_HIGH_COVERAGE:
            coverage_split["high"] += 1
        elif coverage >= BODY_LOSS_MEDIUM_COVERAGE:
            coverage_split["medium"] += 1
        else:
            coverage_split["low"] += 1

    # 装饰类型的细分统计，用于判断"哪一类装饰最容易漏"。
    gt_by_subtype: dict[str, int] = {}
    for element in gt_furniture:
        key = str(element.get("type"))
        gt_by_subtype[key] = gt_by_subtype.get(key, 0) + 1
    detected_by_subtype: dict[str, int] = {}
    for match in alignment["matches"]:
        if match["similarity"] < MATCH_THRESHOLD:
            continue
        if match["gt_element_id"] in gt_furniture_ids and match["parsed_type"] in FURNITURE_TYPES:
            # 用 GT 子类型归因，因为 parser 的子类型命名体系不同。
            gt_element = ground_truth[match["gt_index"]]
            key = str(gt_element.get("type"))
            detected_by_subtype[key] = detected_by_subtype.get(key, 0) + 1

    return {
        "gt_furniture_count": len(gt_furniture),
        "parsed_furniture_count": len(parsed_furniture),
        "furniture_true_positive": matched_furniture,
        "furniture_precision": _ratio(matched_furniture, len(parsed_furniture)),
        "furniture_recall": _ratio(matched_furniture, len(gt_furniture)),
        "gt_furniture_by_type": dict(sorted(gt_by_subtype.items())),
        "detected_furniture_by_type": dict(sorted(detected_by_subtype.items())),
        "furniture_false_removal_count": len(false_removal),
        "furniture_false_removal": _ratio(len(false_removal), len(body_gt)),
        # 成因分解：同一个 false_removal 数值背后可能是完全不同的两件事。
        # 报告必须依据这三层给出归因，而不能只看总比例。
        "body_loss_by_coverage": {
            **coverage_split,
            # 高覆盖率占比 = "看着像丢失、其实是分块差异"的比例。
            "high_coverage_ratio": _ratio(
                coverage_split["high"], len(false_removal)
            ),
            # 中覆盖（0.5 <= 覆盖率 < 0.9）：部分内容在、部分不在，是"分块差异 + 真实缺失"的混合，
            # 单独给出以免它被并入任一侧而让归因看起来比实际更干净。
            "medium_coverage_ratio": _ratio(
                coverage_split["medium"], len(false_removal)
            ),
            "low_coverage_ratio": _ratio(coverage_split["low"], len(false_removal)),
            "thresholds": {
                "high": BODY_LOSS_HIGH_COVERAGE,
                "medium": BODY_LOSS_MEDIUM_COVERAGE,
            },
            "note": (
                "high = 内容都在但未对齐到单个元素（分块/合并差异）；"
                "medium = 0.5 <= 覆盖率 < 0.9，分块差异与真实缺失的混合；"
                "low = 文档级 token 覆盖率 < 0.5，内容真的缺失。"
                "总比例相同不代表问题相同。"
                "注意三个 *_coverage_ratio 都是**本文档内**的比例；arm 级汇总时按均值合并，"
                "因此一个独占大量误删元素的文档（如扫描页）只会与其他文档等权——"
                "要看元素级占比请用计数（high/medium/low）自行现算。"
            ),
        },
        # 「把正文标成装饰」的直接计数（false_removal 测不到这一类错误）。
        "body_matched_to_furniture": body_matched_to_furniture,
        "furniture_false_removal_examples": [
            {"gt_element_id": e["element_id"], "type": e.get("type"), "page": e.get("page")}
            for e in false_removal[:12]
        ],
    }


# --------------------------------------------------------------------------------------
# §11.6 Figure / Caption
# --------------------------------------------------------------------------------------


def figure_caption_metrics(
    ground_truth: Sequence[dict[str, Any]],
    parsed: Sequence[dict[str, Any]],
    alignment: dict[str, Any],
    *,
    regression_figures: Sequence[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """图与图注：检测率、图注文本召回、caption→figure 链接、caption→section 归属。

    ``figure_detection`` 的主判据是**几何**（同页 bbox IoU），不是文本相似度——受控语料
    的图是矢量绘制，GT ``text`` 为空字符串，文本判据在构造上恒为 0。文本判据作为补充：
    少数 parser 会把图片转成带说明文字的文本块，此时文本对齐成立同样是"已检出"的有效
    证据，两者取并集。

    ``caption_to_figure_link_accuracy`` 依赖 parser 提供链接信息：若 GT 的
    ``caption_of`` 指向 figure_id，而 parser 的元素里 caption_of 非空且指向同一 figure
    元素，则算链接正确；parser 不提供链接时该指标为 ``None``。
    """

    gt_figures = [e for e in ground_truth if e.get("type") == "figure"]
    gt_captions = [e for e in ground_truth if e.get("type") == "figure_caption"]
    parsed_figures = [e for e in parsed if e.get("type") == "figure"]
    parsed_captions = [e for e in parsed if e.get("type") == "figure_caption"]

    caption_tp = sum(
        1
        for match in alignment["matches"]
        if match["similarity"] >= MATCH_THRESHOLD and match["gt_type"] == "figure_caption"
    )
    figure_geometry = geometric_detection(ground_truth, parsed, gt_type="figure")
    figure_hits_by_geometry = set(figure_geometry["matched_gt_element_ids"])
    figure_hits_by_text = {
        match["gt_element_id"]
        for match in alignment["matches"]
        if match["similarity"] >= MATCH_THRESHOLD and match["gt_type"] == "figure"
    }
    figure_tp = len(figure_hits_by_geometry | figure_hits_by_text)

    # 图注文本召回：GT 图注 token 被对齐元素覆盖的比例（图注常被"稀释"进正文）。
    agg = token_counter(" ".join(_text(e) for e in gt_captions))
    recovered = Counter()
    for match in alignment["matches"]:
        if match["similarity"] < MATCH_THRESHOLD or match["gt_type"] != "figure_caption":
            continue
        recovered.update(token_counter(_text(parsed[match["parsed_index"]])))
    caption_text_recall = _ratio(sum((agg & recovered).values()), sum(agg.values()))

    # 链接正确性：解析侧 caption 的 caption_of 是否可用。
    linked = [c for c in parsed_captions if c.get("caption_of")]
    link_accuracy: float | None = None
    if parsed_captions:
        gt_caption_link_ids = {e["element_id"] for e in gt_captions if e.get("caption_of")}
        correct_links = 0
        for match in alignment["matches"]:
            if match["gt_element_id"] not in gt_caption_link_ids:
                continue
            parsed_element = parsed[match["parsed_index"]]
            gt_element = ground_truth[match["gt_index"]]
            if parsed_element.get("caption_of") and parsed_element.get("figure_id") == gt_element.get(
                "figure_id"
            ):
                correct_links += 1
        link_accuracy = _ratio(correct_links, len(gt_caption_link_ids))

    # 章节归属：图注所在的 section_path 首项是否与 GT 一致。
    section_correct = 0
    section_total = 0
    for match in alignment["matches"]:
        if match["similarity"] < MATCH_THRESHOLD or match["gt_type"] != "figure_caption":
            continue
        gt_path = list(ground_truth[match["gt_index"]].get("section_path") or [])
        parsed_path = list(parsed[match["parsed_index"]].get("section_path") or [])
        if not gt_path:
            continue
        section_total += 1
        if parsed_path and parsed_path[0] == gt_path[0]:
            section_correct += 1

    payload: dict[str, Any] = {
        "gt_figure_count": len(gt_figures),
        "gt_figure_caption_count": len(gt_captions),
        "parsed_figure_count": len(parsed_figures),
        "parsed_figure_caption_count": len(parsed_captions),
        "figure_detection": _ratio(figure_tp, len(gt_figures)),
        # 几何 / 文本两条证据链分列，便于报告区分"真的没找到"与"找到了但坐标体系不可比"。
        "figure_detection_by_geometry": figure_geometry["detection"],
        "figure_detection_by_text": _ratio(len(figure_hits_by_text), len(gt_figures)),
        # 几何精确率：parser 输出的 figure 元素中有多少真的对应一个 GT 图。
        "figure_detection_precision": figure_geometry["precision"],
        "figure_geometry": figure_geometry,
        "caption_detection": _ratio(caption_tp, len(gt_captions)),
        "caption_text_recall": caption_text_recall,
        "caption_to_figure_link_accuracy": link_accuracy,
        "caption_link_coverage": _ratio(len(linked), len(parsed_captions)),
        "caption_section_assignment": _ratio(section_correct, section_total),
    }

    # ARR-007 回归切片（方案 §11.6）：该页图注在 13B 中被正文稀释，必须作为回归用例。
    if regression_figures:
        payload["regression_slice"] = _caption_regression_slice(
            regression_figures, parsed, alignment
        )
    return payload


def _caption_regression_slice(
    regression_figures: Sequence[dict[str, Any]],
    parsed: Sequence[dict[str, Any]],
    alignment: dict[str, Any],
) -> dict[str, Any]:
    """ARR-007 类回归：指定图注在 parser 输出中是否被隔离为独立元素。"""

    results: list[dict[str, Any]] = []
    for item in regression_figures:
        gt_id = item.get("element_id")
        match = next((m for m in alignment["matches"] if m["gt_element_id"] == gt_id), None)
        if match is None:
            results.append({"gt_element_id": gt_id, "isolated": False, "reason": "NO_MATCH"})
            continue
        parsed_element = parsed[match["parsed_index"]]
        # 隔离判据：以该图注文本为主的元素，正文 token 占比不得超过 40%。
        caption_tokens = token_counter(collapse_whitespace(item.get("text", "")))
        element_tokens = token_counter(_text(parsed_element))
        if not element_tokens:
            results.append({"gt_element_id": gt_id, "isolated": False, "reason": "EMPTY"})
            continue
        dilution = 1 - sum((caption_tokens & element_tokens).values()) / sum(element_tokens.values())
        results.append(
            {
                "gt_element_id": gt_id,
                "parsed_element_id": parsed_element.get("element_id"),
                "parsed_type": parsed_element.get("type"),
                "dilution": round(dilution, 6),
                "isolated": dilution <= 0.40,
                "reason": "OK" if dilution <= 0.40 else "CAPTION_DILUTED",
            }
        )
    return {
        "case_count": len(results),
        "isolated_count": sum(1 for r in results if r["isolated"]),
        "cases": results,
    }


# --------------------------------------------------------------------------------------
# §11.7 Table
# --------------------------------------------------------------------------------------


def table_metrics(
    ground_truth: Sequence[dict[str, Any]],
    parsed: Sequence[dict[str, Any]],
    alignment: dict[str, Any],
) -> dict[str, Any]:
    """表格：检测、caption、行列关系、表头归属、跨列单元格。

    关键设计：**不使用"表中文字是否出现"作为主要判据**（方案 §11.7 明确禁止）。
    因此：

    - ``table_detected`` 的主判据是**同页 bbox IoU**——GT 表格的内容在结构化的
      ``table_structure`` 里、扁平文本为空或只有片段，用文本匹配会让"结构完整的表格"
      反而判 0（实测 Docling 输出了 6 个 table 却被判 0）。文本对齐作为补充证据：
      parser 把表格降级成一段纯文本时，文本命中同样说明它没有完全丢失该表格；
    - ``table_detection_precision`` 是召回的必要补充：parser 输出里带 bbox 的 table 元素中，
      有多少真的对应一个 GT 表。缺了它，"把图统通标成 table"会在召回上拿满分
      （实测 Marker 在 T10 即为此）；
    - ``rows_correct`` / ``columns_correct`` 需要 parser 输出 ``table_structure``；
      P0 这类纯文本检测器没有结构，指标返回 ``None``——这正是它的结构性缺陷；
    - ``column_header_relation`` 检查表头行的每个标签是否与其下方列的第一个数值同列
      （用 GT 的列索引关系判定，parser 必须给出单元格坐标或顺序）。
    """

    gt_tables = [e for e in ground_truth if e.get("type") == "table"]
    gt_captions = [e for e in ground_truth if e.get("type") == "table_caption"]
    parsed_tables = [e for e in parsed if e.get("type") == "table"]
    parsed_captions = [e for e in parsed if e.get("type") == "table_caption"]

    table_geometry = geometric_detection(ground_truth, parsed, gt_type="table")
    table_hits_by_geometry = set(table_geometry["matched_gt_element_ids"])
    table_hits_by_text = {
        match["gt_element_id"]
        for match in alignment["matches"]
        if match["similarity"] >= MATCH_THRESHOLD and match["gt_type"] == "table"
    }
    table_tp = len(table_hits_by_geometry | table_hits_by_text)
    caption_tp = sum(
        1
        for match in alignment["matches"]
        if match["similarity"] >= MATCH_THRESHOLD and match["gt_type"] == "table_caption"
    )

    structure_available = [t for t in parsed_tables if t.get("table_structure")]
    rows_correct = 0
    rows_total = 0
    cols_correct = 0
    cols_total = 0
    cell_correct = 0
    cell_total = 0
    header_correct = 0
    header_total = 0
    spanning_correct = 0
    spanning_total = 0
    geometry_available = False

    for match in alignment["matches"]:
        if match["similarity"] < MATCH_THRESHOLD or match["gt_type"] != "table":
            continue
        gt_structure = ground_truth[match["gt_index"]].get("table_structure")
        parsed_structure = parsed[match["parsed_index"]].get("table_structure")
        if not gt_structure:
            continue
        if not parsed_structure:
            # GT 有结构而 parser 没有 -> 计入分母，分子不加，体现结构性缺失。
            rows_total += len(gt_structure.get("rows") or [])
            cols_total += 1
            cell_total += sum(len(row) for row in gt_structure.get("rows") or [])
            header_total += 1
            if gt_structure.get("has_spanning_cell"):
                spanning_total += 1
            continue
        geometry_available = True
        gt_rows = gt_structure.get("rows") or []
        parsed_rows = parsed_structure.get("rows") or []
        rows_total += len(gt_rows)
        rows_correct += sum(
            1
            for index, row in enumerate(gt_rows)
            if index < len(parsed_rows)
            and [_norm_cell(c) for c in row] == [_norm_cell(c) for c in parsed_rows[index]]
        )
        cols_total += 1
        gt_width = max((len(row) for row in gt_rows), default=0)
        parsed_width = max((len(row) for row in parsed_rows), default=0)
        if gt_width == parsed_width:
            cols_correct += 1
        for r_index, row in enumerate(gt_rows):
            for c_index, cell in enumerate(row):
                cell_total += 1
                if (
                    r_index < len(parsed_rows)
                    and c_index < len(parsed_rows[r_index])
                    and _norm_cell(cell) == _norm_cell(parsed_rows[r_index][c_index])
                ):
                    cell_correct += 1
        # 表头归属：表头行的每个标签，其下方同列第一个数值必须与 GT 一致。
        if gt_structure.get("header_row") and len(gt_rows) >= 2:
            header_total += 1
            gt_header = [_norm_cell(c) for c in gt_rows[0]]
            parsed_header = (
                [_norm_cell(c) for c in parsed_rows[0]] if parsed_rows else []
            )
            if parsed_header[: len(gt_header)] == gt_header:
                header_correct += 1
        if gt_structure.get("has_spanning_cell"):
            spanning_total += 1
            if parsed_structure.get("has_spanning_cell"):
                spanning_correct += 1

    # 表格 → caption 链接：GT 表格的 caption 文本是否被对齐到 caption 类型元素。
    agg = token_counter(" ".join(_text(e) for e in gt_captions))
    recovered = Counter()
    for match in alignment["matches"]:
        if match["similarity"] < MATCH_THRESHOLD or match["gt_type"] != "table_caption":
            continue
        recovered.update(token_counter(_text(parsed[match["parsed_index"]])))
    caption_text_recall = _ratio(sum((agg & recovered).values()), sum(agg.values()))

    # 表体是否被降级为纯文本：GT 表格被对齐到 paragraph 的比例。
    downgraded = sum(
        1
        for match in alignment["matches"]
        if match["similarity"] >= MATCH_THRESHOLD
        and match["gt_type"] == "table"
        and match["parsed_type"] not in {"table"}
    )

    return {
        "gt_table_count": len(gt_tables),
        "gt_table_caption_count": len(gt_captions),
        "parsed_table_count": len(parsed_tables),
        "parsed_table_caption_count": len(parsed_captions),
        "table_detected": _ratio(table_tp, len(gt_tables)),
        # 几何 / 文本两条证据链分列（同 figure_detection 的理由）。
        "table_detected_by_geometry": table_geometry["detection"],
        "table_detected_by_text": _ratio(len(table_hits_by_text), len(gt_tables)),
        # 几何精确率：parser 输出的 table 元素中有多少真的对应一个 GT 表。
        "table_detection_precision": table_geometry["precision"],
        "table_geometry": table_geometry,
        "caption_correct": _ratio(caption_tp, len(gt_captions)),
        "caption_text_recall": caption_text_recall,
        "structure_available_count": len(structure_available),
        "structure_available": bool(structure_available),
        "geometry_available_on_matched": geometry_available,
        "rows_correct": _ratio(rows_correct, rows_total),
        "columns_correct": _ratio(cols_correct, cols_total),
        "cell_text_accuracy": _ratio(cell_correct, cell_total),
        "row_column_relation_accuracy": _ratio(rows_correct, rows_total),
        "header_assignment_accuracy": _ratio(header_correct, header_total),
        "spanning_cell_accuracy": _ratio(spanning_correct, spanning_total),
        "table_downgraded_to_text_count": downgraded,
        "table_downgraded_to_text_rate": _ratio(
            downgraded, max(1, len([m for m in alignment["matches"] if m["gt_type"] == "table"]))
        ),
    }


def _norm_cell(text: str) -> str:
    """单元格归一：折叠空白、统一大小写、去掉包裹的空白与标点差异。"""

    return collapse_whitespace(str(text)).strip().lower()


def tables_to_html(table_structure: dict[str, Any] | None) -> str | None:
    """把结构化表格序列化为 HTML（MinerU/Docling 的输出形态），用于 TEDS 类比较。"""

    if not table_structure:
        return None
    rows = table_structure.get("rows") or []
    if not rows:
        return None
    parts = ["<table>"]
    for r_index, row in enumerate(rows):
        tag = "th" if r_index == 0 and table_structure.get("header_row") else "td"
        cells = "".join(f"<{tag}>{_norm_cell(cell)}</{tag}>" for cell in row)
        parts.append(f"<tr>{cells}</tr>")
    parts.append("</table>")
    return "".join(parts)


def html_structure_similarity(left_html: str | None, right_html: str | None) -> float | None:
    """HTML 结构相似度（TEDS 的轻量替代）：标签序列的最长公共子序列占比。

    不做完整 TEDS 的原因：TEDS 需要 tree-edit-distance 实现，而本阶段要回答的是
    "表格结构有没有被保住"，用标签序列 LCS 已足够区分"结构保住"与"结构塌成纯文本"。
    """

    if not left_html or not right_html:
        return None
    left_tags = re.findall(r"</?[a-z]+>", left_html)
    right_tags = re.findall(r"</?[a-z]+>", right_html)
    if not left_tags or not right_tags:
        return None
    previous = [0] * (len(right_tags) + 1)
    for left_tag in left_tags:
        current = [0] * (len(right_tags) + 1)
        for j, right_tag in enumerate(right_tags, start=1):
            if left_tag == right_tag:
                current[j] = previous[j - 1] + 1
            else:
                current[j] = max(previous[j], current[j - 1])
        previous = current
    return round(previous[-1] / max(len(left_tags), len(right_tags)), 6)


# --------------------------------------------------------------------------------------
# §11.8 Equation
# --------------------------------------------------------------------------------------


def equation_metrics(
    ground_truth: Sequence[dict[str, Any]],
    parsed: Sequence[dict[str, Any]],
    alignment: dict[str, Any],
    *,
    leakage_slice: Sequence[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """公式：检测召回、行间/行内判别、编号保留、LaTeX 相似度、破碎率。

    ``display_vs_inline_accuracy`` 依赖 GT 的 ``display_mode`` 字段（``display`` /
    ``inline``）：parser 无此字段时该指标为 ``None``，此时报告必须显式写明
    "该 parser 不区分行内/行间公式"，而不是给一个 0 分。

    **检出判定用几何（bbox IoU）而不是文本**，理由与 figure / table 相同但更尖锐：
    各 parser 对公式的**文本约定**不同。同一篇 T09（GT 11 个公式，asciiMath 约定
    ``L_enc = - sum_i log P( e_i | e_less_i ) (1)``）实测：Docling 输出同约定文本
    （文本召回 0.958），而 MinerU 输出原始 LaTeX
    （``$$ \\mathrm { L } \\operatorname { e n c } = ...``）——它与 GT 逐 token 不可比，
    文本召回只有 0.071，**但 MinerU 确实检出了全部 11 个公式**
    （``parsed_equation_count == 11``）。也就是说文本判据在这里测的是"选了哪种输出约定"，
    不是"有没有检出"。几何判据与约定无关，才是检出的正确判据。
    """

    gt_equations = [e for e in ground_truth if e.get("type") == "equation"]
    parsed_equations = [e for e in parsed if e.get("type") == "equation"]

    equation_geometry = geometric_detection(
        ground_truth,
        parsed,
        gt_type="equation",
        parsed_type="equation",
        # 公式用重叠系数而非 IoU：GT 是行级框（占满栏宽、含右侧编号），parser 是字形框，
        # IoU 会被并集分母系统性压低（实测对应块中位仅 0.24，低于 0.5 阈值）。
        # 分布与阈值依据见 GEOMETRIC_OVERLAP_THRESHOLD。
        criterion="overlap",
    )
    eq_hits_by_geometry = set(equation_geometry["matched_gt_element_ids"])

    # 文本命中**必须同时约束两侧类型**。旧实现只判 ``gt_type == "equation"``，
    # 于是"把公式输出成段落"也能算命中：实测 baseline 在 T09 只把 1 个元素标成公式，
    # 却因 11 个 GT 公式各自匹配到一个段落元素而得到 recall = 1.0、precision = 11.0。
    # 分子是 GT 侧计数、分母是 parser 侧计数，属不同计数空间，precision > 1 在数学上
    # 不可能——这类缺陷必须由实现消除（§35 也会把它记成越界异常）。
    eq_hits_by_text = {
        match["gt_element_id"]
        for match in alignment["matches"]
        if match["similarity"] >= MATCH_THRESHOLD
        and match["gt_type"] == "equation"
        and match["parsed_type"] == "equation"
    }
    eq_tp = len(eq_hits_by_geometry | eq_hits_by_text)

    number_preserved = 0
    number_total = 0
    display_correct = 0
    display_total = 0
    latex_scores: list[float] = []
    broken = 0
    for match in alignment["matches"]:
        if match["similarity"] < MATCH_THRESHOLD or match["gt_type"] != "equation":
            continue
        if match["parsed_type"] != "equation":
            # 公式被输出成了其它类型：类型判定本身已失败，不该继续拿它去评估
            # "编号保留"与"行内/行间判别"——那样只会去读一个段落元素。
            # 正确做法是让这两个指标的分母不包含它，从而如实变成 None 或更低。
            continue
        gt_element = ground_truth[match["gt_index"]]
        parsed_element = parsed[match["parsed_index"]]
        if gt_element.get("equation_number"):
            number_total += 1
            if gt_element["equation_number"] in _text(parsed_element):
                number_preserved += 1
        if gt_element.get("display_mode"):
            display_total += 1
            if parsed_element.get("display_mode") == gt_element["display_mode"]:
                display_correct += 1
        # LaTeX 相似度在"本阶段无法渲染真实 LaTeX"的约束下退化为字符级相似度，
        # 报告必须携带该降级说明（见 metrics["latex_similarity_mode"]）。
        latex_scores.append(match["similarity"])
        if _is_broken_formula(_text(parsed_element)):
            broken += 1

    # 破碎率：公式元素过短（只剩编号或单个符号）视为破碎。
    broken_overall = sum(1 for e in parsed_equations if _is_broken_formula(_text(e)))

    # 文本约定可比性披露：GT 与 parser 各自的公式文本里有多少是 LaTeX 源码形态。
    # 两侧计数都不为 0 时文本级公式指标可比；一侧为 0 时文本指标主要反映表示约定差异，
    # 此时必须优先采信 equation_detection_by_geometry（报告据此写明结论）。
    gt_latex_like = sum(1 for e in gt_equations if _looks_like_latex(_text(e)))
    parsed_latex_like = sum(1 for e in parsed_equations if _looks_like_latex(_text(e)))

    payload: dict[str, Any] = {
        "gt_equation_count": len(gt_equations),
        "parsed_equation_count": len(parsed_equations),
        "equation_true_positive": eq_tp,
        "equation_detection_recall": _ratio(eq_tp, len(gt_equations)),
        # 几何精确率与几何召回同处"一对一命中对"这一个计数空间，因此恒 ≤ 1。
        "equation_detection_precision": equation_geometry["precision"],
        "equation_detection_by_geometry": equation_geometry["detection"],
        "equation_detection_by_text": _ratio(len(eq_hits_by_text), len(gt_equations)),
        "equation_geometry": equation_geometry,
        "latex_convention": {
            "gt_latex_like_count": gt_latex_like,
            "parsed_latex_like_count": parsed_latex_like,
            "comparable": bool(gt_latex_like) == bool(parsed_latex_like),
            "note": (
                "公式文本约定的可比性检查。两侧都为 0（均为纯文本约定）或都不为 0 时，"
                "文本级公式指标可比；一侧为 0 时文本指标主要反映表示约定差异，"
                "结论须以 equation_detection_by_geometry 为准。"
            ),
        },
        "display_vs_inline_accuracy": _ratio(display_correct, display_total),
        "formula_number_preservation": _ratio(number_preserved, number_total),
        "normalized_latex_similarity": (
            round(sum(latex_scores) / len(latex_scores), 6) if latex_scores else None
        ),
        "latex_similarity_mode": "token_multiset_f1(DEGRADED: 无真实 LaTeX 渲染，非 AST 比较)",
        "broken_formula_rate": _ratio(broken_overall, len(parsed_equations)),
        "broken_matched_formula_count": broken,
    }

    # 219 处 LaTeX leakage 回归切片（方案 §11.8）。
    if leakage_slice is not None:
        leaked_in_body = 0
        for match in alignment["matches"]:
            if match["gt_type"] in {"paragraph", "list_item"} and match["similarity"] >= MATCH_THRESHOLD:
                if _looks_like_latex(_text(parsed[match["parsed_index"]])):
                    leaked_in_body += 1
        payload["regression_slice"] = {
            "leakage_reference_count": len(leakage_slice),
            "body_elements_with_latex_leakage": leaked_in_body,
            "note": "参照 Phase 13A 统计的 219 处 LaTeX leakage 切片；本阶段只统计正文元素比例。",
        }
    return payload


def _looks_like_latex(text: str) -> bool:
    """判断文本是否含 LaTeX 命令泄漏。"""

    return bool(re.search(r"\\[a-zA-Z]{2,}", text))


def _is_broken_formula(text: str) -> bool:
    """判断公式是否破碎：规范化后只剩编号、或长度 < 8 且不含运算符。"""

    stripped = collapse_whitespace(text)
    if not stripped:
        return True
    if re.fullmatch(r"\(\d{1,3}\)", stripped):
        return True
    return len(stripped) < 8 and not re.search(r"[=+\-×÷∑∈≤≥/^_]", stripped)


# --------------------------------------------------------------------------------------
# §11.9 References
# --------------------------------------------------------------------------------------


def reference_metrics(
    ground_truth: Sequence[dict[str, Any]],
    parsed: Sequence[dict[str, Any]],
    alignment: dict[str, Any],
) -> dict[str, Any]:
    """参考文献：章节检测、条目切分、引用标注保留、对正文的污染。"""

    gt_refs = [e for e in ground_truth if e.get("type") == "reference_entry"]
    parsed_refs = [e for e in parsed if e.get("type") == "reference_entry"]

    tp = sum(
        1
        for match in alignment["matches"]
        if match["similarity"] >= MATCH_THRESHOLD and match["gt_type"] == "reference_entry"
    )

    # 污染：GT 正文元素里出现了 GT 参考文献条目的高比例 token（说明参考文献被塞进正文块）。
    gt_ref_token_union = Counter()
    for element in gt_refs:
        gt_ref_token_union.update(token_counter(_text(element)))
    polluted = 0
    pollution_examples: list[dict[str, Any]] = []
    for match in alignment["matches"]:
        if match["similarity"] < MATCH_THRESHOLD:
            continue
        if match["gt_type"] not in {"paragraph", "list_item"}:
            continue
        parsed_tokens = token_counter(_text(parsed[match["parsed_index"]]))
        if not parsed_tokens:
            continue
        ref_share = sum((gt_ref_token_union & parsed_tokens).values()) / sum(parsed_tokens.values())
        if ref_share >= 0.30:
            polluted += 1
            if len(pollution_examples) < 12:
                pollution_examples.append(
                    {
                        "gt_element_id": match["gt_element_id"],
                        "parsed_element_id": match["parsed_element_id"],
                        "reference_token_share": round(ref_share, 6),
                    }
                )

    # 引用标注保留：GT 中带 citation callout 的正文元素，其 callout 是否在 parser 文本中保留。
    callout_total = 0
    callout_kept = 0
    callout_missing: list[dict[str, Any]] = []
    for match in alignment["matches"]:
        if match["similarity"] < MATCH_THRESHOLD:
            continue
        gt_element = ground_truth[match["gt_index"]]
        callouts = gt_element.get("citation_callouts") or []
        if not callouts:
            continue
        parsed_text = collapse_whitespace(_text(parsed[match["parsed_index"]]))
        for callout in callouts:
            callout_total += 1
            if callout in parsed_text:
                callout_kept += 1
            elif len(callout_missing) < 12:
                callout_missing.append(
                    {"gt_element_id": match["gt_element_id"], "callout": callout}
                )

    return {
        "gt_reference_entry_count": len(gt_refs),
        "parsed_reference_entry_count": len(parsed_refs),
        "reference_section_detection": _ratio(tp, len(gt_refs)),
        "reference_entry_segmentation": _ratio(
            len([m for m in alignment["fragments"] if m["gt_type"] == "reference_entry"]),
            max(1, len(gt_refs)),
        ),
        "reference_segmentation_accuracy": _ratio(tp, len(gt_refs)),
        "citation_callout_preservation": _ratio(callout_kept, callout_total),
        "citation_callout_total": callout_total,
        "citation_callout_missing_examples": callout_missing,
        "reference_pollution_into_body_count": polluted,
        "reference_pollution_into_body_rate": _ratio(polluted, max(1, len(parsed_refs) + polluted)),
        "reference_pollution_examples": pollution_examples,
    }


# --------------------------------------------------------------------------------------
# §11.10 Provenance
# --------------------------------------------------------------------------------------


def provenance_metrics(parsed: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """来源可追溯性：页码与 bbox 覆盖率。

    分母只算**保留元素**（正文 + 结构元素），因为装饰元素本来就要被丢弃，
    把它们计入分母会人为拉高覆盖率。
    """

    retained = [
        e
        for e in parsed
        if e.get("type") not in FURNITURE_TYPES and e.get("type") != "unknown"
    ]
    with_page = [e for e in retained if isinstance(e.get("page"), int)]
    with_bbox = [
        e
        for e in retained
        if isinstance(e.get("bbox"), list)
        and len(e["bbox"]) == 4
        and all(isinstance(v, (int, float)) for v in e["bbox"])
    ]
    with_ref = [e for e in retained if e.get("source_ref")]
    return {
        "retained_element_count": len(retained),
        "page_provenance_count": len(with_page),
        "page_provenance_coverage": _ratio(len(with_page), len(retained)),
        "bbox_provenance_count": len(with_bbox),
        "bbox_provenance_coverage": _ratio(len(with_bbox), len(retained)),
        "source_ref_coverage": _ratio(len(with_ref), len(retained)),
    }


# --------------------------------------------------------------------------------------
# §30 成本与工程指标
# --------------------------------------------------------------------------------------


def cost_metrics(
    *,
    elapsed_seconds: float,
    page_count: int,
    output_bytes: int,
    element_count: int,
    peak_ram_mb: float | None,
    ocr_pages: int = 0,
    gpu_required: bool = False,
    device: str = "cpu",
) -> dict[str, Any]:
    """解析成本与工程指标（方案 §30）。"""

    return {
        "parse_seconds_total": round(elapsed_seconds, 3),
        "parse_time_per_page": round(elapsed_seconds / page_count, 4) if page_count else None,
        "page_count": page_count,
        "output_size_bytes": output_bytes,
        "output_bytes_per_page": round(output_bytes / page_count, 1) if page_count else None,
        "element_count": element_count,
        "peak_ram_mb": round(peak_ram_mb, 1) if peak_ram_mb is not None else None,
        "ocr_pages": ocr_pages,
        "gpu_required": gpu_required,
        "device": device,
    }


# --------------------------------------------------------------------------------------
# 内部文本取用（与 align._text_of 保持同一口径）
# --------------------------------------------------------------------------------------


def _text(element: dict[str, Any]) -> str:
    """取 element 的比对文本，口径与 :mod:`phase13c.align` 完全一致。"""

    for key in ("normalized_text", "text", "raw_text"):
        value = element.get(key)
        if isinstance(value, str) and value.strip():
            return collapse_whitespace(value)
    return ""


def mean(values: Sequence[float | None]) -> float | None:
    """忽略 ``None`` 的均值；全为 ``None`` 时返回 ``None``。"""

    present = [v for v in values if v is not None]
    if not present:
        return None
    return round(sum(present) / len(present), 6)


def aggregate_metric_dicts(dicts: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """把多篇文档的同名数值指标求均值。

    语义与 :func:`mean` 一致：``None`` 表示"该文档没有提供这项指标所需的信息"，
    不参与平均；某指标在所有文档上都是 ``None`` 时聚合结果保持 ``None``（而不是 0）。
    只有当某键存在非数值、非 ``None`` 的内容（例如列表型诊断字段）时，才退化为
    占位字符串，提示调用方去逐文档结果里查看。
    """

    keys = sorted({key for payload in dicts for key in payload})
    aggregated: dict[str, Any] = {}
    for key in keys:
        values = [payload.get(key) for payload in dicts]
        numeric = [v for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)]
        others = [
            v for v in values if v is not None and not (isinstance(v, (int, float)) and not isinstance(v, bool))
        ]
        if others:
            aggregated[key] = "<per-document; see metrics[*].%s>" % key
        else:
            aggregated[key] = round(sum(numeric) / len(numeric), 6) if numeric else None
    return aggregated
