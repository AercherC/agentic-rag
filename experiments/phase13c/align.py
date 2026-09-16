"""Ground Truth ↔ parser 输出的元素对齐引擎。

Parser Benchmark 的核心困难是：parser 输出的块边界与 Ground Truth 的段落边界**不一定
相同**（可能被切碎、被合并、被错分类型）。因此所有"文本保真度 / 段落完整性 / 阅读顺序"
指标都必须建立在一次**显式的对齐**之上，而不能靠位置索引硬比。

## 两层判定

**第一层：一对一匹配**。两侧以 token 多重集合表示，用多重集合 F1 作为相似度，按 F1 降序
做贪心一对一匹配（同分用 element_id 字典序打破平局，保证跨运行完全一致）。

**第二层：单元完整性（碎片 / 合并 / 缺失）**。仅靠"F1 ≥ 某个阈值"判定碎片会产生大量
假阳性：同一主题的相邻段落在词表上高度重叠，任意两段都能轻松越过 0.20 的相似度线。
因此这里改用**覆盖率**语义，它直接对应"知识单元是否被完整保住"：

- 某 GT 元素被**单个** parser 元素覆盖 ≥ :data:`COMPLETE_COVERAGE` -> 单元完整；
- 否则用贪心算法求覆盖该 GT 所需的最少 parser 元素数，若 ≥ 2 且并集覆盖 ≥
  :data:`UNION_COVERAGE` -> **碎片化**（内容都在，只是被切开）；
- 若最少元素数仍达不到并集覆盖 -> **缺失**；
- 某 parser 元素**完整包含** ≥ 2 个正文 GT 元素（每个 ≥ :data:`MIN_MERGE_UNIT_TOKENS`
  个 token、且各自的独占 token 不为空），且这些 GT 的 token 占该元素 ≥
  :data:`MERGED_DOMINANCE` -> **误合并**。装饰元素不参与该判定（它们本应被剔除，
  属于 cleaning 问题）。

这套定义不依赖任何调参阈值区间，因此结果的解释是"内容有没有被保住"，而不是
"相似度有没有越过某条线"。
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable, Sequence

from .schema import FURNITURE_TYPES, collapse_whitespace, tokenize

# 匹配阈值：多重集合 F1 >= 0.50 视为同一元素。
MATCH_THRESHOLD = 0.50
# 局部重叠阈值：仅用于"相邻元素有实质重叠"的诊断，不参与碎片/合并判定。
PARTIAL_OVERLAP_THRESHOLD = 0.20
# 强匹配阈值：用于"精确命中"统计。
STRONG_THRESHOLD = 0.85
# 单元完整性阈值（见模块 docstring）。
COMPLETE_COVERAGE = 0.90
UNION_COVERAGE = 0.90
MERGED_CONTAINMENT = 0.90
MERGED_DOMINANCE = 0.60
# parser 元素相对"被包含 GT 的 token 并集"允许的最大体积倍数。超过它说明该元素
# 还装了大量其它内容，属于"长块顺带覆盖了短 GT"，不是真正的误合并。
MERGED_MAX_SIZE_RATIO = 2.0
# 参与误合并判定的 GT 单元至少要有的 token 数。单个 token 的单元（最典型的是页码
# "1" / "2" / "3"）会被几乎任何含数字的正文块"完整包含"，从而制造大量假合并。
MIN_MERGE_UNIT_TOKENS = 2
# 求最少覆盖元素数时的迭代上限，防止病态输入导致长循环。
_MAX_PIECES = 8


# --------------------------------------------------------------------------------------
# token 多重集合运算
# --------------------------------------------------------------------------------------


def token_counter(text: str) -> Counter[str]:
    """返回文本的 token 多重集合；空文本返回空 Counter。"""

    return Counter(tokenize(text))


def multiset_union(counters: Iterable[Counter[str]]) -> Counter[str]:
    """多重集合的"并"：逐 token **计数相加**。

    注意 ``Counter.__or__`` 是"逐键取最大值"（集合语义）。当同一个词在参考文本里出现
    多次、却被不等地分到两个片段时（例：14 个 token 的段落里 ``the`` 出现 3 次，被切成
    两份后分别贡献 2 次与 1 次），取最大值只会记 2 次，并集覆盖率会被低估成 13/14，
    从而把一个"内容完整但被切开"的单元误判为"内容不全"。因此这里必须用加法。
    """

    total: Counter[str] = Counter()
    for counter in counters:
        total = total + counter
    return total


def capped_coverage(reference: Counter[str], hypothesis: Counter[str]) -> float:
    """hypothesis 对 reference 的多重集合覆盖率，逐 token 上限为 reference 的实际计数。"""

    total = sum(reference.values())
    if total == 0:
        return 0.0
    covered = sum(min(count, reference[token]) for token, count in hypothesis.items())
    return covered / total


def multiset_f1(left: Counter[str], right: Counter[str]) -> float:
    """计算两个 token 多重集合的 F1。

    使用多重集合（而非集合）是因为段落里同一个词重复出现会改变长度，集合会低估差异；
    F1 的调和平均同时对 precision 与 recall 敏感。
    """

    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    overlap = sum((left & right).values())
    if overlap == 0:
        return 0.0
    precision = overlap / sum(right.values())
    recall = overlap / sum(left.values())
    return 2 * precision * recall / (precision + recall)


def multiset_recall(reference: Counter[str], hypothesis: Counter[str]) -> float:
    """reference 的 token 中被 hypothesis 覆盖的比例。"""

    if not reference:
        return 1.0
    if not hypothesis:
        return 0.0
    return sum((reference & hypothesis).values()) / sum(reference.values())


def _text_of(element: dict[str, Any]) -> str:
    """取 element 的比对文本：优先 ``normalized_text``，退回 ``text``/``raw_text``。"""

    for key in ("normalized_text", "text", "raw_text"):
        value = element.get(key)
        if isinstance(value, str) and value.strip():
            return collapse_whitespace(value)
    return ""


def _key_of(element: dict[str, Any], prefix: str) -> str:
    """取用于打破平局的稳定键。"""

    eid = element.get("element_id")
    if isinstance(eid, str) and eid:
        return eid
    return f"{prefix}-{_text_of(element)[:32]}"


# --------------------------------------------------------------------------------------
# 第二层：单元完整性
# --------------------------------------------------------------------------------------


def _greedy_pieces(gt_counter: Counter[str], parsed_counters: Sequence[Counter[str]]) -> list[int]:
    """贪心求覆盖 GT token 所需的最少 parser 元素序列（按剩余重叠最大者优先）。"""

    total = sum(gt_counter.values())
    if total == 0:
        return []
    remaining = Counter(gt_counter)
    pieces: list[int] = []
    while sum(remaining.values()) > (1.0 - UNION_COVERAGE) * total:
        best_index = -1
        best_overlap = 0
        for index, counter in enumerate(parsed_counters):
            if index in pieces or not counter:
                continue
            overlap = sum((remaining & counter).values())
            if overlap > best_overlap:
                best_index, best_overlap = index, overlap
        if best_index < 0 or best_overlap == 0:
            break
        # Counter 的 ``-`` 会丢弃非正值，正好等价于"扣除已覆盖的 token"。
        remaining = remaining - parsed_counters[best_index]
        pieces.append(best_index)
        if len(pieces) >= _MAX_PIECES:
            break
    return pieces


def analyze_unit_integrity(
    ground_truth: Sequence[dict[str, Any]],
    parsed: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """判定每个 GT 单元的完整性状态与每个 parser 元素的误合并状态。"""

    gt_counters = [token_counter(_text_of(element)) for element in ground_truth]
    parsed_counters = [token_counter(_text_of(element)) for element in parsed]

    complete: list[int] = []
    fragmented: list[dict[str, Any]] = []
    incomplete: list[dict[str, Any]] = []

    for gt_index, gt_counter in enumerate(gt_counters):
        if not gt_counter:
            continue
        best = 0.0
        for counter in parsed_counters:
            if not counter:
                continue
            recall = multiset_recall(gt_counter, counter)
            if recall > best:
                best = recall
        if best >= COMPLETE_COVERAGE:
            complete.append(gt_index)
            continue
        pieces = _greedy_pieces(gt_counter, parsed_counters)
        # 并集覆盖：各"片段贡献给 GT 的 token"按多重集合相加，再做逐 token 上限截断。
        # 不能用 ``Counter.__or__``（逐键 max）——重复词被不等分到两个片段时会低估覆盖。
        union_covered = multiset_union(gt_counter & parsed_counters[index] for index in pieces)
        union_coverage = capped_coverage(gt_counter, union_covered)
        record = {
            "gt_element_id": _key_of(ground_truth[gt_index], "gt"),
            "gt_type": ground_truth[gt_index].get("type"),
            "page": ground_truth[gt_index].get("page"),
            "best_single_coverage": round(best, 6),
            "piece_count": len(pieces),
            "union_coverage": round(union_coverage, 6),
            "parsed_element_ids": [_key_of(parsed[i], "pr") for i in pieces],
            "text_head": _text_of(ground_truth[gt_index])[:100],
        }
        if len(pieces) >= 2 and union_coverage >= UNION_COVERAGE:
            fragmented.append(record)
        else:
            incomplete.append(record)

    merged: list[dict[str, Any]] = []
    for parsed_index, parsed_counter in enumerate(parsed_counters):
        if not parsed_counter:
            continue
        # 参与判定的 GT 单元必须是"实质内容"：
        # ① 排除装饰元素（页眉/页脚/页码/水印/侧栏）。装饰本就应该被剔除，
        #    "正文块里出现了页码"属于 cleaning 问题，由 furniture_* 指标承担；
        # ② 排除 token 数 < MIN_MERGE_UNIT_TOKENS 的单元。否则页码 "1" 这种单 token
        #    单元会被任何含数字的正文块（"...1 Science Park Road..."）瞬间"完整包含"，
        #    制造大量假合并——这是首版 merged 率虚高到 0.46 的唯一原因。
        contained = [
            gt_index
            for gt_index, gt_counter in enumerate(gt_counters)
            if gt_counter
            and sum(gt_counter.values()) >= MIN_MERGE_UNIT_TOKENS
            and ground_truth[gt_index].get("type") not in FURNITURE_TYPES
            and multiset_recall(gt_counter, parsed_counter) >= MERGED_CONTAINMENT
        ]
        if len(contained) < 2:
            continue
        # 独占贡献过滤：一个被包含的 GT 单元若其 token 全部已被其它被包含单元覆盖，
        # 说明它只是"碰巧也在里面"，不是真正被合并进来的一段内容，必须剔除。
        # 例：`2 Related Work` 与紧随的段落共享 related/work，但编号 token "2" 是独占的，
        # 因此仍会被判为真实合并；而页码 "1" 的 token 已被作者单位覆盖，会被剔除。
        survivors: list[int] = []
        for gt_index in contained:
            others: Counter = Counter()
            for other in contained:
                if other != gt_index:
                    others = others | gt_counters[other]
            if sum((gt_counters[gt_index] - others).values()) > 0:
                survivors.append(gt_index)
        if len(survivors) < 2:
            continue
        contained = survivors
        # 被包含 GT 的整体 token 并集。用多重集合相加（而不是逐键 max），否则同一个词
        # 在多个 GT 单元中重复出现时并集偏小，size_ratio 会虚高、真实合并被判漏。
        union_full = multiset_union(gt_counters[gt_index] for gt_index in contained)
        covered = union_full & parsed_counter
        dominance = sum(covered.values()) / sum(parsed_counter.values())
        if dominance < MERGED_DOMINANCE:
            continue
        # 尺寸约束：只有当 parser 元素"基本就是这些 GT 的拼接"时才算误合并。
        # 没有这条约束时，一个包含大量其他内容的长块只要顺带覆盖了两个短 GT 单元
        #（例如正文里出现了 "1 Introduction" 的全部词）就会被误判为合并。
        size_ratio = sum(parsed_counter.values()) / max(1, sum(union_full.values()))
        if size_ratio > MERGED_MAX_SIZE_RATIO:
            continue
        merged.append(
            {
                "parsed_element_id": _key_of(parsed[parsed_index], "pr"),
                "parsed_type": parsed[parsed_index].get("type"),
                "page": parsed[parsed_index].get("page"),
                "gt_element_ids": [_key_of(ground_truth[i], "gt") for i in contained],
                "merged_count": len(contained),
                "dominance": round(dominance, 6),
                "size_ratio": round(size_ratio, 6),
            }
        )

    return {
        "complete_unit_count": len(complete),
        "fragmented": fragmented,
        "incomplete": incomplete,
        "merged": merged,
    }


# --------------------------------------------------------------------------------------
# 第一层：一对一匹配
# --------------------------------------------------------------------------------------


def align_elements(
    ground_truth: Sequence[dict[str, Any]],
    parsed: Sequence[dict[str, Any]],
    *,
    match_threshold: float = MATCH_THRESHOLD,
    partial_threshold: float = PARTIAL_OVERLAP_THRESHOLD,
) -> dict[str, Any]:
    """对齐 Ground Truth 与 parser 元素，返回匹配、单元完整性、缺失与多余清单。"""

    gt_tokens = [token_counter(_text_of(element)) for element in ground_truth]
    pr_tokens = [token_counter(_text_of(element)) for element in parsed]

    # 计算全部相似度对。复杂度 O(G*P)，受控语料规模下（G<200, P<800）完全可接受。
    pairs: list[tuple[float, int, int]] = []
    for gi, g_tokens in enumerate(gt_tokens):
        if not g_tokens:
            continue
        for pi, p_tokens in enumerate(pr_tokens):
            if not p_tokens:
                continue
            score = multiset_f1(g_tokens, p_tokens)
            if score >= partial_threshold:
                pairs.append((score, gi, pi))

    # 贪心一对一：分数降序，同分时用两侧稳定键字典序，保证跨运行完全一致。
    pairs.sort(
        key=lambda item: (
            -item[0],
            _key_of(ground_truth[item[1]], "gt"),
            _key_of(parsed[item[2]], "pr"),
        )
    )
    gt_taken: dict[int, int] = {}
    pr_taken: dict[int, int] = {}
    for score, gi, pi in pairs:
        if gi in gt_taken or pi in pr_taken or score < match_threshold:
            continue
        gt_taken[gi] = pi
        pr_taken[pi] = gi

    matches: list[dict[str, Any]] = [
        {
            "gt_index": gi,
            "parsed_index": pi,
            "gt_element_id": _key_of(ground_truth[gi], "gt"),
            "parsed_element_id": _key_of(parsed[pi], "pr"),
            "gt_type": ground_truth[gi].get("type"),
            "parsed_type": parsed[pi].get("type"),
            "similarity": round(multiset_f1(gt_tokens[gi], pr_tokens[pi]), 6),
            "strong": multiset_f1(gt_tokens[gi], pr_tokens[pi]) >= STRONG_THRESHOLD,
            "gt_reading_order": ground_truth[gi].get("reading_order"),
            "parsed_reading_order": parsed[pi].get("reading_order"),
            "gt_page": ground_truth[gi].get("page"),
            "parsed_page": parsed[pi].get("page"),
        }
        for gi, pi in sorted(gt_taken.items())
    ]

    # 局部重叠诊断：GT 与多个 parser 元素都达到 partial 阈值（不代表碎片，仅供参考）。
    partial_overlap: list[dict[str, Any]] = []
    for gi, g_tokens in enumerate(gt_tokens):
        if not g_tokens:
            continue
        related = [
            pi
            for pi, p_tokens in enumerate(pr_tokens)
            if p_tokens and multiset_f1(g_tokens, p_tokens) >= partial_threshold
        ]
        if len(related) >= 2:
            partial_overlap.append(
                {
                    "gt_element_id": _key_of(ground_truth[gi], "gt"),
                    "gt_type": ground_truth[gi].get("type"),
                    "parsed_element_count": len(related),
                }
            )

    integrity = analyze_unit_integrity(ground_truth, parsed)

    matched_gt_ids = {m["gt_element_id"] for m in matches}
    missing = [
        {
            "gt_element_id": _key_of(element, "gt"),
            "gt_type": element.get("type"),
            "page": element.get("page"),
            "text_head": _text_of(element)[:120],
        }
        for gi, element in enumerate(ground_truth)
        if gt_tokens[gi] and _key_of(element, "gt") not in matched_gt_ids
    ]
    spurious = [
        {
            "parsed_element_id": _key_of(element, "pr"),
            "parsed_type": element.get("type"),
            "page": element.get("page"),
            "text_head": _text_of(element)[:120],
        }
        for pi, element in enumerate(parsed)
        if pr_tokens[pi] and pi not in pr_taken
    ]

    return {
        "match_threshold": match_threshold,
        "partial_threshold": partial_threshold,
        "gt_count": len(ground_truth),
        "parsed_count": len(parsed),
        "matches": matches,
        "missing": missing,
        "spurious": spurious,
        # 单元完整性（碎片 / 缺失 / 误合并）—— 主要判据。
        "fragments": integrity["fragmented"],
        "incomplete": integrity["incomplete"],
        "merged": integrity["merged"],
        "complete_unit_count": integrity["complete_unit_count"],
        # 局部重叠统计 —— 辅助诊断，不参与质量判定。
        "partial_overlap": partial_overlap,
        "match_rate": round(len(matches) / max(1, sum(1 for t in gt_tokens if t)), 6),
        "complete_unit_rate": round(
            integrity["complete_unit_count"] / max(1, sum(1 for t in gt_tokens if t)), 6
        ),
    }


# --------------------------------------------------------------------------------------
# 阅读顺序
# --------------------------------------------------------------------------------------


def matched_pairs_for_reading_order(
    alignment: dict[str, Any],
    *,
    min_similarity: float = MATCH_THRESHOLD,
) -> list[tuple[int, int]]:
    """从对齐结果抽出用于阅读顺序比较的 ``(gt_reading_order, parsed_reading_order)`` 对。"""

    pairs: list[tuple[int, int]] = []
    for match in alignment["matches"]:
        if match["similarity"] < min_similarity:
            continue
        gt_order = match.get("gt_reading_order")
        parsed_order = match.get("parsed_reading_order")
        if isinstance(gt_order, int) and isinstance(parsed_order, int):
            pairs.append((gt_order, parsed_order))
    return pairs


def kendall_tau(pairs: Sequence[tuple[int, int]]) -> float | None:
    """在匹配对上计算 Kendall tau（顺序一致性）。样本不足时返回 ``None``。

    只统计两侧都出现的可比对；tau = (concordant - discordant) / comparable。
    复杂度 O(n^2)，n 为匹配元素数（受控语料 < 150），可接受。
    """

    n = len(pairs)
    if n < 2:
        return None
    concordant = 0
    discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            gt_delta = pairs[j][0] - pairs[i][0]
            pr_delta = pairs[j][1] - pairs[i][1]
            if gt_delta == 0 or pr_delta == 0:
                continue
            if (gt_delta > 0) == (pr_delta > 0):
                concordant += 1
            else:
                discordant += 1
    comparable = concordant + discordant
    if comparable == 0:
        return None
    return (concordant - discordant) / comparable


def pair_order_accuracy(pairs: Sequence[tuple[int, int]]) -> float | None:
    """块对顺序正确率：两侧相对次序一致的匹配对占比。"""

    n = len(pairs)
    if n < 2:
        return None
    correct = 0
    comparable = 0
    for i in range(n):
        for j in range(i + 1, n):
            gt_delta = pairs[j][0] - pairs[i][0]
            pr_delta = pairs[j][1] - pairs[i][1]
            if gt_delta == 0 or pr_delta == 0:
                continue
            comparable += 1
            if (gt_delta > 0) == (pr_delta > 0):
                correct += 1
    if comparable == 0:
        return None
    return correct / comparable


def detect_column_interleave(
    ground_truth: Sequence[dict[str, Any]],
    parsed: Sequence[dict[str, Any]],
    alignment: dict[str, Any],
) -> dict[str, Any]:
    """检测"左1 → 右1 → 左2 → 右2"式系统性双栏交叉（方案 §11.2 硬门）。

    判定方式：对每一对"GT 认为先出现"的栏内元素 (a, b)，检查 parser 的阅读顺序是否把
    b 排到了 a 之前。逆序对占可比对的比例即 ``interleave_rate``。

    该方法只用 GT 的阅读顺序与 parser 的阅读顺序，不依赖像素坐标，因此对坐标体系
    完全不同的 parser 同样公平。
    """

    order_to_column = {
        element["reading_order"]: element["column"]
        for element in ground_truth
        if element.get("column") in {"L", "R"} and isinstance(element.get("reading_order"), int)
    }
    # gt_reading_order -> parser_reading_order，仅保留匹配成功的元素。
    gt_to_parsed = {
        gt_order: parsed_order
        for gt_order, parsed_order in matched_pairs_for_reading_order(alignment)
    }

    # 只保留同时具备"栏标记"与"成功匹配"的 GT 顺序号，并按 GT 顺序升序排列。
    orders = sorted(order for order in order_to_column if order in gt_to_parsed)

    inverted = 0
    comparable = 0
    examples: list[dict[str, Any]] = []
    for i in range(len(orders)):
        for j in range(i + 1, len(orders)):
            earlier, later = orders[i], orders[j]
            comparable += 1
            if gt_to_parsed[later] < gt_to_parsed[earlier]:
                inverted += 1
                if len(examples) < 12:
                    examples.append(
                        {
                            "gt_earlier": {"order": earlier, "column": order_to_column[earlier]},
                            "gt_later": {"order": later, "column": order_to_column[later]},
                            "parsed_orders": [gt_to_parsed[earlier], gt_to_parsed[later]],
                        }
                    )

    rate = inverted / comparable if comparable else 0.0
    return {
        "comparable_pairs": comparable,
        "inverted_pairs": inverted,
        "interleave_rate": round(rate, 6),
        # 阈值 0.05：正常双栏读取的偶发错位远低于此值；一旦 parser 逐行交叉，该比率
        # 会接近 1.0。硬门判定留给 orchestrator（Gate 而非指标）。
        "threshold": 0.05,
        "interleave_detected": comparable > 0 and rate > 0.05,
        "examples": examples,
    }


def type_confusion(
    alignment: dict[str, Any],
    *,
    min_similarity: float = MATCH_THRESHOLD,
) -> dict[str, dict[str, int]]:
    """统计匹配对的 (GT type -> parser type) 混淆矩阵。"""

    matrix: dict[str, dict[str, int]] = {}
    for match in alignment["matches"]:
        if match["similarity"] < min_similarity:
            continue
        gt_type = str(match["gt_type"])
        parsed_type = str(match["parsed_type"])
        matrix.setdefault(gt_type, {})
        matrix[gt_type][parsed_type] = matrix[gt_type].get(parsed_type, 0) + 1
    return {key: dict(sorted(value.items())) for key, value in sorted(matrix.items())}


def count_gt_by_type(ground_truth: Iterable[dict[str, Any]]) -> dict[str, int]:
    """按类型统计 GT 元素数量，供 precision/recall 分母使用。"""

    counts: dict[str, int] = {}
    for element in ground_truth:
        counts[str(element.get("type"))] = counts.get(str(element.get("type")), 0) + 1
    return dict(sorted(counts.items()))
