"""受控语料的"与解析器无关"有效性自证（Phase 13C-A Step 2）。

设计动机（这一版是对首版的重要修正）
--------------------------------------
首版自证拿 P0（当前工程解析器）当"语料探针"：把 P0 在受控语料上的文本召回当作语料
是否排得对的证据。这个设计有一个结构性缺陷——它把 **parser 的能力边界** 和
**语料的排版质量** 混在同一个数字里：

* P0 不产出表格结构（表格体被当成普通文本块），于是 ``table`` 类 GT 天然失配；
* P0 把页脚页码归入 furniture 并在文本层过滤，于是 ``page_number`` 类 GT 天然失配；
* P0 的块合并会把相邻段落并成一个块，于是段落召回天然偏低。

这些低分反映的是 P0 的局限（本身就是要被测量的结论），却被自证误读成"排版有问题"，
进而阻塞了整个 benchmark。

正确的自证必须与解析器无关：GT 是我们在渲染**之前**写下的内容，所以判据应当回到
PDF 文本层本身，验证三件事：

1. **page token recall** —— GT 声称的文本确实出现在 GT 声称的那一页；
2. **bbox token recall** —— 该文本确实落在 GT 声称的 bbox 矩形内（坐标系与排版正确）；
3. **unclaimed ratio** —— 该页没有 GT 未声明的"幽灵文字"（渲染器没有额外画东西）。

三项同时通过，语料才能作为评测 parser 的基准；否则说明 ``corpus.py`` 的排版引擎或
GT 登记逻辑有 bug，必须先修语料。

坐标系约定：与全项目一致，PDF 点（1/72 inch）、左上原点、``[x0, y0, x1, y1]``。
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

import pymupdf

from . import schema

# bbox 判定容差（PDF 点）。渲染器写入的 bbox 是"文字盒"，而 get_text("words") 返回的
# 是"字形盒"，两者在行距/上伸下延上会有 1pt 级的差异，因此判定单词是否落入 GT 矩形
# 时使用"单词中心点 + 容差"而非严格包含。
BBOX_PAD = 2.0

# 每页允许的未声明 token 比例上限。0 是理想值；留一点余量以吸收 PDF 文本层的
# 连字/符号切分差异。
UNCLAIMED_TOLERANCE = 0.02


def _page_words(page: Any) -> list[dict[str, Any]]:
    """抽取一页的单词及其矩形，并预先分词。

    返回 ``[{"rect": (x0, y0, x1, y1), "tokens": [...]}, ...]``。
    旋转文字（水印）在 PyMuPDF 中返回的是字形外接矩形，中心点判定依然有效。
    """

    words: list[dict[str, Any]] = []
    for item in page.get_text("words"):
        # get_text("words") 的元素布局为 (x0, y0, x1, y1, word, block, line, word_no)
        x0, y0, x1, y1, word = float(item[0]), float(item[1]), float(item[2]), float(item[3]), item[4]
        tokens = schema.tokenize(word)
        if not tokens:
            continue
        words.append({"rect": (x0, y0, x1, y1), "tokens": tokens})
    return words


def _centre_inside(rect: Sequence[float], bbox: Sequence[float], pad: float) -> bool:
    """单词中心点是否落在（带容差的）GT bbox 内。"""

    x0, y0, x1, y1 = float(rect[0]), float(rect[1]), float(rect[2]), float(rect[3])
    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0
    bx0, by0, bx1, by1 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
    return (bx0 - pad) <= cx <= (bx1 + pad) and (by0 - pad) <= cy <= (by1 + pad)


def _recall(reference: Counter, hypothesis: Counter) -> float:
    """多重集合召回：被 hypothesis 覆盖的 reference token 比例。"""

    total = sum(reference.values())
    if total == 0:
        return 1.0
    return sum((reference & hypothesis).values()) / total


def check_document(
    pdf_path: Path,
    gt_document: dict[str, Any],
    *,
    threshold: float = 0.95,
    pad: float = BBOX_PAD,
    unclaimed_tolerance: float = UNCLAIMED_TOLERANCE,
) -> dict[str, Any]:
    """对单个受控模板文档做解析器无关的自证。

    参数
    ----
    pdf_path:
        渲染产出的 PDF。
    gt_document:
        ``corpus.build_controlled_corpus`` 里 ``ground_truth[template_id]`` 结构。
    threshold:
        page / bbox token recall 的通过阈值。
    pad:
        单词中心点判定的容差（PDF 点）。
    unclaimed_tolerance:
        每页允许的未声明 token 比例上限。
    """

    template_id = gt_document["template_id"]
    elements: list[dict[str, Any]] = gt_document["elements"]

    # 扫描件模板在渲染的最后一步被栅格化，文本层按设计就是空的，无法做文本层自证。
    if gt_document.get("rasterized"):
        return {
            "template_id": template_id,
            "excluded": True,
            "reason": "rasterized template has no text layer by design (OCR 专项 O2 的对照输入)",
            "page_count": gt_document.get("page_count"),
            "element_count": len(elements),
        }

    document = pymupdf.open(pdf_path)
    try:
        page_count = document.page_count
        page_words: dict[int, list[dict[str, Any]]] = {
            index + 1: _page_words(document.load_page(index)) for index in range(page_count)
        }
    finally:
        document.close()

    per_element: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    graded_tokens = 0

    for element in elements:
        text = element.get("text") or ""
        tokens = schema.tokenize(text)
        if not tokens:
            # figure 等图形元素没有文本，不参与文本层自证（其正确性由 GT 的 bbox 与
            # 后续 figure 类指标衡量）。
            skipped.append(
                {
                    "element_id": element["element_id"],
                    "type": element["type"],
                    "reason": "no text tokens",
                }
            )
            continue

        page_no = int(element["page"])
        words = page_words.get(page_no, [])
        gt_counter = Counter(tokens)
        graded_tokens += len(tokens)

        # ---- 判据 1：文本是否出现在 GT 声称的页上 ----
        page_counter = Counter()
        for word in words:
            page_counter.update(word["tokens"])
        page_recall = _recall(gt_counter, page_counter)

        # ---- 判据 2：文本是否落在 GT 声称的 bbox 内 ----
        bbox_counter = Counter()
        for word in words:
            if _centre_inside(word["rect"], element["bbox"], pad):
                bbox_counter.update(word["tokens"])
        bbox_recall = _recall(gt_counter, bbox_counter)

        per_element.append(
            {
                "element_id": element["element_id"],
                "type": element["type"],
                "page": page_no,
                "token_count": len(tokens),
                "page_token_recall": round(page_recall, 6),
                "bbox_token_recall": round(bbox_recall, 6),
                "text": text[:160],
            }
        )

    # ---- 判据 3：每页是否有 GT 未声明的"幽灵文字" ----
    claimed_by_page: dict[int, Counter] = {}
    for element in elements:
        page_no = int(element["page"])
        claimed_by_page.setdefault(page_no, Counter()).update(schema.tokenize(element.get("text") or ""))
    per_page: list[dict[str, Any]] = []
    for page_no in sorted(page_words):
        page_counter = Counter()
        for word in page_words[page_no]:
            page_counter.update(word["tokens"])
        total = sum(page_counter.values())
        claimed = claimed_by_page.get(page_no, Counter())
        unclaimed_counter = page_counter - claimed
        unclaimed = sum(unclaimed_counter.values())
        ratio = (unclaimed / total) if total else 0.0
        per_page.append(
            {
                "page": page_no,
                "page_token_count": total,
                "unclaimed_token_count": unclaimed,
                "unclaimed_ratio": round(ratio, 6),
                "unclaimed_samples": [token for token, _ in unclaimed_counter.most_common(12)],
            }
        )

    # 只有"文本型"元素参与打分；figure 之类无文本元素已在 skipped 中。
    min_page_recall = min((row["page_token_recall"] for row in per_element), default=1.0)
    min_bbox_recall = min((row["bbox_token_recall"] for row in per_element), default=1.0)
    max_unclaimed = max((row["unclaimed_ratio"] for row in per_page), default=0.0)
    mean_bbox_recall = (
        sum(row["bbox_token_recall"] for row in per_element) / len(per_element) if per_element else 1.0
    )

    passes = (
        min_page_recall >= threshold
        and min_bbox_recall >= threshold
        and max_unclaimed <= unclaimed_tolerance
    )

    # 只保留失败项，避免自证工件被 768 个元素淹没。
    failures = [
        row for row in per_element if row["page_token_recall"] < threshold or row["bbox_token_recall"] < threshold
    ]
    ghost_pages = [row for row in per_page if row["unclaimed_ratio"] > unclaimed_tolerance]

    return {
        "template_id": template_id,
        "excluded": False,
        "passes": passes,
        "threshold": threshold,
        "bbox_pad": pad,
        "unclaimed_tolerance": unclaimed_tolerance,
        "page_count": page_count,
        "gt_page_max": max((int(e["page"]) for e in elements), default=0),
        "graded_element_count": len(per_element),
        "skipped_element_count": len(skipped),
        "graded_token_count": graded_tokens,
        "min_page_token_recall": round(min_page_recall, 6),
        "min_bbox_token_recall": round(min_bbox_recall, 6),
        "mean_bbox_token_recall": round(mean_bbox_recall, 6),
        "max_unclaimed_ratio": round(max_unclaimed, 6),
        "failure_count": len(failures),
        "failures": failures[:40],
        "ghost_page_count": len(ghost_pages),
        "ghost_pages": ghost_pages[:10],
        "skipped": skipped[:20],
    }


def check_corpus(
    built: dict[str, Any], *, threshold: float = 0.95, pad: float = BBOX_PAD
) -> dict[str, Any]:
    """对整套受控语料做自证，返回可落盘的汇总工件。"""

    from . import sandbox

    checks: list[dict[str, Any]] = []
    for document in built["templates"]:
        template_id = document["template_id"]
        gt_document = built["ground_truth"][template_id]
        checks.append(check_document(Path(document["pdf_path"]), gt_document, threshold=threshold, pad=pad))

    graded = [check for check in checks if not check.get("excluded")]
    failing = [check["template_id"] for check in graded if not check["passes"]]
    return {
        "artifact_type": "phase13c_corpus_self_check",
        "created_at_utc": sandbox.utc_now(),
        "method": (
            "解析器无关的 PDF 文本层自证：①GT 文本是否出现在 GT 声称的页（page token recall）；"
            "②是否落在 GT 声称的 bbox 内（bbox token recall）；③该页是否存在 GT 未声明的幽灵文字"
            "（unclaimed ratio）。三项全部通过才认为语料可用于评测 parser。"
        ),
        "why_not_p0_based": (
            "首版用 P0 当探针会把 parser 的结构性局限（无表格结构、页码归入 furniture）"
            "误判成语料缺陷，因此改为直接校验 PDF 文本层。"
        ),
        "threshold": threshold,
        "unclaimed_tolerance": UNCLAIMED_TOLERANCE,
        "graded_template_count": len(graded),
        "passing_template_count": sum(1 for check in graded if check["passes"]),
        "all_passed": not failing,
        "failing_templates": failing,
        "checks": checks,
    }


def format_summary(payload: dict[str, Any]) -> list[str]:
    """把自证结果格式化成便于日志/报告阅读的短行。"""

    lines: list[str] = []
    for check in payload["checks"]:
        if check.get("excluded"):
            lines.append(f"  {check['template_id']}: EXCLUDED（{check['reason']}）")
            continue
        status = "PASS" if check["passes"] else "FAIL"
        lines.append(
            f"  {check['template_id']}: {status} "
            f"page_recall_min={check['min_page_token_recall']:.4f} "
            f"bbox_recall_min={check['min_bbox_token_recall']:.4f} "
            f"bbox_recall_mean={check['mean_bbox_token_recall']:.4f} "
            f"unclaimed_max={check['max_unclaimed_ratio']:.4f}"
        )
        for row in check["failures"][:6]:
            lines.append(
                f"      ! {row['type']} p{row['page']} page_recall={row['page_token_recall']:.3f} "
                f"bbox_recall={row['bbox_token_recall']:.3f} :: {row['text'][:90]!r}"
            )
        for row in check["ghost_pages"][:3]:
            lines.append(
                f"      ! ghost p{row['page']} ratio={row['unclaimed_ratio']:.4f} "
                f"samples={row['unclaimed_samples'][:8]}"
            )
    return lines
