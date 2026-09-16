"""O2 OCR 对照臂：对每一页做**强制** OCR，产出逐行文本块。

## 这个 arm 回答什么

方案 §12–§13 要求回答"OCR 应该何时启用"，并明确禁止预设「全部 OCR 更好」。
本 arm 提供 O2（force OCR）这一端的行为样本：

- 在**无文本层**的扫描件上，它是唯一能产出内容的路径 -> 相对 O0 是净收益；
- 在**有可靠文本层**的出生数字 PDF 上，它是净损失，因为 OCR 会把连字符、上标、
  特殊符号和公式一起改写，且不提供 block 类型、章节结构、表格结构。

因此它是"何时启用 OCR"这个问题的**否定对照组**。

## 输出

逐页 OCR 行 -> 按纵向邻近 + 横向重叠合并为"伪段落"块，减少毫无信息量的碎片噪声。
合并规则是显式启发式（见 :data:`LINE_GROUP_Y_GAP_RATIO` 等常量），不是黑盒。
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

from phase13c.runners._common import Timer, build_parser, emit_result, fail_result

import pymupdf  # noqa: E402  （_common 已完成 sys.path 注入）

# 行合并启发式：纵向间距小于 0.9 倍行高、且横向重叠超过较短行的 25% 视为同一段。
LINE_GROUP_Y_GAP_RATIO = 0.9
LINE_GROUP_X_OVERLAP_RATIO = 0.25
RENDER_DPI = 200


def _box_to_rect(box: list) -> list[float]:
    """把 4 点多边形包围盒压成 ``[x0, y0, x1, y1]``。"""

    xs = [float(point[0]) for point in box]
    ys = [float(point[1]) for point in box]
    return [min(xs), min(ys), max(xs), max(ys)]


def _merge_lines(lines: list[dict]) -> list[dict]:
    """把 OCR 行合并为伪段落块。

    RapidOCR 只给出行级框，没有版面结构。若直接输出行，任何段落完整性指标都会因为
    "一个段落被切成 6 行"而全部失败——那衡量的是 OCR 引擎的输出粒度，而不是 ingestion
    质量。因此这里做最小限度的行合并，并把这个规则写进结果供审计。
    """

    if not lines:
        return []
    ordered = sorted(lines, key=lambda item: (item["bbox"][1], item["bbox"][0]))
    blocks: list[dict] = []
    for line in ordered:
        x0, y0, x1, y1 = line["bbox"]
        placed = False
        for block in blocks:
            bx0, by0, bx1, by1 = block["bbox"]
            height = max(y1 - y0, 1.0)
            vertical_gap = y0 - by1
            if -0.5 * height <= vertical_gap <= LINE_GROUP_Y_GAP_RATIO * height:
                overlap = min(x1, bx1) - max(x0, bx0)
                shorter = min(x1 - x0, bx1 - bx0) or 1.0
                if overlap / shorter >= LINE_GROUP_X_OVERLAP_RATIO:
                    block["lines"].append(line["text"])
                    block["bbox"] = [min(bx0, x0), min(by0, y0), max(bx1, x1), max(by1, y1)]
                    placed = True
                    break
        if not placed:
            blocks.append({"lines": [line["text"]], "bbox": [x0, y0, x1, y1], "page": line["page"]})
    return [
        {
            "text": " ".join(block["lines"]),
            "bbox": [round(value, 2) for value in block["bbox"]],
            "page": block["page"],
            "line_count": len(block["lines"]),
        }
        for block in blocks
    ]


def main() -> int:
    """执行一次 OCR 转换。"""

    parser = build_parser("O2_ocr")
    args = parser.parse_args()
    pdf_path = Path(args.pdf)
    timer = Timer()

    try:
        from rapidocr_onnxruntime import RapidOCR

        engine = RapidOCR()
    except Exception as exc:
        fail_result(out_path=Path(args.out), arm="O2_ocr", error=f"engine init failed: {exc}", timer=timer)
        return 0

    blocks: list[dict] = []
    ocr_pages = 0
    page_count = 0
    try:
        with pymupdf.open(pdf_path) as pdf:
            page_count = pdf.page_count
            for page_index in range(page_count):
                page = pdf[page_index]
                pixmap = page.get_pixmap(dpi=RENDER_DPI)
                image = pixmap.tobytes("png")
                result, _elapse = engine(image)
                ocr_pages += 1
                lines: list[dict] = []
                for item in result or []:
                    box, text = item[0], item[1]
                    if not str(text).strip():
                        continue
                    lines.append({"bbox": _box_to_rect(box), "text": str(text).strip(), "page": page_index + 1})
                for block in _merge_lines(lines):
                    blocks.append(block)
    except Exception as exc:
        fail_result(
            out_path=Path(args.out),
            arm="O2_ocr",
            error=f"{type(exc).__name__}: {exc}\n{traceback.format_exc()[-1200:]}",
            timer=timer,
        )
        return 0

    emit_result(
        out_path=Path(args.out),
        payload={
            "status": "OK",
            "arm": "O2_ocr",
            "document_id": pdf_path.stem,
            "page_count": page_count,
            "ocr_pages": ocr_pages,
            "block_count": len(blocks),
            "blocks": blocks,
        },
        timer=timer,
        extra={
            "engine": "rapidocr-onnxruntime",
            "render_dpi": RENDER_DPI,
            "line_grouping": {
                "y_gap_ratio": LINE_GROUP_Y_GAP_RATIO,
                "x_overlap_ratio": LINE_GROUP_X_OVERLAP_RATIO,
            },
        },
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
