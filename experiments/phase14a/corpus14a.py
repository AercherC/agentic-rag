"""Phase 14A-A 受控语料定义（方案 §8 / §9）。

设计要点（与 13C selfcheck 同一纪律）：
- **Gold = 生成器的意图本身**，由本模块纯 Python 数据描述；候选 parser 不得参与 Gold 自证。
- 生成器（runners/build_corpus_arm.py）按本 spec 写出 .docx / .xlsx；
- 自证（selfcheck14a.py）用标准库直接读 OOXML，验证"写出的东西确实是意图"，不依赖任何 parser。

每个 case 的 ``blocks`` / ``sheets`` 既驱动文件生成，也是 metrics 的对照 Gold。
"""

from __future__ import annotations

from typing import Any

# --------------------------------------------------------------------------------------
# DOCX 受控语料：W1..W8（方案 §8）
# --------------------------------------------------------------------------------------

DOCX_CASES: list[dict[str, Any]] = [
    {
        "case_id": "W1",
        "description": "标准技术文档：Title / H1 / H2 / Paragraph / List",
        "blocks": [
            {"kind": "title", "text": "Sample Technical Document"},
            {"kind": "heading", "level": 1, "text": "Introduction"},
            {"kind": "paragraph", "text": "This document describes the experiment setup."},
            {"kind": "heading", "level": 2, "text": "Background"},
            {"kind": "paragraph", "text": "Prior work focused on point cloud completion."},
            {"kind": "list_item", "style": "List Bullet", "text": "First bullet point"},
            {"kind": "list_item", "style": "List Bullet", "text": "Second bullet point"},
            {"kind": "list_item", "style": "List Number", "text": "First numbered item"},
        ],
    },
    {
        "case_id": "W2",
        "description": "多级标题 + 自动编号（结构层级）",
        "blocks": [
            {"kind": "title", "text": "Multi-level Heading Document"},
            {"kind": "heading", "level": 1, "text": "1 Introduction"},
            {"kind": "heading", "level": 2, "text": "1.1 Background"},
            {"kind": "heading", "level": 3, "text": "1.1.1 Motivation"},
            {"kind": "paragraph", "text": "Motivation body text."},
            {"kind": "heading", "level": 2, "text": "1.2 Related Work"},
        ],
    },
    {
        "case_id": "W3",
        "description": "Table-heavy：简单表 / 合并单元格 / 多行表头",
        "blocks": [
            {"kind": "title", "text": "Table Heavy Document"},
            {"kind": "heading", "level": 1, "text": "Results"},
            {
                "kind": "table",
                "headers": [["Model", "CD", "F1"], ["Method", "Chamfer", "Score"]],
                "multi_row_header": True,
                "rows": [
                    ["Baseline", "8.7", "0.81"],
                    ["Ours", "7.4", "0.85"],
                ],
                "merges": [["A1", "C1"]],
            },
            {"kind": "paragraph", "text": "A caption-like sentence follows the table."},
        ],
    },
    {
        "case_id": "W4",
        "description": "Image / Caption（图像与题注关系）",
        "blocks": [
            {"kind": "title", "text": "Image Caption Document"},
            {"kind": "heading", "level": 1, "text": "Architecture"},
            {"kind": "image", "text": "architecture_diagram", "width": 1.0},
            {"kind": "caption", "text": "Figure 1: The overall network architecture."},
            {"kind": "paragraph", "text": "The diagram shows the pipeline."},
        ],
    },
    {
        "case_id": "W5",
        "description": "Footnote / Endnote-like citation text",
        "blocks": [
            {"kind": "title", "text": "Footnote Document"},
            {"kind": "heading", "level": 1, "text": "Claims"},
            {"kind": "paragraph", "text": "The method improves SOTA performance."},
            {"kind": "footnote", "text": "See Smith et al. 2020 for the original formulation."},
            {"kind": "paragraph", "text": "Another claim with a second note."},
            {"kind": "footnote", "text": "Additional ablation in the appendix."},
        ],
    },
    {
        "case_id": "W6",
        "description": "Header / Footer / Page Break",
        "blocks": [
            {"kind": "title", "text": "Header Footer Document"},
            {"kind": "header", "text": "Running Header Title"},
            {"kind": "footer", "text": "Page Footer Confidential"},
            {"kind": "heading", "level": 1, "text": "Section One"},
            {"kind": "paragraph", "text": "Content before the page break."},
            {"kind": "page_break"},
            {"kind": "heading", "level": 1, "text": "Section Two"},
            {"kind": "paragraph", "text": "Content after the page break."},
        ],
    },
    {
        "case_id": "W7",
        "description": "Text Box / Complex Layout（文档树顺序 vs 视觉位置）",
        "blocks": [
            {"kind": "title", "text": "Textbox Document"},
            {"kind": "heading", "level": 1, "text": "Main Body"},
            {"kind": "paragraph", "text": "This paragraph appears first in the tree."},
            {"kind": "textbox", "text": "Side note rendered as a floating textbox."},
            {"kind": "paragraph", "text": "This paragraph appears after the textbox in the tree."},
        ],
    },
    {
        "case_id": "W8",
        "description": "Badly Styled Word（无 Heading 样式，仅靠字号/加粗/编号表达标题）",
        "blocks": [
            {"kind": "title", "text": "Badly Styled Document"},
            {"kind": "bad_heading", "level": 1, "text": "Introduction", "bold": True, "size": 16},
            {"kind": "paragraph", "text": "Body text under the fake heading."},
            {"kind": "bad_heading", "level": 2, "text": "1.1 Background", "bold": True, "size": 13, "number": "1.1"},
            {"kind": "paragraph", "text": "More body text."},
        ],
    },
]

# --------------------------------------------------------------------------------------
# XLSX 受控语料：X1..X10（方案 §9）
# --------------------------------------------------------------------------------------

XLSX_CASES: list[dict[str, Any]] = [
    {
        "case_id": "X1",
        "description": "单一干净表格（header + rows）",
        "sheets": [
            {
                "name": "Sheet1",
                "tables": [
                    {
                        "anchor": "A1",
                        "headers": ["Method", "CD", "F1"],
                        "rows": [["Baseline", 8.7, 0.81], ["Ours", 7.4, 0.85]],
                        "is_named": False,
                    }
                ],
            }
        ],
    },
    {
        "case_id": "X2",
        "description": "一个 Sheet 多个表（空行分隔）",
        "sheets": [
            {
                "name": "Data",
                "tables": [
                    {"anchor": "A1", "headers": ["A", "B"], "rows": [[1, 2], [3, 4]], "is_named": False},
                    {"anchor": "A5", "headers": ["X", "Y"], "rows": [[5, 6], [7, 8]], "is_named": False},
                ],
            }
        ],
    },
    {
        "case_id": "X3",
        "description": "多 Sheet（Summary / RawData / Ablation）",
        "sheets": [
            {"name": "Summary", "tables": [{"anchor": "A1", "headers": ["Metric", "Value"], "rows": [["Acc", 0.92]], "is_named": False}]},
            {"name": "RawData", "tables": [{"anchor": "A1", "headers": ["id", "val"], "rows": [[1, 10], [2, 20]], "is_named": False}]},
            {"name": "Ablation", "tables": [{"anchor": "A1", "headers": ["variant", "cd"], "rows": [["v1", 8.0], ["v2", 7.5]], "is_named": False}]},
        ],
    },
    {
        "case_id": "X4",
        "description": "Multi-level Header（年份跨季度合并）",
        "sheets": [
            {
                "name": "Quarterly",
                "tables": [
                    {
                        "anchor": "A1",
                        "multi_level_header": True,
                        "header_top": ["", "2025", "2025", "2026", "2026"],
                        "header_bottom": ["Region", "Q1", "Q2", "Q1", "Q2"],
                        "merges": [["B1", "C1"], ["D1", "E1"]],
                        "rows": [["North", 10, 12, 14, 16], ["South", 9, 11, 13, 15]],
                        "is_named": False,
                    }
                ],
            }
        ],
    },
    {
        "case_id": "X5",
        "description": "Merged Cells",
        "sheets": [
            {
                "name": "Merged",
                "tables": [
                    {
                        "anchor": "A1",
                        "headers": ["Name", "Score"],
                        "rows": [["Team A", 90]],
                        "is_named": False,
                        "merges": [["A3", "B3"]],
                    }
                ],
            }
        ],
    },
    {
        "case_id": "X6",
        "description": "Formula-heavy（formula + cached value 双表示）",
        "sheets": [
            {
                "name": "Calc",
                "tables": [
                    {
                        "anchor": "A1",
                        "headers": ["a", "b", "sum"],
                        "rows": [[10, 20, None], [30, 40, None]],
                        "is_named": False,
                        "formula_cells": {"C2": ("=A2+B2", 30), "C3": ("=A3+B3", 70)},
                    }
                ],
            }
        ],
    },
    {
        "case_id": "X7",
        "description": "Hidden Rows / Columns / Sheets",
        "sheets": [
            {
                "name": "Visible",
                "tables": [{"anchor": "A1", "headers": ["k", "v"], "rows": [[1, 2], [3, 4]], "is_named": False}],
                "hidden_rows": [3],
                "hidden_cols": ["C"],
            },
            {
                "name": "HiddenSheet",
                "state": "hidden",
                "tables": [{"anchor": "A1", "headers": ["secret"], "rows": [["x"]], "is_named": False}],
            },
        ],
    },
    {
        "case_id": "X8",
        "description": "Named Tables（Excel Table object）",
        "sheets": [
            {
                "name": "Results",
                "tables": [
                    {
                        "anchor": "A1",
                        "headers": ["Method", "CD", "F1"],
                        "rows": [["Baseline", 8.7, 0.81], ["Ours", 7.4, 0.85]],
                        "is_named": True,
                        "table_name": "ResultTable",
                    }
                ],
            }
        ],
    },
    {
        "case_id": "X9",
        "description": "Sparse Sheet / Notes（表旁 title/description/notes）",
        "sheets": [
            {
                "name": "Report",
                "notes": [
                    {"cell": "A1", "text": "Experiment Report"},
                    {"cell": "A2", "text": "This sheet summarizes the runs."},
                ],
                "tables": [
                    {"anchor": "A4", "headers": ["Method", "Score"], "rows": [["M1", 0.9], ["M2", 0.88]], "is_named": False}
                ],
                "side_notes": [{"cell": "D4", "text": "Note: M2 uses fewer params."}],
            }
        ],
    },
    {
        "case_id": "X10",
        "description": "Charts / Images / Conditional Formatting（不应误当普通 cells）",
        "sheets": [
            {
                "name": "ChartSheet",
                "tables": [{"anchor": "A1", "headers": ["x", "y"], "rows": [[1, 2], [2, 4], [3, 6]], "is_named": False}],
                "chart": {"type": "bar", "title": "xy", "cat_col": "A", "val_col": "B", "anchor": "D1"},
                "conditional_format": {"range": "B2:B4", "type": "color_scale"},
            }
        ],
    },
]


def build_gold() -> dict[str, Any]:
    """返回受控语料的 Gold（即本模块 spec 的结构化副本，供 metrics 对照）。"""

    return {
        "artifact_type": "phase14a_controlled_corpus_gold",
        "docx_cases": DOCX_CASES,
        "xlsx_cases": XLSX_CASES,
        "docx_case_ids": [c["case_id"] for c in DOCX_CASES],
        "xlsx_case_ids": [c["case_id"] for c in XLSX_CASES],
    }


def all_case_ids() -> list[str]:
    return [c["case_id"] for c in DOCX_CASES] + [c["case_id"] for c in XLSX_CASES]
