"""Phase 14A-B 受控语料扩展（方案 §7 / §9）。

复用 14A-A 的 corpus14a.DOCX_CASES（W1..W8）与 XLSX_CASES（X1..X10），本模块只**追加**
W9..W12 与 X11..X15 的意图 spec。Gold = 本模块数据本身（与 14A-A 同一纪律：生成器按
spec 写文件，候选 arm 不参与 Gold 自证）。

新 block kind（相对 14A-A 生成器的增量）：
- ``nested_table``     ：表格内嵌套子表（验证 nested table traceability）
- ``merged_table``     ：grid_span 合并 + 嵌套 + cell 段落序（W11）
- XLSX 新增 sheet 属性 ``auto_filter``、``display_table``（display-like 文本值）
"""

from __future__ import annotations

from typing import Any

from . import corpus14a

# --------------------------------------------------------------------------------------
# DOCX 追加：W9..W12（方案 §7）
# --------------------------------------------------------------------------------------

DOCX_CASES_B: list[dict[str, Any]] = [
    {
        "case_id": "W9",
        "description": "Poorly Styled：标题无 Heading 样式，仅 bold+大字号+编号；B1 推断须标 semantic_inferred=true",
        "blocks": [
            {"kind": "title", "text": "Poorly Styled Technical Document"},
            {"kind": "bad_heading", "level": 1, "text": "System Overview", "bold": True, "size": 16, "infer": "bold_size"},
            {"kind": "paragraph", "text": "The system provides a modular ingestion pipeline."},
            {"kind": "bad_heading", "level": 2, "text": "1.1 Data Sources", "bold": True, "size": 13, "number": "1.1", "infer": "numbering"},
            {"kind": "paragraph", "text": "Data sources include PDF and DOCX files."},
            {"kind": "bad_heading", "level": 2, "text": "1.2 Cleaning Rules", "bold": True, "size": 13, "number": "1.2", "infer": "numbering"},
            {"kind": "paragraph", "text": "Cleaning rules must be deterministic and auditable."},
        ],
    },
    {
        "case_id": "W10",
        "description": "Mixed Table / Paragraph Order：p → tbl → p → tbl，读取顺序不得被 paragraphs+tables 拼接打乱",
        "blocks": [
            {"kind": "title", "text": "Interleaved Order Document"},
            {"kind": "paragraph", "text": "Intro paragraph before the first table."},
            {"kind": "table", "headers": [["Id", "Name"]], "rows": [["1", "Alpha"]]},
            {"kind": "paragraph", "text": "Middle paragraph between the two tables."},
            {"kind": "table", "headers": [["Id", "Score"]], "rows": [["1", "0.91"]]},
            {"kind": "paragraph", "text": "Closing paragraph after the second table."},
        ],
    },
    {
        "case_id": "W11",
        "description": "Nested / Merged Table：grid_span 合并 + cell 内嵌套子表 + cell 段落顺序",
        "blocks": [
            {"kind": "title", "text": "Nested Merged Table Document"},
            {"kind": "heading", "level": 1, "text": "Complex Tables"},
            {
                "kind": "merged_table",
                "headers": [["Region", "Metric", "Value"]],
                "rows": [["North", "CD", "8.1"], ["North", "F1", "0.84"]],
                "merges": [["A2", "A3"]],
            },
            {
                "kind": "table",
                "headers": [["Model", "Params"]],
                "rows": [["Small", "1M"]],
                "nested": {
                    "headers": [["Detail", "Note"]],
                    "rows": [["Encoder", "frozen"]],
                },
            },
        ],
    },
    {
        "case_id": "W12",
        "description": "Furniture / Notes / Captions：header/footer/caption/footnote/page break 的 role 分离",
        "blocks": [
            {"kind": "title", "text": "Furniture Separation Document"},
            {"kind": "header", "text": "Quarterly Header"},
            {"kind": "footer", "text": "Internal Footer"},
            {"kind": "heading", "level": 1, "text": "Findings"},
            {"kind": "paragraph", "text": "Main finding text lives in the body."},
            {"kind": "caption", "text": "Table 1: Summary of findings."},
            {"kind": "footnote", "text": "Note: figures rounded to two decimals."},
            {"kind": "page_break"},
            {"kind": "heading", "level": 1, "text": "Appendix"},
            {"kind": "paragraph", "text": "Appendix body text after the page break."},
        ],
    },
]

# --------------------------------------------------------------------------------------
# XLSX 追加：X11..X15（方案 §9）
# --------------------------------------------------------------------------------------

XLSX_CASES_B: list[dict[str, Any]] = [
    {
        "case_id": "X11",
        "description": "Multiple Tables in One Sheet：标题 + 表A + 空行 + Notes + 表B（subtable inference）",
        "sheets": [
            {
                "name": "MultiTable",
                "notes": [{"cell": "A1", "text": "Two Regions Report"}],
                "tables": [
                    {"anchor": "A3", "headers": ["Method", "CD"], "rows": [["Base", 8.9]], "is_named": False},
                    {"anchor": "A6", "headers": ["Variant", "CD"], "rows": [["V1", 7.9], ["V2", 7.4]], "is_named": False},
                ],
                "side_notes": [{"cell": "D6", "text": "Note: V2 is the best variant."}],
            }
        ],
    },
    {
        "case_id": "X12",
        "description": "Header Hierarchy：2025/2026 → Q1/Q2 合并表头（merged cell + hierarchy）",
        "sheets": [
            {
                "name": "Hierarchy",
                "tables": [
                    {
                        "anchor": "A1",
                        "multi_level_header": True,
                        "header_top": ["", "2025", "2025", "2026", "2026"],
                        "header_bottom": ["Region", "Q1", "Q2", "Q1", "Q2"],
                        "merges": [["B1", "C1"], ["D1", "E1"]],
                        "rows": [["North", 10, 12, 14, 16]],
                        "is_named": False,
                    }
                ],
            }
        ],
    },
    {
        "case_id": "X13",
        "description": "Formula / Display Format：raw value + number_format + display-like 文本 + formula/cached 并存",
        "sheets": [
            {
                "name": "Formats",
                "tables": [
                    {
                        "anchor": "A1",
                        "headers": ["Item", "RawValue", "Display", "Sum"],
                        "rows": [
                            ["alpha", 0.816, "81.6%", None],
                            ["beta", 0.744, "74.4%", None],
                        ],
                        "number_formats": {"B2": "0.00", "B3": "0.00"},
                        "is_named": False,
                        "formula_cells": {"D2": ("=B2*1.0", 0.816), "D3": ("=B3*1.0", 0.744)},
                    }
                ],
            }
        ],
    },
    {
        "case_id": "X14",
        "description": "Hidden / Filtered：hidden sheet/row/col + auto filter",
        "sheets": [
            {
                "name": "Filtered",
                "tables": [{"anchor": "A1", "headers": ["k", "v"], "rows": [[1, 2], [3, 4], [5, 6]], "is_named": False}],
                "hidden_rows": [3],
                "hidden_cols": ["C"],
                "auto_filter": {"ref": "A1:B4"},
            },
            {
                "name": "Archived",
                "state": "hidden",
                "tables": [{"anchor": "A1", "headers": ["legacy"], "rows": [["old"]], "is_named": False}],
            },
        ],
    },
    {
        "case_id": "X15",
        "description": "Sparse Business Sheet：标题/说明/KPI key-value/小表/大表/备注混排，连续非空≠必然 table",
        "sheets": [
            {
                "name": "Business",
                "notes": [
                    {"cell": "A1", "text": "Business Review"},
                    {"cell": "A2", "text": "Prepared by the evaluation team."},
                ],
                "kv_pairs": [
                    {"cell": "A4", "text": "Total Runs"},
                    {"cell": "B4", "text": 42},
                    {"cell": "A5", "text": "Owner"},
                    {"cell": "B5", "text": "eval"},
                ],
                "tables": [
                    {"anchor": "A7", "headers": ["Metric", "Value"], "rows": [["Acc", 0.92], ["F1", 0.88]], "is_named": False},
                    {"anchor": "A11", "headers": ["Run", "CD", "F1", "Note"],
                     "rows": [["R1", 8.7, 0.81, "base"], ["R2", 7.4, 0.85, "ours"], ["R3", 7.1, 0.86, "tuned"]]},
                ],
                "side_notes": [{"cell": "A15", "text": "Note: R3 uses the tuned cleaner."}],
            }
        ],
    },
]


def build_gold_b() -> dict[str, Any]:
    """返回 14A-B 受控语料 Gold（14A-A 全量 + 本模块追加）。"""

    return {
        "artifact_type": "phase14a_b_controlled_corpus_gold",
        "schema_version": "phase14a-b-structured-document-v1",
        "docx_cases": corpus14a.DOCX_CASES + DOCX_CASES_B,
        "xlsx_cases": corpus14a.XLSX_CASES + XLSX_CASES_B,
        "docx_case_ids": [c["case_id"] for c in corpus14a.DOCX_CASES + DOCX_CASES_B],
        "xlsx_case_ids": [c["case_id"] for c in corpus14a.XLSX_CASES + XLSX_CASES_B],
    }


def all_case_ids_b() -> list[str]:
    return [c["case_id"] for c in corpus14a.DOCX_CASES + DOCX_CASES_B] + [
        c["case_id"] for c in corpus14a.XLSX_CASES + XLSX_CASES_B
    ]
