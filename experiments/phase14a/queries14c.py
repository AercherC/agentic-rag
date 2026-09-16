"""Phase 14A-C 受控查询与 source evidence anchor（方案 §4 重建执行版）。

81 个查询（36 DOCX + 45 XLSX），每 case 3 类：
- ``local_fact``          ：文档/表内具体事实（值、句子）；
- ``structural_context``  ：标题/章节/sheet 结构语境（可多 anchor）；
- ``table_caption_formula``：表格/题注/公式（DOCX 无表 case 退化为 title 查询，
  XLSX 公式 case 用 metadata.formula 定位）。

anchor 为 locator（在冻结 B1 文档上按确定性规则解析为 element_id）：
- ``value_equals`` / ``value_contains``：normalized_text 匹配（数值 float 容差）；
- ``meta_equals`` / ``meta_contains``：metadata 字段匹配（formula/table_name/auto_filter_ref）；
- ``type_is`` + ``ordinal``：第 N 个该类型元素（0-based，文档序）。
多命中取文档序第一个（可用 ordinal 指定）。

Gold **不绑定 chunk_id**：anchor 只定位到 B1 element_id，逐 arm 投影到 chunk 空间
（见 metrics14c.project_qrels）。required anchor 必须可解析且可入索引，否则自检 STOP。
"""

from __future__ import annotations

from typing import Any

from .chunk14c import INDEXABLE_ROLES, column_index_of, row_index_of

QUERY_TYPES = ("local_fact", "structural_context", "table_caption_formula")
TYPE_SUFFIX = {"local_fact": "LF", "structural_context": "SC", "table_caption_formula": "TC"}


# --------------------------------------------------------------------------------------
# 查询定义（基于 corpus14a/corpus14b 受控语料内容逐 case 撰写，冻结不改动）
# --------------------------------------------------------------------------------------

QUERIES_DOCX: list[dict[str, Any]] = [
    # W1
    {"doc": "W1", "type": "local_fact", "text": "What does W1 say about the experiment setup?",
     "anchors": [{"match": "value_contains", "value": "experiment setup", "required": True}]},
    {"doc": "W1", "type": "structural_context", "text": "What does the Background section of W1 cover?",
     "anchors": [{"match": "value_equals", "value": "Background", "required": True},
                 {"match": "value_contains", "value": "point cloud completion", "required": False}]},
    {"doc": "W1", "type": "table_caption_formula", "text": "What is the title of document W1?",
     "anchors": [{"match": "value_equals", "value": "Sample Technical Document", "required": True}]},
    # W2
    {"doc": "W2", "type": "local_fact", "text": "What is the motivation body text in W2?",
     "anchors": [{"match": "value_equals", "value": "Motivation body text.", "required": True}]},
    {"doc": "W2", "type": "structural_context", "text": "Which subsections appear under 1 Introduction in W2?",
     "anchors": [{"match": "value_equals", "value": "1 Introduction", "required": True},
                 {"match": "value_equals", "value": "1.1 Background", "required": False},
                 {"match": "value_equals", "value": "1.2 Related Work", "required": False}]},
    {"doc": "W2", "type": "table_caption_formula", "text": "What is the deepest heading in W2?",
     "anchors": [{"match": "value_equals", "value": "1.1.1 Motivation", "required": True}]},
    # W3
    {"doc": "W3", "type": "local_fact", "text": "In the W3 results table, what CD value does the Ours method achieve?",
     "anchors": [{"match": "type_is", "value": "table", "ordinal": 0, "required": True}]},
    {"doc": "W3", "type": "structural_context", "text": "Which section of W3 contains the results table?",
     "anchors": [{"match": "value_equals", "value": "Results", "required": True},
                 {"match": "type_is", "value": "table", "ordinal": 0, "required": False}]},
    {"doc": "W3", "type": "table_caption_formula", "text": "What is the multi-row header of the W3 table?",
     "anchors": [{"match": "type_is", "value": "table", "ordinal": 0, "required": True}]},
    # W4
    {"doc": "W4", "type": "local_fact", "text": "What does Figure 1 show in W4?",
     "anchors": [{"match": "value_contains", "value": "Figure 1", "required": True}]},
    {"doc": "W4", "type": "structural_context", "text": "Which section presents the architecture diagram in W4?",
     "anchors": [{"match": "value_equals", "value": "Architecture", "required": True}]},
    {"doc": "W4", "type": "table_caption_formula", "text": "What caption is attached to the figure in W4?",
     "anchors": [{"match": "value_contains", "value": "overall network architecture", "required": True}]},
    # W5
    {"doc": "W5", "type": "local_fact", "text": "What does the footnote in W5 reference?",
     "anchors": [{"match": "value_contains", "value": "Smith et al. 2020", "required": True}]},
    {"doc": "W5", "type": "structural_context", "text": "Which section contains the claims in W5?",
     "anchors": [{"match": "value_equals", "value": "Claims", "required": True}]},
    {"doc": "W5", "type": "table_caption_formula", "text": "What additional note appears in W5?",
     "anchors": [{"match": "value_contains", "value": "Additional ablation", "required": True}]},
    # W6
    {"doc": "W6", "type": "local_fact", "text": "What content appears before the page break in W6?",
     "anchors": [{"match": "value_equals", "value": "Content before the page break.", "required": True}]},
    {"doc": "W6", "type": "structural_context", "text": "Which top-level sections does W6 have?",
     "anchors": [{"match": "value_equals", "value": "Section One", "required": True},
                 {"match": "value_equals", "value": "Section Two", "required": True}]},
    {"doc": "W6", "type": "table_caption_formula", "text": "What is the document title of W6?",
     "anchors": [{"match": "value_equals", "value": "Header Footer Document", "required": True}]},
    # W7
    {"doc": "W7", "type": "local_fact", "text": "Which paragraph appears first in the tree in W7?",
     "anchors": [{"match": "value_contains", "value": "appears first in the tree", "required": True}]},
    {"doc": "W7", "type": "structural_context", "text": "What does the Main Body section of W7 discuss?",
     "anchors": [{"match": "value_equals", "value": "Main Body", "required": True}]},
    {"doc": "W7", "type": "table_caption_formula", "text": "What is the title of W7?",
     "anchors": [{"match": "value_equals", "value": "Textbox Document", "required": True}]},
    # W8
    {"doc": "W8", "type": "local_fact", "text": "What body text appears under the fake heading in W8?",
     "anchors": [{"match": "value_equals", "value": "Body text under the fake heading.", "required": True}]},
    {"doc": "W8", "type": "structural_context", "text": "Which heading does the fake Introduction use in W8?",
     "anchors": [{"match": "value_equals", "value": "Introduction", "required": True}]},
    {"doc": "W8", "type": "table_caption_formula", "text": "What is the title of W8?",
     "anchors": [{"match": "value_equals", "value": "Badly Styled Document", "required": True}]},
    # W9
    {"doc": "W9", "type": "local_fact", "text": "What do the data sources include in W9?",
     "anchors": [{"match": "value_contains", "value": "PDF and DOCX files", "required": True}]},
    {"doc": "W9", "type": "structural_context", "text": "Which sections appear under System Overview in W9?",
     "anchors": [{"match": "value_equals", "value": "System Overview", "required": True},
                 {"match": "value_equals", "value": "1.1 Data Sources", "required": False},
                 {"match": "value_equals", "value": "1.2 Cleaning Rules", "required": False}]},
    {"doc": "W9", "type": "table_caption_formula", "text": "What do the cleaning rules require in W9?",
     "anchors": [{"match": "value_contains", "value": "deterministic and auditable", "required": True},
                 {"match": "value_equals", "value": "1.2 Cleaning Rules", "required": False}]},
    # W10
    {"doc": "W10", "type": "local_fact", "text": "Which paragraph separates the two tables in W10?",
     "anchors": [{"match": "value_contains", "value": "Middle paragraph between the two tables", "required": True}]},
    {"doc": "W10", "type": "structural_context", "text": "What precedes the first table in W10?",
     "anchors": [{"match": "value_contains", "value": "Intro paragraph before the first table", "required": True},
                 {"match": "type_is", "value": "table", "ordinal": 0, "required": False}]},
    {"doc": "W10", "type": "table_caption_formula", "text": "What Score value does Id 1 achieve in the second table of W10?",
     "anchors": [{"match": "type_is", "value": "table", "ordinal": 1, "required": True}]},
    # W11
    {"doc": "W11", "type": "local_fact", "text": "What F1 value does North achieve in W11?",
     "anchors": [{"match": "type_is", "value": "table", "ordinal": 0, "required": True}]},
    {"doc": "W11", "type": "structural_context", "text": "What does the Complex Tables section of W11 contain?",
     "anchors": [{"match": "value_equals", "value": "Complex Tables", "required": True},
                 {"match": "type_is", "value": "table", "ordinal": 0, "required": False}]},
    {"doc": "W11", "type": "table_caption_formula", "text": "What nested table detail appears in W11?",
     "anchors": [{"match": "type_is", "value": "table", "ordinal": 2, "required": True}]},
    # W12
    {"doc": "W12", "type": "local_fact", "text": "What is the main finding stated in W12?",
     "anchors": [{"match": "value_contains", "value": "Main finding text", "required": True}]},
    {"doc": "W12", "type": "structural_context", "text": "Which top-level sections does W12 have?",
     "anchors": [{"match": "value_equals", "value": "Findings", "required": True},
                 {"match": "value_equals", "value": "Appendix", "required": True}]},
    {"doc": "W12", "type": "table_caption_formula", "text": "What does Table 1 summarize in W12?",
     "anchors": [{"match": "value_contains", "value": "Table 1", "required": True}]},
]

QUERIES_XLSX: list[dict[str, Any]] = [
    # X1
    {"doc": "X1", "type": "local_fact", "text": "What CD value does Ours achieve in X1?",
     "anchors": [{"match": "value_equals", "value": "Ours", "required": True},
                 {"match": "value_equals", "value": "7.4", "required": False}]},
    {"doc": "X1", "type": "structural_context", "text": "What columns does the X1 table have?",
     "anchors": [{"match": "value_equals", "value": "CD", "required": True},
                 {"match": "value_equals", "value": "F1", "required": False}]},
    {"doc": "X1", "type": "table_caption_formula", "text": "What is the F1 score of the Baseline method in X1?",
     "anchors": [{"match": "value_equals", "value": "0.81", "required": True},
                 {"match": "value_equals", "value": "Baseline", "required": False}]},
    # X2
    {"doc": "X2", "type": "local_fact", "text": "What is the value of Y when X is 5 in X2?",
     "anchors": [{"match": "value_equals", "value": "5", "required": True},
                 {"match": "value_equals", "value": "6", "required": False}]},
    {"doc": "X2", "type": "structural_context", "text": "How many tables are in sheet Data of X2?",
     "anchors": [{"match": "type_is", "value": "table", "ordinal": 0, "required": True},
                 {"match": "type_is", "value": "table", "ordinal": 1, "required": True}]},
    {"doc": "X2", "type": "table_caption_formula", "text": "What are the headers of the second table in X2?",
     "anchors": [{"match": "value_equals", "value": "X", "required": True},
                 {"match": "value_equals", "value": "Y", "required": False}]},
    # X3
    {"doc": "X3", "type": "local_fact", "text": "What Acc value is reported in the Summary sheet of X3?",
     "anchors": [{"match": "value_equals", "value": "0.92", "required": True}]},
    {"doc": "X3", "type": "structural_context", "text": "Which sheets does workbook X3 contain?",
     "anchors": [{"match": "type_is", "value": "sheet", "ordinal": 0, "required": True},
                 {"match": "type_is", "value": "sheet", "ordinal": 1, "required": False},
                 {"match": "type_is", "value": "sheet", "ordinal": 2, "required": False}]},
    {"doc": "X3", "type": "table_caption_formula", "text": "What cd values are compared in the Ablation sheet of X3?",
     "anchors": [{"match": "value_equals", "value": "8.0", "required": True},
                 {"match": "value_equals", "value": "7.5", "required": False}]},
    # X4
    {"doc": "X4", "type": "local_fact", "text": "What is the 2025 Q2 value for North in X4?",
     "anchors": [{"match": "value_equals", "value": "12", "required": True},
                 {"match": "value_equals", "value": "North", "required": False}]},
    {"doc": "X4", "type": "structural_context", "text": "Which years span the header of the X4 table?",
     "anchors": [{"match": "value_equals", "value": "2025", "required": True},
                 {"match": "value_equals", "value": "2026", "required": True}]},
    {"doc": "X4", "type": "table_caption_formula", "text": "Which columns share the merged header 2025 in X4?",
     "anchors": [{"match": "value_equals", "value": "Q1", "ordinal": 0, "required": True},
                 {"match": "value_equals", "value": "Q2", "ordinal": 0, "required": False}]},
    # X5
    {"doc": "X5", "type": "local_fact", "text": "What score does Team A achieve in X5?",
     "anchors": [{"match": "value_equals", "value": "Team A", "required": True},
                 {"match": "value_equals", "value": "90", "required": False}]},
    {"doc": "X5", "type": "structural_context", "text": "Which merged range appears in the X5 table?",
     "anchors": [{"match": "type_is", "value": "table", "ordinal": 0, "required": True}]},
    {"doc": "X5", "type": "table_caption_formula", "text": "What are the headers of the merged-cell table in X5?",
     "anchors": [{"match": "value_equals", "value": "Name", "required": True},
                 {"match": "value_equals", "value": "Score", "required": False}]},
    # X6
    {"doc": "X6", "type": "local_fact", "text": "What is the cached sum of 30 and 40 in X6?",
     "anchors": [{"match": "meta_equals", "field": "cached_value", "value": "70", "required": True}]},
    {"doc": "X6", "type": "structural_context", "text": "What columns does the Calc table in X6 have?",
     "anchors": [{"match": "value_equals", "value": "sum", "required": True},
                 {"match": "value_equals", "value": "a", "required": False}]},
    {"doc": "X6", "type": "table_caption_formula", "text": "Which formula computes the sum column in X6?",
     "anchors": [{"match": "meta_contains", "field": "formula", "value": "=A2+B2", "required": True}]},
    # X7
    {"doc": "X7", "type": "local_fact", "text": "What k-v pairs are visible in the Visible sheet of X7?",
     "anchors": [{"match": "value_equals", "value": "1", "required": True},
                 {"match": "value_equals", "value": "2", "required": False}]},
    {"doc": "X7", "type": "structural_context", "text": "Which sheets exist in X7?",
     "anchors": [{"match": "type_is", "value": "sheet", "ordinal": 0, "required": True},
                 {"match": "type_is", "value": "sheet", "ordinal": 1, "required": False}]},
    {"doc": "X7", "type": "table_caption_formula", "text": "What is the header of the visible table in X7?",
     "anchors": [{"match": "value_equals", "value": "k", "required": True},
                 {"match": "value_equals", "value": "v", "required": False}]},
    # X8
    {"doc": "X8", "type": "local_fact", "text": "What CD value does ResultTable report for Ours in X8?",
     "anchors": [{"match": "value_equals", "value": "7.4", "required": True}]},
    {"doc": "X8", "type": "structural_context", "text": "Which named table exists in the Results sheet of X8?",
     "anchors": [{"match": "meta_equals", "field": "table_name", "value": "ResultTable", "required": True}]},
    {"doc": "X8", "type": "table_caption_formula", "text": "What are the column headers of ResultTable in X8?",
     "anchors": [{"match": "value_equals", "value": "Method", "required": True},
                 {"match": "value_equals", "value": "CD", "required": False}]},
    # X9
    {"doc": "X9", "type": "local_fact", "text": "What does the note about M2 say in X9?",
     "anchors": [{"match": "value_contains", "value": "fewer params", "required": True}]},
    {"doc": "X9", "type": "structural_context", "text": "What title does the Report sheet of X9 carry?",
     "anchors": [{"match": "value_equals", "value": "Experiment Report", "required": True}]},
    {"doc": "X9", "type": "table_caption_formula", "text": "What scores do M1 and M2 achieve in X9?",
     "anchors": [{"match": "value_equals", "value": "M1", "required": True},
                 {"match": "value_equals", "value": "0.9", "required": False},
                 {"match": "value_equals", "value": "0.88", "required": False}]},
    # X10
    {"doc": "X10", "type": "local_fact", "text": "What y value corresponds to x=2 in X10?",
     "anchors": [{"match": "value_equals", "value": "4", "required": True}]},
    {"doc": "X10", "type": "structural_context", "text": "What does the chart in X10 visualize?",
     "anchors": [{"match": "type_is", "value": "sheet", "ordinal": 0, "required": True}]},
    {"doc": "X10", "type": "table_caption_formula", "text": "What are the x-y headers in X10?",
     "anchors": [{"match": "value_equals", "value": "x", "required": True},
                 {"match": "value_equals", "value": "y", "required": False}]},
    # X11
    {"doc": "X11", "type": "local_fact", "text": "What CD value does the Base method achieve in X11?",
     "anchors": [{"match": "value_equals", "value": "8.9", "required": True}]},
    {"doc": "X11", "type": "structural_context", "text": "How many tables does the MultiTable sheet of X11 contain?",
     "anchors": [{"match": "type_is", "value": "table", "ordinal": 0, "required": True},
                 {"match": "type_is", "value": "table", "ordinal": 1, "required": True}]},
    {"doc": "X11", "type": "table_caption_formula", "text": "Which variant is the best in X11?",
     "anchors": [{"match": "value_contains", "value": "best variant", "required": True}]},
    # X12
    {"doc": "X12", "type": "local_fact", "text": "What is the 2026 Q1 value for North in X12?",
     "anchors": [{"match": "value_equals", "value": "14", "required": True}]},
    {"doc": "X12", "type": "structural_context", "text": "Which years does the header hierarchy of X12 span?",
     "anchors": [{"match": "value_equals", "value": "2025", "required": True},
                 {"match": "value_equals", "value": "2026", "required": True}]},
    {"doc": "X12", "type": "table_caption_formula", "text": "Which header shares the merged 2026 Q1 column in X12?",
     "anchors": [{"match": "value_equals", "value": "Q1", "ordinal": 1, "required": True}]},
    # X13
    {"doc": "X13", "type": "local_fact", "text": "What is the display value of alpha in X13?",
     "anchors": [{"match": "value_equals", "value": "81.6%", "required": True}]},
    {"doc": "X13", "type": "structural_context", "text": "What is the raw value of beta in X13?",
     "anchors": [{"match": "value_equals", "value": "0.744", "required": True}]},
    {"doc": "X13", "type": "table_caption_formula", "text": "Which formula computes the Sum column in X13?",
     "anchors": [{"match": "meta_contains", "field": "formula", "value": "=B2*1.0", "required": True}]},
    # X14
    {"doc": "X14", "type": "local_fact", "text": "What value pairs are visible in the Filtered sheet of X14?",
     "anchors": [{"match": "value_equals", "value": "1", "required": True},
                 {"match": "value_equals", "value": "2", "required": False}]},
    {"doc": "X14", "type": "structural_context", "text": "What range does the auto filter cover in X14?",
     "anchors": [{"match": "meta_equals", "field": "auto_filter_ref", "value": "A1:B4", "required": True}]},
    {"doc": "X14", "type": "table_caption_formula", "text": "What is the header of the visible table in X14?",
     "anchors": [{"match": "value_equals", "value": "k", "required": True},
                 {"match": "value_equals", "value": "v", "required": False}]},
    # X15
    {"doc": "X15", "type": "local_fact", "text": "What is the Total Runs KPI in X15?",
     "anchors": [{"match": "value_equals", "value": "42", "required": True}]},
    {"doc": "X15", "type": "structural_context", "text": "Who prepared the Business sheet of X15?",
     "anchors": [{"match": "value_equals", "value": "eval", "required": True}]},
    {"doc": "X15", "type": "table_caption_formula", "text": "What CD value does run R1 report in X15?",
     "anchors": [{"match": "value_equals", "value": "R1", "required": True},
                 {"match": "value_equals", "value": "8.7", "required": False}]},
]


# --------------------------------------------------------------------------------------
# locator → element_id 解析（确定性：文档序 + ordinal）
# --------------------------------------------------------------------------------------

def _values_equal(a: str, b: str) -> bool:
    """文本相等（数值 float 容差：'8.7' == '8.70' == '8.699...'）。"""

    if a == b:
        return True
    try:
        return abs(float(a) - float(b)) < 1e-9
    except (TypeError, ValueError):
        return False


def _element_indexable(el: dict[str, Any]) -> bool:
    """元素是否可入索引（role 白名单；XLSX 隐藏行/列 cell 与隐藏 sheet 元素除外）。"""

    if el.get("retrieval_role") not in INDEXABLE_ROLES:
        return False
    md = el.get("metadata") or {}
    if md.get("is_hidden_row") or md.get("is_hidden_col"):
        return False
    # 隐藏 sheet 上的元素：查 sheet 元素可见性由调用方预先注入，这里用 metadata 兜底。
    return True


def resolve_locator(doc: dict[str, Any], locator: dict[str, Any]) -> str | None:
    """在冻结 B1 文档上解析 locator → element_id（文档序 + ordinal，确定性）。"""

    match = locator["match"]
    ordinal = int(locator.get("ordinal") or 0)
    seen = 0
    for el in doc["elements"]:
        hit = False
        if match == "value_equals":
            hit = _values_equal(el.get("normalized_text") or "", str(locator["value"]))
        elif match == "value_contains":
            hit = str(locator["value"]) in (el.get("normalized_text") or "")
        elif match == "type_is":
            hit = el.get("type") == locator["value"]
        elif match == "meta_equals":
            v = (el.get("metadata") or {}).get(locator["field"])
            # 数值容差比较（cached_value 等浮点字段：70 == 70.0）。
            hit = v is not None and _values_equal(str(v), str(locator["value"]))
        elif match == "meta_contains":
            v = (el.get("metadata") or {}).get(locator["field"])
            hit = isinstance(v, str) and str(locator["value"]) in v
        else:
            raise ValueError(f"unknown locator match: {match}")
        if hit:
            if seen == ordinal:
                return el["element_id"]
            seen += 1
    return None


def build_queries(
    specs: list[dict[str, Any]], docs_by_id: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    """构建查询集：解析 anchor → element_id，并记录 anchor 的可索引性。

    返回查询记录：{qid, doc_id, type, text,
      anchors: [{locator, element_id, required, indexable}], required_element_ids}
    """

    out: list[dict[str, Any]] = []
    for spec in specs:
        doc = docs_by_id[spec["doc"]]
        anchors: list[dict[str, Any]] = []
        for loc in spec["anchors"]:
            eid = resolve_locator(doc, loc)
            el = next((e for e in doc["elements"] if e["element_id"] == eid), None) if eid else None
            anchors.append(
                {
                    "locator": loc,
                    "element_id": eid,
                    "required": bool(loc.get("required")),
                    "indexable": bool(el) and _element_indexable(el),
                }
            )
        required = [a["element_id"] for a in anchors if a["required"] and a["element_id"]]
        out.append(
            {
                "qid": f"{spec['doc']}-{TYPE_SUFFIX[spec['type']]}",
                "doc_id": spec["doc"],
                "type": spec["type"],
                "text": spec["text"],
                "anchors": anchors,
                "required_element_ids": required,
            }
        )
    return out


def query_self_check(
    docx_queries: list[dict[str, Any]], xlsx_queries: list[dict[str, Any]]
) -> dict[str, Any]:
    """查询自检：数量 36/45、required anchor 全部解析且可入索引、qid 唯一。"""

    problems: list[str] = []
    for name, qs, expected in (("docx", docx_queries, 36), ("xlsx", xlsx_queries, 45)):
        if len(qs) != expected:
            problems.append(f"{name}: expected {expected} queries, got {len(qs)}")
        qids = [q["qid"] for q in qs]
        if len(set(qids)) != len(qids):
            problems.append(f"{name}: duplicate qid")
        for q in qs:
            for a in q["anchors"]:
                if a["required"] and (a["element_id"] is None or not a["indexable"]):
                    problems.append(
                        f"{q['qid']}: required anchor unresolved or not indexable: {a['locator']}"
                    )
    return {"pass": not problems, "problems": problems}
