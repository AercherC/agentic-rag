"""验证结构解析纯函数（Phase 13A/13B 移植层）的判定规则与确定性。"""

import pytest

from evidenceflow.domain.document import DocumentSegment
from evidenceflow.ingestion.structure import (
    _classify_block_text,
    _is_table_body,
    detect_headings,
    remove_furniture_elements,
    resolve_section,
)


def _segments(*contents: str) -> list[DocumentSegment]:
    """构造单页/多页测试 Segment（页码从 1 起）。"""

    return [
        DocumentSegment(content=content, page_number=index + 1)
        for index, content in enumerate(contents)
    ]


def test_detect_headings_numbered_and_named_with_dedup() -> None:
    """编号标题识别 + 具名标题全文唯一性 + 一级编号不重复。"""

    segments = _segments(
        "1 Introduction\nSome paragraph text.\n2 Background\nMore text.\n"
        "2 Other Thing\nDuplicated major number is rejected.\n",
        "Methods\nTable 1\nA B\n",
        "Methods\nNamed heading repeated across pages is rejected.\n",
    )

    headings = detect_headings(segments)
    titles = [heading["title"] for heading in headings]

    assert "1 Introduction" in titles
    assert "2 Background" in titles
    # 一级编号 "2" 重复 → 第二个 "2 Other Thing" 被拒。
    assert "2 Other Thing" not in titles
    # 具名标题 "Methods" 跨页出现两次 → 全部拒绝（表格列头噪声规则）。
    assert "Methods" not in titles


def test_resolve_section_builds_two_level_path() -> None:
    """section 路径 = 最近一级标题 + 其下最近的更深标题。"""

    segments = _segments("1 Introduction\nBody line.\n2.1 Setup\nBody line 2.\n")
    headings = detect_headings(segments)

    top = resolve_section(headings, 0, 0)  # 偏移 0 即标题行起点：标题管辖自身位置
    assert top["section"] == "1 Introduction" and top["section_path"] == ["1 Introduction"]

    intro = resolve_section(headings, 0, segments[0].content.index("Body line."))
    assert intro["section"] == "1 Introduction" and intro["subsection"] is None

    setup = resolve_section(headings, 0, segments[0].content.index("Body line 2."))
    assert setup["section"] == "1 Introduction"
    assert setup["subsection"] == "2.1 Setup"
    assert setup["section_path"] == ["1 Introduction", "2.1 Setup"]


def test_table_body_signals() -> None:
    """表格体判定：多短行 / 数值密集任一命中；正文段落不命中。"""

    multi_short_lines = "Category A\nPlane 1.2\nCabinet 3.4\n"
    dense_numbers = "0.91 0.87 0.95 0.99"
    paragraph = (
        "The method processes point clouds with a learned backbone and "
        "reports consistent improvements across all benchmarks in our study."
    )

    assert _is_table_body(multi_short_lines)
    assert _is_table_body(dense_numbers)
    assert not _is_table_body(paragraph)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Figure 2: Overall pipeline of the framework", "figure_caption"),
        ("Table 3: Ablation results", "table_caption"),
        ("TABLE 3\nAblation results on three benchmarks", "table_caption"),
        ("[12] A. Author, B. Author. Some title. 2023.", "reference_entry"),
        ("\\frac{a}{b} + \\sum_{i} x_i", "equation"),
        ("(3)\nwhere the loss is defined", "equation"),
        ("The proposed method achieves state of the art performance.", "paragraph"),
    ],
)
def test_classify_block_text(text: str, expected: str) -> None:
    """块级类型分类按 13B 冻结优先级：caption > reference > equation > paragraph。"""

    assert _classify_block_text(text) == expected


def test_remove_furniture_elements_keeps_everything_else() -> None:
    """元素级装饰移除只删 furniture，其余类型（含 caption/table/equation）保留。"""

    elements = [
        {"element_id": "a", "element_type": "paragraph"},
        {"element_id": "b", "element_type": "furniture"},
        {"element_id": "c", "element_type": "table_caption"},
        {"element_id": "d", "element_type": "furniture"},
    ]

    kept = remove_furniture_elements(elements)

    assert [element["element_id"] for element in kept] == ["a", "c"]
