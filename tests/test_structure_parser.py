"""Verify the decision rules and determinism of the structure-parsing pure functions (Phase 13A/13B port layer)."""

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
    """Build single-page/multi-page test Segments (page numbers start at 1)."""

    return [
        DocumentSegment(content=content, page_number=index + 1)
        for index, content in enumerate(contents)
    ]


def test_detect_headings_numbered_and_named_with_dedup() -> None:
    """Numbered heading detection + full-text uniqueness of named headings + no repeated top-level number."""

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
    # The top-level number "2" repeats -> the second "2 Other Thing" is rejected.
    assert "2 Other Thing" not in titles
    # The named heading "Methods" appears twice across pages -> both are rejected (the table-column
    # noise rule).
    assert "Methods" not in titles


def test_resolve_section_builds_two_level_path() -> None:
    """section path = the nearest top-level heading + the nearest deeper heading under it."""

    segments = _segments("1 Introduction\nBody line.\n2.1 Setup\nBody line 2.\n")
    headings = detect_headings(segments)

    top = resolve_section(headings, 0, 0)  # offset 0 is the heading line itself: a heading governs its own position
    assert top["section"] == "1 Introduction" and top["section_path"] == ["1 Introduction"]

    intro = resolve_section(headings, 0, segments[0].content.index("Body line."))
    assert intro["section"] == "1 Introduction" and intro["subsection"] is None

    setup = resolve_section(headings, 0, segments[0].content.index("Body line 2."))
    assert setup["section"] == "1 Introduction"
    assert setup["subsection"] == "2.1 Setup"
    assert setup["section_path"] == ["1 Introduction", "2.1 Setup"]


def test_table_body_signals() -> None:
    """Table-body decision: either many short lines or a dense numeric run triggers it; a body paragraph does not."""

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
    """Classify block types by the 13B frozen priority: caption > reference > equation > paragraph."""

    assert _classify_block_text(text) == expected


def test_remove_furniture_elements_keeps_everything_else() -> None:
    """Element-level furniture removal deletes only furniture; all other types (including caption/table/equation) are kept."""

    elements = [
        {"element_id": "a", "element_type": "paragraph"},
        {"element_id": "b", "element_type": "furniture"},
        {"element_id": "c", "element_type": "table_caption"},
        {"element_id": "d", "element_type": "furniture"},
    ]

    kept = remove_furniture_elements(elements)

    assert [element["element_id"] for element in kept] == ["a", "c"]
