"""测试 Text Cleaner 的保守规范化与数据保持规则。"""

import unicodedata
from typing import Literal

import pytest

from evidenceflow.domain import Document, DocumentSegment
from evidenceflow.ingestion import DocumentCleaningError, clean_document


def make_document(
    contents: list[str],
    *,
    file_type: Literal["pdf", "md", "txt"] = "txt",
    page_numbers: list[int | None] | None = None,
) -> Document:
    """使用固定来源字段创建 Cleaner 测试文档。"""

    if page_numbers is None:
        page_numbers = [None] * len(contents)
    segments = [
        DocumentSegment(
            content=content,
            page_number=page_number,
            metadata={"position": index},
        )
        for index, (content, page_number) in enumerate(zip(contents, page_numbers))
    ]
    return Document(
        document_id="document-1",
        source_path=f"notes.{file_type}",
        file_name=f"notes.{file_type}",
        file_type=file_type,
        content_hash="original-bytes-hash",
        segments=segments,
        metadata={"file_size": 128},
    )


def test_unicode_is_normalized_to_nfc_and_chinese_is_preserved() -> None:
    decomposed = "Cafe\u0301 与智能体"

    cleaned = clean_document(make_document([decomposed]))

    assert cleaned.segments[0].content == "Café 与智能体"
    assert unicodedata.is_normalized("NFC", cleaned.segments[0].content)


def test_newlines_are_normalized_and_leading_bom_is_removed() -> None:
    cleaned = clean_document(make_document(["\ufeffA\r\nB\rC\nD"]))

    assert cleaned.segments[0].content == "A\nB\nC\nD"


@pytest.mark.parametrize("file_type", ["pdf", "txt"])
def test_plain_text_whitespace_is_cleaned_conservatively(file_type: str) -> None:
    source = "\n\nParagraph A   \t\r\n\r\n\r\n\r\nParagraph B\t\r\n\n"

    cleaned = clean_document(make_document([source], file_type=file_type))

    assert cleaned.segments[0].content == "Paragraph A\n\nParagraph B"


def test_markdown_structure_spacing_and_indentation_are_preserved() -> None:
    markdown = (
        "\n# Agent  \r\n\r\n- State\r\n- Tool\r\n\r\n\r\n"
        "```python\r\ndef run():\r\n    return \"agent\"\r\n```\r\n\n"
    )

    cleaned = clean_document(make_document([markdown], file_type="md"))
    content = cleaned.segments[0].content

    assert content.startswith("# Agent  \n")
    assert "- State\n- Tool" in content
    assert "\n\n\n```python" in content
    assert "    return \"agent\"" in content
    assert content.endswith("```")


def test_page_numbers_segment_metadata_and_order_are_preserved() -> None:
    original = make_document(
        [" Page 1 \n", " Page 3 \n", " Page 5 \n"],
        file_type="pdf",
        page_numbers=[1, 3, 5],
    )

    cleaned = clean_document(original)

    assert [segment.page_number for segment in cleaned.segments] == [1, 3, 5]
    assert [segment.metadata for segment in cleaned.segments] == [
        {"position": 0},
        {"position": 1},
        {"position": 2},
    ]


def test_document_source_fields_remain_unchanged() -> None:
    original = make_document([" Agent \n"])

    cleaned = clean_document(original)

    for field_name in (
        "document_id",
        "source_path",
        "file_name",
        "file_type",
        "content_hash",
        "metadata",
    ):
        assert getattr(cleaned, field_name) == getattr(original, field_name)


def test_empty_segments_are_removed_without_renumbering_pages() -> None:
    original = make_document(
        ["Agent", "   \n\n ", "Tool"],
        file_type="pdf",
        page_numbers=[1, 2, 3],
    )

    cleaned = clean_document(original)

    assert [segment.content for segment in cleaned.segments] == ["Agent", "Tool"]
    assert [segment.page_number for segment in cleaned.segments] == [1, 3]


def test_document_without_usable_text_is_rejected() -> None:
    original = make_document([" \n\t ", "\r\n"])

    with pytest.raises(
        DocumentCleaningError,
        match="Document contains no usable text after cleaning",
    ):
        clean_document(original)


def test_cleaner_returns_new_data_without_modifying_input() -> None:
    original = make_document(["\ufeffAgent   \r\n\r\n\r\nTool\t"])
    original_content = original.segments[0].content

    cleaned = clean_document(original)

    assert cleaned is not original
    assert cleaned.segments[0] is not original.segments[0]
    assert original.segments[0].content == original_content
    assert cleaned.segments[0].content == "Agent\n\nTool"
