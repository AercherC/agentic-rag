"""Test the Storage Inspector's record filtering, limiting, and metadata rendering."""

import pytest

from evidenceflow.inspection import format_storage_record, select_records


def test_select_records_filters_file_sorts_and_limits() -> None:
    """Script-level filtering must not require changing the formal VectorStore Contract."""

    records = select_records(
        ["z", "a", "b"],
        ["z text", "a text", "b text"],
        [
            {"file_name": "other.md"},
            {"file_name": "target.md"},
            {"file_name": "target.md"},
        ],
        limit=1,
        file_name="target.md",
    )

    assert [item["chunk_id"] for item in records] == ["a"]
    with pytest.raises(ValueError):
        select_records([], [], [], limit=0)


def test_storage_format_includes_content_and_metadata_without_embedding() -> None:
    """The output proves the Store keeps text and metadata but does not print unreadable full vectors."""

    rendered = format_storage_record(
        {
            "chunk_id": "c1",
            "content": "stored content",
            "metadata": {
                "document_id": "d1",
                "file_name": "source.pdf",
                "chunk_index": 2,
                "page_start": 4,
                "page_end": 5,
                "token_count": 20,
                "domain_metadata_json": '{"kind":"guide"}',
            },
        }
    )

    assert "Page: 4-5" in rendered
    assert '"kind": "guide"' in rendered
    assert "Content: stored content" in rendered
    assert "embedding" not in rendered.lower()
