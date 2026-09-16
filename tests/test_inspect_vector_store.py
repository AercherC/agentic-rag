"""测试 Storage Inspector 的记录过滤、限制和 Metadata 展示。"""

import pytest

from evidenceflow.inspection import format_storage_record, select_records


def test_select_records_filters_file_sorts_and_limits() -> None:
    """脚本层过滤不要求修改正式 VectorStore Contract。"""

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
    """输出证明 Store 保存文本与 Metadata，但不打印不可读的完整向量。"""

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
