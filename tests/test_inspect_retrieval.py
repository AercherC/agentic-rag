"""测试 Retrieval Inspector 的纯格式化函数，不触发交互或真实模型。"""

from evidenceflow.domain import DocumentChunk, RetrievedChunk
from evidenceflow.inspection import (
    content_preview,
    format_page,
    format_retrieval_result,
)


def _chunk() -> DocumentChunk:
    """构造无页码 Markdown Chunk 以覆盖 Inspector 的常见展示路径。"""

    return DocumentChunk(
        chunk_id="chunk-a",
        document_id="doc-a",
        content="A" * 20,
        token_count=5,
        chunk_index=3,
        file_name="notes.md",
        content_hash="hash",
    )


def test_content_preview_and_page_formatting() -> None:
    """默认预览会截断，而完整模式和页码格式保持可读。"""

    assert content_preview(" abcdef ", 3) == "abc…"
    assert content_preview(" abcdef ", 3, full=True) == "abcdef"
    assert format_page(None, None) == "None"
    assert format_page(3, 3) == "3"
    assert format_page(3, 4) == "3-4"


def test_dense_evidence_display() -> None:
    """Dense Inspector 显示 cosine、来源定位与正文预览。"""

    original = format_retrieval_result(
        RetrievedChunk(chunk=_chunk(), rank=1, similarity_score=0.8), preview_chars=5
    )
    assert "Cosine: 0.800000" in original
    assert "Page: None" in original
