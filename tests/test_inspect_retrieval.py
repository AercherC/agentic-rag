"""Test the pure formatting functions of the Retrieval Inspector, without triggering interaction or a real model."""

from evidenceflow.domain import DocumentChunk, RetrievedChunk
from evidenceflow.inspection import (
    content_preview,
    format_page,
    format_retrieval_result,
)


def _chunk() -> DocumentChunk:
    """Build a page-less Markdown Chunk to cover the Inspector's common display path."""

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
    """The default preview truncates, while full mode and page formatting stay readable."""

    assert content_preview(" abcdef ", 3) == "abc…"
    assert content_preview(" abcdef ", 3, full=True) == "abcdef"
    assert format_page(None, None) == "None"
    assert format_page(3, 3) == "3"
    assert format_page(3, 4) == "3-4"


def test_dense_evidence_display() -> None:
    """The Dense Inspector shows the cosine, the source locator and the body preview."""

    original = format_retrieval_result(
        RetrievedChunk(chunk=_chunk(), rank=1, similarity_score=0.8), preview_chars=5
    )
    assert "Cosine: 0.800000" in original
    assert "Page: None" in original
