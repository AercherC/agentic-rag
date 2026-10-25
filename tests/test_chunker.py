"""Verify the token sliding window and page-tracking algorithm with a deterministic Fake Tokenizer."""

from copy import deepcopy

import pytest

from evidenceflow.config import Settings
from evidenceflow.domain.document import Document, DocumentSegment
from evidenceflow.ingestion.chunker import ChunkingError, chunk_document
from evidenceflow.ingestion.tokenizer import BgeM3Tokenizer


class FakeTokenizer:
    """Treat space-separated integers as tokens and use -1 to mark a Segment boundary."""

    def encode(self, text: str) -> list[int]:
        if text == "\n\n":
            return [-1]
        return [int(part) for part in text.split()]

    def decode(self, token_ids: list[int]) -> str:
        # Restore a boundary as two newlines; ordinary tokens stay as readable, assertable integer
        # text.
        result = ""
        for token_id in token_ids:
            if token_id == -1:
                result = result.rstrip() + "\n\n"
            else:
                if result and not result.endswith("\n\n"):
                    result += " "
                result += str(token_id)
        return result


def _document(
    *segments: tuple[str, int | None],
    file_type: str = "pdf",
    metadata: dict[str, object] | None = None,
) -> Document:
    """Build a test document containing only the fields the chunking algorithm needs."""

    return Document(
        document_id="document-id",
        source_path=f"docs/example.{file_type}",
        file_name=f"example.{file_type}",
        file_type=file_type,
        content_hash="original-content-hash",
        segments=[
            DocumentSegment(content=content, page_number=page)
            for content, page in segments
        ],
        metadata=metadata or {"language": "zh"},
    )


@pytest.mark.parametrize(
    ("chunk_size", "chunk_overlap"),
    [(0, 0), (-1, 0), (4, -1), (4, 4), (4, 5)],
)
def test_invalid_chunk_parameters_fail(chunk_size: int, chunk_overlap: int) -> None:
    """An invalid window must fail before the step is computed, to avoid stalling or advancing backwards."""

    with pytest.raises(ChunkingError):
        chunk_document(
            _document(("0 1 2", 1)), FakeTokenizer(), chunk_size, chunk_overlap
        )


def test_sliding_window_overlap_and_exact_ending() -> None:
    """The window advances by size-overlap, and no redundant tail chunk appears when it exactly covers the end."""

    chunks = chunk_document(
        _document(("0 1 2 3 4 5 6 7 8 9", 1)),
        FakeTokenizer(),
        chunk_size=4,
        chunk_overlap=1,
    )

    assert [chunk.content for chunk in chunks] == [
        "0 1 2 3",
        "3 4 5 6",
        "6 7 8 9",
    ]
    assert [chunk.token_count for chunk in chunks] == [4, 4, 4]
    assert [chunk.chunk_index for chunk in chunks] == [0, 1, 2]


def test_final_partial_window_and_short_document_are_preserved() -> None:
    """A final valid body shorter than chunk_size must still be kept, and a short document yields only one chunk."""

    chunks = chunk_document(
        _document(("0 1 2 3 4 5 6 7 8 9 10", 1)),
        FakeTokenizer(),
        chunk_size=4,
        chunk_overlap=1,
    )
    short_chunks = chunk_document(
        _document(("0 1 2", 1)), FakeTokenizer(), chunk_size=4, chunk_overlap=1
    )

    assert chunks[-1].content == "9 10"
    assert chunks[-1].token_count == 2
    assert len(short_chunks) == 1
    assert short_chunks[0].content == "0 1 2"


def test_overlap_is_exact_and_no_source_token_is_lost() -> None:
    """Adjacent complete windows share the exact overlap, and every source token appears at least once."""

    chunks = chunk_document(
        _document(("0 1 2 3 4 5 6 7 8 9 10 11 12 13", 1)),
        FakeTokenizer(),
        chunk_size=8,
        chunk_overlap=2,
    )
    decoded_ids = [[int(part) for part in chunk.content.split()] for chunk in chunks]

    assert all(left[-2:] == right[:2] for left, right in zip(decoded_ids, decoded_ids[1:]))
    assert set().union(*(set(ids) for ids in decoded_ids)) == set(range(14))


def test_page_tracking_for_single_cross_page_and_page_gap() -> None:
    """Page numbers come from the real body tokens inside the window; boundary tokens must not pollute the range."""

    single_page = chunk_document(
        _document(("0 1 2 3 4", 5)), FakeTokenizer(), 3, 1
    )
    cross_page = chunk_document(
        _document(("0 1 2", 1), ("3 4 5", 2)), FakeTokenizer(), 5, 1
    )
    page_gap = chunk_document(
        _document(("0 1", 1), ("2 3", 3)), FakeTokenizer(), 5, 0
    )

    assert all((chunk.page_start, chunk.page_end) == (5, 5) for chunk in single_page)
    assert (cross_page[0].page_start, cross_page[0].page_end) == (1, 2)
    assert (page_gap[0].page_start, page_gap[0].page_end) == (1, 3)
    assert "\n\n" in cross_page[0].content


def test_bge_m3_cross_segment_chunk_does_not_glue_text() -> None:
    """When the real BGE-M3 ignores the whitespace boundary, cross-page body text must still not be glued into a new word."""

    settings = Settings(_env_file=None)
    tokenizer = BgeM3Tokenizer(settings.embedding_model)
    document = _document(
        ("The first page ends with alpha", 1),
        ("beta starts the second page", 2),
    )

    # BGE-M3's SentencePiece ignores pure whitespace, but when each Segment is encoded
    # independently the later segment
    # still carries word-initial information; this test pins that real combined behaviour.
    assert tokenizer.encode("\n\n") == []
    total_tokens = sum(
        len(tokenizer.encode(segment.content)) for segment in document.segments
    )
    chunks = chunk_document(
        document,
        tokenizer,
        chunk_size=total_tokens + 1,
        chunk_overlap=0,
    )

    assert len(chunks) == 1
    assert chunks[0].page_start == 1
    assert chunks[0].page_end == 2
    content = chunks[0].content
    assert "alpha" in content
    assert "beta" in content
    assert "alphabeta" not in content


def test_markdown_page_numbers_remain_none() -> None:
    """Markdown without page numbers must keep an empty page range even when chunked across Segments."""

    chunks = chunk_document(
        _document(("0 1", None), ("2 3", None), file_type="md"),
        FakeTokenizer(),
        chunk_size=3,
        chunk_overlap=1,
    )

    assert all(chunk.page_start is None and chunk.page_end is None for chunk in chunks)


def test_chunk_ids_are_deterministic_and_include_window_configuration() -> None:
    """IDs stay stable across repeated runs on the same input, and change with the window size or overlap."""

    document = _document(("0 1 2 3 4 5 6 7 8 9", 1))
    first = chunk_document(document, FakeTokenizer(), 4, 1)
    repeated = chunk_document(document, FakeTokenizer(), 4, 1)
    changed_size = chunk_document(document, FakeTokenizer(), 5, 1)
    changed_overlap = chunk_document(document, FakeTokenizer(), 4, 2)

    assert [chunk.chunk_id for chunk in first] == [chunk.chunk_id for chunk in repeated]
    assert first[0].chunk_id != changed_size[0].chunk_id
    assert first[0].chunk_id != changed_overlap[0].chunk_id


def test_document_is_not_mutated_and_metadata_is_copied() -> None:
    """The Chunker is a pure transformation boundary, so output metadata must not share mutable objects with the input."""

    document = _document(
        ("0 1 2 3", 1), metadata={"nested": {"source": "test"}}
    )
    original = deepcopy(document)
    chunks = chunk_document(document, FakeTokenizer(), 3, 1)
    chunks[0].metadata["nested"]["source"] = "changed"

    assert document == original
    assert document.metadata["nested"]["source"] == "test"
    assert chunks[0].file_name == document.file_name
    assert chunks[0].content_hash == document.content_hash


def test_empty_document_and_invalid_tokenizer_outputs_fail() -> None:
    """Missing body text or a broken Tokenizer contract must yield an explicit chunking error."""

    empty_document = _document()
    with pytest.raises(ChunkingError):
        chunk_document(empty_document, FakeTokenizer(), 4, 1)

    class InvalidTokenizer(FakeTokenizer):
        def encode(self, text: str) -> tuple[int, ...]:
            return (1, 2)

    with pytest.raises(ChunkingError):
        chunk_document(
            _document(("0 1", 1)), InvalidTokenizer(), chunk_size=4, chunk_overlap=1
        )
