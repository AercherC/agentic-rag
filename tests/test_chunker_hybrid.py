"""Verify the hybrid chunking algorithm contract (Phase 13F §9.3) with a deterministic Fake Tokenizer."""

import pytest

from evidenceflow.domain.document import Document
from evidenceflow.ingestion.chunker import (
    HYBRID_STANDALONE_TYPES,
    ChunkingError,
    chunk_document_structured,
)


class FakeTokenizer:
    """Treat whitespace-separated words as Tokens (word count = token count, so no counting drift)."""

    def encode(self, text: str) -> list[int]:
        return [0] * len(text.split())

    def decode(self, token_ids: list[int]) -> str:
        return " ".join(str(token_id) for token_id in token_ids)


def _document() -> Document:
    """Build the minimal PDF document needed for hybrid chunking."""

    return Document(
        document_id="document-id",
        source_path="docs/example.pdf",
        file_name="example.pdf",
        file_type="pdf",
        content_hash="original-content-hash",
        segments=[],
        metadata={"language": "en"},
    )


def _element(
    element_id: str,
    element_type: str,
    text: str,
    page: int = 1,
    section_path: tuple[str, ...] = (),
) -> dict:
    """Build an element dict in the 13B convention (the chunker consumes only text/element_type/element_id/page/section_path)."""

    return {
        "element_id": element_id,
        "element_type": element_type,
        "text": text,
        "page": page,
        "section_path": list(section_path),
    }


def test_standalone_elements_become_isolated_leading_chunks() -> None:
    """Standalone types become isolated Chunks at the front of the sequence and never enter the body flow."""

    elements = [
        _element("p1", "paragraph", "body one two three", section_path=("1 Intro",)),
        _element("f1", "figure_caption", "Figure 1: architecture", page=2),
        _element("t1", "table", "a b\nc d", page=2),
        _element("p2", "paragraph", "body four five six", section_path=("1 Intro",)),
    ]

    chunks = chunk_document_structured(_document(), elements, FakeTokenizer(), chunk_size=16)

    # Standalone Chunks come first: figure_caption and table each stand alone.
    assert chunks[0].metadata["element_ids"] == ["f1"]
    assert chunks[1].metadata["element_ids"] == ["t1"]
    # Body Chunks contain no standalone elements.
    body_ids = [chunk.metadata["element_ids"] for chunk in chunks[2:]]
    assert all("f1" not in ids and "t1" not in ids for ids in body_ids)
    # chunk_index is contiguous and the page numbers come from the owning element's real page.
    assert [chunk.chunk_index for chunk in chunks] == list(range(len(chunks)))
    assert chunks[0].page_start == 2 and chunks[0].page_end == 2


def test_body_chunks_group_by_continuous_section_path() -> None:
    """Body text is grouped by continuous section_path, and group boundaries force a break (different sections never share a Chunk)."""

    elements = [
        _element("p1", "paragraph", "intro text", section_path=("1 Intro",)),
        _element("p2", "paragraph", "more intro", section_path=("1 Intro",)),
        _element("p3", "paragraph", "method text", section_path=("2 Method", "2.1 Setup")),
        _element("p4", "paragraph", "unattributed text", section_path=()),
    ]

    chunks = chunk_document_structured(_document(), elements, FakeTokenizer(), chunk_size=64)

    grouped_ids = [chunk.metadata["element_ids"] for chunk in chunks]
    # p1/p2 share a group and may share a window; p3 and p4 each get their own window (group
    # boundaries break).
    assert ["p1", "p2"] == grouped_ids[0]
    assert ["p3"] == grouped_ids[1]
    assert ["p4"] == grouped_ids[2]
    # heading_path comes from section_path; a group with no attribution writes no heading_path key.
    assert chunks[0].metadata["heading_path"] == "1 Intro"
    assert chunks[1].metadata["heading_path"] == "2 Method > 2.1 Setup"
    assert "heading_path" not in chunks[2].metadata


def test_hard_budget_never_exceeded_and_windows_do_not_overlap() -> None:
    """No Chunk exceeds chunk_size tokens; hybrid windows have zero overlap (the tail is not carried into the next window)."""

    paragraphs = [
        _element(f"p{i}", "paragraph", " ".join(["word"] * 6), section_path=("1 Intro",))
        for i in range(6)
    ]

    chunks = chunk_document_structured(_document(), paragraphs, FakeTokenizer(), chunk_size=10)

    assert all(chunk.token_count <= 10 for chunk in chunks)
    # Zero overlap: adjacent windows' element_ids do not intersect (each element appears in exactly
    # one window).
    seen: set[str] = set()
    for chunk in chunks:
        ids = set(chunk.metadata["element_ids"])
        assert ids.isdisjoint(seen)
        seen |= ids


def test_oversized_element_is_word_split_within_budget() -> None:
    """An oversized element is word-split across multiple windows, each within chunk_size tokens."""

    element = _element("big", "paragraph", " ".join(["tok"] * 25))

    chunks = chunk_document_structured(_document(), [element], FakeTokenizer(), chunk_size=10)

    assert len(chunks) > 1
    assert all(chunk.token_count <= 10 for chunk in chunks)
    assert all(chunk.metadata["element_ids"] == ["big"] for chunk in chunks)


def test_chunk_ids_are_deterministic_and_metadata_contract_holds() -> None:
    """Chunking twice yields an identical chunk_id sequence; metadata records the strategy and provenance."""

    elements = [
        _element("p1", "paragraph", "body text", section_path=("1 Intro",)),
        _element("e1", "equation", "(1)", page=3),
    ]

    first = chunk_document_structured(_document(), elements, FakeTokenizer(), chunk_size=16)
    second = chunk_document_structured(_document(), elements, FakeTokenizer(), chunk_size=16)

    assert [chunk.chunk_id for chunk in first] == [chunk.chunk_id for chunk in second]
    assert all(chunk.metadata["chunk_strategy"] == "hybrid" for chunk in first)
    # equation is standalone, so it comes first in the sequence; body paragraphs follow.
    assert first[0].metadata["element_ids"] == ["e1"]
    assert first[1].metadata["element_ids"] == ["p1"]


def test_invalid_inputs_fail_fast() -> None:
    """An illegal chunk_size, an empty element stream, and a missing element_id must all fail explicitly."""

    with pytest.raises(ChunkingError):
        chunk_document_structured(_document(), [_element("p1", "paragraph", "x")], FakeTokenizer(), 0)
    with pytest.raises(ChunkingError):
        chunk_document_structured(_document(), [], FakeTokenizer(), 16)
    with pytest.raises(ChunkingError):
        broken = [{"element_type": "paragraph", "text": "x"}]
        chunk_document_structured(_document(), broken, FakeTokenizer(), 16)


def test_standalone_type_set_matches_frozen_ph1_semantics() -> None:
    """The standalone type set must stay on the frozen 13E PH1 convention, preventing accidental spread."""

    assert HYBRID_STANDALONE_TYPES == {"table", "table_caption", "figure_caption", "equation"}
