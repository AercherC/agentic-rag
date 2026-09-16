"""使用确定性 Fake Tokenizer 验证混合切分（Phase 13F §9.3）算法合同。"""

import pytest

from evidenceflow.domain.document import Document
from evidenceflow.ingestion.chunker import (
    HYBRID_STANDALONE_TYPES,
    ChunkingError,
    chunk_document_structured,
)


class FakeTokenizer:
    """把空格分隔的单词视为 Token（词数 = token 数，无计数漂移）。"""

    def encode(self, text: str) -> list[int]:
        return [0] * len(text.split())

    def decode(self, token_ids: list[int]) -> str:
        return " ".join(str(token_id) for token_id in token_ids)


def _document() -> Document:
    """构造混合切分所需的最小 PDF 文档。"""

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
    """构造 13B 口径的元素 dict（chunker 只消费 text/element_type/element_id/page/section_path）。"""

    return {
        "element_id": element_id,
        "element_type": element_type,
        "text": text,
        "page": page,
        "section_path": list(section_path),
    }


def test_standalone_elements_become_isolated_leading_chunks() -> None:
    """standalone 类型独立成 Chunk、位于序列前部，且不进入正文流。"""

    elements = [
        _element("p1", "paragraph", "body one two three", section_path=("1 Intro",)),
        _element("f1", "figure_caption", "Figure 1: architecture", page=2),
        _element("t1", "table", "a b\nc d", page=2),
        _element("p2", "paragraph", "body four five six", section_path=("1 Intro",)),
    ]

    chunks = chunk_document_structured(_document(), elements, FakeTokenizer(), chunk_size=16)

    # standalone Chunk 在前：figure_caption 与 table 各自独立。
    assert chunks[0].metadata["element_ids"] == ["f1"]
    assert chunks[1].metadata["element_ids"] == ["t1"]
    # 正文 Chunk 不含 standalone 元素。
    body_ids = [chunk.metadata["element_ids"] for chunk in chunks[2:]]
    assert all("f1" not in ids and "t1" not in ids for ids in body_ids)
    # chunk_index 连续、页码取 owner 元素真实页码。
    assert [chunk.chunk_index for chunk in chunks] == list(range(len(chunks)))
    assert chunks[0].page_start == 2 and chunks[0].page_end == 2


def test_body_chunks_group_by_continuous_section_path() -> None:
    """正文按连续 section_path 分组，组边界强制断开（不同 section 不共 Chunk）。"""

    elements = [
        _element("p1", "paragraph", "intro text", section_path=("1 Intro",)),
        _element("p2", "paragraph", "more intro", section_path=("1 Intro",)),
        _element("p3", "paragraph", "method text", section_path=("2 Method", "2.1 Setup")),
        _element("p4", "paragraph", "unattributed text", section_path=()),
    ]

    chunks = chunk_document_structured(_document(), elements, FakeTokenizer(), chunk_size=64)

    grouped_ids = [chunk.metadata["element_ids"] for chunk in chunks]
    # p1/p2 同组可同窗；p3、p4 各自成窗（组边界断开）。
    assert ["p1", "p2"] == grouped_ids[0]
    assert ["p3"] == grouped_ids[1]
    assert ["p4"] == grouped_ids[2]
    # heading_path 来自 section_path；无归属组不写 heading_path 键。
    assert chunks[0].metadata["heading_path"] == "1 Intro"
    assert chunks[1].metadata["heading_path"] == "2 Method > 2.1 Setup"
    assert "heading_path" not in chunks[2].metadata


def test_hard_budget_never_exceeded_and_windows_do_not_overlap() -> None:
    """任何 Chunk token 数 ≤ chunk_size；hybrid 窗口零重叠（尾部不带入下一窗）。"""

    paragraphs = [
        _element(f"p{i}", "paragraph", " ".join(["word"] * 6), section_path=("1 Intro",))
        for i in range(6)
    ]

    chunks = chunk_document_structured(_document(), paragraphs, FakeTokenizer(), chunk_size=10)

    assert all(chunk.token_count <= 10 for chunk in chunks)
    # 零重叠：相邻窗口的 element_ids 无交叉（每个元素只出现在一个窗口）。
    seen: set[str] = set()
    for chunk in chunks:
        ids = set(chunk.metadata["element_ids"])
        assert ids.isdisjoint(seen)
        seen |= ids


def test_oversized_element_is_word_split_within_budget() -> None:
    """超长元素词级拆分为多个窗口，每个窗口 token 数 ≤ chunk_size。"""

    element = _element("big", "paragraph", " ".join(["tok"] * 25))

    chunks = chunk_document_structured(_document(), [element], FakeTokenizer(), chunk_size=10)

    assert len(chunks) > 1
    assert all(chunk.token_count <= 10 for chunk in chunks)
    assert all(chunk.metadata["element_ids"] == ["big"] for chunk in chunks)


def test_chunk_ids_are_deterministic_and_metadata_contract_holds() -> None:
    """重复切分得到完全一致的 chunk_id 序列；metadata 记录策略与溯源。"""

    elements = [
        _element("p1", "paragraph", "body text", section_path=("1 Intro",)),
        _element("e1", "equation", "(1)", page=3),
    ]

    first = chunk_document_structured(_document(), elements, FakeTokenizer(), chunk_size=16)
    second = chunk_document_structured(_document(), elements, FakeTokenizer(), chunk_size=16)

    assert [chunk.chunk_id for chunk in first] == [chunk.chunk_id for chunk in second]
    assert all(chunk.metadata["chunk_strategy"] == "hybrid" for chunk in first)
    # equation 是 standalone → 排在序列最前；正文段落随后。
    assert first[0].metadata["element_ids"] == ["e1"]
    assert first[1].metadata["element_ids"] == ["p1"]


def test_invalid_inputs_fail_fast() -> None:
    """非法 chunk_size、空元素流与缺失 element_id 必须显式失败。"""

    with pytest.raises(ChunkingError):
        chunk_document_structured(_document(), [_element("p1", "paragraph", "x")], FakeTokenizer(), 0)
    with pytest.raises(ChunkingError):
        chunk_document_structured(_document(), [], FakeTokenizer(), 16)
    with pytest.raises(ChunkingError):
        broken = [{"element_type": "paragraph", "text": "x"}]
        chunk_document_structured(_document(), broken, FakeTokenizer(), 16)


def test_standalone_type_set_matches_frozen_ph1_semantics() -> None:
    """standalone 类型集合必须保持 13E PH1 冻结口径，防止无意扩散。"""

    assert HYBRID_STANDALONE_TYPES == {"table", "table_caption", "figure_caption", "equation"}
