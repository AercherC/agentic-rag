"""使用确定性 Fake Tokenizer 验证 Token 滑动窗口和页码追踪算法。"""

from copy import deepcopy

import pytest

from evidenceflow.config import Settings
from evidenceflow.domain.document import Document, DocumentSegment
from evidenceflow.ingestion.chunker import ChunkingError, chunk_document
from evidenceflow.ingestion.tokenizer import BgeM3Tokenizer


class FakeTokenizer:
    """把空格分隔的整数视为 Token，并用 -1 表示 Segment 边界。"""

    def encode(self, text: str) -> list[int]:
        if text == "\n\n":
            return [-1]
        return [int(part) for part in text.split()]

    def decode(self, token_ids: list[int]) -> str:
        # 边界恢复为两个换行；普通 Token 保持可读且可断言的整数文本。
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
    """构造只包含分块算法所需字段的测试文档。"""

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
    """非法窗口必须在计算 step 前失败，避免停滞或反向推进。"""

    with pytest.raises(ChunkingError):
        chunk_document(
            _document(("0 1 2", 1)), FakeTokenizer(), chunk_size, chunk_overlap
        )


def test_sliding_window_overlap_and_exact_ending() -> None:
    """窗口按 size-overlap 推进，刚好覆盖末尾时不产生多余尾块。"""

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
    """最后不足 chunk_size 的有效正文仍需保留，短文档只生成一个块。"""

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
    """相邻完整窗口共享精确 overlap，所有原始 Token 至少出现一次。"""

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
    """页码来自窗口内真实正文 Token，边界 Token 不应污染范围。"""

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
    """真实 BGE-M3 忽略空白边界时，跨页正文仍不得粘连成新单词。"""

    settings = Settings(_env_file=None)
    tokenizer = BgeM3Tokenizer(settings.embedding_model)
    document = _document(
        ("The first page ends with alpha", 1),
        ("beta starts the second page", 2),
    )

    # BGE-M3 的 SentencePiece 会忽略纯空白，但每个 Segment 独立编码时，
    # 后一段开头仍携带词首信息；本测试固定验证这条真实组合行为。
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
    """无页码 Markdown 即使跨 Segment 分块也必须保持空页码范围。"""

    chunks = chunk_document(
        _document(("0 1", None), ("2 3", None), file_type="md"),
        FakeTokenizer(),
        chunk_size=3,
        chunk_overlap=1,
    )

    assert all(chunk.page_start is None and chunk.page_end is None for chunk in chunks)


def test_chunk_ids_are_deterministic_and_include_window_configuration() -> None:
    """同输入重复运行 ID 稳定，窗口大小或重叠变化会改变 ID。"""

    document = _document(("0 1 2 3 4 5 6 7 8 9", 1))
    first = chunk_document(document, FakeTokenizer(), 4, 1)
    repeated = chunk_document(document, FakeTokenizer(), 4, 1)
    changed_size = chunk_document(document, FakeTokenizer(), 5, 1)
    changed_overlap = chunk_document(document, FakeTokenizer(), 4, 2)

    assert [chunk.chunk_id for chunk in first] == [chunk.chunk_id for chunk in repeated]
    assert first[0].chunk_id != changed_size[0].chunk_id
    assert first[0].chunk_id != changed_overlap[0].chunk_id


def test_document_is_not_mutated_and_metadata_is_copied() -> None:
    """Chunker 是纯转换边界，输出元数据不得与输入共享可变对象。"""

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
    """缺少正文或 Tokenizer 合同被破坏时应返回明确的分块错误。"""

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
