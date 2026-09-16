"""测试 Formal Context Builder 的预算、顺序和来源映射合同。"""

from copy import deepcopy

import pytest

from evidenceflow.context import ContextBuilder
from evidenceflow.domain import (
    DocumentChunk,
    RerankedRetrievedChunk,
    RetrievedChunk,
)


class CharacterTokenizer:
    """以 Unicode 字符为 Token，提供完全可预测的预算边界。"""

    def encode(self, text: str) -> list[int]:
        """让最终渲染文本长度直接等于 Token 数。"""

        return [ord(character) for character in text]

    def decode(self, token_ids: list[int]) -> str:
        """满足项目 TokenizerProtocol；Builder 不需要调用解码。"""

        return "".join(chr(token_id) for token_id in token_ids)


def make_result(
    rank: int,
    content: str,
    *,
    chunk_id: str | None = None,
    file_name: str = "guide.pdf",
    page_start: int | None = 12,
    page_end: int | None = 13,
) -> RetrievedChunk:
    """创建确定性的 Dense 检索候选。"""

    chunk = DocumentChunk(
        chunk_id=chunk_id or f"chunk-{rank}",
        document_id=f"document-{rank}",
        content=content,
        # 此字段故意与字符 Tokenizer 不同，验证 Builder 不复用 Chunk 旧计数。
        token_count=1,
        chunk_index=rank - 1,
        page_start=page_start,
        page_end=page_end,
        file_name=file_name,
        content_hash=f"hash-{rank}",
    )
    return RetrievedChunk(
        chunk=chunk,
        rank=rank,
        similarity_score=0.9 - rank / 100,
    )


def render(label: str, item: RetrievedChunk) -> str:
    """构造规范中的预期 Context 文本，避免依赖私有实现。"""

    lines = [label, f"File: {item.chunk.file_name}"]
    if item.chunk.page_start is not None:
        pages = (
            str(item.chunk.page_start)
            if item.chunk.page_start == item.chunk.page_end
            else f"{item.chunk.page_start}-{item.chunk.page_end}"
        )
        lines.append(f"Pages: {pages}")
    return "\n".join([*lines, "Content:", item.chunk.content])


def test_build_orders_by_retrieval_rank_and_maps_pdf_and_markdown_sources() -> None:
    pdf = make_result(1, "PDF evidence")
    markdown = make_result(
        2,
        "Markdown evidence",
        file_name="notes.md",
        page_start=None,
        page_end=None,
    )
    builder = ContextBuilder(CharacterTokenizer(), max_context_tokens=1000)

    built = builder.build([markdown, pdf])

    assert built.context_text == f"{render('[S1]', pdf)}\n\n{render('[S2]', markdown)}"
    assert "Pages: 12-13" in built.context_text
    assert "Pages: None" not in built.context_text
    assert built.entries[0].source_reference.model_dump() == {
        "source_label": "[S1]",
        "document_id": "document-1",
        "file_name": "guide.pdf",
        "page_start": 12,
        "page_end": 13,
        "chunk_id": "chunk-1",
    }
    assert built.entries[1].source_reference.page_start is None
    assert built.total_tokens == len(built.context_text)


def test_exact_fit_is_included_and_one_token_less_is_skipped() -> None:
    item = make_result(1, "exact")
    exact_tokens = len(render("[S1]", item))
    builder = ContextBuilder(CharacterTokenizer(), max_context_tokens=exact_tokens)

    assert builder.build([item]).included_count == 1
    too_small = builder.build([item], max_tokens=exact_tokens - 1)
    assert too_small.included_count == 0
    assert too_small.skipped_count == 1


def test_oversized_candidate_is_skipped_then_later_shorter_candidate_is_included() -> None:
    first = make_result(1, "first")
    oversized = make_result(2, "x" * 300)
    later = make_result(3, "later")
    expected = f"{render('[S1]', first)}\n\n{render('[S2]', later)}"
    builder = ContextBuilder(CharacterTokenizer(), max_context_tokens=len(expected))

    built = builder.build([first, oversized, later])

    assert built.context_text == expected
    assert [entry.source_label for entry in built.entries] == ["[S1]", "[S2]"]
    assert [entry.retrieval_rank for entry in built.entries] == [1, 3]
    assert built.candidate_count == 3
    assert built.included_count == 2
    assert built.skipped_count == 1


def test_builder_never_truncates_content_and_does_not_mutate_input() -> None:
    items = [make_result(2, "second"), make_result(1, "first")]
    original = deepcopy(items)
    builder = ContextBuilder(CharacterTokenizer(), max_context_tokens=1000)

    built = builder.build(items)

    assert [entry.content for entry in built.entries] == ["first", "second"]
    assert items == original
    assert [item.rank for item in items] == [2, 1]


def test_empty_and_all_oversized_inputs_return_consistent_counts() -> None:
    builder = ContextBuilder(CharacterTokenizer(), max_context_tokens=1)

    empty = builder.build([])
    oversized = builder.build([make_result(1, "a"), make_result(2, "b")])

    assert empty.model_dump() == {
        "context_text": "",
        "entries": [],
        "total_tokens": 0,
        "candidate_count": 0,
        "included_count": 0,
        "skipped_count": 0,
    }
    assert oversized.context_text == ""
    assert oversized.entries == []
    assert oversized.total_tokens == 0
    assert oversized.candidate_count == oversized.skipped_count == 2


def test_budget_counts_full_rendered_entry_instead_of_document_token_count() -> None:
    item = make_result(1, "abc")
    assert item.chunk.token_count == 1
    builder = ContextBuilder(CharacterTokenizer(), max_context_tokens=len(item.chunk.content))

    built = builder.build([item])

    assert built.included_count == 0
    assert len(render("[S1]", item)) > item.chunk.token_count


def test_single_pdf_page_uses_compact_page_header() -> None:
    item = make_result(1, "single page", page_start=7, page_end=7)
    built = ContextBuilder(CharacterTokenizer(), 1000).build([item])

    assert "Pages: 7\n" in built.context_text
    assert "Pages: 7-7" not in built.context_text


def test_dense_score_is_not_exposed_in_context_models() -> None:
    item = make_result(1, "evidence")
    built = ContextBuilder(CharacterTokenizer(), 1000).build([item])
    serialized = str(built.model_dump())

    assert "similarity" not in serialized
    assert str(item.similarity_score) not in built.context_text


def test_phase9_reranker_scores_are_not_exposed_to_context() -> None:
    """Context 只消费 Final Rank 与 Chunk，不泄露 Dense/Reranker 内部分数。"""

    base = make_result(1, "reranked evidence")
    item = RerankedRetrievedChunk(
        chunk=base.chunk,
        rank=1,
        candidate_rank=8,
        similarity_score=0.456789,
        reranker_score=9.876543,
    )

    built = ContextBuilder(CharacterTokenizer(), 1000).build([item])
    serialized = str(built.model_dump())

    assert built.included_count == 1
    assert "candidate_rank" not in serialized
    assert "reranker_score" not in serialized
    assert "0.456789" not in built.context_text
    assert "9.876543" not in built.context_text


def test_constructor_uses_default_budget_and_call_can_override_it() -> None:
    item = make_result(1, "settings")
    builder = ContextBuilder(CharacterTokenizer(), 1000)

    assert builder.build([item]).included_count == 1
    assert builder.build([item], max_tokens=1).included_count == 0


@pytest.mark.parametrize("budget", [0, -1, True, 1.5])
def test_invalid_budget_is_rejected(budget: object) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        ContextBuilder(CharacterTokenizer(), budget)  # type: ignore[arg-type]


def test_duplicate_rank_or_chunk_identity_is_rejected() -> None:
    builder = ContextBuilder(CharacterTokenizer(), 1000)
    with pytest.raises(ValueError, match="duplicate ranks"):
        builder.build([make_result(1, "a"), make_result(1, "b", chunk_id="other")])
    with pytest.raises(ValueError, match="duplicate chunk_id"):
        builder.build([make_result(1, "a"), make_result(2, "b", chunk_id="chunk-1")])
