"""使用确定性 Domain Objects 验证 Retrieval Gold 匹配与指标公式。"""

import pytest

from evidenceflow.domain.chunk import DocumentChunk, RetrievedChunk
from evidenceflow.domain.evaluation import EvaluationSource
from evidenceflow.eval.metrics import (
    aggregate_question_metrics,
    calculate_question_metrics,
    is_relevant,
    is_relevant_to_any,
)


def make_result(
    rank: int,
    *,
    document_id: str = "doc-a",
    page_start: int | None = 12,
    page_end: int | None = 13,
) -> RetrievedChunk:
    """构造不依赖 Embedding 或 Chroma 的固定检索结果。"""

    return RetrievedChunk(
        chunk=DocumentChunk(
            chunk_id=f"chunk-{rank}-{document_id}",
            document_id=document_id,
            content=f"content at rank {rank}",
            token_count=10,
            chunk_index=rank - 1,
            page_start=page_start,
            page_end=page_end,
            file_name="source.pdf" if page_start is not None else "source.md",
            content_hash="content-hash",
            metadata={},
        ),
        rank=rank,
        similarity_score=1.0 - rank / 10,
    )


def test_same_document_with_pdf_page_overlap_is_relevant() -> None:
    gold = EvaluationSource(document_id="doc-a", page_start=12, page_end=13)
    retrieved = make_result(1, page_start=13, page_end=14)

    assert is_relevant(retrieved, gold)


def test_same_pdf_without_page_overlap_is_not_relevant() -> None:
    gold = EvaluationSource(document_id="doc-a", page_start=12, page_end=13)
    retrieved = make_result(1, page_start=14, page_end=15)

    assert not is_relevant(retrieved, gold)


def test_wrong_document_is_not_relevant_even_when_pages_overlap() -> None:
    gold = EvaluationSource(document_id="doc-a", page_start=12, page_end=13)
    retrieved = make_result(
        1,
        document_id="doc-b",
        page_start=12,
        page_end=13,
    )

    assert not is_relevant(retrieved, gold)


def test_pdf_gold_rejects_retrieved_chunk_without_pages() -> None:
    gold = EvaluationSource(document_id="doc-a", page_start=12, page_end=13)
    retrieved = make_result(1, page_start=None, page_end=None)

    assert not is_relevant(retrieved, gold)


def test_markdown_gold_matches_same_document_at_document_level() -> None:
    gold = EvaluationSource(document_id="doc-a", page_start=None, page_end=None)

    assert is_relevant(make_result(1, document_id="doc-a"), gold)
    assert not is_relevant(make_result(1, document_id="doc-b"), gold)


def test_retrieved_chunk_can_match_second_of_multiple_gold_sources() -> None:
    gold_sources = [
        EvaluationSource(document_id="doc-a"),
        EvaluationSource(document_id="doc-b"),
    ]

    assert is_relevant_to_any(
        make_result(1, document_id="doc-b", page_start=None, page_end=None),
        gold_sources,
    )


def test_rank_three_sets_recall_at_one_three_and_five() -> None:
    gold = [EvaluationSource(document_id="gold-doc")]
    results = [
        make_result(1, document_id="wrong-1", page_start=None, page_end=None),
        make_result(2, document_id="wrong-2", page_start=None, page_end=None),
        make_result(3, document_id="gold-doc", page_start=None, page_end=None),
    ]

    metrics = calculate_question_metrics(results, gold)

    assert metrics.first_relevant_rank == 3
    assert metrics.recall_at_1 == 0.0
    assert metrics.recall_at_3 == 1.0
    assert metrics.recall_at_5 == 1.0
    assert metrics.reciprocal_rank == pytest.approx(1 / 3)


@pytest.mark.parametrize(
    ("rank", "expected_rr"),
    [(1, 1.0), (2, 0.5), (4, 0.25)],
)
def test_reciprocal_rank_for_relevant_result(rank: int, expected_rr: float) -> None:
    metrics = calculate_question_metrics(
        [make_result(rank, document_id="gold-doc", page_start=None, page_end=None)],
        [EvaluationSource(document_id="gold-doc")],
    )

    assert metrics.reciprocal_rank == expected_rr


def test_empty_results_produce_zero_recall_and_rr() -> None:
    metrics = calculate_question_metrics([], [EvaluationSource(document_id="doc-a")])

    assert metrics.first_relevant_rank is None
    assert metrics.recall_at_1 == 0.0
    assert metrics.recall_at_3 == 0.0
    assert metrics.recall_at_5 == 0.0
    assert metrics.reciprocal_rank == 0.0


def test_aggregate_metrics_match_manual_calculation() -> None:
    gold = [EvaluationSource(document_id="gold-doc")]
    first_ranks = [1, 2, None, 4]
    per_question = [
        calculate_question_metrics(
            []
            if rank is None
            else [
                make_result(
                    rank,
                    document_id="gold-doc",
                    page_start=None,
                    page_end=None,
                )
            ],
            gold,
        )
        for rank in first_ranks
    ]

    aggregate = aggregate_question_metrics(per_question)

    assert aggregate.question_count == 4
    assert aggregate.recall_at_1 == 0.25
    assert aggregate.recall_at_3 == 0.5
    assert aggregate.recall_at_5 == 0.75
    assert aggregate.mrr == pytest.approx(0.4375)


def test_metrics_reject_empty_gold_or_empty_aggregate() -> None:
    with pytest.raises(ValueError, match="gold_sources"):
        calculate_question_metrics([], [])
    with pytest.raises(ValueError, match="metrics"):
        aggregate_question_metrics([])
