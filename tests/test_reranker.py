"""验证 Learned Reranker 的候选不变性、稳定排序与批处理合同。"""

import pytest

from evidenceflow.domain import DocumentChunk, RetrievedChunk
from evidenceflow.retrieval import (
    DenseRerankerRetriever,
    LearnedReranker,
    RerankerError,
    rank_scored_candidates,
)


def _chunk(chunk_id: str, content: str | None = None) -> DocumentChunk:
    """构造最小候选 Chunk。"""

    return DocumentChunk(
        chunk_id=chunk_id,
        document_id=f"doc-{chunk_id}",
        content=content or chunk_id,
        token_count=1,
        chunk_index=0,
        file_name="source.pdf",
        content_hash="hash",
    )


class FakeScorer:
    """按 passage 返回确定性分数并记录批次。"""

    def __init__(self, scores: dict[str, float]) -> None:
        self.scores = scores
        self.batches: list[list[tuple[str, str]]] = []

    def score_pairs(self, pairs):
        """以内容映射生成与 batch 无关的分数。"""

        self.batches.append(list(pairs))
        return [self.scores[passage] for _, passage in pairs]


def test_reranker_empty_single_and_score_order() -> None:
    """空候选不调用 scorer；单候选及异分候选按 Learned score 排序。"""

    scorer = FakeScorer({"a": .1, "b": .9})
    reranker = LearnedReranker(scorer, batch_size=2)

    assert reranker.rerank("query", []) == []
    assert scorer.batches == []
    result = reranker.rerank("query", [_chunk("a"), _chunk("b")])
    assert [row.chunk.chunk_id for row in result] == ["b", "a"]
    assert [row.original_rank for row in result] == [2, 1]


def test_equal_score_tie_break_preserves_frozen_order() -> None:
    """同分时保留 Candidate Replay 顺序，重复运行得到相同结果。"""

    candidates = [_chunk("b"), _chunk("a")]
    reranker = LearnedReranker(FakeScorer({"a": 1.0, "b": 1.0}))

    first = reranker.rerank("query", candidates)
    second = reranker.rerank("query", candidates)

    assert [row.chunk.chunk_id for row in first] == ["b", "a"]
    assert first == second


def test_batch_size_does_not_change_scores_or_order() -> None:
    """评分函数与 batch 无关时，batch_size 只改变 forward 分组。"""

    candidates = [_chunk(str(i)) for i in range(5)]
    scores = {str(i): float(i) for i in range(5)}
    one = LearnedReranker(FakeScorer(scores), batch_size=1).rerank("q", candidates)
    three = LearnedReranker(FakeScorer(scores), batch_size=3).rerank("q", candidates)

    assert one == three


def test_reranker_accumulates_optional_scorer_timing() -> None:
    """性能观测只汇总 batch 耗时，不能参与 Learned 排序。"""

    class TimedScorer(FakeScorer):
        """为确定性测试替身提供单批 tokenizer 与 forward 耗时。"""

        def score_pairs(self, pairs):
            scores = super().score_pairs(pairs)
            self.last_timing_seconds = {"model_forward": 0.25}
            return scores

    reranker = LearnedReranker(
        TimedScorer({"a": 1.0, "b": 2.0, "c": 3.0}), batch_size=2
    )
    result = reranker.rerank("q", [_chunk("a"), _chunk("b"), _chunk("c")])

    assert [row.chunk.chunk_id for row in result] == ["c", "b", "a"]
    assert reranker.last_timing_seconds == {"model_forward": 0.5}


def test_precomputed_scores_preserve_all_candidate_ids() -> None:
    """预算实验复用分数时只能重排，不能改变完整候选集合。"""

    candidates = [_chunk("a"), _chunk("b"), _chunk("c")]
    result = rank_scored_candidates(candidates, [.2, .9, .1])

    assert [row.chunk.chunk_id for row in result] == ["b", "a", "c"]
    assert {row.chunk.chunk_id for row in result} == {chunk.chunk_id for chunk in candidates}


def test_candidate_limit_duplicates_and_bad_score_count_are_rejected() -> None:
    """候选上限、重复身份和 scorer 输出数量错误必须显式失败。"""

    with pytest.raises(ValueError, match="max_candidates"):
        LearnedReranker(FakeScorer({"a": 1.0}), max_candidates=1).rerank(
            "q", [_chunk("a"), _chunk("b")]
        )
    with pytest.raises(ValueError, match="unique"):
        LearnedReranker(FakeScorer({"a": 1.0})).rerank(
            "q", [_chunk("a"), _chunk("a")]
        )

    class MissingScore:
        """模拟返回数量错误的第三方模型。"""

        def score_pairs(self, pairs):
            return []

    with pytest.raises(ValueError, match="score count"):
        LearnedReranker(MissingScore()).rerank("q", [_chunk("a")])


class FakeDenseRetriever:
    """返回固定 ANN 候选并记录请求深度。"""

    def __init__(self, candidates: list[RetrievedChunk]) -> None:
        self.candidates = candidates
        self.calls: list[tuple[str, int]] = []

    def retrieve(self, query: str, top_k: int) -> list[RetrievedChunk]:
        """模拟 Chroma ANN 已按 Dense cosine 排序的结果。"""

        self.calls.append((query, top_k))
        return self.candidates[:top_k]


def test_dense_reranker_pipeline_preserves_dense_evidence_and_returns_top5() -> None:
    """正式路径请求 20 个 Dense 候选，并只把 Learned Top5 交给 Context。"""

    chunks = [_chunk(str(index)) for index in range(20)]
    dense_rows = [
        RetrievedChunk(chunk=chunk, rank=index, similarity_score=1 - index / 100)
        for index, chunk in enumerate(chunks, 1)
    ]
    dense = FakeDenseRetriever(dense_rows)
    reranker = LearnedReranker(
        FakeScorer({str(index): float(index) for index in range(20)}),
        batch_size=4,
        max_candidates=20,
    )
    pipeline = DenseRerankerRetriever(
        dense,
        reranker,
        candidate_pool_k=20,
        reranker_input_k=20,
        final_top_k=5,
    )

    result = pipeline.retrieve("query")

    assert dense.calls == [("query", 20)]
    assert [row.chunk.chunk_id for row in result] == ["19", "18", "17", "16", "15"]
    assert [row.rank for row in result] == [1, 2, 3, 4, 5]
    assert result[0].candidate_rank == 20
    assert result[0].similarity_score == dense_rows[-1].similarity_score
    assert result[0].reranker_score == 19.0


def test_dense_reranker_failure_is_explicit_and_never_returns_dense_order() -> None:
    """Scorer 失败必须终止请求，禁止静默退回未经验证的 Dense 排名。"""

    class FailingScorer:
        """模拟 Cross-Encoder 推理失败。"""

        def score_pairs(self, pairs):
            raise RuntimeError("model unavailable")

    dense = FakeDenseRetriever(
        [RetrievedChunk(chunk=_chunk("a"), rank=1, similarity_score=0.8)]
    )
    pipeline = DenseRerankerRetriever(
        dense,
        LearnedReranker(FailingScorer()),
        candidate_pool_k=1,
        reranker_input_k=1,
        final_top_k=1,
    )

    with pytest.raises(RerankerError, match="learned reranker failed"):
        pipeline.retrieve("query")
