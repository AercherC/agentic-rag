"""Verify the candidate invariance, stable ordering and batching contract of the Learned Reranker."""

import pytest

from evidenceflow.domain import DocumentChunk, RetrievedChunk
from evidenceflow.retrieval import (
    DenseRerankerRetriever,
    LearnedReranker,
    RerankerError,
    rank_scored_candidates,
)


def _chunk(chunk_id: str, content: str | None = None) -> DocumentChunk:
    """Build a minimal candidate Chunk."""

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
    """Return deterministic scores per passage and record the batches."""

    def __init__(self, scores: dict[str, float]) -> None:
        self.scores = scores
        self.batches: list[list[tuple[str, str]]] = []

    def score_pairs(self, pairs):
        """Produce scores from the content mapping, independent of the batch."""

        self.batches.append(list(pairs))
        return [self.scores[passage] for _, passage in pairs]


def test_reranker_empty_single_and_score_order() -> None:
    """Empty candidates never call the scorer; single and differently scored candidates are ordered by Learned score."""

    scorer = FakeScorer({"a": .1, "b": .9})
    reranker = LearnedReranker(scorer, batch_size=2)

    assert reranker.rerank("query", []) == []
    assert scorer.batches == []
    result = reranker.rerank("query", [_chunk("a"), _chunk("b")])
    assert [row.chunk.chunk_id for row in result] == ["b", "a"]
    assert [row.original_rank for row in result] == [2, 1]


def test_equal_score_tie_break_preserves_frozen_order() -> None:
    """Equal scores keep the Candidate Replay order, so repeated runs give the same result."""

    candidates = [_chunk("b"), _chunk("a")]
    reranker = LearnedReranker(FakeScorer({"a": 1.0, "b": 1.0}))

    first = reranker.rerank("query", candidates)
    second = reranker.rerank("query", candidates)

    assert [row.chunk.chunk_id for row in first] == ["b", "a"]
    assert first == second


def test_batch_size_does_not_change_scores_or_order() -> None:
    """When the scoring function is batch-independent, batch_size only changes how forwards are grouped."""

    candidates = [_chunk(str(i)) for i in range(5)]
    scores = {str(i): float(i) for i in range(5)}
    one = LearnedReranker(FakeScorer(scores), batch_size=1).rerank("q", candidates)
    three = LearnedReranker(FakeScorer(scores), batch_size=3).rerank("q", candidates)

    assert one == three


def test_reranker_accumulates_optional_scorer_timing() -> None:
    """The performance observation only sums batch durations and must never take part in Learned ordering."""

    class TimedScorer(FakeScorer):
        """Provide single-batch tokenizer and forward durations for the deterministic test double."""

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
    """When a budget experiment reuses scores, only the order may change, never the complete candidate set."""

    candidates = [_chunk("a"), _chunk("b"), _chunk("c")]
    result = rank_scored_candidates(candidates, [.2, .9, .1])

    assert [row.chunk.chunk_id for row in result] == ["b", "a", "c"]
    assert {row.chunk.chunk_id for row in result} == {chunk.chunk_id for chunk in candidates}


def test_candidate_limit_duplicates_and_bad_score_count_are_rejected() -> None:
    """The candidate limit, duplicate identities and a wrong scorer output count must all fail explicitly."""

    with pytest.raises(ValueError, match="max_candidates"):
        LearnedReranker(FakeScorer({"a": 1.0}), max_candidates=1).rerank(
            "q", [_chunk("a"), _chunk("b")]
        )
    with pytest.raises(ValueError, match="unique"):
        LearnedReranker(FakeScorer({"a": 1.0})).rerank(
            "q", [_chunk("a"), _chunk("a")]
        )

    class MissingScore:
        """Simulate a third-party model that returns the wrong count."""

        def score_pairs(self, pairs):
            return []

    with pytest.raises(ValueError, match="score count"):
        LearnedReranker(MissingScore()).rerank("q", [_chunk("a")])


class FakeDenseRetriever:
    """Return fixed ANN candidates and record the requested depth."""

    def __init__(self, candidates: list[RetrievedChunk]) -> None:
        self.candidates = candidates
        self.calls: list[tuple[str, int]] = []

    def retrieve(self, query: str, top_k: int) -> list[RetrievedChunk]:
        """Simulate the result of Chroma ANN already ordered by Dense cosine."""

        self.calls.append((query, top_k))
        return self.candidates[:top_k]


def test_dense_reranker_pipeline_preserves_dense_evidence_and_returns_top5() -> None:
    """The formal path requests 20 Dense candidates and hands only the Learned Top5 to the Context."""

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
    """A scorer failure must terminate the request; silently falling back to the unverified Dense ranking is forbidden."""

    class FailingScorer:
        """Simulate a Cross-Encoder inference failure."""

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
