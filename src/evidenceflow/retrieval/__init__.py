"""Public interfaces for formal Dense Retrieval and the Learned Reranker."""

from evidenceflow.retrieval.reranker import (
    DenseRerankerRetriever,
    LearnedReranker,
    RerankerError,
    RerankedChunk,
    TransformersCrossEncoderScorer,
    rank_scored_candidates,
)
from evidenceflow.retrieval.retriever import DenseRetriever, InvalidQueryError
from evidenceflow.retrieval.exact_dense import ExactDenseIndex, ExactDenseIndexError

__all__ = [
    "DenseRetriever",
    "ExactDenseIndex",
    "ExactDenseIndexError",
    "DenseRerankerRetriever",
    "InvalidQueryError",
    "LearnedReranker",
    "RerankerError",
    "RerankedChunk",
    "TransformersCrossEncoderScorer",
    "rank_scored_candidates",
]
