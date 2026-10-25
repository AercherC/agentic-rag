"""Provides the Learned Cross-Encoder rerank boundary and the Transformers implementation used by the online retrieval chain (Exact Dense20 -> Reranker -> Top5)."""

from __future__ import annotations

import math
import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from evidenceflow.domain.chunk import (
    DocumentChunk,
    RerankedRetrievedChunk,
    RetrievedChunk,
)
from evidenceflow.model_device import resolve_model_device
from evidenceflow.retrieval.retriever import DenseRetriever, InvalidQueryError
from evidenceflow.tracing import TraceRecorder, safe_record


logger = logging.getLogger(__name__)


class RerankerError(RuntimeError):
    """Signals that candidates were recalled but the Learned Reranker could not produce a trustworthy ordering."""


class PairScorerProtocol(Protocol):
    """Defines the minimal Query-Passage batch scoring interface the Reranker needs."""

    def score_pairs(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        """Return finite relevance scores in one-to-one correspondence with the input order."""


@dataclass(frozen=True)
class RerankedChunk:
    """Holds the Learned score, the final rank and the candidate's original order."""

    chunk: DocumentChunk
    rank: int
    relevance_score: float
    original_rank: int


class LearnedReranker:
    """Batch-scores candidates with a replaceable PairScorer and applies a stable deterministic ordering."""

    def __init__(
        self,
        scorer: PairScorerProtocol,
        *,
        batch_size: int = 4,
        max_candidates: int = 100,
    ) -> None:
        """Store the batch size and candidate cap; reject booleans posing as positive integers."""

        for value, name in ((batch_size, "batch_size"), (max_candidates, "max_candidates")):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        self.scorer = scorer
        self.batch_size = batch_size
        self.max_candidates = max_candidates
        self.last_truncated_pair_count = 0
        # Aggregate only the stage timings the scorer exposes; test doubles missing the field are
        # treated as zero.
        self.last_timing_seconds: dict[str, float] = {}

    def rerank(
        self,
        query: str,
        candidates: Sequence[DocumentChunk],
        *,
        top_k: int | None = None,
    ) -> list[RerankedChunk]:
        """Rerank the frozen candidates; only the order changes, chunks are never added, removed or modified."""

        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must not be blank")
        if len(candidates) > self.max_candidates:
            raise ValueError("candidate count exceeds max_candidates")
        ids = [chunk.chunk_id for chunk in candidates]
        if len(ids) != len(set(ids)):
            raise ValueError("candidate chunk IDs must be unique")
        if top_k is None:
            effective_top_k = len(candidates)
        else:
            if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
                raise ValueError("top_k must be a positive integer")
            effective_top_k = min(top_k, len(candidates))

        scores: list[float] = []
        self.last_truncated_pair_count = 0
        self.last_timing_seconds = {}
        # Batching only bounds memory; all scores are collected and sorted together, so batch size
        # never changes the result.
        for start in range(0, len(candidates), self.batch_size):
            batch = candidates[start : start + self.batch_size]
            batch_scores = self.scorer.score_pairs(
                [(query, chunk.content) for chunk in batch]
            )
            # The real Transformers scorer exposes the per-batch truncation count; a test double
            # without that field counts as 0.
            self.last_truncated_pair_count += int(
                getattr(self.scorer, "last_truncated_pair_count", 0)
            )
            for name, seconds in getattr(self.scorer, "last_timing_seconds", {}).items():
                self.last_timing_seconds[name] = self.last_timing_seconds.get(name, 0.0) + float(seconds)
            if len(batch_scores) != len(batch):
                raise ValueError("pair scorer returned an unexpected score count")
            if any(not math.isfinite(score) for score in batch_scores):
                raise ValueError("pair scorer returned a non-finite score")
            scores.extend(float(score) for score in batch_scores)

        return rank_scored_candidates(candidates, scores, top_k=effective_top_k)


def rank_scored_candidates(
    candidates: Sequence[DocumentChunk],
    scores: Sequence[float],
    *,
    top_k: int | None = None,
) -> list[RerankedChunk]:
    """Sort precomputed Learned scores so budget experiments can reuse a single model forward pass."""

    if len(candidates) != len(scores):
        raise ValueError("candidates and scores must have the same length")
    if any(not math.isfinite(score) for score in scores):
        raise ValueError("scores must be finite")
    effective_top_k = len(candidates) if top_k is None else top_k
    if isinstance(effective_top_k, bool) or not isinstance(effective_top_k, int) or effective_top_k < 0:
        raise ValueError("top_k must be a non-negative integer")
    indexed = list(enumerate(zip(candidates, scores), start=1))
    # Ties keep the frozen candidate order first; chunk_id is the final stable-identity tiebreaker.
    indexed.sort(key=lambda item: (-item[1][1], item[0], item[1][0].chunk_id))
    return [
        RerankedChunk(
            chunk=chunk.model_copy(deep=True),
            rank=rank,
            relevance_score=score,
            original_rank=original_rank,
        )
        for rank, (original_rank, (chunk, score)) in enumerate(
            indexed[: min(effective_top_k, len(indexed))], start=1
        )
    ]


class TransformersCrossEncoderScorer:
    """Compute Cross-Encoder logits with a Hugging Face Sequence Classification model."""

    def __init__(
        self,
        model_name: str,
        *,
        device: str = "auto",
        dtype: str = "float32",
        max_length: int = 512,
        inference_mode: bool = False,
    ) -> None:
        """Load the frozen model; the device and truncation length must be recorded explicitly by the experiment manifest."""

        if not model_name.strip():
            raise ValueError("model_name must not be blank")
        if isinstance(max_length, bool) or not isinstance(max_length, int) or max_length <= 0:
            raise ValueError("max_length must be a positive integer")
        if dtype != "float32":
            # Phase 9 promotes only the validated CPU float32 path; lower precision belongs to a
            # later performance phase.
            raise ValueError("dtype must be float32 in the promoted runtime")
        effective_device = resolve_model_device(device)
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.model_name = model_name
        self.device = effective_device
        self.dtype = dtype
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        # The Tokenizer stays on CPU; only the encoded batch is moved to the model device before
        # the forward pass.
        self.model.to(device=self.device, dtype=torch.float32)
        self.model.eval()
        self._torch = torch
        self.forward_passes = 0
        self.last_truncated_pair_count = 0
        self.inference_mode = inference_mode
        # Only the stage timings of the most recent batch are kept, to locate bottlenecks in
        # performance experiments;
        # they take no part in scoring or ranking.
        self.last_timing_seconds = {
            "untruncated_tokenization": 0.0,
            "model_tokenization": 0.0,
            "model_forward": 0.0,
            "score_postprocess": 0.0,
        }

    def score_pairs(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        """Run one forward pass over a single batch and return the raw logits."""

        if not pairs:
            self.last_truncated_pair_count = 0
            return []
        # First obtain the untruncated length for observability only; reaching the limit must never
        # be misreported as truncation.
        tokenization_started = time.perf_counter()
        untruncated = self.tokenizer(
            list(pairs),
            padding=False,
            truncation=False,
        )
        self.last_timing_seconds["untruncated_tokenization"] = (
            time.perf_counter() - tokenization_started
        )
        self.last_truncated_pair_count = sum(
            len(input_ids) > self.max_length
            for input_ids in untruncated["input_ids"]
        )
        tokenization_started = time.perf_counter()
        inputs = self.tokenizer(
            list(pairs),
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)
        self.last_timing_seconds["model_tokenization"] = time.perf_counter() - tokenization_started
        forward_started = time.perf_counter()
        # inference_mode only removes autograd metadata; the model, inputs and float32 logits stay
        # unchanged.
        context = self._torch.inference_mode if self.inference_mode else self._torch.no_grad
        with context():
            logits = self.model(**inputs, return_dict=True).logits.view(-1).float()
        self.last_timing_seconds["model_forward"] = time.perf_counter() - forward_started
        self.forward_passes += 1
        postprocess_started = time.perf_counter()
        scores = logits.cpu().tolist()
        self.last_timing_seconds["score_postprocess"] = time.perf_counter() - postprocess_started
        return scores


class DenseRerankerRetriever:
    """Execute Original Dense Candidate Generation followed by final retrieval with the Learned Reranker."""

    retrieval_strategy = "dense_reranker"

    def __init__(
        self,
        dense_retriever: DenseRetriever,
        reranker: LearnedReranker,
        *,
        candidate_pool_k: int = 20,
        reranker_input_k: int = 20,
        final_top_k: int = 5,
        trace_recorder: TraceRecorder | None = None,
    ) -> None:
        """Freeze the layered contract of candidate pool, Reranker input and final depth.

        ``trace_recorder`` is optional (default None): it only observes retrieval.dense.* and rerank.* events and does not
        affect candidates, ordering, scores, exceptions or logging (PROJECT_SPEC §12.5).
        """

        for value, name in (
            (candidate_pool_k, "candidate_pool_k"),
            (reranker_input_k, "reranker_input_k"),
            (final_top_k, "final_top_k"),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if reranker_input_k > candidate_pool_k:
            raise ValueError("reranker_input_k must not exceed candidate_pool_k")
        if final_top_k > reranker_input_k:
            raise ValueError("final_top_k must not exceed reranker_input_k")
        self.dense_retriever = dense_retriever
        self.reranker = reranker
        self.candidate_pool_k = candidate_pool_k
        self.reranker_input_k = reranker_input_k
        self.final_top_k = final_top_k
        # Public attribute: callers (including the B2 trace comparison experiment) may swap the
        # recorder between two retrieve calls.
        self.trace_recorder = trace_recorder

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[RerankedRetrievedChunk]:
        """Generate candidates via ANN Top-K and rerank them; on failure raise explicitly and never return the Dense order."""

        if not isinstance(query, str) or not query.strip():
            raise InvalidQueryError("query must not be empty or whitespace")
        effective_top_k = self.final_top_k if top_k is None else top_k
        if (
            isinstance(effective_top_k, bool)
            or not isinstance(effective_top_k, int)
            or effective_top_k <= 0
        ):
            raise ValueError("top_k must be a positive integer")
        if effective_top_k > self.reranker_input_k:
            raise ValueError("top_k must not exceed reranker_input_k")

        # This call pins a single recorder; when it is None no payload is built below, so the cost
        # matches an untraced run.
        recorder = self.trace_recorder
        safe_record(recorder, "retrieval.dense.started")
        candidates = self.dense_retriever.retrieve(query, top_k=self.candidate_pool_k)
        if recorder is not None:
            # Record only IDs and the raw float32 similarity, never the chunk body.
            safe_record(
                recorder,
                "retrieval.dense.completed",
                candidate_count=len(candidates),
                candidates=[
                    {
                        "rank": item.rank,
                        "chunk_id": item.chunk.chunk_id,
                        "similarity_score": item.similarity_score,
                    }
                    for item in candidates
                ],
            )
        selected = candidates[: self.reranker_input_k]
        candidate_ids = [item.chunk.chunk_id for item in selected]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("dense candidates must have unique chunk IDs")

        safe_record(recorder, "rerank.started")
        started = time.perf_counter()
        try:
            reranked = self.reranker.rerank(
                query,
                [item.chunk for item in selected],
                top_k=effective_top_k,
            )
        except Exception as exc:
            # An unvalidated raw Dense ordering must not pass itself off as a successful Reranker
            # output.
            raise RerankerError("learned reranker failed") from exc
        latency = time.perf_counter() - started
        candidate_by_id = {item.chunk.chunk_id: item for item in selected}
        results = [
            RerankedRetrievedChunk(
                chunk=item.chunk,
                rank=item.rank,
                candidate_rank=candidate_by_id[item.chunk.chunk_id].rank,
                similarity_score=candidate_by_id[item.chunk.chunk_id].similarity_score,
                reranker_score=item.relevance_score,
            )
            for item in reranked
        ]
        logger.info(
            "retrieval completed",
            extra={
                "retrieval_strategy": self.retrieval_strategy,
                "candidate_count": len(candidates),
                "reranker_enabled": True,
                "reranker_input_count": len(selected),
                "reranker_output_count": len(results),
                "reranker_latency": latency,
                "reranker_truncated_pair_count": self.reranker.last_truncated_pair_count,
            },
        )
        if recorder is not None:
            # truncated_pair_count is one value for the whole batch; results are recorded with ID
            # and score by reranked rank.
            safe_record(
                recorder,
                "rerank.completed",
                input_count=len(selected),
                output_count=len(results),
                truncated_pair_count=self.reranker.last_truncated_pair_count,
                results=[
                    {
                        "rank": item.rank,
                        "chunk_id": item.chunk.chunk_id,
                        "candidate_rank": item.candidate_rank,
                        "reranker_score": item.reranker_score,
                    }
                    for item in results
                ],
            )
        return results
