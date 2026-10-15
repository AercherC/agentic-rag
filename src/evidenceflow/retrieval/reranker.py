"""提供线上检索链路使用的 Learned Cross-Encoder 重排边界与 Transformers 实现（Exact Dense20 → Reranker → Top5）。"""

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
    """表示候选已召回但 Learned Reranker 无法产生可信排序。"""


class PairScorerProtocol(Protocol):
    """定义 Reranker 所需的最小 Query–Passage 批量评分接口。"""

    def score_pairs(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        """返回与输入顺序一一对应的有限 relevance scores。"""


@dataclass(frozen=True)
class RerankedChunk:
    """保存 Learned score、最终排名与候选原始顺序。"""

    chunk: DocumentChunk
    rank: int
    relevance_score: float
    original_rank: int


class LearnedReranker:
    """以可替换 PairScorer 对候选批量评分，并执行稳定确定性排序。"""

    def __init__(
        self,
        scorer: PairScorerProtocol,
        *,
        batch_size: int = 4,
        max_candidates: int = 100,
    ) -> None:
        """保存批量与候选上限；拒绝布尔值冒充正整数。"""

        for value, name in ((batch_size, "batch_size"), (max_candidates, "max_candidates")):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        self.scorer = scorer
        self.batch_size = batch_size
        self.max_candidates = max_candidates
        self.last_truncated_pair_count = 0
        # 仅汇总 scorer 暴露的阶段耗时，缺失字段的测试替身按零处理。
        self.last_timing_seconds: dict[str, float] = {}

    def rerank(
        self,
        query: str,
        candidates: Sequence[DocumentChunk],
        *,
        top_k: int | None = None,
    ) -> list[RerankedChunk]:
        """重排冻结候选；只改变顺序，不增删或修改 Chunk。"""

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
        # 分批只控制内存；所有分数汇总后统一排序，保证 batch size 不改变结果。
        for start in range(0, len(candidates), self.batch_size):
            batch = candidates[start : start + self.batch_size]
            batch_scores = self.scorer.score_pairs(
                [(query, chunk.content) for chunk in batch]
            )
            # 正式 Transformers scorer 暴露每批截断数；测试替身没有该字段时按 0 处理。
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
    """对已计算的 Learned scores 排序，供预算实验复用同一次模型前向。"""

    if len(candidates) != len(scores):
        raise ValueError("candidates and scores must have the same length")
    if any(not math.isfinite(score) for score in scores):
        raise ValueError("scores must be finite")
    effective_top_k = len(candidates) if top_k is None else top_k
    if isinstance(effective_top_k, bool) or not isinstance(effective_top_k, int) or effective_top_k < 0:
        raise ValueError("top_k must be a non-negative integer")
    indexed = list(enumerate(zip(candidates, scores), start=1))
    # 同分优先保留冻结候选顺序，chunk_id 作为最终稳定身份决胜键。
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
    """使用 Hugging Face Sequence Classification 模型计算 Cross-Encoder logits。"""

    def __init__(
        self,
        model_name: str,
        *,
        device: str = "auto",
        dtype: str = "float32",
        max_length: int = 512,
        inference_mode: bool = False,
    ) -> None:
        """加载冻结模型；设备与截断长度必须由实验 manifest 显式记录。"""

        if not model_name.strip():
            raise ValueError("model_name must not be blank")
        if isinstance(max_length, bool) or not isinstance(max_length, int) or max_length <= 0:
            raise ValueError("max_length must be a positive integer")
        if dtype != "float32":
            # Phase 9 只晋升已验证的 CPU float32 路径；低精度属于后续性能阶段。
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
        # Tokenizer 保持 CPU；只有编码后的 batch 在前向前迁移到模型设备。
        self.model.to(device=self.device, dtype=torch.float32)
        self.model.eval()
        self._torch = torch
        self.forward_passes = 0
        self.last_truncated_pair_count = 0
        self.inference_mode = inference_mode
        # 仅保存最近一批的阶段耗时，供性能实验定位瓶颈，不参与评分或排序。
        self.last_timing_seconds = {
            "untruncated_tokenization": 0.0,
            "model_tokenization": 0.0,
            "model_forward": 0.0,
            "score_postprocess": 0.0,
        }

    def score_pairs(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        """对一个 batch 执行一次前向，并返回原始 logit。"""

        if not pairs:
            self.last_truncated_pair_count = 0
            return []
        # 先取得未截断长度，只用于可观测性；不得把达到上限误报为已截断。
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
        # inference_mode 仅移除 autograd 元数据；模型、输入和 float32 logits 保持不变。
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
    """执行 Original Dense Candidate Generation 与 Learned Reranker 最终检索。"""

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
        """冻结候选池、Reranker 输入与最终深度的分层合同。

        ``trace_recorder`` 可选（默认 None）：只观测 retrieval.dense.* 与 rerank.* 事件，
        不影响候选、排序、分数、异常或日志（PROJECT_SPEC §12.5）。
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
        # 公开属性：调用方（含 B2 trace 对照实验）可在两次 retrieve 之间替换 recorder。
        self.trace_recorder = trace_recorder

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[RerankedRetrievedChunk]:
        """以 ANN Top-K 生成候选并重排；失败时明确报错且不返回 Dense 顺序。"""

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

        # 本次调用固定使用同一个 recorder；为 None 时下面不构造任何 payload，开销与未接 trace 一致。
        recorder = self.trace_recorder
        safe_record(recorder, "retrieval.dense.started")
        candidates = self.dense_retriever.retrieve(query, top_k=self.candidate_pool_k)
        if recorder is not None:
            # 只记 ID 与原始 float32 相似度，不记 chunk 正文。
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
            # 未验证的 Dense 原排序不能伪装成成功的 Reranker 输出。
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
            # truncated_pair_count 是整批一个值；结果按重排后 rank 记录 ID 与分数。
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
