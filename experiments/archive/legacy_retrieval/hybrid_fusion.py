"""Phase 15B.5 归档的历史 Rank-only Equal RRF。

§17.3.1 Hybrid 集成把该模块引入 Runtime 路径，位于 ``retrieval/`` 包内；原先位于
``evidenceflow.eval.hybrid`` 的实现已迁到这里，``eval.hybrid`` 现在仅 re-export，便于
既有实验脚本继续使用同一组函数。``RankedChunkCandidate`` 与 ``HybridRRFChunk`` 同样是
生产算法产物，搬到 retrieval 防止 Runtime 反向依赖 eval 命名空间。
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from evidenceflow.domain.chunk import DocumentChunk


@dataclass(frozen=True)
class RankedChunkCandidate:
    """表示不暴露异构原始分数的单路 Chunk 排名。"""

    chunk: DocumentChunk
    rank: int

    def __post_init__(self) -> None:
        """排名必须是正整数，避免布尔值被当作 1。"""

        if isinstance(self.rank, bool) or not isinstance(self.rank, int) or self.rank <= 0:
            raise ValueError("rank must be a positive integer")


@dataclass(frozen=True)
class HybridRRFChunk:
    """保存 Hybrid RRF 排名、贡献分支排名与稳定 Chunk 身份。"""

    chunk: DocumentChunk
    rank: int
    fusion_score: float
    dense_rank: int | None
    bm25_rank: int | None


def equal_ranked_rrf_fusion(
    dense_results: Sequence[RankedChunkCandidate],
    bm25_results: Sequence[RankedChunkCandidate],
    *,
    rrf_k: int = 60,
    candidate_budget: int = 20,
) -> list[HybridRRFChunk]:
    """按 chunk_id 去重，将两路 Top-20 排名等权融合并压回固定预算。

    该函数只使用 Rank，不把 Dense fusion score、cosine 或 BM25 score 互相冒充，
    因而不会污染正式 Domain Model 的分数语义。
    """

    for value, field_name in (
        (rrf_k, "rrf_k"),
        (candidate_budget, "candidate_budget"),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{field_name} must be a positive integer")

    records: dict[str, dict[str, object]] = {}

    def add_branch(
        results: Sequence[RankedChunkCandidate], branch_name: str
    ) -> None:
        """登记单路唯一 ID/Rank，并拒绝同 ID 不同内容的候选。"""

        seen_ids: set[str] = set()
        seen_ranks: set[int] = set()
        for result in results:
            chunk_id = result.chunk.chunk_id
            if chunk_id in seen_ids:
                raise ValueError(
                    f"duplicate chunk_id in {branch_name} branch: {chunk_id}"
                )
            if result.rank in seen_ranks:
                raise ValueError(
                    f"duplicate rank in {branch_name} branch: {result.rank}"
                )
            seen_ids.add(chunk_id)
            seen_ranks.add(result.rank)
            record = records.setdefault(
                chunk_id,
                {"chunk": result.chunk, "dense_rank": None, "bm25_rank": None},
            )
            if record["chunk"] != result.chunk:
                raise ValueError(f"chunk payload mismatch for chunk_id: {chunk_id}")
            record[f"{branch_name}_rank"] = result.rank

    add_branch(dense_results, "dense")
    add_branch(bm25_results, "bm25")

    scored: list[tuple[float, int, str, dict[str, object]]] = []
    for chunk_id, record in records.items():
        dense_rank = record["dense_rank"]
        bm25_rank = record["bm25_rank"]
        ranks = [rank for rank in (dense_rank, bm25_rank) if isinstance(rank, int)]
        # Equal RRF 沿用 k=60 和最小分支排名、chunk_id 的稳定 tie-break。
        score = math.fsum(1.0 / (rrf_k + rank) for rank in ranks)
        scored.append((score, min(ranks), chunk_id, record))
    scored.sort(key=lambda item: (-item[0], item[1], item[2]))

    return [
        HybridRRFChunk(
            chunk=record["chunk"].model_copy(deep=True),
            rank=rank,
            fusion_score=score,
            dense_rank=record["dense_rank"],
            bm25_rank=record["bm25_rank"],
        )
        for rank, (score, _, _, record) in enumerate(
            scored[:candidate_budget], start=1
        )
    ]
