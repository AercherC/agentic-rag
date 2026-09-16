"""Phase 15B.5 归档的历史 Weighted RRF 排序逻辑。"""

import math
from collections.abc import Sequence

from evidenceflow.config import (
    DEFAULT_MULTILINGUAL_FINAL_TOP_K,
    DEFAULT_MULTILINGUAL_ORIGINAL_WEIGHT,
    DEFAULT_MULTILINGUAL_RRF_K,
    DEFAULT_MULTILINGUAL_TRANSLATION_WEIGHT,
)
from evidenceflow.domain.chunk import MultilingualRetrievedChunk, RetrievedChunk


def _validate_positive_integer(value: int, *, field_name: str) -> None:
    """拒绝布尔值和非正整数，避免无效排名深度或 RRF 常数。"""

    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")


def _validate_weight(value: float, *, field_name: str) -> None:
    """权重必须为有限正数，确保每个可用分支都产生正贡献。"""

    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"{field_name} must be a finite positive number")


def weighted_rrf_fusion(
    original_results: Sequence[RetrievedChunk],
    translated_results: Sequence[RetrievedChunk],
    *,
    original_weight: float = DEFAULT_MULTILINGUAL_ORIGINAL_WEIGHT,
    translation_weight: float = DEFAULT_MULTILINGUAL_TRANSLATION_WEIGHT,
    rrf_k: int = DEFAULT_MULTILINGUAL_RRF_K,
    final_top_k: int = DEFAULT_MULTILINGUAL_FINAL_TOP_K,
) -> list[MultilingualRetrievedChunk]:
    """按 chunk_id 去重并使用固定、可审计的 Weighted RRF 规则融合两路结果。"""

    _validate_weight(original_weight, field_name="original_weight")
    _validate_weight(translation_weight, field_name="translation_weight")
    _validate_positive_integer(rrf_k, field_name="rrf_k")
    _validate_positive_integer(final_top_k, field_name="final_top_k")

    records: dict[str, dict[str, object]] = {}

    def add_branch(results: Sequence[RetrievedChunk], branch: str) -> None:
        """登记单路候选；同一路重复 ID 会造成双重计分，因此必须拒绝。"""

        seen: set[str] = set()
        for result in results:
            chunk_id = result.chunk.chunk_id
            if chunk_id in seen:
                raise ValueError(f"duplicate chunk_id in {branch} branch: {chunk_id}")
            seen.add(chunk_id)
            record = records.setdefault(
                chunk_id,
                {
                    "chunk": result.chunk,
                    "original_rank": None,
                    "translated_rank": None,
                    "original_similarity_score": None,
                    "translated_similarity_score": None,
                },
            )
            if record["chunk"] != result.chunk:
                raise ValueError(f"chunk payload mismatch for chunk_id: {chunk_id}")
            record[f"{branch}_rank"] = result.rank
            record[f"{branch}_similarity_score"] = result.similarity_score

    add_branch(original_results, "original")
    add_branch(translated_results, "translated")

    scored: list[tuple[float, int, str, dict[str, object]]] = []
    for chunk_id, record in records.items():
        contributions: list[float] = []
        original_rank = record["original_rank"]
        translated_rank = record["translated_rank"]
        if isinstance(original_rank, int):
            contributions.append(original_weight / (rrf_k + original_rank))
        if isinstance(translated_rank, int):
            contributions.append(translation_weight / (rrf_k + translated_rank))
        ranks = [rank for rank in (original_rank, translated_rank) if isinstance(rank, int)]
        # fsum 稳定累加；后续严格按已冻结的三层 tie-break 排序。
        scored.append((math.fsum(contributions), min(ranks), chunk_id, record))

    scored.sort(key=lambda item: (-item[0], item[1], item[2]))
    return [
        MultilingualRetrievedChunk(
            chunk=record["chunk"].model_copy(deep=True),
            rank=rank,
            fusion_score=score,
            original_rank=record["original_rank"],
            translated_rank=record["translated_rank"],
            original_similarity_score=record["original_similarity_score"],
            translated_similarity_score=record["translated_similarity_score"],
        )
        for rank, (score, _, _, record) in enumerate(scored[:final_top_k], start=1)
    ]
