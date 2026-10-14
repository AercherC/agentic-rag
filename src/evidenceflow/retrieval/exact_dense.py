"""提供确定性的、进程内 exact Dense candidate search primitive（numpy.exact 后端）。"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np

from evidenceflow.domain.chunk import DocumentChunk, RetrievedChunk

# 后端固定为 numpy.exact（Stage A §S.1）：817 个向量无需 FAISS，且线上后端不应由
# 运行环境是否安装 FAISS 决定。
EXACT_BACKEND = "numpy.exact"


class ExactDenseIndexError(ValueError):
    """表示 ExactDenseIndex 输入不满足向量或身份合同。"""


def _as_normalized_float32(
    vector: Any,
    *,
    field_name: str,
    expected_dimension: int | None = None,
) -> np.ndarray:
    """把外部向量转换为连续 float32，并验证有限值、维度和 L2 归一化。"""

    try:
        array = np.asarray(vector, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ExactDenseIndexError(f"{field_name} must be a numeric vector") from exc
    if array.ndim != 1 or array.size == 0:
        raise ExactDenseIndexError(f"{field_name} must be a non-empty 1-D vector")
    if expected_dimension is not None and array.size != expected_dimension:
        raise ExactDenseIndexError(
            f"{field_name} dimension does not match the index"
        )
    if not np.isfinite(array).all():
        raise ExactDenseIndexError(f"{field_name} contains non-finite values")
    # 用 float64 计算校验范数，避免 float32 累加误差把合法向量判为未归一化。
    norm = float(np.linalg.norm(array.astype(np.float64)))
    if not math.isfinite(norm) or not math.isclose(norm, 1.0, abs_tol=1e-4):
        raise ExactDenseIndexError(f"{field_name} must be L2-normalized")
    return np.ascontiguousarray(array, dtype=np.float32)


class ExactDenseIndex:
    """从 Chroma 快照构建进程内 exact Dense 索引并返回稳定候选。"""

    def __init__(
        self,
        chunks: Sequence[DocumentChunk],
        embeddings: Sequence[Sequence[float]],
    ) -> None:
        """按 chunk_id 固定记录顺序，构建 NumPy float32 全量内积矩阵。"""

        chunk_list = list(chunks)
        embedding_list = list(embeddings)
        if len(chunk_list) != len(embedding_list):
            raise ExactDenseIndexError("chunks and embeddings must have the same length")
        chunk_ids = [chunk.chunk_id for chunk in chunk_list]
        if len(chunk_ids) != len(set(chunk_ids)):
            raise ExactDenseIndexError("duplicate chunk IDs are not allowed")

        records: list[tuple[str, DocumentChunk, np.ndarray]] = []
        dimension: int | None = None
        for chunk, embedding in zip(chunk_list, embedding_list):
            vector = _as_normalized_float32(
                embedding,
                field_name=f"embedding[{chunk.chunk_id}]",
                expected_dimension=dimension,
            )
            dimension = int(vector.size) if dimension is None else dimension
            records.append((chunk.chunk_id, chunk, vector))
        records.sort(key=lambda item: item[0])

        self._chunk_ids = tuple(item[0] for item in records)
        self._chunks = tuple(item[1] for item in records)
        self._dimension = dimension
        self._matrix = (
            np.ascontiguousarray(
                np.vstack([item[2] for item in records]),
                dtype=np.float32,
            )
            if records
            else np.empty((0, 0), dtype=np.float32)
        )
        # 矩阵只在进程内持有，不写出任何索引文件；backend 恒为 numpy.exact。
        self.backend = EXACT_BACKEND

    @property
    def chunk_ids(self) -> tuple[str, ...]:
        """返回构建时冻结的 Chunk ID 顺序。"""

        return self._chunk_ids

    @property
    def dimension(self) -> int | None:
        """返回向量维度；空索引没有可推断的维度。"""

        return self._dimension

    @property
    def size(self) -> int:
        """返回当前索引的 Chunk 数量。"""

        return len(self._chunk_ids)

    @property
    def matrix_nbytes(self) -> int:
        """返回内存矩阵字节数，用于架构评测而非排序逻辑。"""

        return int(self._matrix.nbytes)

    def search(
        self,
        query_embedding: Sequence[float],
        top_k: int,
    ) -> list[RetrievedChunk]:
        """执行全量 Inner Product 并应用 similarity DESC、chunk_id ASC 排序。"""

        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        if not self._chunk_ids:
            return []
        query = _as_normalized_float32(
            query_embedding,
            field_name="query_embedding",
            expected_dimension=self._dimension,
        )

        # float32 矩阵乘法得到全量内积；候选顺序完全由下面的项目层排序决定。
        scores = self._matrix @ query
        scored = [
            (float(score), chunk_id)
            for score, chunk_id in zip(scores.tolist(), self._chunk_ids)
        ]

        scored.sort(key=lambda item: (-item[0], item[1]))
        by_id = dict(zip(self._chunk_ids, self._chunks))
        return [
            RetrievedChunk(
                chunk=by_id[chunk_id].model_copy(deep=True),
                rank=rank,
                similarity_score=score,
            )
            for rank, (score, chunk_id) in enumerate(
                scored[: min(top_k, len(scored))],
                start=1,
            )
        ]

