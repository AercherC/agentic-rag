"""验证 canonical Exact Dense index 的向量、排序和错误合同。"""

import numpy as np
import pytest

from evidenceflow.domain.chunk import DocumentChunk
from evidenceflow.retrieval.exact_dense import ExactDenseIndex


def _chunk(chunk_id: str, index: int = 0) -> DocumentChunk:
    """构造最小但可完整恢复来源的 Domain Chunk。"""

    return DocumentChunk(
        chunk_id=chunk_id,
        document_id="DOC",
        content=chunk_id,
        token_count=1,
        chunk_index=index,
        file_name="doc.txt",
        content_hash="HASH",
    )


def test_exact_index_sorts_ties_by_chunk_id_and_preserves_float32_score() -> None:
    """相同相似度必须由项目层 chunk_id 决胜，不能依赖后端返回顺序。"""

    index = ExactDenseIndex(
        [_chunk("B", 1), _chunk("A", 0), _chunk("C", 2)],
        [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
    )

    results = index.search(np.array([1.0, 0.0], dtype=np.float32), top_k=3)

    assert [item.chunk.chunk_id for item in results] == ["A", "B", "C"]
    assert [item.rank for item in results] == [1, 2, 3]
    assert results[0].similarity_score == results[1].similarity_score


def test_exact_index_matches_independent_numpy_order() -> None:
    """普通分数场景与独立矩阵内积的稳定排序一致。"""

    vectors = np.asarray(
        [[1.0, 0.0], [0.8, 0.6], [0.0, 1.0]],
        dtype=np.float32,
    )
    query = np.asarray([0.8, 0.6], dtype=np.float32)
    index = ExactDenseIndex(
        [_chunk("C"), _chunk("A"), _chunk("B")],
        vectors[[2, 0, 1]],
    )
    expected_scores = vectors @ query
    expected = sorted(
        zip(expected_scores.tolist(), ["A", "B", "C"]),
        key=lambda item: (-item[0], item[1]),
    )

    actual = index.search(query, top_k=3)

    assert [item.chunk.chunk_id for item in actual] == [item[1] for item in expected]
    assert [item.similarity_score for item in actual] == pytest.approx(
        [item[0] for item in expected], abs=1e-6
    )


@pytest.mark.parametrize(
    ("chunks", "embeddings", "message"),
    [
        ([_chunk("A"), _chunk("A")], [[1.0, 0.0], [1.0, 0.0]], "duplicate"),
        ([_chunk("A"), _chunk("B")], [[1.0, 0.0], [1.0, 0.0, 0.0]], "dimension"),
        ([_chunk("A")], [[float("nan"), 0.0]], "non-finite"),
    ],
)
def test_exact_index_rejects_invalid_records(chunks, embeddings, message: str) -> None:
    """身份重复、维度漂移和非有限向量必须在建索引时失败。"""

    with pytest.raises(ValueError, match=message):
        ExactDenseIndex(chunks, embeddings)


def test_exact_index_rejects_wrong_query_dimension_and_empty_search() -> None:
    """查询维度必须匹配；空索引可安全返回空结果。"""

    index = ExactDenseIndex([], [])
    assert index.search([1.0, 0.0], top_k=1) == []

    populated = ExactDenseIndex([_chunk("A")], [[1.0, 0.0]])
    with pytest.raises(ValueError, match="dimension"):
        populated.search([1.0, 0.0, 0.0], top_k=1)


def test_exact_backend_is_fixed_to_numpy_regardless_of_faiss(monkeypatch) -> None:
    """后端固定为 numpy.exact：环境里即使存在 faiss 也不得被使用（Stage A §S.1）。"""

    import sys

    import evidenceflow.retrieval.exact_dense as module

    class _ExplodingFaiss:
        def __getattr__(self, name: str):
            raise AssertionError("faiss must not be used")

    monkeypatch.setitem(sys.modules, "faiss", _ExplodingFaiss())
    index = ExactDenseIndex([_chunk("A")], [[1.0, 0.0]])

    assert index.backend == "numpy.exact"
    # 模块不再导入 faiss，因此没有 faiss 属性可被误用。
    assert not hasattr(module, "faiss")
    assert index.search([1.0, 0.0], top_k=1)[0].chunk.chunk_id == "A"


@pytest.mark.parametrize("top_k", [1, 2, 3, 7, 25, 40, 41])
def test_partial_selection_matches_full_sort_including_boundary_ties(top_k: int) -> None:
    """Top-K 先选阈值再排序，结果必须与“全量按 (score DESC, chunk_id ASC) 排序后截断”一致。

    向量只取少数几个方向，制造大量同分，并让同分组跨越第 K 名的边界。
    """

    rng = np.random.default_rng(7)
    directions = np.eye(4, dtype=np.float32)
    picks = rng.integers(0, 4, size=40)
    chunk_ids = [f"c{value:02d}" for value in rng.permutation(40)]
    embeddings = [directions[pick].tolist() for pick in picks]
    index = ExactDenseIndex([_chunk(chunk_id, i) for i, chunk_id in enumerate(chunk_ids)], embeddings)
    query = np.array([0.8, 0.6, 0.0, 0.0], dtype=np.float32)

    results = index.search(query, top_k=top_k)

    scores = (np.asarray(embeddings, dtype=np.float32) @ query).tolist()
    expected = sorted(zip(scores, chunk_ids), key=lambda item: (-item[0], item[1]))[:top_k]
    assert [(item.similarity_score, item.chunk.chunk_id) for item in results] == expected
    assert [item.rank for item in results] == list(range(1, len(expected) + 1))
