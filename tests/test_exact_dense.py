"""Verify the vectors, ordering, and error contract of the canonical Exact Dense index."""

import numpy as np
import pytest

from evidenceflow.domain.chunk import DocumentChunk
from evidenceflow.retrieval.exact_dense import ExactDenseIndex


def _chunk(chunk_id: str, index: int = 0) -> DocumentChunk:
    """Build a minimal Domain Chunk that can still fully recover its source."""

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
    """Equal similarity must be tie-broken by the project-level chunk_id, never by the backend's return order."""

    index = ExactDenseIndex(
        [_chunk("B", 1), _chunk("A", 0), _chunk("C", 2)],
        [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
    )

    results = index.search(np.array([1.0, 0.0], dtype=np.float32), top_k=3)

    assert [item.chunk.chunk_id for item in results] == ["A", "B", "C"]
    assert [item.rank for item in results] == [1, 2, 3]
    assert results[0].similarity_score == results[1].similarity_score


def test_exact_index_matches_independent_numpy_order() -> None:
    """The ordinary-score case matches a stable ordering computed from an independent matrix inner product."""

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
    """Duplicate identity, dimension drift, and non-finite vectors must fail at index construction."""

    with pytest.raises(ValueError, match=message):
        ExactDenseIndex(chunks, embeddings)


def test_exact_index_rejects_wrong_query_dimension_and_empty_search() -> None:
    """The query dimension must match; an empty index may safely return an empty result."""

    index = ExactDenseIndex([], [])
    assert index.search([1.0, 0.0], top_k=1) == []

    populated = ExactDenseIndex([_chunk("A")], [[1.0, 0.0]])
    with pytest.raises(ValueError, match="dimension"):
        populated.search([1.0, 0.0, 0.0], top_k=1)


def test_exact_backend_is_fixed_to_numpy_regardless_of_faiss(monkeypatch) -> None:
    """The backend is fixed to numpy.exact: even if faiss is present in the environment it must not be used (Stage A §S.1)."""

    import sys

    import evidenceflow.retrieval.exact_dense as module

    class _ExplodingFaiss:
        def __getattr__(self, name: str):
            raise AssertionError("faiss must not be used")

    monkeypatch.setitem(sys.modules, "faiss", _ExplodingFaiss())
    index = ExactDenseIndex([_chunk("A")], [[1.0, 0.0]])

    assert index.backend == "numpy.exact"
    # The module no longer imports faiss, so there is no faiss attribute that could be misused.
    assert not hasattr(module, "faiss")
    assert index.search([1.0, 0.0], top_k=1)[0].chunk.chunk_id == "A"


@pytest.mark.parametrize("top_k", [1, 2, 3, 7, 25, 40, 41])
def test_partial_selection_matches_full_sort_including_boundary_ties(top_k: int) -> None:
    """Top-K selects a threshold before sorting; the result must match a full
    (score DESC, chunk_id ASC) sort followed by truncation.

    Only a few vector directions are used, which creates many ties and lets a tie group straddle
    the K-th boundary.
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
