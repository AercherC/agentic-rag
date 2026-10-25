"""Deterministic in-process exact Dense candidate search primitive (numpy.exact backend)."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np

from evidenceflow.domain.chunk import DocumentChunk, RetrievedChunk

# The backend is pinned to numpy.exact (Stage A §S.1): 817 vectors do not need FAISS, and the
# online backend must not depend on whether the runtime environment happens to have FAISS
# installed.
EXACT_BACKEND = "numpy.exact"


class ExactDenseIndexError(ValueError):
    """Signals that the ExactDenseIndex input violates the vector or identity contract."""


def _as_normalized_float32(
    vector: Any,
    *,
    field_name: str,
    expected_dimension: int | None = None,
) -> np.ndarray:
    """Convert an external vector to contiguous float32 and verify finite values, dimension, and L2 normalization."""

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
    # Compute the validation norm in float64 so float32 accumulation error cannot flag a valid
    # vector as not normalized.
    norm = float(np.linalg.norm(array.astype(np.float64)))
    if not math.isfinite(norm) or not math.isclose(norm, 1.0, abs_tol=1e-4):
        raise ExactDenseIndexError(f"{field_name} must be L2-normalized")
    return np.ascontiguousarray(array, dtype=np.float32)


class ExactDenseIndex:
    """Build an in-process exact Dense index from a Chroma snapshot and return stable candidates."""

    def __init__(
        self,
        chunks: Sequence[DocumentChunk],
        embeddings: Sequence[Sequence[float]],
    ) -> None:
        """Fix the record order by chunk_id, then build a NumPy float32 full inner-product matrix."""

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
        # The matrix lives only in the process and no index file is written; the backend is always
        # numpy.exact.
        self.backend = EXACT_BACKEND

    @property
    def chunk_ids(self) -> tuple[str, ...]:
        """Return the Chunk ID order frozen at build time."""

        return self._chunk_ids

    @property
    def dimension(self) -> int | None:
        """Return the vector dimension; an empty index has no inferable dimension."""

        return self._dimension

    @property
    def size(self) -> int:
        """Return the number of Chunks currently indexed."""

        return len(self._chunk_ids)

    @property
    def matrix_nbytes(self) -> int:
        """Return the byte size of the in-memory matrix, used for architecture evaluation rather than ranking."""

        return int(self._matrix.nbytes)

    def search(
        self,
        query_embedding: Sequence[float],
        top_k: int,
    ) -> list[RetrievedChunk]:
        """Run the full Inner Product and apply similarity DESC, chunk_id ASC ordering."""

        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        if not self._chunk_ids:
            return []
        query = _as_normalized_float32(
            query_embedding,
            field_name="query_embedding",
            expected_dimension=self._dimension,
        )

        # float32 matrix multiplication yields the full inner products; the candidate order is
        # decided entirely by the project-level sort below.
        scores = self._matrix @ query
        size = len(self._chunk_ids)
        effective_top_k = min(top_k, size)
        if effective_top_k < size:
            # Only candidates that can reach the Top-K go through the Python sort: take the K-th
            # largest score as the threshold and let every candidate at or above it (ties included)
            # take part, so ties still break strictly on chunk_id and the result matches a full sort
            # exactly.
            threshold = np.partition(scores, size - effective_top_k)[size - effective_top_k]
            positions = np.flatnonzero(scores >= threshold).tolist()
        else:
            positions = range(size)
        scored = [(float(scores[position]), self._chunk_ids[position], position) for position in positions]

        scored.sort(key=lambda item: (-item[0], item[1]))
        return [
            RetrievedChunk(
                chunk=self._chunks[position].model_copy(deep=True),
                rank=rank,
                similarity_score=score,
            )
            for rank, (score, _chunk_id, position) in enumerate(
                scored[:effective_top_k],
                start=1,
            )
        ]

