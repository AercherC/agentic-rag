"""Define the unified in-project Vector Store contract."""

from typing import Iterator, Protocol, Sequence, runtime_checkable

from evidenceflow.domain.chunk import DocumentChunk, RetrievedChunk


@runtime_checkable
class VectorStore(Protocol):
    """Thin interface that persists Chunk vectors and restores project Domain Models."""

    def add_chunks(
        self,
        chunks: Sequence[DocumentChunk],
        embeddings: Sequence[Sequence[float]],
    ) -> None:
        """Write one-to-one corresponding Chunks and Dense Vectors into storage."""

        ...

    def search(
        self,
        query_embedding: Sequence[float],
        top_k: int,
    ) -> list[RetrievedChunk]:
        """Return project retrieval results ranked 1-based by vector similarity."""

        ...

    def contains_document(
        self,
        document_id: str,
        content_hash: str | None = None,
    ) -> bool:
        """Check whether a logical document, or the given content version of it, already exists."""

        ...

    def delete_document(self, document_id: str) -> None:
        """Delete every Chunk of the given logical document."""

        ...

    def iter_all_chunks(self) -> Iterator[DocumentChunk]:
        """Iterate all formal Chunks in the current Collection in ascending Chunk ID order.

        Current uses: index-lifecycle verification (chunk ID set and size checks after a staging
        build) and the full read before building the Exact Dense index; ascending Chunk ID order
        keeps the result deterministic.
        """

        ...

    def iter_all_chunks_with_embeddings(
        self,
    ) -> Iterator[tuple[DocumentChunk, list[float]]]:
        """Read-only iteration over complete Chunks together with their stored Dense vectors, in ascending Chunk ID order."""

        ...

    def clear(self) -> None:
        """Clear the current Collection while keeping the Adapter usable."""

        ...
