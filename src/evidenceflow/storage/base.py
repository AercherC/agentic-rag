"""定义 EvidenceFlow 内部统一的 Vector Store 合同。"""

from typing import Iterator, Protocol, Sequence, runtime_checkable

from evidenceflow.domain.chunk import DocumentChunk, RetrievedChunk


@runtime_checkable
class VectorStore(Protocol):
    """持久化 Chunk 向量并恢复项目 Domain Model 的薄接口。"""

    def add_chunks(
        self,
        chunks: Sequence[DocumentChunk],
        embeddings: Sequence[Sequence[float]],
    ) -> None:
        """将数量一一对应的 Chunk 和 Dense Vector 写入存储。"""

        ...

    def search(
        self,
        query_embedding: Sequence[float],
        top_k: int,
    ) -> list[RetrievedChunk]:
        """按向量相似度返回 1-based 排名后的项目检索结果。"""

        ...

    def contains_document(
        self,
        document_id: str,
        content_hash: str | None = None,
    ) -> bool:
        """检查逻辑文档或其指定内容版本是否已存在。"""

        ...

    def delete_document(self, document_id: str) -> None:
        """删除指定逻辑文档的全部 Chunk。"""

        ...

    def iter_all_chunks(self) -> Iterator[DocumentChunk]:
        """按 Chunk ID 升序遍历当前 Collection 中的全部正式 Chunk。

        当前用途：索引生命周期校验（staging 构建后的 chunk ID 集合与规模校验）以及
        Exact Dense 索引构建前的全量读取；按 Chunk ID 升序保证结果确定。
        """

        ...

    def iter_all_chunks_with_embeddings(
        self,
    ) -> Iterator[tuple[DocumentChunk, list[float]]]:
        """按 Chunk ID 升序只读遍历完整 Chunk 与已存 Dense 向量。"""

        ...

    def clear(self) -> None:
        """清空当前 Collection，并保持 Adapter 可继续使用。"""

        ...
