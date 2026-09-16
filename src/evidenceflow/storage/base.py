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

        §17.3.1 Hybrid 集成依赖该接口把正式 Chunk 灌入 BM25Index，确保离线实验
        与 Runtime 共用同一份 Chunk 集，避免 document_id/chunk_id 漂移。
        """

        ...

    def clear(self) -> None:
        """清空当前 Collection，并保持 Adapter 可继续使用。"""

        ...
