"""定义 EvidenceFlow 内部统一的 Dense Embedding 合同。"""

from typing import Protocol, Sequence, runtime_checkable


@runtime_checkable
class EmbeddingProvider(Protocol):
    """把文档正文和查询转换到同一 Dense Vector 空间。"""

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """批量生成文档向量，并返回普通 Python 浮点数列表。"""

        ...

    def embed_query(self, query: str) -> list[float]:
        """生成单个查询向量，并返回普通 Python 浮点数列表。"""

        ...
