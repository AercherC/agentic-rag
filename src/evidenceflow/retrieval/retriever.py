"""组合 EmbeddingProvider 与 VectorStore 实现 Dense Retrieval。"""

from evidenceflow.domain.chunk import RetrievedChunk
from evidenceflow.embedding.base import EmbeddingProvider
from evidenceflow.storage.base import VectorStore


class InvalidQueryError(ValueError):
    """表示用户 Query 为空、纯空白或不是字符串。"""


def _validate_top_k(top_k: int) -> None:
    """保证 Dense Top-K 是正整数，避免无意义的底层查询。"""

    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")


class DenseRetriever:
    """把原始 Query 编码为向量，并委托 VectorStore 返回 Dense Top-K。"""

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStore,
        top_k: int = 5,
    ) -> None:
        """注入已有 Provider 和 Store，并保存经过校验的默认 Top-K。"""

        _validate_top_k(top_k)
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store
        self.top_k = top_k

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        """使用原始 Query 完成一次 Embedding 和一次 Vector Store 查询。"""

        if not isinstance(query, str) or not query.strip():
            # strip() 仅用于判断空白，不把处理后的文本传给 Embedding Provider。
            raise InvalidQueryError("query must not be empty or whitespace")
        effective_top_k = self.top_k if top_k is None else top_k
        _validate_top_k(effective_top_k)

        # Retriever 不加载模型、计算 cosine 或重排结果，只编排现有稳定接口。
        query_embedding = self.embedding_provider.embed_query(query)
        return self.vector_store.search(
            query_embedding=query_embedding,
            top_k=effective_top_k,
        )
