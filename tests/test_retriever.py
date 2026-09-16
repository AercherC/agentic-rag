"""使用 Fake Provider 和 Store 验证 DenseRetriever 的调用编排。"""

import pytest

from evidenceflow.domain.chunk import DocumentChunk, RetrievedChunk
from evidenceflow.retrieval import DenseRetriever, InvalidQueryError


class FakeEmbeddingProvider:
    """返回固定 Query Vector，并记录收到的原始 Query。"""

    def __init__(self, vector: list[float] | None = None) -> None:
        self.vector = vector or [1.0, 0.0, 0.0]
        self.queries: list[str] = []

    def embed_query(self, query: str) -> list[float]:
        self.queries.append(query)
        return self.vector


class FakeVectorStore:
    """返回预设检索结果，并记录 Query Vector 和 Top-K。"""

    def __init__(self, results: list[RetrievedChunk] | None = None) -> None:
        self.results = results if results is not None else []
        self.search_calls: list[tuple[list[float], int]] = []

    def search(
        self,
        query_embedding: list[float],
        top_k: int,
    ) -> list[RetrievedChunk]:
        self.search_calls.append((query_embedding, top_k))
        return self.results


def _result(rank: int, score: float) -> RetrievedChunk:
    """构造用于验证排名和分数保持不变的检索结果。"""

    chunk = DocumentChunk(
        chunk_id=f"chunk-{rank}",
        document_id="document-id",
        content=f"result content {rank}",
        token_count=10,
        chunk_index=rank - 1,
        page_start=None,
        page_end=None,
        file_name="notes.md",
        content_hash="content-hash",
        metadata={},
    )
    return RetrievedChunk(chunk=chunk, rank=rank, similarity_score=score)


def test_retrieve_passes_original_query_vector_and_default_top_k_once() -> None:
    """一次调用只编码和搜索一次，并原样返回 Store 的排名结果。"""

    provider = FakeEmbeddingProvider([1.0, 0.0, 0.0])
    expected = [_result(1, 0.8), _result(2, 0.7)]
    store = FakeVectorStore(expected)
    retriever = DenseRetriever(provider, store, top_k=5)
    query = "How Does LangGraph Manage State?"

    results = retriever.retrieve(query)

    assert provider.queries == [query]
    assert store.search_calls == [([1.0, 0.0, 0.0], 5)]
    assert results is expected
    assert [result.rank for result in results] == [1, 2]
    assert [result.similarity_score for result in results] == [0.8, 0.7]


def test_top_k_override_does_not_change_retriever_default() -> None:
    """单次 Top-K 覆盖只影响当前请求，后续仍使用构造时默认值。"""

    provider = FakeEmbeddingProvider()
    store = FakeVectorStore()
    retriever = DenseRetriever(provider, store, top_k=5)

    retriever.retrieve("What is ReAct?", top_k=3)
    retriever.retrieve("What is an agent?")

    assert [top_k for _, top_k in store.search_calls] == [3, 5]
    assert retriever.top_k == 5
    assert len(provider.queries) == 2


@pytest.mark.parametrize("query", ["", "   ", "\n\t"])
def test_invalid_query_fails_before_provider_and_store(query: str) -> None:
    """空白 Query 必须在任何模型或数据库调用前失败。"""

    provider = FakeEmbeddingProvider()
    store = FakeVectorStore()
    retriever = DenseRetriever(provider, store)

    with pytest.raises(InvalidQueryError):
        retriever.retrieve(query)

    assert provider.queries == []
    assert store.search_calls == []


@pytest.mark.parametrize("top_k", [0, -1, True])
def test_invalid_default_top_k_is_rejected(top_k: int) -> None:
    """构造阶段必须拒绝非正整数默认 Top-K。"""

    provider = FakeEmbeddingProvider()
    store = FakeVectorStore()
    with pytest.raises(ValueError, match="positive integer"):
        DenseRetriever(provider, store, top_k=top_k)
    assert provider.queries == []
    assert store.search_calls == []


@pytest.mark.parametrize("top_k", [0, -1, True])
def test_invalid_override_top_k_fails_before_calls(top_k: int) -> None:
    """单次非法 Top-K 不得触发 Query Embedding 或 Vector Search。"""

    provider = FakeEmbeddingProvider()
    store = FakeVectorStore()
    retriever = DenseRetriever(provider, store)

    with pytest.raises(ValueError, match="positive integer"):
        retriever.retrieve("What is ReAct?", top_k=top_k)

    assert provider.queries == []
    assert store.search_calls == []


def test_empty_store_result_is_returned_without_error() -> None:
    """空知识库结果保持空列表，不在 Retriever 生成用户级错误。"""

    retriever = DenseRetriever(FakeEmbeddingProvider(), FakeVectorStore())
    assert retriever.retrieve("What is ReAct?") == []


def test_provider_error_propagates_and_store_is_not_called() -> None:
    """Embedding 错误保留原始类型和原因，且不得继续查询 Store。"""

    class FailingProvider(FakeEmbeddingProvider):
        def embed_query(self, query: str) -> list[float]:
            raise RuntimeError("embedding failed")

    store = FakeVectorStore()
    retriever = DenseRetriever(FailingProvider(), store)

    with pytest.raises(RuntimeError, match="embedding failed"):
        retriever.retrieve("What is ReAct?")
    assert store.search_calls == []


def test_store_error_propagates_after_one_embedding_call() -> None:
    """Storage 错误保留原始类型，同时 Query 只生成一次向量。"""

    class FailingStore(FakeVectorStore):
        def search(
            self,
            query_embedding: list[float],
            top_k: int,
        ) -> list[RetrievedChunk]:
            self.search_calls.append((query_embedding, top_k))
            raise RuntimeError("search failed")

    provider = FakeEmbeddingProvider()
    store = FailingStore()
    retriever = DenseRetriever(provider, store)

    with pytest.raises(RuntimeError, match="search failed"):
        retriever.retrieve("What is ReAct?")
    assert provider.queries == ["What is ReAct?"]
    assert len(store.search_calls) == 1
