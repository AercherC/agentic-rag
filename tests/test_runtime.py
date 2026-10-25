"""测试 Composition Root 的正式组件选择与 Settings 映射，不加载真实模型。"""

from types import SimpleNamespace

import pytest

from evidenceflow.config import Settings
from evidenceflow.rag import RAGService
from evidenceflow.retrieval import ExactDenseIndexError
from evidenceflow.runtime import (
    build_ingestion_service,
    build_rag_service,
    build_retriever,
)


@pytest.fixture(autouse=True)
def _no_lifecycle_state(monkeypatch, tmp_path) -> None:
    """模拟没有 lifecycle 状态文件的全新环境。

    切换到临时目录，使默认的相对路径 ``artifacts/phase15a/index_state.json`` 不存在；
    不能显式传入一个不存在的路径，因为显式设置的 INDEX_STATE_PATH 缺失会被视为配置错误。
    """

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("INDEX_STATE_PATH", raising=False)
    monkeypatch.delenv("COLLECTION_NAME", raising=False)


class _FakeExactIndex:
    """记录构建参数的 Exact 索引替身；size/dimension 可由用例覆盖。"""

    size = 2
    dimension = 1024

    def __init__(self, chunks, embeddings):
        self.chunks = list(chunks)
        self.embeddings = list(embeddings)


def test_runtime_factory_wires_formal_components_from_settings(monkeypatch, tmp_path) -> None:
    events: list[tuple[str, object]] = []

    class FakeEmbedding:
        def __init__(self, model, *, device): events.append(("embedding", (model, device)))

    class FakeStore:
        count = 2

        def __init__(self, path, collection): events.append(("store", (path, collection)))

        def iter_all_chunks_with_embeddings(self):
            return iter([("chunk-a", [1.0]), ("chunk-b", [2.0])])

    class FakeDense:
        def __init__(self, embedding, search, top_k):
            events.append(("dense", top_k))
            # DenseRetriever 收到的 search 对象必须是 Exact 索引，而不是 Chroma Store。
            events.append(("dense_search_is_exact", isinstance(search, _FakeExactIndex)))

    class FakeScorer:
        def __init__(self, model, *, device, dtype, max_length):
            events.append(("scorer", (model, device, dtype, max_length)))

    class FakeReranker:
        def __init__(self, scorer, *, batch_size, max_candidates):
            events.append(("reranker", (batch_size, max_candidates)))

    class FakeDenseReranker:
        def __init__(self, dense, reranker, **contract):
            events.append(("dense_reranker", contract))

    class FakeLLM:
        def __init__(self, *, api_key, base_url, model, temperature, timeout_seconds, max_attempts, thinking, trust_env):
            events.append(("llm", (model, temperature, timeout_seconds, max_attempts, thinking, trust_env)))

    class FakeTokenizer:
        def __init__(self, model): events.append(("tokenizer", model))

    class FakeContext:
        def __init__(self, tokenizer, max_context_tokens):
            events.append(("context", max_context_tokens))

    class FakeGenerator:
        def __init__(self, client, *, trace_recorder=None):
            events.append(("generator", type(client).__name__))
            events.append(("generator_trace_recorder", trace_recorder))

    monkeypatch.setattr("evidenceflow.runtime.BgeM3EmbeddingProvider", FakeEmbedding)
    monkeypatch.setattr("evidenceflow.runtime.ChromaVectorStore", FakeStore)
    monkeypatch.setattr("evidenceflow.runtime.ExactDenseIndex", _FakeExactIndex)
    monkeypatch.setattr("evidenceflow.runtime.DenseRetriever", FakeDense)
    monkeypatch.setattr("evidenceflow.runtime.TransformersCrossEncoderScorer", FakeScorer)
    monkeypatch.setattr("evidenceflow.runtime.LearnedReranker", FakeReranker)
    monkeypatch.setattr("evidenceflow.runtime.DenseRerankerRetriever", FakeDenseReranker)
    monkeypatch.setattr("evidenceflow.runtime.OpenAICompatibleLLMClient", FakeLLM)
    monkeypatch.setattr("evidenceflow.runtime.BgeM3Tokenizer", FakeTokenizer)
    monkeypatch.setattr("evidenceflow.runtime.ContextBuilder", FakeContext)
    monkeypatch.setattr("evidenceflow.runtime.Generator", FakeGenerator)

    settings = Settings(
        _env_file=None,
        embedding_model="test-bge",
        vector_db_path=tmp_path,
        collection_name="test-collection",
        reranker_model="test-reranker",
        model_device="cpu",
        reranker_input_k=20,
        reranker_final_k=5,
        max_context_tokens=1234,
        llm_model="test-llm",
    )
    service = build_rag_service(settings)

    assert isinstance(service, RAGService)
    assert events == [
        ("embedding", ("test-bge", "cpu")),
        ("store", (tmp_path, "test-collection")),
        ("dense", 20),
        ("dense_search_is_exact", True),
        ("scorer", ("test-reranker", "cpu", "float32", 512)),
        ("reranker", (4, 20)),
        (
            "dense_reranker",
            {
                "candidate_pool_k": 20,
                "reranker_input_k": 20,
                "final_top_k": 5,
                "trace_recorder": None,
            },
        ),
        ("tokenizer", "test-bge"),
        ("context", 1234),
        ("llm", ("test-llm", 0.0, 60.0, 2, "disabled", False)),
        ("generator", "FakeLLM"),
        ("generator_trace_recorder", None),
    ]


def test_runtime_factory_wires_ingestion_from_settings(monkeypatch, tmp_path) -> None:
    """摄取生产装配也集中在 runtime.py，CLI 不再选择具体依赖。"""

    events: list[tuple[str, object]] = []

    class FakeTokenizer:
        def __init__(self, model): events.append(("tokenizer", model))

    class FakeEmbedding:
        def __init__(self, model, *, device): events.append(("embedding", (model, device)))

    class FakeStore:
        def __init__(self, path, collection): events.append(("store", (path, collection)))

    class FakeIngestionService:
        def __init__(
            self, root, tokenizer, embedding, store, *, chunk_size, chunk_overlap, chunk_strategy
        ):
            # Phase 13F：runtime 装配必须把 Settings.chunk_strategy 传入摄取服务。
            events.append(("ingestion", (root, chunk_size, chunk_overlap, chunk_strategy)))

    monkeypatch.setattr("evidenceflow.runtime.BgeM3Tokenizer", FakeTokenizer)
    monkeypatch.setattr("evidenceflow.runtime.BgeM3EmbeddingProvider", FakeEmbedding)
    monkeypatch.setattr("evidenceflow.runtime.ChromaVectorStore", FakeStore)
    monkeypatch.setattr("evidenceflow.runtime.IngestionService", FakeIngestionService)
    settings = Settings(
        _env_file=None,
        source_docs_path=tmp_path / "docs",
        vector_db_path=tmp_path / "db",
        collection_name="test-collection",
        embedding_model="test-bge",
        model_device="cpu",
        chunk_size=512,
        chunk_overlap=64,
    )

    build_ingestion_service(settings)

    assert events == [
        ("tokenizer", "test-bge"),
        ("embedding", ("test-bge", "cpu")),
        ("store", (tmp_path / "db", "test-collection")),
        ("ingestion", (tmp_path / "docs", 512, 64, "hybrid")),
    ]


def _patch_retriever_deps(
    monkeypatch,
    *,
    store_count: int,
    index_size: int,
    dimension: int,
    provider_dimension: int | None = None,
):
    """为 build_retriever 的启动校验用例替换重依赖；provider_dimension 为 None 表示不声明维度。"""

    class FakeStore:
        count = store_count

        def __init__(self, path, collection): ...

        def iter_all_chunks_with_embeddings(self):
            return iter([("chunk-a", [1.0])])

    class FakeIndex(_FakeExactIndex):
        pass

    FakeIndex.size = index_size
    FakeIndex.dimension = dimension
    provider = object() if provider_dimension is None else SimpleNamespace(dimension=provider_dimension)
    monkeypatch.setattr("evidenceflow.runtime.BgeM3EmbeddingProvider", lambda m, *, device: provider)
    monkeypatch.setattr("evidenceflow.runtime.ChromaVectorStore", FakeStore)
    monkeypatch.setattr("evidenceflow.runtime.ExactDenseIndex", FakeIndex)
    monkeypatch.setattr("evidenceflow.runtime.DenseRetriever", lambda e, s, top_k: object())
    monkeypatch.setattr(
        "evidenceflow.runtime.TransformersCrossEncoderScorer",
        lambda m, *, device, dtype, max_length: object(),
    )
    monkeypatch.setattr(
        "evidenceflow.runtime.LearnedReranker", lambda scorer, *, batch_size, max_candidates: object()
    )
    monkeypatch.setattr("evidenceflow.runtime.DenseRerankerRetriever", lambda *a, **k: "retriever")


def test_build_retriever_rejects_index_size_mismatch(monkeypatch, tmp_path) -> None:
    """Exact 索引大小与 collection 记录数不一致时必须拒绝启动。"""

    _patch_retriever_deps(monkeypatch, store_count=817, index_size=816, dimension=1024)

    with pytest.raises(ExactDenseIndexError, match="size"):
        build_retriever(Settings(_env_file=None, vector_db_path=tmp_path, model_device="cpu"))


def test_build_retriever_rejects_wrong_dimension(monkeypatch, tmp_path) -> None:
    """维度不是 1024 说明 collection 与 BGE-M3 不匹配，必须拒绝启动。"""

    _patch_retriever_deps(monkeypatch, store_count=2, index_size=2, dimension=768)

    with pytest.raises(ExactDenseIndexError, match="dimension"):
        build_retriever(Settings(_env_file=None, vector_db_path=tmp_path, model_device="cpu"))


def test_build_retriever_checks_dimension_against_embedding_model(monkeypatch, tmp_path) -> None:
    """Provider 声明维度时以模型维度为准，而不是写死的 BGE-M3 1024。"""

    settings = Settings(_env_file=None, vector_db_path=tmp_path, model_device="cpu")
    _patch_retriever_deps(
        monkeypatch, store_count=2, index_size=2, dimension=768, provider_dimension=768
    )
    assert build_retriever(settings) == "retriever"

    _patch_retriever_deps(
        monkeypatch, store_count=2, index_size=2, dimension=1024, provider_dimension=768
    )
    with pytest.raises(ExactDenseIndexError, match="embedding model dimension 768"):
        build_retriever(settings)


def test_build_retriever_returns_retriever_when_validation_passes(monkeypatch, tmp_path) -> None:
    """校验通过时返回装配好的检索器。"""

    _patch_retriever_deps(monkeypatch, store_count=2, index_size=2, dimension=1024)

    retriever = build_retriever(
        Settings(_env_file=None, vector_db_path=tmp_path, model_device="cpu")
    )

    assert retriever == "retriever"


def test_build_rag_service_accepts_injected_retriever_and_generator(monkeypatch, tmp_path) -> None:
    """评测脚本可注入检索器与 Generator；注入时不得再装配正式检索或创建 LLM Client。"""

    class FakeTokenizer:
        def __init__(self, model): ...

    class FakeContext:
        def __init__(self, tokenizer, max_context_tokens): ...

    monkeypatch.setattr("evidenceflow.runtime.BgeM3Tokenizer", FakeTokenizer)
    monkeypatch.setattr("evidenceflow.runtime.ContextBuilder", FakeContext)
    monkeypatch.setattr(
        "evidenceflow.runtime.build_retriever",
        lambda settings: (_ for _ in ()).throw(AssertionError("must not build retriever")),
    )
    monkeypatch.setattr(
        "evidenceflow.runtime.build_llm_client",
        lambda settings: (_ for _ in ()).throw(AssertionError("must not build LLM client")),
    )
    retriever, generator = object(), object()

    service = build_rag_service(
        Settings(_env_file=None, vector_db_path=tmp_path, model_device="cpu"),
        retriever=retriever,
        generator=generator,
    )

    assert service._retriever is retriever
    assert service._generator is generator
