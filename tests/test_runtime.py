"""测试 Composition Root 的正式组件选择与 Settings 映射，不加载真实模型。"""

from types import SimpleNamespace

from evidenceflow.config import Settings
from evidenceflow.rag import RAGService
from evidenceflow.runtime import build_ingestion_service, build_rag_service


def test_runtime_factory_wires_formal_components_from_settings(monkeypatch, tmp_path) -> None:
    events: list[tuple[str, object]] = []

    class FakeEmbedding:
        def __init__(self, model, *, device): events.append(("embedding", (model, device)))

    class FakeStore:
        def __init__(self, path, collection): events.append(("store", (path, collection)))

    class FakeDense:
        def __init__(self, embedding, store, top_k): events.append(("dense", top_k))

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
        def __init__(self, *, api_key, base_url, model, temperature, timeout_seconds, max_attempts):
            events.append(("llm", (model, temperature, timeout_seconds, max_attempts)))

    class FakeTokenizer:
        def __init__(self, model): events.append(("tokenizer", model))

    class FakeContext:
        def __init__(self, tokenizer, max_context_tokens):
            events.append(("context", max_context_tokens))

    class FakeGenerator:
        def __init__(self, client): events.append(("generator", type(client).__name__))

    monkeypatch.setattr("evidenceflow.runtime.BgeM3EmbeddingProvider", FakeEmbedding)
    monkeypatch.setattr("evidenceflow.runtime.ChromaVectorStore", FakeStore)
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
        ("scorer", ("test-reranker", "cpu", "float32", 512)),
        ("reranker", (4, 20)),
        (
            "dense_reranker",
            {
                "candidate_pool_k": 20,
                "reranker_input_k": 20,
                "final_top_k": 5,
            },
        ),
        ("tokenizer", "test-bge"),
        ("context", 1234),
        ("llm", ("test-llm", 0.0, 60.0, 2)),
        ("generator", "FakeLLM"),
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
