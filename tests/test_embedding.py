"""验证 BGE-M3 Embedding Provider 的接口、归一化与真实模型输出。"""

import math

import pytest

from evidenceflow.config import Settings
from evidenceflow.embedding import BgeM3EmbeddingProvider, EmbeddingProvider
from evidenceflow.embedding.bge_m3 import (
    EmbeddingLoadError,
    EmbeddingNormalizationError,
    _normalize_vector,
)


class _FakeSentenceTransformer:
    """返回确定向量并记录调用，用于隔离模型下载和推理。"""

    def __init__(self) -> None:
        self.calls: list[tuple[object, dict[str, object]]] = []

    def encode(self, texts: object, **kwargs: object) -> list[float] | list[list[float]]:
        self.calls.append((texts, kwargs))
        if isinstance(texts, list):
            return [[3.0, 4.0] for _ in texts]
        return [0.0, 5.0]


def _norm(vector: list[float]) -> float:
    """计算测试向量的 L2 norm。"""

    return math.sqrt(sum(value * value for value in vector))


def test_normalize_vector_and_reject_zero_or_non_finite_values() -> None:
    """显式归一化应保持方向，并拒绝零向量及非有限数值。"""

    assert _normalize_vector([3.0, 4.0]) == pytest.approx([0.6, 0.8])
    with pytest.raises(EmbeddingNormalizationError):
        _normalize_vector([0.0, 0.0])
    with pytest.raises(EmbeddingNormalizationError):
        _normalize_vector([math.inf, 1.0])


def test_provider_loads_once_batches_documents_and_returns_python_floats(
    monkeypatch,
) -> None:
    """模型只在初始化时加载一次，文档 Batch 只调用一次底层 encode。"""

    model = _FakeSentenceTransformer()
    loaded_names: list[str] = []

    def fake_loader(model_name: str, *, device: str) -> _FakeSentenceTransformer:
        loaded_names.append(model_name)
        return model

    monkeypatch.setattr(
        "evidenceflow.embedding.bge_m3.SentenceTransformer", fake_loader
    )
    provider = BgeM3EmbeddingProvider("configured/model")
    texts = ["first document", "second document", "third document"]
    vectors = provider.embed_documents(texts)
    query_vector = provider.embed_query("original query")

    assert isinstance(provider, EmbeddingProvider)
    assert loaded_names == ["configured/model"]
    assert model.calls[0][0] == texts
    assert len(model.calls) == 2
    assert len(vectors) == 3
    assert all(vector == pytest.approx([0.6, 0.8]) for vector in vectors)
    assert query_vector == pytest.approx([0.0, 1.0])
    assert all(type(value) is float for vector in vectors for value in vector)
    assert all(type(value) is float for value in query_vector)
    assert all(
        call[1]
        == {
            "convert_to_numpy": True,
            "normalize_embeddings": False,
            "show_progress_bar": False,
            "device": "cpu",
        }
        for call in model.calls
    )


def test_empty_inputs_are_handled_without_model_inference(monkeypatch) -> None:
    """空文档批次不推理，Query 或批次内空文本应明确拒绝。"""

    model = _FakeSentenceTransformer()
    monkeypatch.setattr(
        "evidenceflow.embedding.bge_m3.SentenceTransformer",
        lambda model_name, *, device: model,
    )
    provider = BgeM3EmbeddingProvider("configured/model")

    assert provider.embed_documents([]) == []
    assert model.calls == []
    with pytest.raises(ValueError):
        provider.embed_query("")
    with pytest.raises(ValueError):
        provider.embed_query("   ")
    with pytest.raises(ValueError):
        provider.embed_documents(["valid", "\t"])
    assert model.calls == []


def test_model_loading_failure_is_wrapped(monkeypatch) -> None:
    """底层模型无法加载时应提供带模型上下文的项目异常。"""

    def fail_loading(model_name: str, *, device: str) -> None:
        raise OSError(model_name)

    monkeypatch.setattr(
        "evidenceflow.embedding.bge_m3.SentenceTransformer", fail_loading
    )
    with pytest.raises(EmbeddingLoadError, match="missing/model"):
        BgeM3EmbeddingProvider("missing/model")


def test_embedding_model_and_encoding_share_selected_device(monkeypatch) -> None:
    """模型初始化与 Query/Document 编码必须使用同一解析设备。"""

    model = _FakeSentenceTransformer()
    loaded_devices: list[str] = []

    def fake_loader(model_name: str, *, device: str) -> _FakeSentenceTransformer:
        loaded_devices.append(device)
        return model

    monkeypatch.setattr(
        "evidenceflow.embedding.bge_m3.resolve_model_device",
        lambda requested: "cuda:0",
    )
    monkeypatch.setattr("evidenceflow.embedding.bge_m3.SentenceTransformer", fake_loader)
    provider = BgeM3EmbeddingProvider("configured/model", device="auto")

    document_vector = provider.embed_documents(["document"])[0]
    query_vector = provider.embed_query("query")

    assert provider.device == "cuda:0"
    assert loaded_devices == ["cuda:0"]
    assert all(call[1]["device"] == "cuda:0" for call in model.calls)
    assert all(type(value) is float for value in document_vector + query_vector)


@pytest.fixture(scope="module")
def real_bge_m3_provider() -> BgeM3EmbeddingProvider:
    """加载一次 Settings 指定的真实 BGE-M3，供集成测试共享。"""

    settings = Settings(_env_file=None)
    return BgeM3EmbeddingProvider(settings.embedding_model)


@pytest.mark.parametrize(
    "query",
    [
        "An AI agent can use tools to complete tasks.",
        "智能体可以根据任务调用外部工具。",
    ],
)
def test_real_bge_m3_embeds_english_and_chinese_queries(
    real_bge_m3_provider: BgeM3EmbeddingProvider,
    query: str,
) -> None:
    """真实 BGE-M3 应返回非空、有限且归一化的 Python 查询向量。"""

    vector = real_bge_m3_provider.embed_query(query)

    assert vector
    assert all(type(value) is float and math.isfinite(value) for value in vector)
    assert _norm(vector) == pytest.approx(1.0, abs=1e-6)


def test_real_bge_m3_batch_and_query_dimensions_match(
    real_bge_m3_provider: BgeM3EmbeddingProvider,
) -> None:
    """真实文档 Batch 与 Query 必须处于维度一致的归一化向量空间。"""

    texts = [
        "LangGraph supports stateful agent workflows.",
        "Agents can call external tools.",
        "Vector databases support similarity search.",
    ]
    document_vectors = real_bge_m3_provider.embed_documents(texts)
    query_vector = real_bge_m3_provider.embed_query(
        "How does LangGraph manage agent state?"
    )

    assert len(document_vectors) == 3
    dimension = len(query_vector)
    assert dimension > 0
    assert all(len(vector) == dimension for vector in document_vectors)
    assert all(_norm(vector) == pytest.approx(1.0, abs=1e-6) for vector in document_vectors)
