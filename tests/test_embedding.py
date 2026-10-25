"""Verify the BGE-M3 Embedding Provider's interface, normalization and real model output."""

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
    """Return deterministic vectors and record calls, isolating model download and inference."""

    def __init__(self) -> None:
        self.calls: list[tuple[object, dict[str, object]]] = []

    def encode(self, texts: object, **kwargs: object) -> list[float] | list[list[float]]:
        self.calls.append((texts, kwargs))
        if isinstance(texts, list):
            return [[3.0, 4.0] for _ in texts]
        return [0.0, 5.0]


def _norm(vector: list[float]) -> float:
    """Compute the L2 norm of a test vector."""

    return math.sqrt(sum(value * value for value in vector))


def test_normalize_vector_and_reject_zero_or_non_finite_values() -> None:
    """Explicit normalization must preserve direction and reject zero vectors and non-finite values."""

    assert _normalize_vector([3.0, 4.0]) == pytest.approx([0.6, 0.8])
    with pytest.raises(EmbeddingNormalizationError):
        _normalize_vector([0.0, 0.0])
    with pytest.raises(EmbeddingNormalizationError):
        _normalize_vector([math.inf, 1.0])


def test_provider_loads_once_batches_documents_and_returns_python_floats(
    monkeypatch,
) -> None:
    """The model is loaded only once at initialization, and a document Batch calls the underlying encode once."""

    model = _FakeSentenceTransformer()
    loaded_names: list[str] = []

    def fake_loader(model_name: str, *, device: str) -> _FakeSentenceTransformer:
        loaded_names.append(model_name)
        return model

    monkeypatch.setattr(
        "evidenceflow.embedding.bge_m3.SentenceTransformer", fake_loader
    )
    # Specify CPU explicitly: the default auto resolves to cuda:0 on machines with CUDA, which
    # would make
    # the assertions depend on the runtime environment.
    provider = BgeM3EmbeddingProvider("configured/model", device="cpu")
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
    """An empty document batch performs no inference, while a blank Query or blank text inside a batch
    must be rejected explicitly."""

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
    """When the underlying model cannot load, a project exception carrying the model context is provided."""

    def fail_loading(model_name: str, *, device: str) -> None:
        raise OSError(model_name)

    monkeypatch.setattr(
        "evidenceflow.embedding.bge_m3.SentenceTransformer", fail_loading
    )
    with pytest.raises(EmbeddingLoadError, match="missing/model"):
        BgeM3EmbeddingProvider("missing/model")


def test_embedding_model_and_encoding_share_selected_device(monkeypatch) -> None:
    """Model initialization and Query/Document encoding must use the same resolved device."""

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
    """Load the real BGE-M3 specified by Settings once, shared by the integration tests."""

    settings = Settings(_env_file=None)
    return BgeM3EmbeddingProvider(settings.embedding_model)


@pytest.mark.requires_model
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
    """The real BGE-M3 must return a non-empty, finite and normalized Python query vector."""

    vector = real_bge_m3_provider.embed_query(query)

    assert vector
    assert all(type(value) is float and math.isfinite(value) for value in vector)
    assert _norm(vector) == pytest.approx(1.0, abs=1e-6)


@pytest.mark.requires_model
def test_real_bge_m3_batch_and_query_dimensions_match(
    real_bge_m3_provider: BgeM3EmbeddingProvider,
) -> None:
    """Real document Batches and Queries must land in a normalized vector space of matching dimension."""

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


def test_provider_reports_model_dimension_or_none(monkeypatch) -> None:
    """dimension comes from the model's declaration; None when it declares none or an invalid value,
    leaving the fallback choice to the caller."""

    class DeclaringModel(_FakeSentenceTransformer):
        def get_sentence_embedding_dimension(self) -> int | None:
            return self.declared

    model = DeclaringModel()
    monkeypatch.setattr(
        "evidenceflow.embedding.bge_m3.SentenceTransformer",
        lambda model_name, *, device: model,
    )
    provider = BgeM3EmbeddingProvider("configured/model", device="cpu")

    model.declared = 1024
    assert provider.dimension == 1024
    model.declared = None
    assert provider.dimension is None
    model.declared = True
    assert provider.dimension is None
