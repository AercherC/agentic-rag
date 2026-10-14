"""测试线上 collection 跟随 lifecycle active 指针（A3）。

全部使用临时目录中的 state 文件和 Fake store，不加载真实模型、不打开真实 Chroma。
"""

import pytest

from evidenceflow.config import Settings
from evidenceflow.knowledge import (
    ActiveCollectionError,
    ActiveIndexProtectedError,
    CollectionValidationReport,
    IndexLifecycle,
    PipelineIdentity,
    SourceRecord,
    build_manifest,
)
from evidenceflow.runtime import (
    build_ingestion_service,
    build_retriever,
    guard_ingest_target,
    resolve_active_collection,
)


def _manifest(collection: str, *, status: str):
    """生成测试用 manifest；corpus 固定，仅 collection 与状态变化。"""

    pipeline = PipelineIdentity(
        parser_version="p1",
        cleaner_version="c1",
        chunker_version="k1",
        chunker_config={"strategy": "hybrid"},
        embedding_model="m",
        embedding_version="e1",
    )
    return build_manifest(
        knowledge_base_id="kb-test",
        sources=[SourceRecord(document_id="d", relative_path="a.pdf", content_hash="h")],
        pipeline=pipeline,
        collection_name=collection,
        document_count=1,
        chunk_count=2,
        status=status,  # type: ignore[arg-type]
    )


def _lifecycle(tmp_path, active: str = "col-active") -> IndexLifecycle:
    """在临时目录建立只含 active 的状态文件。"""

    lifecycle = IndexLifecycle(tmp_path / "index_state.json", "kb-test")
    lifecycle.initialize_active(_manifest(active, status="ACTIVE"))
    return lifecycle


def _settings(tmp_path, **overrides) -> Settings:
    """不读取 .env 的 Settings，状态文件指向临时目录。"""

    return Settings(
        _env_file=None,
        vector_db_path=tmp_path / "db",
        index_state_path=tmp_path / "index_state.json",
        **overrides,
    )


@pytest.fixture(autouse=True)
def _no_ambient_collection_name(monkeypatch) -> None:
    """进程环境里的 COLLECTION_NAME 会被视为“显式设置”，用例需要自己控制它。"""

    monkeypatch.delenv("COLLECTION_NAME", raising=False)


def test_resolve_uses_active_pointer_when_collection_name_not_explicit(tmp_path) -> None:
    _lifecycle(tmp_path, "col-active")

    assert resolve_active_collection(_settings(tmp_path)) == "col-active"


def test_resolve_rejects_explicit_collection_name_that_conflicts_with_active(tmp_path) -> None:
    _lifecycle(tmp_path, "col-active")

    with pytest.raises(ActiveCollectionError, match="conflicts"):
        resolve_active_collection(_settings(tmp_path, collection_name="col-other"))


def test_resolve_accepts_explicit_collection_name_equal_to_active(tmp_path) -> None:
    _lifecycle(tmp_path, "col-active")

    assert resolve_active_collection(_settings(tmp_path, collection_name="col-active")) == "col-active"


def test_resolve_rejects_conflicting_collection_name_from_environment(tmp_path, monkeypatch) -> None:
    """环境变量也算“显式设置”，过期的 .env/环境变量不能静默盖过 active。"""

    _lifecycle(tmp_path, "col-active")
    monkeypatch.setenv("COLLECTION_NAME", "col-stale")

    with pytest.raises(ActiveCollectionError):
        resolve_active_collection(_settings(tmp_path))


def test_resolve_follows_activate_then_rollback(tmp_path) -> None:
    """activate 与 rollback 之后，解析结果随 pointer 改变，无需改配置。"""

    lifecycle = _lifecycle(tmp_path, "col-v1")
    staging = _manifest("col-v2", status="STAGING")
    lifecycle.prepare_staging(staging)
    passed = CollectionValidationReport(
        passed=True,
        manifest_document_count=1,
        actual_document_count=1,
        manifest_chunk_count=2,
        actual_chunk_count=2,
    )
    settings = _settings(tmp_path)
    assert resolve_active_collection(settings) == "col-v1"

    lifecycle.activate(staging, passed)
    assert resolve_active_collection(settings) == "col-v2"

    lifecycle.rollback()
    assert resolve_active_collection(settings) == "col-v1"


def test_resolve_falls_back_to_settings_without_state_file(tmp_path) -> None:
    settings = _settings(tmp_path, collection_name="col-from-settings")

    assert not settings.index_state_path.exists()
    assert resolve_active_collection(settings) == "col-from-settings"


def test_resolve_rejects_corrupt_state_file_instead_of_falling_back(tmp_path) -> None:
    (tmp_path / "index_state.json").write_text("{not json", encoding="utf-8")

    with pytest.raises(ActiveCollectionError, match="Cannot read lifecycle state"):
        resolve_active_collection(_settings(tmp_path))


class _FakeExactIndex:
    """Exact 索引替身：大小与维度满足 build_retriever 的启动校验。"""

    size = 2
    dimension = 1024

    def __init__(self, chunks, embeddings):
        self.chunks = list(chunks)


def _patch_retriever_deps(monkeypatch, *, metadata: dict, opened: list[str]) -> None:
    """替换 build_retriever 的重依赖，并记录实际打开的 collection 名称。"""

    class FakeStore:
        count = 2

        def __init__(self, path, collection):
            opened.append(collection)
            self.metadata = metadata

        def iter_all_chunks_with_embeddings(self):
            return iter([("chunk-a", [1.0])])

    monkeypatch.setattr("evidenceflow.runtime.BgeM3EmbeddingProvider", lambda m, *, device: object())
    monkeypatch.setattr("evidenceflow.runtime.ChromaVectorStore", FakeStore)
    monkeypatch.setattr("evidenceflow.runtime.ExactDenseIndex", _FakeExactIndex)
    monkeypatch.setattr("evidenceflow.runtime.DenseRetriever", lambda e, s, top_k: object())
    monkeypatch.setattr(
        "evidenceflow.runtime.TransformersCrossEncoderScorer",
        lambda m, *, device, dtype, max_length: object(),
    )
    monkeypatch.setattr(
        "evidenceflow.runtime.LearnedReranker", lambda scorer, *, batch_size, max_candidates: object()
    )
    monkeypatch.setattr("evidenceflow.runtime.DenseRerankerRetriever", lambda *a, **k: "retriever")


def test_build_retriever_opens_active_collection_with_active_lifecycle_metadata(
    tmp_path, monkeypatch
) -> None:
    _lifecycle(tmp_path, "col-active")
    opened: list[str] = []
    _patch_retriever_deps(monkeypatch, metadata={"lifecycle": "ACTIVE"}, opened=opened)

    assert build_retriever(_settings(tmp_path)) == "retriever"
    assert opened == ["col-active"]


@pytest.mark.parametrize("metadata", [{"lifecycle": "STAGING"}, {"lifecycle": "PREVIOUS"}, {}])
def test_build_retriever_rejects_collection_whose_lifecycle_is_not_active(
    tmp_path, monkeypatch, metadata
) -> None:
    _lifecycle(tmp_path, "col-active")
    _patch_retriever_deps(monkeypatch, metadata=metadata, opened=[])

    with pytest.raises(ActiveCollectionError, match="lifecycle"):
        build_retriever(_settings(tmp_path))


def test_build_retriever_does_not_require_lifecycle_metadata_without_state_file(
    tmp_path, monkeypatch
) -> None:
    """全新环境（无状态文件）下 collection 没有 lifecycle 标记，不应被拒绝。"""

    opened: list[str] = []
    _patch_retriever_deps(monkeypatch, metadata={}, opened=opened)

    assert build_retriever(_settings(tmp_path, collection_name="col-fresh")) == "retriever"
    assert opened == ["col-fresh"]


def test_guard_ingest_target_rejects_active_and_allows_others(tmp_path) -> None:
    _lifecycle(tmp_path, "col-active")
    settings = _settings(tmp_path)

    with pytest.raises(ActiveIndexProtectedError, match="build_staging_index"):
        guard_ingest_target(settings, "col-active")
    guard_ingest_target(settings, "col-staging")  # 非 active 不受影响


def test_guard_ingest_target_is_noop_without_state_file(tmp_path) -> None:
    guard_ingest_target(_settings(tmp_path), "anything")


def test_build_ingestion_service_refuses_active_before_loading_any_model(tmp_path, monkeypatch) -> None:
    """保护必须先于模型加载与 Chroma 打开：任何构造器被调用都说明保护失效。"""

    _lifecycle(tmp_path, "col-active")

    def boom(*args, **kwargs):
        raise AssertionError("must not construct runtime dependencies")

    for name in ("BgeM3Tokenizer", "BgeM3EmbeddingProvider", "ChromaVectorStore", "IngestionService"):
        monkeypatch.setattr(f"evidenceflow.runtime.{name}", boom)

    with pytest.raises(ActiveIndexProtectedError):
        build_ingestion_service(_settings(tmp_path))
