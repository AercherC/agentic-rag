"""测试线上 collection 跟随 lifecycle active 指针（A3）。

全部使用临时目录中的 state 文件；Fake store 或临时目录里的真实 Chroma，不加载真实模型，
不触碰仓库的 data/chroma。
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
from evidenceflow.__main__ import main
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


def _fresh_settings(tmp_path, monkeypatch, **overrides) -> Settings:
    """没有状态文件的全新环境：不显式设置 index_state_path，并切到临时目录使默认相对路径不存在。"""

    monkeypatch.chdir(tmp_path)
    overrides.setdefault("vector_db_path", tmp_path / "db")
    return Settings(_env_file=None, **overrides)


@pytest.fixture(autouse=True)
def _no_ambient_collection_name(monkeypatch) -> None:
    """进程环境里的 COLLECTION_NAME / INDEX_STATE_PATH 会被视为“显式设置”，用例需要自己控制它们。"""

    monkeypatch.delenv("COLLECTION_NAME", raising=False)
    monkeypatch.delenv("INDEX_STATE_PATH", raising=False)


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


def test_resolve_falls_back_to_settings_without_state_file(tmp_path, monkeypatch) -> None:
    settings = _fresh_settings(tmp_path, monkeypatch, collection_name="col-from-settings")

    assert "index_state_path" not in settings.model_fields_set
    assert not settings.index_state_path.exists()
    assert resolve_active_collection(settings) == "col-from-settings"


def test_explicit_index_state_path_that_does_not_exist_is_a_configuration_error(tmp_path) -> None:
    """A3.1：显式设置的状态文件缺失时报错，不能静默回落到默认 collection（默认值就是线上 v2）。"""

    settings = _settings(tmp_path)  # 显式指向 tmp_path/index_state.json，但没有创建
    assert "index_state_path" in settings.model_fields_set

    with pytest.raises(ActiveCollectionError, match="set explicitly but the lifecycle state file"):
        resolve_active_collection(settings)
    with pytest.raises(ActiveCollectionError, match="set explicitly"):
        guard_ingest_target(settings, "anything")


def test_explicit_index_state_path_from_environment_that_does_not_exist_is_rejected(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("INDEX_STATE_PATH", str(tmp_path / "gone.json"))

    with pytest.raises(ActiveCollectionError, match="gone.json"):
        resolve_active_collection(Settings(_env_file=None))


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


def _patch_retriever_deps(
    monkeypatch, *, metadata: dict | None, opened: list, events: list[str] | None = None
) -> None:
    """替换 build_retriever 的重依赖，并记录实际打开的 collection 名称与打开参数。

    ``metadata`` 是只读 metadata 查询的返回值（None 表示 collection 不存在）。
    """

    events = events if events is not None else []

    class FakeStore:
        count = 2

        def __init__(self, path, collection, **kwargs):
            events.append("store")
            opened.append((collection, kwargs))
            self.metadata = metadata or {}

        def iter_all_chunks_with_embeddings(self):
            return iter([("chunk-a", [1.0])])

    def fake_read_metadata(path, collection):
        events.append("read_metadata")
        return metadata

    def fake_embedding(model, *, device):
        events.append("embedding")
        return object()

    monkeypatch.setattr("evidenceflow.runtime.read_collection_metadata", fake_read_metadata)
    monkeypatch.setattr("evidenceflow.runtime.BgeM3EmbeddingProvider", fake_embedding)
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
    opened: list = []
    events: list[str] = []
    _patch_retriever_deps(monkeypatch, metadata={"lifecycle": "ACTIVE"}, opened=opened, events=events)

    assert build_retriever(_settings(tmp_path)) == "retriever"
    # 指针解析出的 collection 只打开、不创建；lifecycle 校验先于 Embedding 模型加载。
    assert opened == [("col-active", {"create_if_missing": False})]
    assert events == ["read_metadata", "store", "embedding"]


@pytest.mark.parametrize("metadata", [{"lifecycle": "STAGING"}, {"lifecycle": "PREVIOUS"}, {}])
def test_build_retriever_rejects_collection_whose_lifecycle_is_not_active(
    tmp_path, monkeypatch, metadata
) -> None:
    _lifecycle(tmp_path, "col-active")
    events: list[str] = []
    _patch_retriever_deps(monkeypatch, metadata=metadata, opened=[], events=events)

    with pytest.raises(ActiveCollectionError, match="lifecycle"):
        build_retriever(_settings(tmp_path))
    assert events == ["read_metadata"]  # 拒绝发生在打开 store 与加载 Embedding 模型之前


def test_build_retriever_does_not_require_lifecycle_metadata_without_state_file(
    tmp_path, monkeypatch
) -> None:
    """全新环境（无状态文件）下 collection 没有 lifecycle 标记，不应被拒绝；打开方式不变。"""

    opened: list = []
    events: list[str] = []
    _patch_retriever_deps(monkeypatch, metadata={}, opened=opened, events=events)

    settings = _fresh_settings(tmp_path, monkeypatch, collection_name="col-fresh")
    assert build_retriever(settings) == "retriever"
    assert opened == [("col-fresh", {})]  # 仍是 get_or_create 语义，不传 create_if_missing
    assert events == ["embedding", "store"]  # 不做 metadata 查询，构造顺序与 A3 相同


def test_build_retriever_rejects_missing_active_collection_without_creating_it(
    tmp_path, monkeypatch
) -> None:
    """A3.1：指针指向的 collection 不存在时给出配置错误，且不能留下空 collection（真实 Chroma）。"""

    import chromadb

    _lifecycle(tmp_path, "col-missing")
    db = tmp_path / "db"
    chromadb.PersistentClient(path=str(db))  # 存储目录存在，但没有任何 collection

    def boom(*args, **kwargs):
        raise AssertionError("must not load the embedding model before validating the collection")

    monkeypatch.setattr("evidenceflow.runtime.BgeM3EmbeddingProvider", boom)

    with pytest.raises(ActiveCollectionError, match="does not exist"):
        build_retriever(_settings(tmp_path))
    assert chromadb.PersistentClient(path=str(db)).list_collections() == []


def test_build_retriever_opens_real_active_collection_by_pointer(tmp_path, monkeypatch) -> None:
    """真实 Chroma：lifecycle=ACTIVE 的 collection 经 create_if_missing=False 正常打开。"""

    from evidenceflow.storage import ChromaVectorStore

    _lifecycle(tmp_path, "col-active")
    ChromaVectorStore(tmp_path / "db", "col-active", collection_metadata={"lifecycle": "ACTIVE"})
    opened: list[str] = []

    class Stop(Exception):
        pass

    def stop_at_embedding(model, *, device):
        raise Stop

    real_store = ChromaVectorStore

    def recording_store(path, collection, **kwargs):
        store = real_store(path, collection, **kwargs)
        opened.append(store.collection_name)
        return store

    monkeypatch.setattr("evidenceflow.runtime.ChromaVectorStore", recording_store)
    monkeypatch.setattr("evidenceflow.runtime.BgeM3EmbeddingProvider", stop_at_embedding)

    with pytest.raises(Stop):
        build_retriever(_settings(tmp_path))
    assert opened == ["col-active"]


def test_guard_ingest_target_rejects_active_and_allows_others(tmp_path) -> None:
    _lifecycle(tmp_path, "col-active")
    settings = _settings(tmp_path)

    with pytest.raises(ActiveIndexProtectedError, match="build_staging_index"):
        guard_ingest_target(settings, "col-active")
    guard_ingest_target(settings, "col-staging")  # 非 active 不受影响


def test_guard_ingest_target_is_noop_without_state_file(tmp_path, monkeypatch) -> None:
    guard_ingest_target(_fresh_settings(tmp_path, monkeypatch), "anything")
    assert not (tmp_path / "db").exists()  # 只读检查不为查询而创建 Chroma 目录


def _make_real_collection(db, name: str, metadata: dict | None) -> None:
    """在临时目录的真实 Chroma 中建 collection（只用于测试）。"""

    from evidenceflow.storage import ChromaVectorStore

    ChromaVectorStore(db, name, collection_metadata=metadata)


def test_ingest_cli_refuses_collection_marked_active_without_state_file(tmp_path, monkeypatch, capsys) -> None:
    """A3.1 纵深防御：状态文件缺失，但目标 collection 的 metadata 是 lifecycle=ACTIVE → 退出码 2。"""

    settings = _fresh_settings(tmp_path, monkeypatch, collection_name="col-live")
    _make_real_collection(settings.vector_db_path, "col-live", {"lifecycle": "ACTIVE"})

    def boom(*args, **kwargs):
        raise AssertionError("must not construct runtime dependencies")

    for name in ("BgeM3Tokenizer", "BgeM3EmbeddingProvider", "IngestionService"):
        monkeypatch.setattr(f"evidenceflow.runtime.{name}", boom)

    exit_code = main(["ingest"], settings=settings)
    captured = capsys.readouterr()

    assert exit_code == 2
    assert "col-live" in captured.err and "lifecycle=ACTIVE" in captured.err
    assert captured.out == ""


@pytest.mark.parametrize("db_state", ["no_db_dir", "db_without_collection", "collection_without_lifecycle"])
def test_fresh_environment_first_ingest_is_still_allowed(tmp_path, monkeypatch, db_state) -> None:
    """全新环境：collection 不存在（或没有 lifecycle 标记）时放行，且保护检查本身不创建 collection。"""

    import chromadb

    settings = _fresh_settings(tmp_path, monkeypatch, collection_name="col-new")
    db = settings.vector_db_path
    if db_state == "db_without_collection":
        chromadb.PersistentClient(path=str(db))
    elif db_state == "collection_without_lifecycle":
        _make_real_collection(db, "col-new", None)
    built: list[str] = []
    monkeypatch.setattr("evidenceflow.runtime.BgeM3Tokenizer", lambda model: object())
    monkeypatch.setattr("evidenceflow.runtime.BgeM3EmbeddingProvider", lambda model, *, device: object())
    monkeypatch.setattr("evidenceflow.runtime.ChromaVectorStore", lambda path, name: built.append(name))
    monkeypatch.setattr("evidenceflow.runtime.IngestionService", lambda *a, **k: "service")

    assert build_ingestion_service(settings) == "service"
    assert built == ["col-new"]
    if db_state == "no_db_dir":
        assert not db.exists()
    elif db_state == "db_without_collection":
        assert chromadb.PersistentClient(path=str(db)).list_collections() == []


def test_build_ingestion_service_refuses_active_before_loading_any_model(tmp_path, monkeypatch) -> None:
    """保护必须先于模型加载与 Chroma 打开：任何构造器被调用都说明保护失效。"""

    _lifecycle(tmp_path, "col-active")

    def boom(*args, **kwargs):
        raise AssertionError("must not construct runtime dependencies")

    for name in ("BgeM3Tokenizer", "BgeM3EmbeddingProvider", "ChromaVectorStore", "IngestionService"):
        monkeypatch.setattr(f"evidenceflow.runtime.{name}", boom)

    with pytest.raises(ActiveIndexProtectedError):
        build_ingestion_service(_settings(tmp_path))
