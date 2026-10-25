"""Test that the live collection follows the lifecycle active pointer (A3).

Everything uses state files inside temporary directories, a fake store or a real Chroma in a temp
directory; no real model is loaded and the repository's data/chroma is never touched.
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
    """Build a test manifest; the corpus is fixed and only the collection and status change."""

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
    """Create a state file containing only an active entry inside the temp directory."""

    lifecycle = IndexLifecycle(tmp_path / "index_state.json", "kb-test")
    lifecycle.initialize_active(_manifest(active, status="ACTIVE"))
    return lifecycle


def _settings(tmp_path, **overrides) -> Settings:
    """Settings that do not read .env, with the state file pointing into the temp directory."""

    return Settings(
        _env_file=None,
        vector_db_path=tmp_path / "db",
        index_state_path=tmp_path / "index_state.json",
        **overrides,
    )


def _fresh_settings(tmp_path, monkeypatch, **overrides) -> Settings:
    """A brand-new environment with no state file: index_state_path is not set explicitly, and the cwd is moved to the temp directory so the default relative path does not exist."""

    monkeypatch.chdir(tmp_path)
    overrides.setdefault("vector_db_path", tmp_path / "db")
    return Settings(_env_file=None, **overrides)


@pytest.fixture(autouse=True)
def _no_ambient_collection_name(monkeypatch) -> None:
    """COLLECTION_NAME / INDEX_STATE_PATH in the process environment count as "explicitly set", so each test controls them itself."""

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
    """Environment variables also count as "explicitly set"; a stale .env/environment variable must not silently override active."""

    _lifecycle(tmp_path, "col-active")
    monkeypatch.setenv("COLLECTION_NAME", "col-stale")

    with pytest.raises(ActiveCollectionError):
        resolve_active_collection(_settings(tmp_path))


def test_resolve_follows_activate_then_rollback(tmp_path) -> None:
    """After activate and rollback the resolved result follows the pointer, with no configuration change."""

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
    """A3.1: when an explicitly set state file is missing, raise an error rather than silently falling back to the default collection (the default is the live v2)."""

    settings = _settings(tmp_path)  # explicitly points at tmp_path/index_state.json, which was never created
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
    """Exact-index stand-in: its size and dimension satisfy build_retriever's startup validation."""

    size = 2
    dimension = 1024

    def __init__(self, chunks, embeddings):
        self.chunks = list(chunks)


def _patch_retriever_deps(
    monkeypatch, *, metadata: dict | None, opened: list, events: list[str] | None = None
) -> None:
    """Replace build_retriever's heavy dependencies and record the collection name actually opened plus the open arguments.

    ``metadata`` is the return value of the read-only metadata query (None means the collection does not exist).
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
    # The collection resolved from the pointer is only opened, never created; the lifecycle check
    # precedes
    # loading the Embedding model.
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
    assert events == ["read_metadata"]  # the refusal happens before opening the store and loading the Embedding model


def test_build_retriever_does_not_require_lifecycle_metadata_without_state_file(
    tmp_path, monkeypatch
) -> None:
    """In a fresh environment (no state file) the collection has no lifecycle marker and must not be refused; the open behavior is unchanged."""

    opened: list = []
    events: list[str] = []
    _patch_retriever_deps(monkeypatch, metadata={}, opened=opened, events=events)

    settings = _fresh_settings(tmp_path, monkeypatch, collection_name="col-fresh")
    assert build_retriever(settings) == "retriever"
    assert opened == [("col-fresh", {})]  # still get_or_create semantics, no create_if_missing is passed
    assert events == ["embedding", "store"]  # no metadata query is made; construction order is the same as A3


def test_build_retriever_rejects_missing_active_collection_without_creating_it(
    tmp_path, monkeypatch
) -> None:
    """A3.1: when the collection the pointer refers to does not exist, raise a configuration error and leave no empty collection behind (real Chroma)."""

    import chromadb

    _lifecycle(tmp_path, "col-missing")
    db = tmp_path / "db"
    chromadb.PersistentClient(path=str(db))  # the storage directory exists, but it holds no collection

    def boom(*args, **kwargs):
        raise AssertionError("must not load the embedding model before validating the collection")

    monkeypatch.setattr("evidenceflow.runtime.BgeM3EmbeddingProvider", boom)

    with pytest.raises(ActiveCollectionError, match="does not exist"):
        build_retriever(_settings(tmp_path))
    assert chromadb.PersistentClient(path=str(db)).list_collections() == []


def test_build_retriever_opens_real_active_collection_by_pointer(tmp_path, monkeypatch) -> None:
    """Real Chroma: a collection with lifecycle=ACTIVE opens normally through create_if_missing=False."""

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
    guard_ingest_target(settings, "col-staging")  # nothing is protected when the target is not active


def test_guard_ingest_target_is_noop_without_state_file(tmp_path, monkeypatch) -> None:
    guard_ingest_target(_fresh_settings(tmp_path, monkeypatch), "anything")
    assert not (tmp_path / "db").exists()  # the read-only check does not create a Chroma directory just to query


def _make_real_collection(db, name: str, metadata: dict | None) -> None:
    """Create a collection inside the temp directory's real Chroma (test use only)."""

    from evidenceflow.storage import ChromaVectorStore

    ChromaVectorStore(db, name, collection_metadata=metadata)


def test_ingest_cli_refuses_collection_marked_active_without_state_file(tmp_path, monkeypatch, capsys) -> None:
    """A3.1 defense in depth: the state file is missing, but the target collection's metadata is lifecycle=ACTIVE => exit code 2."""

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
    """Fresh environment: ingest is allowed when the collection does not exist (or has no lifecycle marker), and the guard itself creates no collection."""

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
    """The guard must precede model loading and opening Chroma: any constructor being called means the guard failed."""

    _lifecycle(tmp_path, "col-active")

    def boom(*args, **kwargs):
        raise AssertionError("must not construct runtime dependencies")

    for name in ("BgeM3Tokenizer", "BgeM3EmbeddingProvider", "ChromaVectorStore", "IngestionService"):
        monkeypatch.setattr(f"evidenceflow.runtime.{name}", boom)

    with pytest.raises(ActiveIndexProtectedError):
        build_ingestion_service(_settings(tmp_path))
