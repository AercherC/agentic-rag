"""Test the build, validation, discard, and protection logic of scripts/build_staging_index.py (A3).

Everything uses state files in temporary directories, a Fake builder, and Fake collection
operations; the real-Chroma cases only verify lifecycle metadata rewriting and --discard-staging
in a temporary directory and never touch the repository's data/chroma.
"""

import json

import pytest

from evidenceflow.config import Settings
from evidenceflow.domain.chunk import DocumentChunk
from evidenceflow.knowledge import (
    IndexLifecycle,
    LifecycleState,
    SourceRecord,
    build_manifest,
    current_pipeline_from_settings,
)
from evidenceflow.storage import ChromaVectorStore
from scripts.build_staging_index import (
    Observed,
    ProtectedCollectionError,
    StageReporter,
    StagingBuildRefused,
    StagingPorts,
    discard_staging_collection,
    main,
    mark_collection_lifecycle,
    run_staging_build,
)

SOURCES = [SourceRecord(document_id="doc-a", relative_path="a.pdf", content_hash="hash-a")]


def _chunk(chunk_id: str, index: int) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        document_id="doc-a",
        content=f"content-{index}",
        token_count=2,
        chunk_index=index,
        file_name="a.pdf",
        content_hash="hash-a",
        metadata={"chunk_strategy": "hybrid", "element_ids": [f"el-{index}"]},
    )


class FakeStore:
    """A read-only Collection stand-in that satisfies the read interface of validate_collection."""

    def __init__(self, chunk_ids: list[str]) -> None:
        self.chunks = [_chunk(chunk_id, index) for index, chunk_id in enumerate(chunk_ids)]

    def iter_all_chunks(self):
        return iter(self.chunks)

    @property
    def embedding_dimension(self) -> int:
        return 1024

    def search(self, query_embedding, top_k):
        return [self.chunks[0]]


class Env:
    """Fake environment for one script run: records every side effect on Chroma and on the stage callbacks."""

    def __init__(self, tmp_path, *, staging_chunk_ids=("c1", "c2"), active_chunk_ids=("c1", "c2")) -> None:
        self.tmp_path = tmp_path
        self.settings = Settings(_env_file=None, index_state_path=tmp_path / "index_state.json")
        # # The active manifest's pipeline matches the one the script uses, so the new staging
        # index_signature equals its own;
        # # only the collection name differs.
        pipeline = current_pipeline_from_settings(self.settings)
        self.lifecycle = IndexLifecycle(tmp_path / "index_state.json", "kb-test")
        self.lifecycle.initialize_active(
            build_manifest(
                knowledge_base_id="kb-test",
                sources=SOURCES,
                pipeline=pipeline,
                collection_name="col-active",
                document_count=1,
                chunk_count=2,
                status="ACTIVE",
            )
        )
        self.stages: list[str] = []
        self.deleted: list[str] = []
        self.marked_active: list[str] = []
        self.existing = {"col-active"}
        self.staging_chunk_ids = list(staging_chunk_ids)
        self.active_ids = list(active_chunk_ids)
        self.calls: list[str] = []

    def ports(self) -> StagingPorts:
        def build(name, report_stage):
            reporter = StageReporter(lambda stage: (self.stages.append(stage), report_stage(stage)))
            reporter.reach("PARSING")
            reporter.reach("CHUNKING")
            reporter.reach("EMBEDDING")
            reporter.reach("INDEXING")
            self.existing.add(name)
            return FakeStore(self.staging_chunk_ids)

        return StagingPorts(
            scan_sources=lambda: list(SOURCES),
            list_collections=lambda: set(self.existing),
            active_chunk_ids=lambda name: list(self.active_ids),
            build=build,
            smoke_embedding=lambda: [1.0, 0.0],
            delete_collection=lambda name: (self.deleted.append(name), self.existing.discard(name)),
            mark_collection_active=lambda name: (
                self.calls.append("mark_active"),
                self.marked_active.append(name),
            ),
        )

    def run(self, **flags):
        options = {"expected_chunk_ids_from_active": True, "activate": False, "discard_after_validate": False}
        options.update(flags)
        return run_staging_build(self.settings, self.ports(), job_dir=self.tmp_path / "jobs", **options)

    def state(self) -> LifecycleState:
        return self.lifecycle.load_state()


def test_full_build_walks_all_stages_to_ready_and_does_not_activate_by_default(tmp_path) -> None:
    env = Env(tmp_path)
    before = env.state()

    code, summary = env.run()

    assert code == 0
    assert summary["outcome"] == "VALIDATED_STAGING_KEPT"
    assert summary["job"]["status"] == "READY"
    assert summary["validation"]["passed"] is True
    assert env.stages == ["PARSING", "CHUNKING", "EMBEDDING", "INDEXING"]
    after = env.state()
    assert after.active == before.active and after.previous == before.previous
    assert after.staging is not None  # # The staging pointer is kept by default: not activated, not deleted
    assert env.deleted == [] and env.marked_active == []
    assert json.loads((tmp_path / "jobs" / "staging_build_job.json").read_text(encoding="utf-8"))["status"] == "READY"


def test_discard_after_validate_deletes_collection_then_clears_pointer(tmp_path) -> None:
    env = Env(tmp_path)
    before = env.state()

    code, summary = env.run(discard_after_validate=True)

    assert code == 0 and summary["outcome"] == "VALIDATED_AND_DISCARDED"
    assert len(env.deleted) == 1 and env.deleted[0] == summary["staging_collection"]
    assert env.deleted[0] != "col-active"
    after = env.state()
    assert after.staging is None
    assert after.active == before.active and after.previous == before.previous


def test_discard_after_validate_keeps_staging_when_validation_fails(tmp_path) -> None:
    """chunk IDs differ from the active index: the staging collection and pointer must be kept; evidence is never deleted."""

    env = Env(tmp_path, staging_chunk_ids=("c1", "cX"))

    code, summary = env.run(discard_after_validate=True)

    assert code == 1 and summary["outcome"] == "FAILED_KEPT_STAGING"
    assert summary["validation"]["passed"] is False
    assert summary["validation"]["missing_chunk_ids"] == ["c2"]
    assert summary["validation"]["unexpected_chunk_ids"] == ["cX"]
    assert env.deleted == []
    assert env.state().staging is not None


def test_builder_exception_keeps_staging_and_exits_nonzero(tmp_path) -> None:
    env = Env(tmp_path)
    ports = env.ports()

    def failing_build(name, report_stage):
        report_stage("PARSING")
        raise RuntimeError("controlled failure")

    ports.build = failing_build
    code, summary = run_staging_build(
        env.settings, ports, job_dir=tmp_path / "jobs",
        expected_chunk_ids_from_active=True, activate=False, discard_after_validate=True,
    )

    assert code == 1
    assert summary["job"]["status"] == "FAILED"
    assert env.deleted == [] and env.state().staging is not None


@pytest.mark.parametrize("protected", ["active", "previous"])
def test_discard_refuses_to_delete_active_or_previous_collection(tmp_path, protected) -> None:
    """Even when the staging pointer is tampered with to point at the active/previous names, they must not be deleted."""

    env = Env(tmp_path)
    # # Create previous first: activate a staging index so the old active becomes previous.
    code, _ = env.run(activate=True)
    assert code == 0
    state = env.state()
    assert state.previous is not None
    target = state.active if protected == "active" else state.previous
    tampered = state.model_copy(update={"staging": target.model_copy(update={"status": "STAGING"})})
    (tmp_path / "index_state.json").write_text(tampered.model_dump_json(indent=2), encoding="utf-8")
    deleted: list[str] = []

    with pytest.raises(ProtectedCollectionError):
        discard_staging_collection(env.lifecycle, deleted.append, lambda: set(env.existing))

    assert deleted == []
    assert env.state().staging is not None  # # The pointer is untouched and nothing was deleted


def test_discard_without_staging_pointer_is_refused(tmp_path) -> None:
    env = Env(tmp_path)

    with pytest.raises(StagingBuildRefused):
        discard_staging_collection(env.lifecycle, lambda name: None, lambda: set())


def test_discard_is_idempotent_when_collection_was_already_deleted(tmp_path) -> None:
    """A3.1: the previous run deleted the collection but failed to clear the pointer; discarding again only clears the pointer and succeeds."""

    env = Env(tmp_path)
    code, summary = env.run()  # # Validation passed; staging is kept
    assert code == 0
    name = summary["staging_collection"]
    env.existing.discard(name)  # # Simulate a collection already deleted while the pointer remains
    deleted: list[str] = []

    state = discard_staging_collection(env.lifecycle, deleted.append, lambda: set(env.existing))

    assert deleted == []  # # No further attempt to delete a collection that does not exist
    assert state.staging is None and env.state().staging is None
    env.run()  # # The pointer is cleared, so the build can start over


def test_discard_after_validate_retry_recovers_from_pointer_clear_failure(tmp_path, monkeypatch) -> None:
    """Deletion succeeded but clearing the pointer failed -> running discard once more recovers, and it never gets stuck on "a staging pointer already exists"."""

    env = Env(tmp_path)
    original = IndexLifecycle.discard_staging

    def failing_discard(self):
        raise OSError("disk full")

    monkeypatch.setattr(IndexLifecycle, "discard_staging", failing_discard)
    with pytest.raises(OSError):
        env.run(discard_after_validate=True)
    assert len(env.deleted) == 1 and env.state().staging is not None

    monkeypatch.setattr(IndexLifecycle, "discard_staging", original)
    discard_staging_collection(env.lifecycle, env.ports().delete_collection, lambda: set(env.existing))

    assert env.state().staging is None
    assert len(env.deleted) == 1  # # The second run deletes nothing twice


def test_protected_refusal_after_build_uses_exit_code_3(tmp_path, monkeypatch) -> None:
    """A post-build protection refusal must no longer use "2 = refused to start": it returns 3, and the staging collection and pointer are kept."""

    import scripts.build_staging_index as module

    env = Env(tmp_path)

    def tampered_discard(lifecycle, delete_collection, list_collections):
        raise ProtectedCollectionError("state changed during build")

    monkeypatch.setattr(module, "discard_staging_collection", tampered_discard)

    code, summary = env.run(discard_after_validate=True)

    assert code == 3
    assert summary["outcome"] == "VALIDATED_DISCARD_REFUSED"
    assert env.deleted == [] and env.state().staging is not None
    saved = json.loads((tmp_path / "jobs" / "staging_build_summary.json").read_text(encoding="utf-8"))
    assert saved["outcome"] == "VALIDATED_DISCARD_REFUSED"


def _activate_once(env: Env) -> str:
    """Run --activate once and return the name of the staging index that became the new active one."""

    code, summary = env.run(activate=True)
    assert code == 0
    return summary["staging_collection"]


def test_rebuild_after_activation_does_not_reuse_active_name(tmp_path) -> None:
    """A3.1: rebuilding with the same signature after activation must not name staging after the new active or the previous index."""

    env = Env(tmp_path)
    first = _activate_once(env)
    state = env.state()
    assert state.active.collection_name == first

    code, summary = env.run()

    assert code == 0
    assert summary["staging_collection"] not in {first, "col-active"}
    assert env.marked_active == [first]  # # The rebuild rewrote no existing collection


def test_rebuild_after_rollback_does_not_reuse_previous_name(tmp_path) -> None:
    """A3.1: rebuilding with the same signature after a rollback makes the preferred name exactly previous (the last staging), which must be avoided."""

    env = Env(tmp_path)
    first = _activate_once(env)
    env.lifecycle.rollback()
    state = env.state()
    assert state.active.collection_name == "col-active" and state.previous.collection_name == first

    code, summary = env.run(discard_after_validate=True)

    assert code == 0
    assert summary["staging_collection"] not in {first, "col-active"}
    assert env.deleted == [summary["staging_collection"]]  # # Only the newly created staging index was deleted
    assert {"col-active", first} <= env.existing


@pytest.mark.parametrize("protected", ["active", "previous"])
def test_refuses_to_build_when_generated_name_equals_active_or_previous(
    tmp_path, monkeypatch, protected
) -> None:
    """Defence in depth: when the generated name equals active/previous, refuse before building, and the message must not suggest deleting anything."""

    import scripts.build_staging_index as module

    env = Env(tmp_path)
    _activate_once(env)
    state = env.state()
    target = state.active.collection_name if protected == "active" else state.previous.collection_name
    monkeypatch.setattr(module, "build_staging_collection_name", lambda *a, **k: target)
    before = (tmp_path / "index_state.json").read_bytes()

    with pytest.raises(StagingBuildRefused) as excinfo:
        env.run()

    message = str(excinfo.value)
    assert "equals the active or previous collection" in message
    assert "Do not delete" in message and "remove it" not in message
    assert (tmp_path / "index_state.json").read_bytes() == before  # # The staging pointer was never registered
    assert env.deleted == []


def test_cli_maps_pre_build_refusal_to_exit_code_2(tmp_path, monkeypatch, capsys) -> None:
    import scripts.build_staging_index as module

    env = Env(tmp_path)
    env.run()  # # Leaving a staging pointer behind -> the next build is refused before it starts
    monkeypatch.setattr(module, "Settings", lambda: env.settings)
    monkeypatch.setattr(module, "make_real_ports", lambda settings: env.ports())

    assert main(["--job-dir", str(tmp_path / "jobs")]) == 2
    assert "REFUSED" in capsys.readouterr().err


def test_cli_discard_staging_is_idempotent_in_real_chroma(tmp_path, monkeypatch, capsys) -> None:
    """--discard-staging (real Chroma in a temporary directory): delete the registered staging index and clear the pointer; a repeat run is refused without erroring."""

    import chromadb

    import scripts.build_staging_index as module

    env = Env(tmp_path)
    code, summary = env.run()
    name = summary["staging_collection"]
    db = tmp_path / "chroma"
    ChromaVectorStore(db, name, collection_metadata={"lifecycle": "STAGING"})
    ChromaVectorStore(db, "col-active", collection_metadata={"lifecycle": "ACTIVE"})
    settings = Settings(_env_file=None, index_state_path=tmp_path / "index_state.json", vector_db_path=db)
    monkeypatch.setattr(module, "Settings", lambda: settings)

    assert main(["--discard-staging"]) == 0
    names = {item.name for item in chromadb.PersistentClient(path=str(db)).list_collections()}
    assert names == {"col-active"}
    assert env.state().staging is None
    # # Run once more: no staging index is registered, which counts as "refused to start", and
    # nothing changes.
    assert main(["--discard-staging"]) == 2
    assert {item.name for item in chromadb.PersistentClient(path=str(db)).list_collections()} == {"col-active"}
    assert main(["--discard-staging", "--activate"]) == 2


def test_activate_marks_collection_active_before_flipping_pointer(tmp_path) -> None:
    env = Env(tmp_path)
    old_active = env.state().active

    code, summary = env.run(activate=True)

    assert code == 0 and summary["outcome"] == "ACTIVATED"
    state = env.state()
    assert env.marked_active == [summary["staging_collection"]]
    assert state.active.collection_name == summary["staging_collection"] and state.active.status == "ACTIVE"
    assert state.previous is not None and state.previous.collection_name == old_active.collection_name
    assert state.staging is None
    assert summary["actions"] == ["activated"]
    saved = json.loads((tmp_path / "jobs" / "staging_build_job.json").read_text(encoding="utf-8"))
    assert saved["status"] == "ACTIVATED"


def test_refuses_to_start_when_staging_pointer_already_registered(tmp_path) -> None:
    env = Env(tmp_path)
    env.run()  # # Leave one staging pointer behind

    with pytest.raises(StagingBuildRefused, match="already registered"):
        env.run()


def test_refuses_to_start_when_staging_collection_already_exists(tmp_path) -> None:
    env = Env(tmp_path)
    first_code, summary = env.run(discard_after_validate=True)
    assert first_code == 0
    env.existing.add(summary["staging_collection"])  # # Simulate a same-named collection left over from the previous run

    with pytest.raises(StagingBuildRefused, match="already exists"):
        env.run()
    assert env.state().staging is None  # # The refusal happens before the staging pointer is registered


def test_activate_and_discard_are_mutually_exclusive(tmp_path) -> None:
    env = Env(tmp_path)

    with pytest.raises(StagingBuildRefused, match="mutually exclusive"):
        env.run(activate=True, discard_after_validate=True)
    assert env.state().staging is None


def test_stage_reporter_is_monotonic_and_fills_skipped_stages() -> None:
    seen: list[str] = []
    reporter = StageReporter(seen.append)

    reporter.reach("PARSING")
    reporter.reach("EMBEDDING")  # # Backfill the report when CHUNKING is skipped
    reporter.reach("CHUNKING")  # # Fallback events are ignored (the same stage repeats in the per-file loop)
    reporter.reach("PARSING")
    reporter.reach("INDEXING")

    assert seen == ["PARSING", "CHUNKING", "EMBEDDING", "INDEXING"]


def test_observed_proxy_fires_hook_and_forwards_calls_and_attributes() -> None:
    class Inner:
        label = "inner"

        def encode(self, text):
            return [len(text)]

        def decode(self, ids):
            return "x" * ids[0]

    fired: list[str] = []
    observed = Observed(Inner(), {"encode": lambda: fired.append("encode")})

    assert observed.encode("abc") == [3]
    assert observed.decode([2]) == "xx"
    assert observed.label == "inner"
    assert fired == ["encode"]  # # The decode hook of an unregistered hook does not fire


def test_mark_collection_lifecycle_rewrites_only_lifecycle_in_real_chroma(tmp_path) -> None:
    """Real Chroma (temporary directory): after rewriting, only the lifecycle changes; the cosine configuration is unaffected."""

    import chromadb

    metadata = {
        "chunk_strategy": "hybrid",
        "storage_contract": "domain_chunk_metadata_v2",
        "lifecycle": "STAGING",
    }
    ChromaVectorStore(tmp_path / "chroma", "col-stg", collection_metadata=metadata)
    client = chromadb.PersistentClient(path=str(tmp_path / "chroma"))

    mark_collection_lifecycle(client, "col-stg", "ACTIVE")

    reopened = ChromaVectorStore(tmp_path / "chroma", "col-stg")
    assert reopened.metadata == {**metadata, "lifecycle": "ACTIVE"}
    # # cosine comes from configuration rather than metadata, so rewriting metadata must not affect
    # it.
    fresh = chromadb.PersistentClient(path=str(tmp_path / "chroma")).get_collection("col-stg")
    assert fresh.configuration["hnsw"]["space"] == "cosine"


def test_mark_collection_lifecycle_handles_legacy_hnsw_space_metadata(tmp_path) -> None:
    """A3.1: a cosine collection carrying the legacy hnsw:space key (such as v2) can also have its lifecycle rewritten, and cosine is unchanged."""

    import chromadb

    client = chromadb.PersistentClient(path=str(tmp_path / "chroma"))
    client.create_collection(
        name="col-legacy",
        metadata={"hnsw:space": "cosine", "chunk_strategy": "hybrid", "lifecycle": "ACTIVE"},
        embedding_function=None,
    )
    assert client.get_collection("col-legacy").configuration["hnsw"]["space"] == "cosine"

    mark_collection_lifecycle(client, "col-legacy", "PREVIOUS")

    fresh = chromadb.PersistentClient(path=str(tmp_path / "chroma")).get_collection("col-legacy")
    assert fresh.metadata["lifecycle"] == "PREVIOUS"
    assert fresh.metadata["chunk_strategy"] == "hybrid"
    assert fresh.configuration["hnsw"]["space"] == "cosine"


# ---------------------------------------------------------------------------
# # A3.2: --discard-staging error handling and the naming backstop check
# ---------------------------------------------------------------------------


def _discard_settings(tmp_path, state_path):
    return Settings(_env_file=None, index_state_path=state_path, vector_db_path=tmp_path / "chroma")


def _assert_single_line(err: str, prefix: str) -> None:
    """Error output must be a single line starting with the prefix, with no traceback."""

    assert "Traceback" not in err
    lines = [line for line in err.splitlines() if line.strip()]
    assert len(lines) == 1 and lines[0].startswith(prefix)


def test_cli_discard_staging_missing_state_file_is_refused_without_traceback(tmp_path, monkeypatch, capsys) -> None:
    import scripts.build_staging_index as module

    missing = tmp_path / "missing_state.json"
    monkeypatch.setattr(module, "Settings", lambda: _discard_settings(tmp_path, missing))

    assert main(["--discard-staging"]) == 2

    err = capsys.readouterr().err
    _assert_single_line(err, f"REFUSED: cannot read lifecycle state {missing}: FileNotFoundError: ")
    assert not missing.exists()
    assert not (tmp_path / "chroma").exists()  # # Chroma has not been connected yet when reading state fails


def test_cli_discard_staging_corrupt_state_file_is_refused_without_traceback(tmp_path, monkeypatch, capsys) -> None:
    import scripts.build_staging_index as module

    corrupt = tmp_path / "index_state.json"
    corrupt.write_text("{not json", encoding="utf-8")
    monkeypatch.setattr(module, "Settings", lambda: _discard_settings(tmp_path, corrupt))

    assert main(["--discard-staging"]) == 2

    err = capsys.readouterr().err
    _assert_single_line(err, f"REFUSED: cannot read lifecycle state {corrupt}: ValidationError: ")
    assert corrupt.read_text(encoding="utf-8") == "{not json"  # # Kept as is; not rewritten


def test_cli_discard_staging_delete_failure_returns_1_and_keeps_pointer(tmp_path, monkeypatch, capsys) -> None:
    """Chroma raises while deleting the collection: exit code 1, one FAILED line, and the staging pointer is left unchanged."""

    import chromadb

    import scripts.build_staging_index as module

    env = Env(tmp_path)
    code, summary = env.run()  # # Validation passed; the staging pointer is kept
    assert code == 0
    name = summary["staging_collection"]
    before = (tmp_path / "index_state.json").read_bytes()

    class Item:
        def __init__(self, item_name: str) -> None:
            self.name = item_name

    class FailingClient:
        """List returns the staging index, deleting raises, simulating a Chroma deletion failure."""

        def __init__(self, path: str) -> None:
            self.path = path

        def list_collections(self):
            return [Item("col-active"), Item(name)]

        def delete_collection(self, item_name: str) -> None:
            raise RuntimeError(f"cannot delete {item_name}")

    monkeypatch.setattr(chromadb, "PersistentClient", FailingClient)
    monkeypatch.setattr(module, "Settings", lambda: _discard_settings(tmp_path, tmp_path / "index_state.json"))

    assert main(["--discard-staging"]) == 1

    err = capsys.readouterr().err
    _assert_single_line(err, f"FAILED: RuntimeError: cannot delete {name}")
    assert env.state().staging is not None and env.state().staging.collection_name == name
    assert (tmp_path / "index_state.json").read_bytes() == before


@pytest.mark.parametrize("protected", ["active", "previous"])
def test_naming_guard_refuses_before_any_build_or_write_port_is_called(tmp_path, monkeypatch, protected) -> None:
    """A3.2: the current generator never produces an active/previous name, so this check is a backstop; a generator regression must be refused before any build or write."""

    import scripts.build_staging_index as module

    env = Env(tmp_path)
    if protected == "previous":
        _activate_once(env)  # # Produce previous
    state = env.state()
    target = state.active.collection_name if protected == "active" else state.previous.collection_name
    monkeypatch.setattr(module, "build_staging_collection_name", lambda *a, **k: target)
    marked_before = list(env.marked_active)
    called: list[str] = []
    ports = env.ports()

    def record(port_name):
        def port(*args, **kwargs):
            called.append(port_name)
            raise AssertionError(f"port {port_name} must not be called")

        return port

    ports.build = record("build")
    ports.delete_collection = record("delete_collection")
    ports.mark_collection_active = record("mark_collection_active")
    ports.active_chunk_ids = record("active_chunk_ids")
    ports.smoke_embedding = record("smoke_embedding")
    before = (tmp_path / "index_state.json").read_bytes()
    job_dir = tmp_path / "jobs_guard"

    with pytest.raises(StagingBuildRefused, match="equals the active or previous collection"):
        run_staging_build(
            env.settings, ports, job_dir=job_dir,
            expected_chunk_ids_from_active=True, activate=False, discard_after_validate=True,
        )

    assert called == []
    assert env.marked_active == marked_before
    assert (tmp_path / "index_state.json").read_bytes() == before
    assert not job_dir.exists()  # # No BuildJob or summary was written
