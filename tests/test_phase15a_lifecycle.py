"""Verify Phase 15A staging validation, activation/rollback and BuildJob."""

import pytest

from evidenceflow.domain.chunk import DocumentChunk
from evidenceflow.knowledge import (
    BuildJobStore,
    BuildJobTracker,
    CollectionValidationReport,
    IndexLifecycle,
    PipelineIdentity,
    SourceRecord,
    StagingBuildCoordinator,
    build_manifest,
    build_staging_collection_name,
    validate_collection,
)
from evidenceflow.storage import ChromaVectorStore


def _pipeline() -> PipelineIdentity:
    """Build the same fixed indexing pipeline used by the manifest tests."""

    return PipelineIdentity(
        parser_version="parser-v1",
        cleaner_version="cleaner-v1",
        chunker_version="chunker-v1",
        chunker_config={"strategy": "hybrid", "chunk_size": 512},
        embedding_model="BAAI/bge-m3",
        embedding_version="embedding-v1",
    )


def _manifest(collection: str, *, status: str, chunk_count: int = 2):
    """Produce a staging/active manifest for tests."""

    return build_manifest(
        knowledge_base_id="kb-test",
        sources=[SourceRecord(document_id="doc-a", relative_path="a.pdf", content_hash="hash-a")],
        pipeline=_pipeline(),
        collection_name=collection,
        document_count=1,
        chunk_count=chunk_count,
        status=status,  # type: ignore[arg-type]
    )


def _chunk(chunk_id: str, index: int) -> DocumentChunk:
    """Build a Domain Chunk carrying the current runtime provenance metadata."""

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
    """Read-only Collection simulation; it records smoke calls without touching real Chroma."""

    def __init__(self, chunks: list[DocumentChunk], dimension: int = 2) -> None:
        self.chunks = chunks
        self.dimension = dimension
        self.search_calls = 0

    def iter_all_chunks(self):
        """Return a fixed Chunk collection, simulating the VectorStore recovery interface."""

        return iter(self.chunks)

    @property
    def embedding_dimension(self) -> int:
        """Expose the vector dimension validation needs."""

        return self.dimension

    def search(self, query_embedding, top_k: int):
        """Record one Dense smoke call and return a non-empty result."""

        self.search_calls += 1
        return [self.chunks[0]] if self.chunks else []


def _passed_report() -> CollectionValidationReport:
    """Build the passed validation report activation requires."""

    return CollectionValidationReport(
        passed=True,
        manifest_document_count=1,
        actual_document_count=1,
        manifest_chunk_count=2,
        actual_chunk_count=2,
        embedding_dimension=2,
        expected_embedding_dimension=2,
        embedding_dimension_consistent=True,
        collection_readable=True,
        dense_retrieval_smoke=True,
    )


def test_validation_checks_identity_metadata_dimension_and_dense_smoke() -> None:
    """A qualifying Collection must pass every pre-activation check."""

    manifest = _manifest("staging", status="STAGING")
    store = FakeStore([_chunk("c1", 0), _chunk("c2", 1)])
    report = validate_collection(
        manifest,
        store,
        expected_chunk_ids=["c1", "c2"],
        smoke_query_embedding=[1.0, 0.0],
        expected_embedding_dimension=2,
    )

    assert report.passed is True
    assert report.actual_document_count == 1
    assert store.search_calls == 1


def test_validation_rejects_orphan_duplicate_and_missing_metadata() -> None:
    """Missing records, duplicate IDs or missing runtime metadata all block activation."""

    manifest = _manifest("staging", status="STAGING")
    bad = _chunk("c1", 0)
    bad.metadata = {}
    report = validate_collection(
        manifest,
        FakeStore([bad, bad]),
        expected_chunk_ids=["c1", "c2"],
        smoke_query_embedding=[1.0, 0.0],
        expected_embedding_dimension=2,
    )

    assert report.passed is False
    assert report.duplicate_chunk_ids == ("c1",)
    assert report.missing_chunk_ids == ("c2",)
    assert report.missing_metadata


def test_activation_preserves_previous_and_rollback_only_switches_pointers(tmp_path) -> None:
    """After activation the old active becomes previous, and a rollback needs no rebuild."""

    lifecycle = IndexLifecycle(tmp_path / "index_state.json", "kb-test")
    active_manifest = _manifest("active-a", status="ACTIVE", chunk_count=1)
    staging_manifest = _manifest("staging-b", status="STAGING")
    lifecycle.initialize_active(active_manifest)
    lifecycle.prepare_staging(staging_manifest)

    activated = lifecycle.activate(staging_manifest, _passed_report())
    assert activated.active.collection_name == "staging-b"
    assert activated.previous is not None
    assert activated.previous.collection_name == "active-a"
    assert activated.staging is None

    rolled_back = lifecycle.rollback()
    assert rolled_back.active.collection_name == "active-a"
    assert rolled_back.previous is not None
    assert rolled_back.previous.collection_name == "staging-b"


def test_failed_validation_leaves_active_and_staging_state_untouched(tmp_path) -> None:
    """When validation fails, activation must not change the active pointer."""

    lifecycle = IndexLifecycle(tmp_path / "index_state.json", "kb-test")
    active_manifest = _manifest("active-a", status="ACTIVE", chunk_count=1)
    staging_manifest = _manifest("staging-b", status="STAGING")
    lifecycle.initialize_active(active_manifest)
    lifecycle.prepare_staging(staging_manifest)
    before = lifecycle.load_state()
    failed = _passed_report().model_copy(update={"passed": False, "errors": ("controlled failure",)})

    with pytest.raises(ValueError, match="failed validation"):
        lifecycle.activate(staging_manifest, failed)

    after = lifecycle.load_state()
    assert after.active == before.active
    assert after.staging == before.staging


def test_staging_build_coordinator_records_real_stages_and_never_touches_active(tmp_path) -> None:
    """The staging builder writes only the target Collection, and the Job stages really advance to READY."""

    lifecycle = IndexLifecycle(tmp_path / "index_state.json", "kb-test")
    active_manifest = _manifest("active-a", status="ACTIVE", chunk_count=1)
    staging_manifest = _manifest("staging-b", status="STAGING")
    lifecycle.initialize_active(active_manifest)
    coordinator = StagingBuildCoordinator(lifecycle, BuildJobStore(tmp_path / "job.json"))
    seen: list[str] = []

    def build(collection_name: str, report_stage) -> FakeStore:
        """Simulate a controlled build callback that reuses IngestionService."""

        assert collection_name == "staging-b"
        for stage in ("PARSING", "CHUNKING", "EMBEDDING", "INDEXING"):
            seen.append(stage)
            report_stage(stage)
        return FakeStore([_chunk("c1", 0), _chunk("c2", 1)])

    job, report = coordinator.build(
        staging_manifest,
        build,
        expected_chunk_ids=["c1", "c2"],
        smoke_query_embedding=[1.0, 0.0],
        expected_embedding_dimension=2,
    )

    assert job.status == "READY"
    assert report is not None and report.passed is True
    assert seen == ["PARSING", "CHUNKING", "EMBEDDING", "INDEXING"]
    assert lifecycle.load_state().active.collection_name == "active-a"
    assert lifecycle.load_state().staging is not None


def test_staging_build_failure_is_failed_and_keeps_active_pointer(tmp_path) -> None:
    """A controlled build exception only ends the Job and does not pollute active."""

    lifecycle = IndexLifecycle(tmp_path / "index_state.json", "kb-test")
    active_manifest = _manifest("active-a", status="ACTIVE", chunk_count=1)
    staging_manifest = _manifest("staging-b", status="STAGING")
    lifecycle.initialize_active(active_manifest)
    coordinator = StagingBuildCoordinator(lifecycle, BuildJobStore(tmp_path / "job.json"))

    def build(_collection_name: str, report_stage) -> FakeStore:
        """Simulate a controlled failure during the Embedding stage."""

        report_stage("PARSING")
        report_stage("CHUNKING")
        raise RuntimeError("controlled embedding failure")

    job, report = coordinator.build(staging_manifest, build)

    assert job.status == "FAILED"
    assert report is None
    assert lifecycle.load_state().active.collection_name == "active-a"


def test_build_job_distinguishes_failed_and_blocked_and_persists(tmp_path) -> None:
    """The BuildJob status and error class are persistable, and FAILED is not confused with BLOCKED."""

    failed_store = BuildJobStore(tmp_path / "failed.json")
    failed = BuildJobTracker.create(
        knowledge_base_id="kb-test",
        source_corpus_signature="corpus",
        target_index_signature="index",
        target_collection_name="staging",
        store=failed_store,
    )
    failed.advance("SCANNING")
    failed_job = failed.fail("EMBEDDING_ERROR", "controlled failure")
    assert failed_job.status == "FAILED"
    assert failed_store.load().status == "FAILED"

    blocked_store = BuildJobStore(tmp_path / "blocked.json")
    blocked = BuildJobTracker.create(
        knowledge_base_id="kb-test",
        source_corpus_signature="corpus",
        target_index_signature="index",
        target_collection_name="staging",
        store=blocked_store,
    )
    blocked_job = blocked.block("AMBIGUOUS_RENAME", "controlled block")
    assert blocked_job.status == "BLOCKED"
    assert blocked_store.load().status == "BLOCKED"


def test_staging_collection_name_is_deterministic() -> None:
    """The same base/signature must yield the same staging name, with no reliance on a random UUID."""

    first = build_staging_collection_name("ef_research_bge_m3_hybrid_c512_v2", "a" * 64)
    second = build_staging_collection_name("ef_research_bge_m3_hybrid_c512_v2", "a" * 64)
    assert first == second
    assert "staging" in first


def test_staging_name_regenerated_from_its_own_output_differs() -> None:
    """A3.1: after activation, active is the previous staging name; regenerating from it as base must give a different name.

    The old implementation only truncated the first 32 characters of base, so ``…c512_v2`` became ``…c512_v``, and
    regenerating from the result returned the very same name.
    """

    v2 = "ef_research_bge_m3_hybrid_c512_v2"
    signature = "b4e9432c7170615a" + "0" * 48
    first = build_staging_collection_name(v2, signature)
    second = build_staging_collection_name(first, signature)
    third = build_staging_collection_name(second, signature)

    assert len({v2, first, second, third}) == 4
    assert all(3 <= len(name) <= 63 for name in (first, second, third))  # Chroma name length limit


def test_staging_name_differs_for_bases_sharing_the_truncated_prefix() -> None:
    """Different bases sharing the truncated prefix (such as …_v2 and …_v3) must get different names: the hash uses the full base."""

    signature = "c" * 64
    names = {
        build_staging_collection_name(f"ef_research_bge_m3_hybrid_c512_v{n}", signature) for n in range(2, 6)
    }
    assert len(names) == 4


def test_staging_name_avoids_reserved_names_deterministically() -> None:
    """After a rollback the preferred name for the same base+signature is previous; reserved makes it deterministically move to the next candidate."""

    base, signature = "ef_research_bge_m3_hybrid_c512_v2", "d" * 64
    preferred = build_staging_collection_name(base, signature)
    alternative = build_staging_collection_name(base, signature, reserved={base, preferred})

    assert alternative not in {base, preferred}
    assert alternative == build_staging_collection_name(base, signature, reserved={base, preferred})
    assert build_staging_collection_name(base, signature, reserved={"unrelated"}) == preferred


def test_staging_name_raises_when_every_candidate_is_reserved(monkeypatch) -> None:
    import evidenceflow.knowledge.lifecycle as module

    base, signature = "kb", "e" * 64
    monkeypatch.setattr(module, "_STAGING_NAME_ATTEMPTS", 2)
    taken = {build_staging_collection_name(base, signature)}
    taken.add(build_staging_collection_name(base, signature, reserved=taken))

    with pytest.raises(ValueError, match="cannot derive"):
        build_staging_collection_name(base, signature, reserved=taken)


def test_chroma_staging_metadata_is_explicit_and_readable(tmp_path) -> None:
    """A newly created staging Collection can carry lifecycle metadata without changing the old Adapter semantics."""

    store = ChromaVectorStore(
        tmp_path / "chroma",
        "staging_collection",
        collection_metadata={"lifecycle": "STAGING", "index_signature": "index"},
    )

    assert store.metadata["lifecycle"] == "STAGING"
    assert store.metadata["index_signature"] == "index"
    assert store.count == 0
    assert store.embedding_dimension is None


def _lifecycle_with_staging(tmp_path):
    """Build a state file that has active, previous and staging pointers."""

    lifecycle = IndexLifecycle(tmp_path / "index_state.json", "kb-test")
    lifecycle.initialize_active(_manifest("active-a", status="ACTIVE", chunk_count=1))
    staging = _manifest("staging-b", status="STAGING")
    lifecycle.prepare_staging(staging)
    lifecycle.activate(staging, _passed_report())
    lifecycle.prepare_staging(_manifest("staging-c", status="STAGING"))
    return lifecycle


def test_discard_staging_clears_only_staging_pointer(tmp_path) -> None:
    """discard_staging clears only staging; active and previous stay as they were."""

    lifecycle = _lifecycle_with_staging(tmp_path)
    before = lifecycle.load_state()
    assert before.staging is not None and before.previous is not None

    after = lifecycle.discard_staging()

    assert after.staging is None
    assert after.active == before.active
    assert after.previous == before.previous
    assert lifecycle.load_state() == after  # already persisted


def test_discard_staging_is_noop_and_does_not_rewrite_file_when_staging_is_null(tmp_path) -> None:
    lifecycle = IndexLifecycle(tmp_path / "index_state.json", "kb-test")
    lifecycle.initialize_active(_manifest("active-a", status="ACTIVE", chunk_count=1))
    before_bytes = lifecycle.state_path.read_bytes()
    before_mtime = lifecycle.state_path.stat().st_mtime_ns

    state = lifecycle.discard_staging()

    assert state.staging is None
    assert lifecycle.state_path.read_bytes() == before_bytes
    assert lifecycle.state_path.stat().st_mtime_ns == before_mtime


def test_discard_staging_uses_atomic_write_and_leaves_no_temp_files(tmp_path, monkeypatch) -> None:
    """It reuses _write_json_atomic: when the write fails the original state file is unchanged and no temp file is left behind."""

    import evidenceflow.knowledge.lifecycle as lifecycle_module

    lifecycle = _lifecycle_with_staging(tmp_path)
    before_bytes = lifecycle.state_path.read_bytes()

    def failing_replace(source, destination):
        raise OSError("simulated replace failure")

    monkeypatch.setattr(lifecycle_module.os, "replace", failing_replace)
    with pytest.raises(OSError):
        lifecycle.discard_staging()
    monkeypatch.undo()

    assert lifecycle.state_path.read_bytes() == before_bytes
    assert [path.name for path in tmp_path.iterdir()] == ["index_state.json"]
