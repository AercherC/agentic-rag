"""验证 Phase 15A staging validation、activation/rollback 和 BuildJob。"""

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
    """构造与 manifest 测试一致的固定 indexing pipeline。"""

    return PipelineIdentity(
        parser_version="parser-v1",
        cleaner_version="cleaner-v1",
        chunker_version="chunker-v1",
        chunker_config={"strategy": "hybrid", "chunk_size": 512},
        embedding_model="BAAI/bge-m3",
        embedding_version="embedding-v1",
    )


def _manifest(collection: str, *, status: str, chunk_count: int = 2):
    """生成测试用 staging/active manifest。"""

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
    """构造带当前 runtime 溯源 metadata 的 Domain Chunk。"""

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
    """只读模拟 Collection，记录 smoke 调用但不触碰真实 Chroma。"""

    def __init__(self, chunks: list[DocumentChunk], dimension: int = 2) -> None:
        self.chunks = chunks
        self.dimension = dimension
        self.search_calls = 0

    def iter_all_chunks(self):
        """返回固定 Chunk 集合，模拟 VectorStore 的恢复接口。"""

        return iter(self.chunks)

    @property
    def embedding_dimension(self) -> int:
        """暴露 validation 所需的向量维度。"""

        return self.dimension

    def search(self, query_embedding, top_k: int):
        """记录一次 Dense smoke 并返回非空结果。"""

        self.search_calls += 1
        return [self.chunks[0]] if self.chunks else []


def _passed_report() -> CollectionValidationReport:
    """构造 activation 所需的已通过 validation report。"""

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
    """合格 Collection 必须通过所有 activation 前检查。"""

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
    """缺记录、重复 ID 或运行时 metadata 缺失都不能 activation。"""

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
    """activation 后旧 active 成为 previous，rollback 不需要重建。"""

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
    """validation 失败时 activation 不得改变 active pointer。"""

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
    """staging builder 只写目标 Collection，Job 阶段真实推进到 READY。"""

    lifecycle = IndexLifecycle(tmp_path / "index_state.json", "kb-test")
    active_manifest = _manifest("active-a", status="ACTIVE", chunk_count=1)
    staging_manifest = _manifest("staging-b", status="STAGING")
    lifecycle.initialize_active(active_manifest)
    coordinator = StagingBuildCoordinator(lifecycle, BuildJobStore(tmp_path / "job.json"))
    seen: list[str] = []

    def build(collection_name: str, report_stage) -> FakeStore:
        """模拟复用 IngestionService 的 controlled build callback。"""

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
    """controlled build 异常只结束 Job，不污染 active。"""

    lifecycle = IndexLifecycle(tmp_path / "index_state.json", "kb-test")
    active_manifest = _manifest("active-a", status="ACTIVE", chunk_count=1)
    staging_manifest = _manifest("staging-b", status="STAGING")
    lifecycle.initialize_active(active_manifest)
    coordinator = StagingBuildCoordinator(lifecycle, BuildJobStore(tmp_path / "job.json"))

    def build(_collection_name: str, report_stage) -> FakeStore:
        """在 Embedding 阶段模拟可控失败。"""

        report_stage("PARSING")
        report_stage("CHUNKING")
        raise RuntimeError("controlled embedding failure")

    job, report = coordinator.build(staging_manifest, build)

    assert job.status == "FAILED"
    assert report is None
    assert lifecycle.load_state().active.collection_name == "active-a"


def test_build_job_distinguishes_failed_and_blocked_and_persists(tmp_path) -> None:
    """BuildJob 状态与错误类别可持久化，FAILED 不被 BLOCKED 混淆。"""

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
    """相同 base/signature 必须生成同名 staging，不能依赖随机 UUID。"""

    first = build_staging_collection_name("ef_research_bge_m3_hybrid_c512_v2", "a" * 64)
    second = build_staging_collection_name("ef_research_bge_m3_hybrid_c512_v2", "a" * 64)
    assert first == second
    assert "staging" in first


def test_chroma_staging_metadata_is_explicit_and_readable(tmp_path) -> None:
    """新建 staging Collection 可写入生命周期 metadata，且不改变旧 Adapter 语义。"""

    store = ChromaVectorStore(
        tmp_path / "chroma",
        "staging_collection",
        collection_metadata={"lifecycle": "STAGING", "index_signature": "index"},
    )

    assert store.metadata["lifecycle"] == "STAGING"
    assert store.metadata["index_signature"] == "index"
    assert store.count == 0
    assert store.embedding_dimension is None
