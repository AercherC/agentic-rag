"""测试 scripts/build_staging_index.py 的构建、校验、丢弃与保护逻辑（A3）。

全部使用临时目录中的 state 文件、Fake builder 与 Fake collection 操作；真实 Chroma 用例只在
临时目录里验证 lifecycle metadata 的改写与 --discard-staging，不触碰仓库的 data/chroma。
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
    """只读 Collection 替身，满足 validate_collection 的读取接口。"""

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
    """一次脚本运行的 Fake 环境：记录对 Chroma 与阶段回调的全部副作用。"""

    def __init__(self, tmp_path, *, staging_chunk_ids=("c1", "c2"), active_chunk_ids=("c1", "c2")) -> None:
        self.tmp_path = tmp_path
        self.settings = Settings(_env_file=None, index_state_path=tmp_path / "index_state.json")
        # active manifest 的 pipeline 与脚本使用的一致，因此新 staging 的 index_signature 等于它的；
        # collection 名不同。
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
    assert after.staging is not None  # 默认保留 staging 指针，不激活、不删除
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
    """chunk ID 与 active 不一致：必须保留 staging collection 与指针，不能删除证据。"""

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
    """即使 staging 指针被篡改成指向 active/previous 的名字，也不能删除它们。"""

    env = Env(tmp_path)
    # 先造出 previous：激活一个 staging，旧 active 成为 previous。
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
    assert env.state().staging is not None  # 指针原样，没有任何删除


def test_discard_without_staging_pointer_is_refused(tmp_path) -> None:
    env = Env(tmp_path)

    with pytest.raises(StagingBuildRefused):
        discard_staging_collection(env.lifecycle, lambda name: None, lambda: set())


def test_discard_is_idempotent_when_collection_was_already_deleted(tmp_path) -> None:
    """A3.1：上次删掉了 collection 但清指针失败；再次 discard 只清指针并成功。"""

    env = Env(tmp_path)
    code, summary = env.run()  # 校验通过、保留 staging
    assert code == 0
    name = summary["staging_collection"]
    env.existing.discard(name)  # 模拟 collection 已被删除、指针仍在
    deleted: list[str] = []

    state = discard_staging_collection(env.lifecycle, deleted.append, lambda: set(env.existing))

    assert deleted == []  # 不再尝试删除不存在的 collection
    assert state.staging is None and env.state().staging is None
    env.run()  # 指针已清，可以重新开始构建


def test_discard_after_validate_retry_recovers_from_pointer_clear_failure(tmp_path, monkeypatch) -> None:
    """删除成功、清指针失败 → 再执行一次 discard 即可恢复，不会卡在“已有 staging 指针”。"""

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
    assert len(env.deleted) == 1  # 第二次没有重复删除


def test_protected_refusal_after_build_uses_exit_code_3(tmp_path, monkeypatch) -> None:
    """构建之后的保护拒绝不能再用“2 = 拒绝开始”：返回 3，staging collection 与指针保留。"""

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
    """跑一次 --activate，返回成为新 active 的 staging 名。"""

    code, summary = env.run(activate=True)
    assert code == 0
    return summary["staging_collection"]


def test_rebuild_after_activation_does_not_reuse_active_name(tmp_path) -> None:
    """A3.1：激活后以同一 signature 重建，staging 名不得等于新 active 或 previous。"""

    env = Env(tmp_path)
    first = _activate_once(env)
    state = env.state()
    assert state.active.collection_name == first

    code, summary = env.run()

    assert code == 0
    assert summary["staging_collection"] not in {first, "col-active"}
    assert env.marked_active == [first]  # 重建没有改写任何已有 collection


def test_rebuild_after_rollback_does_not_reuse_previous_name(tmp_path) -> None:
    """A3.1：rollback 后同 signature 重建，首选名恰好是 previous（上一次的 staging），必须避开。"""

    env = Env(tmp_path)
    first = _activate_once(env)
    env.lifecycle.rollback()
    state = env.state()
    assert state.active.collection_name == "col-active" and state.previous.collection_name == first

    code, summary = env.run(discard_after_validate=True)

    assert code == 0
    assert summary["staging_collection"] not in {first, "col-active"}
    assert env.deleted == [summary["staging_collection"]]  # 只删了新建的 staging
    assert {"col-active", first} <= env.existing


@pytest.mark.parametrize("protected", ["active", "previous"])
def test_refuses_to_build_when_generated_name_equals_active_or_previous(
    tmp_path, monkeypatch, protected
) -> None:
    """纵深防御：生成名等于 active/previous 时在构建前拒绝，提示中不得建议删除。"""

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
    assert (tmp_path / "index_state.json").read_bytes() == before  # 未登记 staging 指针
    assert env.deleted == []


def test_cli_maps_pre_build_refusal_to_exit_code_2(tmp_path, monkeypatch, capsys) -> None:
    import scripts.build_staging_index as module

    env = Env(tmp_path)
    env.run()  # 留下 staging 指针 → 下一次构建在开始前被拒绝
    monkeypatch.setattr(module, "Settings", lambda: env.settings)
    monkeypatch.setattr(module, "make_real_ports", lambda settings: env.ports())

    assert main(["--job-dir", str(tmp_path / "jobs")]) == 2
    assert "REFUSED" in capsys.readouterr().err


def test_cli_discard_staging_is_idempotent_in_real_chroma(tmp_path, monkeypatch, capsys) -> None:
    """--discard-staging（临时目录的真实 Chroma）：删除登记的 staging、清指针；重复执行不出错地拒绝。"""

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
    # 再执行一次：没有登记的 staging，属于“拒绝开始”，什么都不改。
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
    env.run()  # 留下一个 staging 指针

    with pytest.raises(StagingBuildRefused, match="already registered"):
        env.run()


def test_refuses_to_start_when_staging_collection_already_exists(tmp_path) -> None:
    env = Env(tmp_path)
    first_code, summary = env.run(discard_after_validate=True)
    assert first_code == 0
    env.existing.add(summary["staging_collection"])  # 模拟上次遗留的同名 collection

    with pytest.raises(StagingBuildRefused, match="already exists"):
        env.run()
    assert env.state().staging is None  # 拒绝发生在登记 staging 指针之前


def test_activate_and_discard_are_mutually_exclusive(tmp_path) -> None:
    env = Env(tmp_path)

    with pytest.raises(StagingBuildRefused, match="mutually exclusive"):
        env.run(activate=True, discard_after_validate=True)
    assert env.state().staging is None


def test_stage_reporter_is_monotonic_and_fills_skipped_stages() -> None:
    seen: list[str] = []
    reporter = StageReporter(seen.append)

    reporter.reach("PARSING")
    reporter.reach("EMBEDDING")  # 跳过 CHUNKING 时补报
    reporter.reach("CHUNKING")  # 回退事件被忽略（逐文件循环时同一阶段会重复出现）
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
    assert fired == ["encode"]  # 未登记钩子的 decode 不触发


def test_mark_collection_lifecycle_rewrites_only_lifecycle_in_real_chroma(tmp_path) -> None:
    """真实 Chroma（临时目录）：改写后只有 lifecycle 变化，cosine 配置不受影响。"""

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
    # cosine 来自 configuration 而不是 metadata，改写 metadata 不应影响它。
    fresh = chromadb.PersistentClient(path=str(tmp_path / "chroma")).get_collection("col-stg")
    assert fresh.configuration["hnsw"]["space"] == "cosine"


def test_mark_collection_lifecycle_handles_legacy_hnsw_space_metadata(tmp_path) -> None:
    """A3.1：带旧键 hnsw:space 的 cosine collection（如 v2）也能改写 lifecycle，cosine 不变。"""

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
