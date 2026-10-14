"""A3：在 staging collection 上全量重建索引，校验后可选激活或丢弃；绝不直接改动 ACTIVE 索引。

复用 ``StagingBuildCoordinator``（阶段推进 + 校验）、``IngestionService``（作为 builder 回调）与
``validate_collection``，不复制 ingestion 逻辑。默认只构建并校验，不激活：

    python scripts/build_staging_index.py --expected-chunk-ids-from-active --discard-after-validate

上次 discard 中途失败（例如 collection 已删、指针未清）时，可单独清理已登记的 staging：

    python scripts/build_staging_index.py --discard-staging

退出码：0 = 校验通过且所选动作完成（或 --discard-staging 完成）；1 = 构建或校验失败（staging 与指针
原样保留供诊断）；2 = 拒绝开始，没有构建、没有删除、没有改动状态文件（已有 staging 指针、生成的
staging 名与 active/previous 或已有 collection 重名、参数冲突、--discard-staging 时没有登记的
staging 或其名称是 active/previous）；3 = 校验已通过，但构建之后的收尾动作被保护拒绝（staging
collection 与指针保留，状态文件在构建期间被外部改动时才可能发生）。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from evidenceflow.config import Settings  # noqa: E402
from evidenceflow.knowledge import (  # noqa: E402
    BuildJobStore,
    IndexLifecycle,
    LifecycleState,
    PipelineIdentity,
    SourceRecord,
    StagingBuildCoordinator,
    build_corpus_signature,
    build_index_signature,
    build_manifest,
    build_staging_collection_name,
    current_pipeline_from_settings,
    manifest_sources_from_documents,
    save_manifest,
)
from evidenceflow.knowledge.job import BuildJobStatus  # noqa: E402
from evidenceflow.knowledge.lifecycle import CollectionReader  # noqa: E402

DEFAULT_JOB_DIR = ROOT / "artifacts/stage_a/A3_staging_jobs"
# BGE-M3 Dense 向量维度；与 runtime.EMBEDDING_DIMENSION 同值（此处不导入 runtime 以免加载模型依赖）。
EMBEDDING_DIMENSION = 1024
SMOKE_QUERY = "PoinTr point cloud representation for Transformer"
# 构建阶段只允许单向前进；ingestion 逐文件循环时同一阶段会重复出现，只在首次出现时上报。
_STAGE_ORDER: tuple[BuildJobStatus, ...] = ("PARSING", "CHUNKING", "EMBEDDING", "INDEXING")


class StagingBuildRefused(RuntimeError):
    """开始构建之前的拒绝（退出码 2）：不会创建 collection，也不会改动状态文件。"""


class ProtectedCollectionError(RuntimeError):
    """试图删除 active 或 previous 的 collection 时抛出；这两个是线上与回滚依赖的索引。

    能前置的场景（生成的 staging 名与它们重名）在构建前以 ``StagingBuildRefused`` 拒绝；
    构建之后仍抛出本异常说明状态文件在构建期间被改动，退出码 3。
    """


class StageReporter:
    """把 ingestion 过程中“首次进入某阶段”的事实单向上报给 Coordinator。"""

    def __init__(self, report: Callable[[BuildJobStatus], None]) -> None:
        self._report = report
        self._reached = -1

    def reach(self, stage: BuildJobStatus) -> None:
        """推进到 stage；逐级补报被跳过的中间阶段，重复或回退的事件直接忽略。"""

        target = _STAGE_ORDER.index(stage)
        while self._reached < target:
            self._reached += 1
            self._report(_STAGE_ORDER[self._reached])


class Observed:
    """透明代理：指定方法首次被调用时触发回调，其余属性与方法原样转发。"""

    def __init__(self, inner: Any, hooks: dict[str, Callable[[], None]]) -> None:
        self._inner = inner
        self._hooks = hooks

    def __getattr__(self, name: str) -> Any:
        attribute = getattr(self._inner, name)
        hook = self._hooks.get(name)
        if hook is None or not callable(attribute):
            return attribute

        def wrapped(*args: Any, **kwargs: Any) -> Any:
            hook()
            return attribute(*args, **kwargs)

        return wrapped


@dataclass
class StagingPorts:
    """脚本与外部世界（磁盘、Chroma、模型）的全部接触点；测试时整体替换为 Fake。"""

    scan_sources: Callable[[], list[SourceRecord]]
    list_collections: Callable[[], set[str]]
    active_chunk_ids: Callable[[str], list[str]]
    build: Callable[[str, Callable[[BuildJobStatus], None]], CollectionReader]
    smoke_embedding: Callable[[], list[float]]
    delete_collection: Callable[[str], None]
    mark_collection_active: Callable[[str], None]


def _protected_names(state: LifecycleState) -> set[str]:
    """active 与 previous 是线上与回滚依赖的 collection，任何删除或复用都必须避开。"""

    names = {state.active.collection_name}
    if state.previous is not None:
        names.add(state.previous.collection_name)
    return names


def discard_staging_collection(
    lifecycle: IndexLifecycle,
    delete_collection: Callable[[str], None],
    list_collections: Callable[[], set[str]],
) -> LifecycleState:
    """删除已登记的 staging collection，然后清空 staging 指针；可重复执行（幂等）。

    删除前先确认目标不是 active 或 previous：这两个 collection 一旦被删，线上或回滚就会失效。
    顺序是先删 collection、后清指针；删除失败时指针保持不变，状态仍与实际一致。
    若上次执行已删掉 collection 但清指针失败，再次执行时 collection 已不存在，只清指针并成功返回，
    否则这种半完成状态会让后续构建永远因“已有 staging 指针”被拒绝。
    """

    state = lifecycle.load_state()
    if state.staging is None:
        raise StagingBuildRefused("no staging pointer is registered; nothing to discard")
    name = state.staging.collection_name
    if name in _protected_names(state):
        raise ProtectedCollectionError(
            f"refusing to delete collection {name!r}: it is the active or previous index"
        )
    # collection 已不存在（上次删除成功、清指针失败）时跳过删除，只补做清指针。
    if name in list_collections():
        delete_collection(name)
    return lifecycle.discard_staging()


def mark_collection_lifecycle(client: Any, collection_name: str, lifecycle: str) -> None:
    """通过 Chroma API 改写 collection metadata 中的 lifecycle 标记，其余业务 metadata 原样保留。

    Chroma 的 modify 以整体替换方式写 metadata，且拒绝**包含** ``hnsw:space`` 键的 metadata。
    合并时去掉该旧键即可让带它创建的历史 collection（如 v2）也能被改写：距离度量由 collection
    的 configuration 决定，modify metadata 不会改变它（Chroma 1.5.9 实测，见对应测试）。
    改写后 metadata 中不再有 ``hnsw:space``；源码中没有任何地方读取该键。
    """

    collection = client.get_collection(name=collection_name)
    merged = {key: value for key, value in (collection.metadata or {}).items() if key != "hnsw:space"}
    collection.modify(metadata={**merged, "lifecycle": lifecycle})


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """与项目 write_json 一致：UTF-8、不转义中文、键排序。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _repo_relative(path: Path) -> str:
    """状态文件里的 manifest_path 使用仓库相对路径，仓库外的路径保持原样。"""

    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def run_staging_build(
    settings: Settings,
    ports: StagingPorts,
    *,
    job_dir: Path,
    expected_chunk_ids_from_active: bool,
    activate: bool,
    discard_after_validate: bool,
) -> tuple[int, dict[str, Any]]:
    """执行一次 staging 全量重建，返回 (退出码, 摘要)；摘要同时写入 job_dir。"""

    if activate and discard_after_validate:
        raise StagingBuildRefused("--activate and --discard-after-validate are mutually exclusive")
    # 状态文件是唯一事实来源；读不出知识库身份时在此失败，而不是自行猜测。
    state = LifecycleState.model_validate_json(Path(settings.index_state_path).read_text(encoding="utf-8"))
    lifecycle = IndexLifecycle(settings.index_state_path, state.knowledge_base_id)
    if state.staging is not None:
        raise StagingBuildRefused(
            f"a staging pointer is already registered ({state.staging.collection_name!r}); "
            "inspect it and discard it before starting another build"
        )

    sources = ports.scan_sources()
    pipeline: PipelineIdentity = current_pipeline_from_settings(settings)
    index_signature = build_index_signature(build_corpus_signature(sources), pipeline)
    protected = _protected_names(state)
    # active/previous 作为 reserved 传入，生成器会避让它们（例如 rollback 后同 signature 重建时，
    # 首选名称恰好是 previous）。
    staging_name = build_staging_collection_name(
        state.active.collection_name, index_signature, reserved=protected
    )
    # 纵深防御：无论生成规则如何，staging 名绝不能等于线上或回滚依赖的 collection，
    # 否则构建会写进 ACTIVE/PREVIOUS 索引，discard 还会删除它。
    if staging_name in protected:
        raise StagingBuildRefused(
            f"generated staging name {staging_name!r} equals the active or previous collection; "
            "refusing to build. Do not delete or modify that collection: it is a live or rollback "
            "index. This indicates a staging naming bug"
        )
    if staging_name in ports.list_collections():
        raise StagingBuildRefused(
            f"collection {staging_name!r} already exists and is neither active nor previous "
            "(likely left over from an earlier build); inspect it before deciding what to do. "
            "This script never reuses or deletes an existing collection"
        )
    expected_ids = (
        ports.active_chunk_ids(state.active.collection_name) if expected_chunk_ids_from_active else None
    )
    # 本脚本只做“同一 corpus 的全量重建”，预期规模取 active 的数量；规模变化会被校验如实报告失败。
    manifest = build_manifest(
        knowledge_base_id=state.knowledge_base_id,
        sources=sources,
        pipeline=pipeline,
        collection_name=staging_name,
        document_count=len(sources),
        chunk_count=state.active.chunk_count,
        status="STAGING",
        parent_index_signature=state.active.index_signature,
    )

    job_store = BuildJobStore(job_dir / "staging_build_job.json")
    job, report = StagingBuildCoordinator(lifecycle, job_store).build(
        manifest,
        ports.build,
        expected_chunk_ids=expected_ids,
        smoke_query_embedding=ports.smoke_embedding(),
        expected_embedding_dimension=EMBEDDING_DIMENSION,
    )
    summary: dict[str, Any] = {
        "staging_collection": staging_name,
        "index_signature": index_signature,
        "active_collection": state.active.collection_name,
        "expected_chunk_ids_from_active": expected_chunk_ids_from_active,
        "job": json.loads(job.model_dump_json()),
        "validation": report.model_dump(mode="json") if report is not None else None,
        "actions": [],
    }
    passed = job.status == "READY" and report is not None and report.passed
    if not passed:
        # 失败时保留 staging collection 与指针：它们是诊断证据，由用户决定如何处置。
        summary["outcome"] = "FAILED_KEPT_STAGING"
        _write_json(job_dir / "staging_build_summary.json", summary)
        return 1, summary

    if activate:
        manifest_path = job_dir / f"{staging_name}.manifest.json"
        save_manifest(manifest_path, manifest.with_index_state(status="ACTIVE"))
        # 先把 collection 的 lifecycle 标记改为 ACTIVE，再切换指针；否则新 active 会被 runtime 的校验拒绝。
        ports.mark_collection_active(staging_name)
        lifecycle.activate(manifest, report, manifest_path=_repo_relative(manifest_path))
        job_store.save(job.transition("ACTIVATED"))
        summary["actions"].append("activated")
        summary["outcome"] = "ACTIVATED"
    elif discard_after_validate:
        try:
            discard_staging_collection(lifecycle, ports.delete_collection, ports.list_collections)
        except ProtectedCollectionError as exc:
            # 构建前已确认 staging 名不是 active/previous；到这里说明状态文件在构建期间被改动。
            # 此时已经构建完成，不能再用“2 = 拒绝开始”，改用退出码 3，staging 与指针保留。
            summary["outcome"] = "VALIDATED_DISCARD_REFUSED"
            summary["error"] = str(exc)
            _write_json(job_dir / "staging_build_summary.json", summary)
            return 3, summary
        summary["actions"].append("staging_collection_deleted")
        summary["actions"].append("staging_pointer_cleared")
        summary["outcome"] = "VALIDATED_AND_DISCARDED"
    else:
        summary["outcome"] = "VALIDATED_STAGING_KEPT"
    _write_json(job_dir / "staging_build_summary.json", summary)
    return 0, summary


def make_real_ports(settings: Settings) -> StagingPorts:
    """用真实的磁盘、Chroma 与本地模型装配端口；模型与 Chroma 依赖延迟到调用时才加载。"""

    import chromadb

    from evidenceflow.embedding import BgeM3EmbeddingProvider
    from evidenceflow.ingestion import BgeM3Tokenizer, IngestionService
    from evidenceflow.ingestion.loader import load_document
    from evidenceflow.ingestion.service import discover_source_files
    from evidenceflow.storage import ChromaVectorStore

    root = Path(settings.source_docs_path)
    client = chromadb.PersistentClient(path=str(settings.vector_db_path))
    provider = BgeM3EmbeddingProvider(settings.embedding_model, device=settings.model_device)

    def scan_sources() -> list[SourceRecord]:
        documents = [load_document(path, root) for path in discover_source_files(root, root)]
        return manifest_sources_from_documents(documents)

    def active_chunk_ids(collection_name: str) -> list[str]:
        # 只读打开 active：ChromaVectorStore 对已存在的 collection 不会改动其 metadata。
        return [chunk.chunk_id for chunk in ChromaVectorStore(settings.vector_db_path, collection_name).iter_all_chunks()]

    def build(collection_name: str, report_stage: Callable[[BuildJobStatus], None]) -> ChromaVectorStore:
        reporter = StageReporter(report_stage)
        reporter.reach("PARSING")
        store = ChromaVectorStore(
            settings.vector_db_path,
            collection_name,
            # 不写旧的 "hnsw:space" 键：源码中没有任何地方读取它，cosine 由 ChromaVectorStore 的
            # configuration 保证。（含该键的 collection 也能改写 lifecycle，见 mark_collection_lifecycle。）
            collection_metadata={
                "chunk_strategy": settings.chunk_strategy,
                "storage_contract": "domain_chunk_metadata_v2",
                "lifecycle": "STAGING",
            },
        )
        service = IngestionService(
            root,
            Observed(BgeM3Tokenizer(settings.embedding_model), {"encode": lambda: reporter.reach("CHUNKING")}),
            Observed(provider, {"embed_documents": lambda: reporter.reach("EMBEDDING")}),
            Observed(store, {"add_chunks": lambda: reporter.reach("INDEXING")}),
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            chunk_strategy=settings.chunk_strategy,
        )
        summary = service.ingest()
        if summary.files_discovered == 0 or summary.failed:
            failures = [f"{item.source_path}: {item.failed_stage}: {item.error_message}" for item in summary.results if item.status == "failed"]
            raise RuntimeError(f"staging ingestion incomplete (files={summary.files_discovered}): {failures}")
        reporter.reach("INDEXING")
        return store

    return StagingPorts(
        scan_sources=scan_sources,
        list_collections=lambda: {item.name for item in client.list_collections()},
        active_chunk_ids=active_chunk_ids,
        build=build,
        smoke_embedding=lambda: provider.embed_query(SMOKE_QUERY),
        delete_collection=lambda name: client.delete_collection(name),
        mark_collection_active=lambda name: mark_collection_lifecycle(client, name, "ACTIVE"),
    )


def run_discard_only(settings: Settings) -> int:
    """--discard-staging：只连接 Chroma 删除已登记的 staging 并清指针，不加载任何模型。"""

    import chromadb

    try:
        state = LifecycleState.model_validate_json(Path(settings.index_state_path).read_text(encoding="utf-8"))
        lifecycle = IndexLifecycle(settings.index_state_path, state.knowledge_base_id)
        client = chromadb.PersistentClient(path=str(settings.vector_db_path))
        discard_staging_collection(
            lifecycle,
            lambda name: client.delete_collection(name),
            lambda: {item.name for item in client.list_collections()},
        )
    except (StagingBuildRefused, ProtectedCollectionError) as exc:
        # 尚未删除任何东西、也没有改动状态文件，属于“拒绝开始”。
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"outcome": "STAGING_DISCARDED"}, ensure_ascii=False))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--expected-chunk-ids-from-active",
        action="store_true",
        help="要求 staging 的 chunk ID 集合与当前 active 完全一致（同一 corpus 全量重建的确定性验证）",
    )
    parser.add_argument("--activate", action="store_true", help="校验通过后激活（默认不激活）")
    parser.add_argument(
        "--discard-after-validate",
        action="store_true",
        help="校验通过后通过 Chroma API 删除 staging collection 并清空 staging 指针；校验失败时保留",
    )
    parser.add_argument(
        "--discard-staging",
        action="store_true",
        help="不构建，只删除已登记的 staging collection 并清空指针（幂等，用于恢复中途失败的 discard）",
    )
    parser.add_argument("--job-dir", type=Path, default=DEFAULT_JOB_DIR, help="BuildJob 与摘要输出目录")
    args = parser.parse_args(argv)

    settings = Settings()
    if args.discard_staging:
        if args.activate or args.discard_after_validate or args.expected_chunk_ids_from_active:
            print("REFUSED: --discard-staging cannot be combined with build options", file=sys.stderr)
            return 2
        return run_discard_only(settings)
    try:
        code, summary = run_staging_build(
            settings,
            make_real_ports(settings),
            job_dir=args.job_dir,
            expected_chunk_ids_from_active=args.expected_chunk_ids_from_active,
            activate=args.activate,
            discard_after_validate=args.discard_after_validate,
        )
    except StagingBuildRefused as exc:
        # 只有构建前的拒绝会走到这里；构建后的保护拒绝由 run_staging_build 以退出码 3 返回。
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    validation = summary["validation"] or {}
    print(
        json.dumps(
            {
                "outcome": summary["outcome"],
                "staging_collection": summary["staging_collection"],
                "job_status": summary["job"]["status"],
                "validation_passed": validation.get("passed"),
                "actual_chunk_count": validation.get("actual_chunk_count"),
                "missing_chunk_ids": len(validation.get("missing_chunk_ids", [])),
                "unexpected_chunk_ids": len(validation.get("unexpected_chunk_ids", [])),
                "errors": validation.get("errors", []),
                "job_error": summary["job"].get("error_message"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return code


if __name__ == "__main__":
    raise SystemExit(main())
