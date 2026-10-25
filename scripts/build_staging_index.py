"""A3: fully rebuild the index into a staging collection, then optionally activate or discard it after
validation; the ACTIVE index itself is never modified directly.

It reuses ``StagingBuildCoordinator`` (stage progression + validation), ``IngestionService`` (as the
builder callback) and ``validate_collection`` instead of duplicating ingestion logic. By default it
only builds and validates, never activates:

    python scripts/build_staging_index.py --expected-chunk-ids-from-active --discard-after-validate

If a previous discard failed halfway (for example the collection was already deleted while the
pointer was not cleared yet), the registered staging can be cleaned up on its own:

    python scripts/build_staging_index.py --discard-staging

Exit codes: 0 = validation passed and the chosen action completed (or --discard-staging completed);
1 = the build or the validation failed (staging and the pointer are kept as-is for diagnosis);
2 = refused to start: nothing was built, deleted or written to the state file (a staging pointer
already exists, the generated staging name collides with active/previous or with an existing
collection, the options conflict, or --discard-staging was requested with no registered staging or
with a target that is active/previous); 3 = validation passed but the wrap-up action after the build
was refused by the guard (staging collection and pointer are kept; this can only happen when the
state file was changed externally during the build).

--discard-staging exit codes (A3.2): 0 = the collection was deleted and the pointer cleared;
2 = refused, nothing was changed (the state file is missing, unreadable or fails validation, with
stderr ``REFUSED: cannot read lifecycle state <path>: <exception type>: <message>``; no registered
staging; the target is active/previous); 1 = an error occurred while connecting to Chroma, listing
or deleting the collection (stderr ``FAILED: <exception type>: <message>``; when the deletion fails
the staging pointer is left unchanged so the operation can be retried after a fix). Both error kinds
print a single line and never raise a traceback.
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
# BGE-M3 Dense vector dimension; the same value as runtime.EMBEDDING_DIMENSION (runtime is not
# imported here, to avoid pulling in the model dependencies).
EMBEDDING_DIMENSION = 1024
SMOKE_QUERY = "PoinTr point cloud representation for Transformer"
# Build stages may only move forward; ingestion loops per file so the same stage recurs, and it is
# reported only on first entry.
_STAGE_ORDER: tuple[BuildJobStatus, ...] = ("PARSING", "CHUNKING", "EMBEDDING", "INDEXING")


class StagingBuildRefused(RuntimeError):
    """Refusal before a build starts (exit code 2): no collection is created and the state file is untouched."""


class ProtectedCollectionError(RuntimeError):
    """Raised when something tries to delete the active or previous collection; those two are the
    indexes that serving and rollback depend on.

    Scenarios that can be detected up front (the generated staging name collides with them) are
    refused before the build via ``StagingBuildRefused``; this exception still being raised after the
    build means the state file changed during the build, and maps to exit code 3.
    """


class StageReporter:
    """Report the "first entry into a stage" facts from ingestion one-way to the Coordinator."""

    def __init__(self, report: Callable[[BuildJobStatus], None]) -> None:
        self._report = report
        self._reached = -1

    def reach(self, stage: BuildJobStatus) -> None:
        """Advance to stage; skipped intermediate stages are reported one by one, and repeated or backward events are ignored."""

        target = _STAGE_ORDER.index(stage)
        while self._reached < target:
            self._reached += 1
            self._report(_STAGE_ORDER[self._reached])


class Observed:
    """Transparent proxy: the configured hooks fire when the named method is first called, and every other attribute or method is forwarded unchanged."""

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
    """Every contact point between the script and the outside world (disk, Chroma, models); tests replace the whole set with Fakes."""

    scan_sources: Callable[[], list[SourceRecord]]
    list_collections: Callable[[], set[str]]
    active_chunk_ids: Callable[[str], list[str]]
    build: Callable[[str, Callable[[BuildJobStatus], None]], CollectionReader]
    smoke_embedding: Callable[[], list[float]]
    delete_collection: Callable[[str], None]
    mark_collection_active: Callable[[str], None]


def _protected_names(state: LifecycleState) -> set[str]:
    """Active and previous are the collections serving and rollback depend on, so any deletion or reuse must avoid them."""

    names = {state.active.collection_name}
    if state.previous is not None:
        names.add(state.previous.collection_name)
    return names


def discard_staging_collection(
    lifecycle: IndexLifecycle,
    delete_collection: Callable[[str], None],
    list_collections: Callable[[], set[str]],
) -> LifecycleState:
    """Delete the registered staging collection, then clear the staging pointer; safe to repeat (idempotent).

    The target is confirmed not to be active or previous first: deleting either of those would break
    serving or rollback. The order is collection first, pointer second, so a failed deletion leaves the
    pointer unchanged and the state still matches reality.
    If a previous run already deleted the collection but failed to clear the pointer, the collection no
    longer exists on the next run, so only the pointer is cleared and the call succeeds; otherwise that
    half-finished state would make every later build fail with "a staging pointer is already registered".
    """

    state = lifecycle.load_state()
    if state.staging is None:
        raise StagingBuildRefused("no staging pointer is registered; nothing to discard")
    name = state.staging.collection_name
    if name in _protected_names(state):
        raise ProtectedCollectionError(
            f"refusing to delete collection {name!r}: it is the active or previous index"
        )
    # When the collection no longer exists (last deletion succeeded, clearing the pointer failed),
    # skip
    # the deletion and only finish the pointer cleanup.
    if name in list_collections():
        delete_collection(name)
    return lifecycle.discard_staging()


def mark_collection_lifecycle(client: Any, collection_name: str, lifecycle: str) -> None:
    """Rewrite the lifecycle flag in the collection metadata through the Chroma API, keeping every other
    business metadata field unchanged.

    Chroma's modify replaces the metadata wholesale and rejects metadata that **contains** the
    ``hnsw:space`` key. Dropping that legacy key while merging lets historical collections created with
    it (such as v2) be rewritten too: the distance metric is decided by the collection configuration and
    modify metadata does not change it (measured on Chroma 1.5.9, see the corresponding test).
    After the rewrite the metadata no longer carries ``hnsw:space``; nothing in the source reads that
    key.
    """

    collection = client.get_collection(name=collection_name)
    merged = {key: value for key, value in (collection.metadata or {}).items() if key != "hnsw:space"}
    collection.modify(metadata={**merged, "lifecycle": lifecycle})


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Match the project's write_json: UTF-8, non-ASCII characters left unescaped, keys sorted."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _repo_relative(path: Path) -> str:
    """Use a repo-relative path for manifest_path in the state file; paths outside the repo stay as they are."""

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
    """Run one full staging rebuild and return (exit code, summary); the summary is also written to job_dir."""

    if activate and discard_after_validate:
        raise StagingBuildRefused("--activate and --discard-after-validate are mutually exclusive")
# The state file is the single source of truth; fail here when the knowledge-base identity cannot
# be
# read instead of guessing it.
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
    # active/previous are passed in as reserved so the generator steers around them (for example
    # when
    # rebuilding with the same signature after a rollback, the preferred name is exactly previous).
    staging_name = build_staging_collection_name(
        state.active.collection_name, index_signature, reserved=protected
    )
    # Defense in depth: whatever the naming rule, the staging name must never equal the collection
    # that
    # serving or rollback depends on, otherwise the build would write into the ACTIVE/PREVIOUS
    # index and
    # discard would even delete it.
    # A3.2 note: the current generator already avoids active/previous through reserved, so this
    # check
    # cannot fire with the current implementation; it is kept on purpose as a backstop against
    # future
    # generator regressions (covered in tests by monkeypatching the generator).
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
    # This script only performs a full rebuild of the same corpus, so the expected magnitude is
    # taken
    # from active; any change in size is faithfully reported as a validation failure.
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
        # On failure keep the staging collection and the pointer: they are diagnostic evidence and
        # the
        # user decides how to handle them.
        summary["outcome"] = "FAILED_KEPT_STAGING"
        _write_json(job_dir / "staging_build_summary.json", summary)
        return 1, summary

    if activate:
        manifest_path = job_dir / f"{staging_name}.manifest.json"
        save_manifest(manifest_path, manifest.with_index_state(status="ACTIVE"))
        # Mark the collection lifecycle ACTIVE before switching the pointer; otherwise runtime
        # validation
        # rejects the new active index.
        ports.mark_collection_active(staging_name)
        lifecycle.activate(manifest, report, manifest_path=_repo_relative(manifest_path))
        job_store.save(job.transition("ACTIVATED"))
        summary["actions"].append("activated")
        summary["outcome"] = "ACTIVATED"
    elif discard_after_validate:
        try:
            discard_staging_collection(lifecycle, ports.delete_collection, ports.list_collections)
        except ProtectedCollectionError as exc:
            # The staging name was confirmed not to be active/previous before the build; reaching
            # this
            # point means the state file changed during the build. The build has already happened,
            # so
            # "2 = refused to start" no longer applies: use exit code 3 and keep staging and the
            # pointer.
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
    """Assemble the ports from the real disk, Chroma and local models; the model and Chroma dependencies are imported lazily when called."""

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
        # Open active read-only: ChromaVectorStore does not modify the metadata of an existing
        # collection.
        return [chunk.chunk_id for chunk in ChromaVectorStore(settings.vector_db_path, collection_name).iter_all_chunks()]

    def build(collection_name: str, report_stage: Callable[[BuildJobStatus], None]) -> ChromaVectorStore:
        reporter = StageReporter(report_stage)
        reporter.reach("PARSING")
        store = ChromaVectorStore(
            settings.vector_db_path,
            collection_name,
            # Do not write the legacy "hnsw:space" key: nothing in the source reads it and cosine
            # is
            # guaranteed by ChromaVectorStore's configuration. (Collections carrying that key can
            # still be
            # rewritten for lifecycle, see mark_collection_lifecycle.)
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


def _one_line(exc: BaseException) -> str:
    """Flatten an exception message into one line: Pydantic-style messages span several lines while the CLI contract is one line per error."""

    return " ".join(str(exc).split())


def run_discard_only(settings: Settings) -> int:
    """--discard-staging: only connect to Chroma to delete the registered staging and clear the pointer, loading no model at all.

    Every error collapses into a single stderr line plus an exit code, never a traceback:
    - the state file is missing, unreadable or fails validation -> ``REFUSED: cannot read lifecycle state ...``, exit code 2 (nothing changed);
    - no registered staging, or the target is protected -> ``REFUSED: ...``, exit code 2 (nothing changed);
    - an error while connecting to Chroma, listing or deleting the collection (or clearing the pointer afterwards) -> ``FAILED: ...``, exit code 1.
      The pointer stays unchanged when the deletion fails: ``discard_staging_collection`` deletes the collection first and clears the pointer second.
    """

    state_path = Path(settings.index_state_path)
    # Step one only reads the state file: Chroma is not connected yet, so any failure here means
    # nothing
    # changed and counts as "refused to start".
    try:
        state = LifecycleState.model_validate_json(state_path.read_text(encoding="utf-8"))
        lifecycle = IndexLifecycle(settings.index_state_path, state.knowledge_base_id)
    except Exception as exc:  # noqa: BLE001 - a missing file / permissions / encoding / JSON / Pydantic validation error all count as a refusal
        print(
            f"REFUSED: cannot read lifecycle state {state_path}: {type(exc).__name__}: {_one_line(exc)}",
            file=sys.stderr,
        )
        return 2
    try:
        import chromadb

        client = chromadb.PersistentClient(path=str(settings.vector_db_path))
        discard_staging_collection(
            lifecycle,
            lambda name: client.delete_collection(name),
            lambda: {item.name for item in client.list_collections()},
        )
    except (StagingBuildRefused, ProtectedCollectionError) as exc:
        # Nothing has been deleted and the state file is untouched, so this counts as "refused to
        # start".
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001 - Chroma connect/list/delete failures are all reported as an execution failure
        # When the deletion fails, discard_staging_collection has not cleared the pointer yet, so
        # the state
        # still matches reality and the operation can be retried after a fix.
        print(f"FAILED: {type(exc).__name__}: {_one_line(exc)}", file=sys.stderr)
        return 1
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
        # Only pre-build refusals reach this point; post-build guard refusals are returned by
        # run_staging_build through exit code 3.
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
