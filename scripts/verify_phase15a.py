"""只读执行 Phase 15A Gate，统一使用 frozen baseline 而不是运行时猜测历史状态。"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import chromadb

from evidenceflow.config import Settings
from evidenceflow.ingestion.loader import load_document
from evidenceflow.ingestion.service import discover_source_files
from evidenceflow.knowledge import (
    build_manifest,
    current_pipeline_from_settings,
    load_manifest,
    manifest_sources_from_documents,
)
from evidenceflow.knowledge.lifecycle import IndexLifecycle
from evidenceflow.storage import ChromaVectorStore

BASELINE_PATH = ROOT / "artifacts/phase15a/baseline_snapshot.json"
MANIFEST_PATH = ROOT / "artifacts/phase15a/active_manifest.json"
STATE_PATH = ROOT / "artifacts/phase15a/index_state.json"
PHASE13F_DIR = ROOT / "evals/evaluation_v3/phase13f_hybrid_runtime_promotion"
REPAIR_REPORT_PATH = ROOT / "artifacts/cleanup/storage_repair/storage_repair_report.json"


def sha256(path: Path) -> str:
    """以分块方式计算文件 hash，避免一次性加载大型评测产物。"""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _env_collection_name() -> str | None:
    """读取 .env 中的非敏感 Collection 配置，不触碰 API Key。"""

    for raw in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line.startswith("COLLECTION_NAME="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


def _hash_checks(paths: dict[str, str], base: Path) -> dict[str, bool]:
    """复算一组冻结文件 hash，并返回逐文件结果。"""

    return {relative: sha256(base / relative) == expected for relative, expected in paths.items()}


def verify_phase15a() -> dict[str, Any]:
    """读取当前 v2 和 frozen snapshot，返回机器可读的 Phase 15A Gate 报告。"""

    baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    settings = Settings()
    env_collection = _env_collection_name()
    client = chromadb.PersistentClient(path=str(settings.vector_db_path))
    collections = {item.name: item for item in client.list_collections()}
    active_names = sorted(
        item.name
        for item in collections.values()
        if (item.metadata or {}).get("lifecycle") == "ACTIVE"
    )
    collection = collections.get(settings.collection_name)
    store = ChromaVectorStore(settings.vector_db_path, settings.collection_name)
    chunks = list(store.iter_all_chunks())
    actual_chunk_ids = sorted(chunk.chunk_id for chunk in chunks)
    actual_document_ids = sorted({chunk.document_id for chunk in chunks})
    source_documents = [
        load_document(path, settings.source_docs_path)
        for path in discover_source_files(settings.source_docs_path, settings.source_docs_path)
    ]
    source_records = manifest_sources_from_documents(source_documents)
    expected_manifest = build_manifest(
        knowledge_base_id=baseline["authoritative_identity"]["knowledge_base_id"],
        sources=source_records,
        pipeline=current_pipeline_from_settings(settings),
        collection_name=settings.collection_name,
        document_count=len(actual_document_ids),
        chunk_count=len(chunks),
        status="ACTIVE",
    )
    active_manifest = load_manifest(MANIFEST_PATH)
    state = IndexLifecycle(STATE_PATH, active_manifest.knowledge_base_id).load_state()
    repair = json.loads(REPAIR_REPORT_PATH.read_text(encoding="utf-8"))

    phase13f_hashes = baseline["frozen_artifacts"]["phase13f_manifest_files_sha256"]
    protected_hashes = baseline["frozen_artifacts"]["gold_and_qrels_files_sha256"]
    phase13f_checks = _hash_checks(phase13f_hashes, PHASE13F_DIR)
    protected_checks = _hash_checks(protected_hashes, ROOT)
    source_hashes = {document.source_path: document.content_hash for document in source_documents}
    expected_source_hashes = baseline["source_corpus"]["source_content_hashes"]
    required_metadata = all(
        "chunk_strategy" in chunk.metadata and chunk.metadata.get("element_ids")
        for chunk in chunks
    )
    gates = {
        "G15A-1_manifest_determinism": (
            expected_manifest.canonical_bytes() == active_manifest.canonical_bytes()
            and state.active.index_signature == active_manifest.index.index_signature
            and state.active.collection_name == active_manifest.index.collection_name
        ),
        "G15A-2_signature_sensitivity": "covered_by_targeted_tests",
        "G15A-3_signature_isolation": "covered_by_targeted_tests",
        "G15A-4_source_reconciliation": (
            len(source_documents) == baseline["source_corpus"]["document_count"] == 10
            and source_hashes == expected_source_hashes
            and sorted(document.document_id for document in source_documents)
            == baseline["source_corpus"]["sorted_document_ids"]
        ),
        "G15A-5_index_reconciliation": (
            len(chunks) == baseline["index"]["chunk_count"] == 817
            and actual_document_ids == baseline["source_corpus"]["sorted_document_ids"]
            and actual_chunk_ids == baseline["index"]["sorted_chunk_ids"]
            and store.embedding_dimension == baseline["index"]["embedding_shape"]["frozen"][1]
        ),
        "G15A-6_no_orphan_state": (
            len(actual_chunk_ids) == len(set(actual_chunk_ids))
            and actual_document_ids == baseline["source_corpus"]["sorted_document_ids"]
            and required_metadata
        ),
        "G15A-7_staging_isolation": "covered_by_targeted_tests",
        "G15A-8_failed_build_safety": "covered_by_targeted_tests",
        "G15A-9_activation": "covered_by_targeted_tests",
        "G15A-10_rollback": "covered_by_targeted_tests",
        "G15A-11_runtime_parity": (
            repair.get("promoted_collection") == settings.collection_name
            and repair.get("retrieval_parity", {}).get("pass") is True
            and repair.get("ir_parity", {}).get("pass") is True
            and repair.get("ir_parity", {}).get("actual") == baseline["phase13f_metric_snapshot"]
            and repair.get("ir_parity", {}).get("differences") == {}
        ),
        "G15A-12_frozen_artifacts_intact": all(phase13f_checks.values()) and all(protected_checks.values()),
    }
    test_covered_gates = {name for name, value in gates.items() if value == "covered_by_targeted_tests"}
    executable_gates = {name: value for name, value in gates.items() if name not in test_covered_gates}
    report = {
        "phase": "15A",
        "status": "PASS" if all(executable_gates.values()) else "FAIL",
        "baseline_snapshot": str(BASELINE_PATH.relative_to(ROOT)).replace("\\", "/"),
        "authoritative_collection": settings.collection_name,
        "config": {
            "settings_collection_name": settings.collection_name,
            "env_collection_name": env_collection,
            "active_collection_names": active_names,
            "collection_metadata": collection.metadata if collection else None,
        },
        "manifest": {
            "path": str(MANIFEST_PATH.relative_to(ROOT)).replace("\\", "/"),
            "corpus_signature": active_manifest.corpus_signature,
            "index_signature": active_manifest.index.index_signature,
            "document_count": active_manifest.index.document_count,
            "chunk_count": active_manifest.index.chunk_count,
            "matches_current_scan": expected_manifest.canonical_bytes() == active_manifest.canonical_bytes(),
        },
        "lifecycle": {
            "state_path": str(STATE_PATH.relative_to(ROOT)).replace("\\", "/"),
            "active_collection": state.active.collection_name,
            "previous_collection": state.previous.collection_name if state.previous else None,
            "staging_collection": state.staging.collection_name if state.staging else None,
        },
        "counts": {
            "source_documents": len(source_documents),
            "indexed_documents": len(actual_document_ids),
            "indexed_chunks": len(chunks),
            "unique_chunk_ids": len(set(actual_chunk_ids)),
        },
        "phase13f_metric_snapshot": baseline["phase13f_metric_snapshot"],
        "gates": gates,
        "phase13f_artifact_hashes_pass": all(phase13f_checks.values()),
        "gold_qrels_hashes_pass": all(protected_checks.values()),
        "phase13f_artifact_hash_checks": phase13f_checks,
        "gold_qrels_hash_checks": protected_checks,
        "known_limitations": [
            "v1 hybrid Collection remains legacy/read-only and was not modified or deleted.",
            "This verification does not rebuild production or rerun model inference; runtime parity uses the frozen v2 storage-repair evidence.",
            "Phase 15B/15C/16/17 were not entered.",
        ],
    }
    return report


def main() -> int:
    """运行只读 Gate 并将报告写入 Phase 15A artifact 目录。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts/phase15a/phase15a_report.json",
        help="报告输出路径",
    )
    args = parser.parse_args()
    report = verify_phase15a()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
