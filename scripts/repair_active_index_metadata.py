# -*- coding: utf-8 -*-
# Historic one-off script (Stage A A4 annotation): it can no longer be run.
# The staging collection it relies on, ef_research_bge_m3_hybrid_c512_v2_staging, was deleted
# during Stage A A4 via
# scripts/prune_chroma_collections.py; for current index changes use
# scripts/build_staging_index.py.
# In addition, experiments/phase13c/metrics13cc.py, which compute_ir_parity imports dynamically
# inside the function, was also removed with A4.3.
# This file is kept for historical record only; the dependencies above can be restored from tag
# stage-a-start when comparing.
"""Repair the metadata contract of the EvidenceFlow active Chroma collection (staging-first).

The old active Collection is kept read-only; this script rebuilds an independent staging Collection from the
Chunk JSONL and embedding NPZ already frozen by Phase 13F, and runs the count, chunk identity, metadata,
Domain reconstruction, Dense Top20, and standard IR parity gates item by item. Only when ``--promote`` is passed
explicitly and every gate passes does it copy into the new active Collection and update the collection name in the local ``.env``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import chromadb
import numpy as np

from evidenceflow.domain.chunk import DocumentChunk
from evidenceflow.storage import ChromaVectorStore, serialize_chunk_metadata

ACTIVE_COLLECTION = "ef_research_bge_m3_hybrid_c512_v1"
STAGING_COLLECTION = "ef_research_bge_m3_hybrid_c512_v2_staging"
PROMOTED_COLLECTION = "ef_research_bge_m3_hybrid_c512_v2"
EXPECTED_COUNT = 817
TOP_K = 20
MODEL_ID = "BAAI/bge-m3"
CHUNK_ARTIFACT = ROOT / "evals" / "evaluation_v3" / "phase13f_hybrid_runtime_promotion" / "chunks" / "runtime_hybrid.jsonl"
EMBEDDING_ARTIFACT = ROOT / "evals" / "evaluation_v3" / "phase13f_hybrid_runtime_promotion" / "embeddings" / "runtime_hybrid.npz"
RUN_ARTIFACT = ROOT / "evals" / "evaluation_v3" / "phase13f_hybrid_runtime_promotion" / "retrieval" / "runtime_hybrid_run.json"
STATUS_ARTIFACT = ROOT / "evals" / "evaluation_v3" / "phase13f_hybrid_runtime_promotion" / "status.json"
QUESTIONS_ARTIFACT = ROOT / "evals" / "research_retrieval" / "research_questions_v1.json"
OUTPUT_DIR = ROOT / "artifacts" / "cleanup" / "storage_repair"


def sha256_of(path: Path) -> str:
    """Stream the input artifact hash so the frozen provenance can be traced before and after the repair."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_reference() -> tuple[list[dict[str, Any]], np.ndarray, list[DocumentChunk]]:
    """Load and validate the 13F frozen records, the vectors, and the Domain Chunk mapping."""

    rows = [
        json.loads(line)
        for line in CHUNK_ARTIFACT.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    npz = np.load(EMBEDDING_ARTIFACT, allow_pickle=False)
    vector_ids = [str(value) for value in npz["chunk_ids"]]
    vectors = np.asarray(npz["embeddings"], dtype=np.float32)
    row_ids = [str(row["chunk_id"]) for row in rows]
    if len(rows) != EXPECTED_COUNT or len(vector_ids) != EXPECTED_COUNT:
        raise ValueError("13F frozen chunk/vector count is not 817")
    if row_ids != vector_ids or vectors.shape[0] != len(rows):
        raise ValueError("13F chunk IDs and embedding IDs are misaligned")
    if vectors.ndim != 2 or not np.isfinite(vectors).all():
        raise ValueError("13F embeddings are not a finite 2-D matrix")

    chunks = [
        DocumentChunk(
            chunk_id=row["chunk_id"],
            document_id=row["document_id"],
            content=row["text"],
            token_count=row["token_count"],
            chunk_index=row["chunk_index"],
            page_start=row.get("page_start"),
            page_end=row.get("page_end"),
            file_name=row["file_name"],
            content_hash=row["content_hash"],
            metadata=row.get("metadata") or {},
        )
        for row in rows
    ]
    if len({chunk.chunk_id for chunk in chunks}) != EXPECTED_COUNT:
        raise ValueError("13F frozen chunks contain duplicate IDs")
    return rows, vectors, chunks


def read_collection_records(collection: Any, *, include_embeddings: bool = False) -> list[dict[str, Any]]:
    """Read raw Chroma records page by page; legacy metadata=None values are preserved verbatim."""

    records: list[dict[str, Any]] = []
    offset = 0
    while True:
        include = ["documents", "metadatas"]
        if include_embeddings:
            include.append("embeddings")
        result = collection.get(limit=1024, offset=offset, include=include)
        ids = list(result.get("ids") or [])
        documents = list(result.get("documents") or [])
        metadatas = result.get("metadatas")
        metadatas = list(metadatas) if metadatas is not None else [None] * len(ids)
        embeddings = result.get("embeddings")
        embeddings = list(embeddings) if embeddings is not None else [None] * len(ids)
        if not ids:
            break
        if not (len(ids) == len(documents) == len(metadatas) == len(embeddings)):
            raise ValueError("Chroma returned misaligned fields")
        records.extend(
            {
                "chunk_id": chunk_id,
                "document": document,
                "metadata": metadata,
                "embedding": embedding,
            }
            for chunk_id, document, metadata, embedding in zip(
                ids, documents, metadatas, embeddings
            )
        )
        if len(ids) < 1024:
            break
        offset += 1024
    return records


def compare_chunk_identity(
    expected_rows: list[dict[str, Any]], actual_records: list[dict[str, Any]]
) -> dict[str, Any]:
    """Compare textual identity exhaustively by chunk_id, independent of the order Chroma returns records in."""

    expected = {row["chunk_id"]: row["text"] for row in expected_rows}
    actual = {record["chunk_id"]: record["document"] for record in actual_records}
    missing = sorted(set(expected) - set(actual))
    extra = sorted(set(actual) - set(expected))
    changed = sorted(
        chunk_id
        for chunk_id in set(expected) & set(actual)
        if expected[chunk_id] != actual[chunk_id]
    )
    return {
        "expected_count": len(expected),
        "actual_count": len(actual),
        "missing_ids": missing[:10],
        "extra_ids": extra[:10],
        "changed_documents": changed[:10],
        "pass": not missing and not extra and not changed,
    }


def check_metadata_completeness(
    expected_rows: list[dict[str, Any]], actual_records: list[dict[str, Any]]
) -> dict[str, Any]:
    """Check whether every new record carries the source information needed to restore a Domain Chunk."""

    rows = {row["chunk_id"]: row for row in expected_rows}
    failures: list[dict[str, Any]] = []
    for record in actual_records:
        chunk_id = record["chunk_id"]
        metadata = record["metadata"]
        row = rows.get(chunk_id)
        valid = isinstance(metadata, dict) and row is not None
        if valid:
            domain_json = metadata.get("domain_metadata_json")
            try:
                domain_metadata = json.loads(domain_json)
            except (TypeError, ValueError):
                domain_metadata = None
            valid = (
                metadata.get("chunk_id") == chunk_id
                and metadata.get("document_id") == row["document_id"]
                and metadata.get("file_name") == row["file_name"]
                and metadata.get("content_hash") == row["content_hash"]
                and metadata.get("chunk_index") == row["chunk_index"]
                and metadata.get("token_count") == row["token_count"]
                and metadata.get("page_start") == row.get("page_start")
                and metadata.get("page_end") == row.get("page_end")
                and isinstance(domain_metadata, dict)
                and domain_metadata.get("chunk_strategy") == "hybrid"
                and bool(domain_metadata.get("element_ids"))
            )
        if not valid:
            failures.append({"chunk_id": chunk_id, "reason": "metadata incomplete or inconsistent"})
    return {
        "checked": len(actual_records),
        "valid": len(actual_records) - len(failures),
        "coverage": (len(actual_records) - len(failures)) / len(actual_records)
        if actual_records
        else 0.0,
        "failures": failures[:10],
        "pass": bool(actual_records) and not failures and len(actual_records) == len(expected_rows),
    }


def load_query_vectors(questions: list[dict[str, Any]]) -> np.ndarray:
    """Generate query vectors for the frozen questions under the 13F BGE-M3 CLS + L2 protocol."""

    # Use only models already present locally, so the integrity verification never accidentally
    # triggers a network download.
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    from transformers import AutoModel, AutoTokenizer
    import torch

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True)
    model = AutoModel.from_pretrained(MODEL_ID, local_files_only=True)
    model.eval()
    vectors: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(questions), 16):
            texts = [q["english_query"] for q in questions[start : start + 16]]
            encoded = tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            cls = torch.nn.functional.normalize(model(**encoded).last_hidden_state[:, 0], dim=-1)
            vectors.append(cls.cpu().numpy().astype("float32"))
    return np.concatenate(vectors, axis=0)


def compare_retrieval(
    old_collection: Any,
    new_collection: Any,
    questions: list[dict[str, Any]],
    expected_run: dict[str, list[str]],
) -> dict[str, Any]:
    """Compare the Dense Top20 rankings of the old collection, the new collection, and the 13F frozen run."""

    query_vectors = load_query_vectors(questions)
    mismatches: list[dict[str, Any]] = []
    artifact_mismatches: list[str] = []
    for question, vector in zip(questions, query_vectors):
        qid = question["question_id"]
        kwargs = {"query_embeddings": [vector.tolist()], "n_results": TOP_K, "include": ["documents"]}
        old_ids = old_collection.query(**kwargs)["ids"][0]
        new_ids = new_collection.query(**kwargs)["ids"][0]
        if old_ids != new_ids:
            mismatches.append({"question_id": qid, "old": old_ids[:5], "new": new_ids[:5]})
        if expected_run.get(qid) != old_ids:
            artifact_mismatches.append(qid)
    return {
        "question_count": len(questions),
        "old_vs_new_mismatches": mismatches[:10],
        "old_vs_frozen_artifact_mismatches": artifact_mismatches,
        "pass": not mismatches and not artifact_mismatches,
    }


def compute_ir_parity(rows: list[dict[str, Any]], run: dict[str, list[str]]) -> dict[str, Any]:
    """Recompute with the same qrels projection and metric implementation as 13F, confirming the metadata repair changed no semantic result."""

    sys.path.insert(0, str(ROOT / "experiments" / "phase13c"))
    import metrics13cc

    questions = json.loads(QUESTIONS_ARTIFACT.read_text(encoding="utf-8"))["questions"]
    projected = metrics13cc.project_qrels_anchor({"runtime_hybrid": rows}, questions)
    actual = metrics13cc.compute_ir_metrics(projected["qrels"], run)["aggregate"]
    expected = json.loads(STATUS_ARTIFACT.read_text(encoding="utf-8"))["ir_aggregate"]
    metric_names = (
        "precision_at_5",
        "recall_at_5",
        "recall_at_10",
        "mrr_at_5",
        "map_at_5",
        "ndcg_at_5",
        "hit_rate_at_20",
    )
    differences = {
        name: {"actual": actual.get(name), "expected": expected.get(name)}
        for name in metric_names
        if not math.isclose(actual.get(name, float("nan")), expected.get(name, float("nan")), abs_tol=1e-12)
    }
    return {"actual": actual, "expected": expected, "differences": differences, "pass": not differences}


def create_collection(client: Any, name: str, *, lifecycle: str) -> Any:
    """Create a new Collection with an explicit identity; existing data of the same name is not deleted, to avoid accidental overwrite."""

    if name in {item.name for item in client.list_collections()}:
        raise RuntimeError(f"Refusing to overwrite existing Collection: {name}")
    return client.create_collection(
        name=name,
        metadata={
            "hnsw:space": "cosine",
            "chunk_strategy": "hybrid",
            "storage_contract": "domain_chunk_metadata_v2",
            "lifecycle": lifecycle,
        },
        configuration={"hnsw": {"space": "cosine"}},
        embedding_function=None,
    )


def update_env_collection(env_path: Path, collection_name: str) -> None:
    """Update only the collection name in the local configuration; no secret field is read or copied."""

    text = env_path.read_text(encoding="utf-8") if env_path.exists() else ""
    lines = text.splitlines()
    replaced = False
    for index, line in enumerate(lines):
        if line.startswith("COLLECTION_NAME="):
            lines[index] = f"COLLECTION_NAME={collection_name}"
            replaced = True
            break
    if not replaced:
        lines.append(f"COLLECTION_NAME={collection_name}")
    env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(*, promote: bool = False) -> dict[str, Any]:
    """Run the staging repair gates and, when requested, complete the promotion to the new active Collection."""

    started = datetime.now(timezone.utc).isoformat()
    rows, vectors, chunks = load_reference()
    client = chromadb.PersistentClient(path=str(ROOT / "data" / "chroma"))
    names = {item.name for item in client.list_collections()}
    if ACTIVE_COLLECTION not in names:
        raise RuntimeError(f"Active source Collection missing: {ACTIVE_COLLECTION}")
    old_collection = client.get_collection(name=ACTIVE_COLLECTION)
    old_records = read_collection_records(old_collection, include_embeddings=True)
    old_identity = compare_chunk_identity(rows, old_records)
    old_metadata_coverage = sum(isinstance(item["metadata"], dict) for item in old_records) / len(old_records)
    if old_collection.count() != EXPECTED_COUNT or not old_identity["pass"]:
        raise RuntimeError("Old active Collection failed count or identity precondition")

    # An existing staging collection is reused read-only and re-verified; it is never silently
    # wiped just to allow a rerun.
    staging = (
        client.get_collection(name=STAGING_COLLECTION)
        if STAGING_COLLECTION in names
        else create_collection(client, STAGING_COLLECTION, lifecycle="STAGING")
    )
    # Write through the project Adapter first, so the real production path and the offline repair
    # share one serializer.
    staging_store = ChromaVectorStore(ROOT / "data" / "chroma", STAGING_COLLECTION)
    if staging.count() == 0:
        staging_store.add_chunks(chunks, vectors.tolist())
    new_records = read_collection_records(staging, include_embeddings=True)
    count_pass = len(new_records) == EXPECTED_COUNT
    identity = compare_chunk_identity(rows, new_records)
    metadata = check_metadata_completeness(rows, new_records)
    try:
        restored = list(staging_store.iter_all_chunks())
        reconstruction = {"count": len(restored), "pass": len(restored) == EXPECTED_COUNT}
    except Exception as exc:
        reconstruction = {"count": 0, "error": f"{type(exc).__name__}: {exc}", "pass": False}

    questions = json.loads(QUESTIONS_ARTIFACT.read_text(encoding="utf-8"))["questions"]
    frozen_run = json.loads(RUN_ARTIFACT.read_text(encoding="utf-8"))
    retrieval = compare_retrieval(old_collection, staging, questions, frozen_run)
    ir = compute_ir_parity(rows, frozen_run)
    phase13f_status = json.loads(STATUS_ARTIFACT.read_text(encoding="utf-8"))
    pytest_baseline = phase13f_status["pytest_baseline"]
    tests = {
        "phase13f_recorded_pass": pytest_baseline.get("failed", 0) == 0
        and pytest_baseline.get("passed", 0) >= 612,
        "note": "root pytest 在 promotion 后单独执行；本 gate 只确认 13F 冻结记录为 PASS",
    }
    gates = {
        "SR-G1_count": count_pass,
        "SR-G2_chunk_identity": identity["pass"],
        "SR-G3_metadata_completeness": metadata["pass"],
        "SR-G4_domain_reconstruction": reconstruction["pass"],
        "SR-G5_retrieval_parity": retrieval["pass"],
        "SR-G6_ir_parity": ir["pass"],
        "SR-G7_tests_recorded": tests["phase13f_recorded_pass"],
        "SR-G8_frozen_input_hashes": sha256_of(CHUNK_ARTIFACT) == json.loads(
            (ROOT / "evals" / "evaluation_v3" / "phase13f_hybrid_runtime_promotion" / "manifest.json").read_text(encoding="utf-8")
        )["files"]["chunks/runtime_hybrid.jsonl"],
    }
    all_pass = all(gates.values())
    promoted = None
    if promote:
        if not all_pass:
            raise RuntimeError("Refusing promotion because one or more storage repair gates failed")
        promoted = create_collection(client, PROMOTED_COLLECTION, lifecycle="ACTIVE")
        promoted.add(
            ids=[record["chunk_id"] for record in new_records],
            embeddings=[list(record["embedding"]) for record in new_records],
            documents=[record["document"] for record in new_records],
            metadatas=[record["metadata"] for record in new_records],
        )
        update_env_collection(ROOT / ".env", PROMOTED_COLLECTION)

    report = {
        "artifact_type": "evidenceflow_storage_contract_repair",
        "generated_at": started,
        "source_collection": ACTIVE_COLLECTION,
        "staging_collection": STAGING_COLLECTION,
        "promoted_collection": PROMOTED_COLLECTION if promoted is not None else None,
        "source_collection_status": "LEGACY_RETRIEVAL_ONLY / BROKEN_METADATA",
        "old_source": {
            "count": len(old_records),
            "metadata_coverage": old_metadata_coverage,
            "read_only": True,
            "identity_pass": old_identity["pass"],
        },
        "staging": {
            "count": len(new_records),
            "identity": identity,
            "metadata": metadata,
            "reconstruction": reconstruction,
        },
        "retrieval_parity": retrieval,
        "ir_parity": ir,
        "tests": tests,
        "gates": gates,
        "pass": all_pass,
        "promotion_performed": promoted is not None,
        "stop_at": "PROJECT_AND_CODEX_HYGIENE_CLEANUP" if all_pass else "STORAGE_CONTRACT_REPAIR",
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "storage_repair_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    return report


def main() -> int:
    """CLI entry point; by default it only builds and verifies staging and never modifies the active configuration."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--promote", action="store_true", help="全部 gate 通过后晋升到 v2 active")
    args = parser.parse_args()
    try:
        report = run(promote=args.promote)
    except Exception as exc:
        print(f"STORAGE_REPAIR: FAIL ({type(exc).__name__}: {exc})")
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    return 0 if report["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
