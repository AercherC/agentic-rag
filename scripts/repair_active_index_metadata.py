# -*- coding: utf-8 -*-
# 历史一次性脚本（Stage A A4 标注）：不能再运行。
# 它依赖的 staging collection ef_research_bge_m3_hybrid_c512_v2_staging 已在 Stage A A4 通过
# scripts/prune_chroma_collections.py 删除；当前索引变更请使用 scripts/build_staging_index.py。
# 另外，compute_ir_parity 在函数内动态导入的 experiments/phase13c/metrics13cc.py 也已随 A4.3 删除。
# 保留本文件仅作历史记录；需要对照时可从 tag stage-a-start 恢复上述依赖。
"""修复 EvidenceFlow active Chroma 的 metadata 契约（staging-first）。

旧 active Collection 只读保留；本脚本从 Phase 13F 已冻结的 Chunk JSONL 与
embedding NPZ 重建独立 staging Collection，逐项执行数量、Chunk 身份、metadata、
Domain 恢复、Dense Top20 和标准 IR parity gate。只有显式传入 ``--promote`` 且
全部 gate 通过时，才复制到新的 active Collection 并更新本地 ``.env`` 的集合名。
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
    """流式计算输入工件 hash，便于修复前后追踪冻结来源。"""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_reference() -> tuple[list[dict[str, Any]], np.ndarray, list[DocumentChunk]]:
    """加载并校验 13F 冻结记录、向量及 Domain Chunk 映射。"""

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
    """按分页读取 Chroma 原始记录；旧 metadata=None 会被原样保留。"""

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
    """按 chunk_id 全量比较文本身份，不依赖 Chroma 返回顺序。"""

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
    """检查每条新记录是否具备可恢复 Domain Chunk 所需的来源信息。"""

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
    """按 13F 的 BGE-M3 CLS + L2 协议生成冻结题目的查询向量。"""

    # 只使用本机已有模型，避免完整性验证意外触发网络下载。
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
    """比较旧库、新库和 13F 冻结 run 的 Dense Top20 排名。"""

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
    """按 13F 同一 qrels 投影与指标实现复算，确认 metadata 修复未改语义结果。"""

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
    """创建明确身份的新 Collection；不删除同名数据，避免误覆盖。"""

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
    """只更新本地配置中的集合名，不读取或复制任何密钥字段。"""

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
    """执行 staging repair gates，并按需完成新 active Collection 晋升。"""

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

    # 已存在的 staging 只读复用并重新验门；绝不为了重跑而静默清空它。
    staging = (
        client.get_collection(name=STAGING_COLLECTION)
        if STAGING_COLLECTION in names
        else create_collection(client, STAGING_COLLECTION, lifecycle="STAGING")
    )
    # 先用项目 Adapter 写入，确保实际生产路径与离线修复共享同一 serializer。
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
    """命令行入口；默认只构建并验证 staging，不会修改 active 配置。"""

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
