"""执行 Phase 15C Dense30 Runtime Promotion 及其最小 Gate。"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evidenceflow.config import Settings
from evidenceflow.embedding.bge_m3 import BgeM3EmbeddingProvider
from evidenceflow.eval.ir_metrics import evaluate_ir_run
from evidenceflow.retrieval.retriever import DenseRetriever
from evidenceflow.retrieval.reranker import (
    LearnedReranker,
    TransformersCrossEncoderScorer,
)
from evidenceflow.storage.chroma import ChromaVectorStore
from experiments.phase15b_active_relevance import project_active_relevance
from experiments.run_phase15b1_dense_candidate_depth import (
    candidate_artifact,
    candidate_metrics,
    classify_failures,
    reranker_artifact,
    standard_summary,
)


OUTPUT_DIR = ROOT / "evals/evaluation_v3/phase15c_dense30_promotion_v1"
SNAPSHOT_PATH = ROOT / "artifacts/phase15a/baseline_snapshot.json"
ACTIVE_MANIFEST_PATH = ROOT / "artifacts/phase15a/active_manifest.json"
QUESTIONS_V1_PATH = ROOT / "evals/research_retrieval/research_questions_v1.json"
QUESTIONS_V2_PATH = ROOT / "evals/evaluation_v2/phase35_gold_stabilization_v1/research_questions_v2_gold_stable.json"
REFERENCE_DIR = ROOT / "evals/evaluation_v3/phase15b1_dense_candidate_depth_v1"
MAPPING_PATH = ROOT / "evals/evaluation_v2/phase35_gold_stabilization_v1/gold_mapping_v3.json"
QRELS_PATH = ROOT / "evals/evaluation_v3/standard_ir_v1/qrels.json"
QRELS_MANIFEST_PATH = ROOT / "evals/evaluation_v3/standard_ir_v1/qrels_manifest.json"

COLLECTION_NAME = "ef_research_bge_m3_hybrid_c512_v2"
CORPUS_SIGNATURE = "bb184726b7f98756bd519d70e1a2a1f1224b5108e689680b2c328023ba1f4332"
INDEX_SIGNATURE = "b4e9432c7170615ade2c53803a887fa86d0f793949934a57db421f0da57739e5"
CHUNK_ID_HASH = "e87a613ebe65b10262996e83974b55839e0b6c27fbe22053ee7b8826dd1f153d"
PROJECTION_HASH = "6ccf0ffeffb8799429baff192e2de75e57199d1868688ca75ac2479cfb0cc4cd"
QUESTION_COUNT = 40
DOCUMENT_COUNT = 10
CHUNK_COUNT = 817
CANDIDATE_K = 30
FINAL_TOP_K = 5

FROZEN_ROOTS = (
    ROOT / "artifacts/phase15a",
    ROOT / "evals/evaluation_v3/phase13f_hybrid_runtime_promotion",
    ROOT / "evals/evaluation_v3/phase15b_hybrid_candidate_v1",
    ROOT / "evals/evaluation_v3/phase15b_hybrid_candidate_v2",
    ROOT / "evals/evaluation_v3/phase15b1_dense_candidate_depth_v1",
    ROOT / "evals/evaluation_v3/phase15b5_project_consolidation_v1",
)
PROTECTED_FILES = (
    ROOT / "evals/evaluation_v2/phase35_gold_stabilization_v1/gold_mapping_v3.json",
    ROOT / "evals/evaluation_v2/phase35_gold_stabilization_v1/research_questions_v2_gold_stable.json",
    ROOT / "evals/evaluation_v3/standard_ir_v1/qrels.json",
    ROOT / "evals/evaluation_v3/standard_ir_v1/qrels_manifest.json",
)
PRODUCTION_SOURCES = (
    ROOT / "src/evidenceflow/config.py",
    ROOT / "src/evidenceflow/runtime.py",
    ROOT / "src/evidenceflow/retrieval/retriever.py",
    ROOT / "src/evidenceflow/retrieval/reranker.py",
    ROOT / "src/evidenceflow/storage/chroma.py",
)


def read_json(path: Path) -> Any:
    """读取 UTF-8 JSON，所有 Promotion 输入均来自本地冻结产物。"""

    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    """以排序键和固定缩进写出可审计的 JSON 产物。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    """计算文件字节 Hash；Hash 用于证明冻结证据没有被覆盖。"""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(payload: Any) -> str:
    """计算排序紧凑 JSON 的 Hash，用于逐题 run 的轻量证据索引。"""

    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def configure_offline_cache() -> None:
    """将模型回放限制在项目已有缓存，避免隐式网络下载。"""

    os.environ.setdefault("HF_HOME", str(ROOT / ".cache/huggingface"))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def resolve_local_reranker_model(settings: Settings) -> str:
    """保持 BAAI/bge-reranker-base 身份不变，优先使用本地 checkpoint。"""

    local = ROOT / ".cache/models/bge-reranker-base"
    if (local / "config.json").is_file() and (local / "model.safetensors").is_file():
        return str(local)
    return settings.reranker_model


def tree_hashes(root: Path) -> dict[str, str]:
    """记录目录内所有文件 Hash，路径统一为项目相对 POSIX 路径。"""

    if not root.is_dir():
        raise RuntimeError(f"required frozen root is missing: {root}")
    return {
        str(path.relative_to(ROOT)).replace("\\", "/"): sha256_file(path)
        for path in sorted(path for path in root.rglob("*") if path.is_file())
    }


def frozen_hashes() -> dict[str, str]:
    """汇总 Phase 13F、15A、15B、15B.1、15B.5 与 Gold/qrels Hash。"""

    hashes: dict[str, str] = {}
    for root in FROZEN_ROOTS:
        hashes.update(tree_hashes(root))
    hashes.update({str(path.relative_to(ROOT)).replace("\\", "/"): sha256_file(path) for path in PROTECTED_FILES})
    return hashes


def production_source_hashes() -> dict[str, str]:
    """记录本阶段允许审计变化的最小 Production source 集合。"""

    return {str(path.relative_to(ROOT)).replace("\\", "/"): sha256_file(path) for path in PRODUCTION_SOURCES}


def active_identity(settings: Settings, store: ChromaVectorStore) -> dict[str, Any]:
    """只读核对 active Collection 身份、数量、metadata 和索引签名。"""

    snapshot = read_json(SNAPSHOT_PATH)
    manifest = read_json(ACTIVE_MANIFEST_PATH)
    chunks = list(store.iter_all_chunks())
    ids = sorted(chunk.chunk_id for chunk in chunks)
    metadata_coverage = sum(bool(chunk.metadata) for chunk in chunks) / len(chunks) if chunks else 0.0
    return {
        "collection_name": settings.collection_name,
        "expected_collection_name": COLLECTION_NAME,
        "collection_metadata": store.metadata,
        "expected_collection_metadata": snapshot["authoritative_identity"]["chroma_active_collection_metadata"],
        "document_count": len({chunk.document_id for chunk in chunks}),
        "chunk_count": len(chunks),
        "metadata_coverage": metadata_coverage,
        "chunk_id_hash": canonical_hash(ids),
        "expected_chunk_id_hash": CHUNK_ID_HASH,
        "corpus_signature": manifest["corpus_signature"],
        "expected_corpus_signature": CORPUS_SIGNATURE,
        "index_signature": manifest["index"]["index_signature"],
        "expected_index_signature": INDEX_SIGNATURE,
        "active_relevance_projection_hash": PROJECTION_HASH,
    }


def identity_pass(identity: Mapping[str, Any]) -> bool:
    """判断 active index 是否满足 Phase 15C 不变 Gate。"""

    return (
        identity["collection_name"] == identity["expected_collection_name"]
        and identity["collection_metadata"] == identity["expected_collection_metadata"]
        and identity["document_count"] == DOCUMENT_COUNT
        and identity["chunk_count"] == CHUNK_COUNT
        and identity["metadata_coverage"] == 1.0
        and identity["chunk_id_hash"] == identity["expected_chunk_id_hash"]
        and identity["corpus_signature"] == identity["expected_corpus_signature"]
        and identity["index_signature"] == identity["expected_index_signature"]
    )


def runtime_composition(settings: Settings) -> dict[str, Any]:
    """输出不含密钥的 Composition Root 快照，便于 baseline 审计。"""

    return {
        "entrypoint": "evidenceflow.runtime.build_rag_service",
        "pipeline": f"Dense Top{settings.reranker_input_k} -> Learned Reranker -> Top5",
        "settings": {
            "embedding_model": settings.embedding_model,
            "collection_name": settings.collection_name,
            "reranker_model": settings.reranker_model,
            "reranker_input_k": settings.reranker_input_k,
            "reranker_final_k": settings.reranker_final_k,
            "reranker_batch_size": settings.reranker_batch_size,
            "reranker_max_length": settings.reranker_max_length,
            "model_device": settings.model_device,
            "reranker_dtype": settings.reranker_dtype,
        },
        "components": [
            "BgeM3EmbeddingProvider",
            "ChromaVectorStore",
            "DenseRetriever",
            "TransformersCrossEncoderScorer",
            "LearnedReranker",
            "DenseRerankerRetriever",
            "ContextBuilder",
            "Generator",
            "RAGService",
        ],
        "candidate_k_source": "Settings.reranker_input_k",
        "final_k_source": "Settings.reranker_final_k",
    }


def run_pytest() -> dict[str, Any]:
    """运行项目完整回归，并保留短尾输出作为 Gate 证据。"""

    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return {
        "command": "python -m pytest -q",
        "returncode": completed.returncode,
        "passed": completed.returncode == 0,
        "stdout_tail": completed.stdout[-2000:],
        "stderr_tail": completed.stderr[-2000:],
    }


def compare_d20_baseline(actual_path: Path) -> dict[str, Any]:
    """将当前未修改 Production D20 replay 与 Phase 15B.1 冻结 D20 对齐。"""

    actual = read_json(actual_path)
    frozen = read_json(REFERENCE_DIR / "production_baseline.json")
    metric_keys = ("P@5", "Recall@5", "MRR@5", "MAP@5", "nDCG@5")
    summary_mismatches = [
        key for key in metric_keys if actual["summary_at_5"].get(key) != frozen["summary_at_5"].get(key)
    ]
    return {
        "status": actual.get("status"),
        "query_count": actual.get("query_count"),
        "candidate_count_values": sorted({len(ids) for ids in actual.get("candidate_run", {}).values()}),
        "final_count_values": sorted({len(ids) for ids in actual.get("final_run", {}).values()}),
        "candidate_parity": actual.get("candidate_run") == frozen.get("candidate_run"),
        "top5_parity": actual.get("final_run") == frozen.get("final_run"),
        "metrics_parity": not summary_mismatches,
        "summary_mismatches": summary_mismatches,
        "candidate_run_hash": canonical_hash(actual.get("candidate_run")),
        "final_run_hash": canonical_hash(actual.get("final_run")),
        "reference_summary_at_5": frozen["summary_at_5"],
        "actual_summary_at_5": actual["summary_at_5"],
    }


def baseline_manifest(d20_replay: Path) -> dict[str, Any]:
    """生成 G0 基线；未通过时不允许继续 Promotion。"""

    configure_offline_cache()
    settings = Settings()
    store = ChromaVectorStore(settings.vector_db_path, settings.collection_name)
    identity = active_identity(settings, store)
    pytest_result = run_pytest()
    d20 = compare_d20_baseline(d20_replay)
    frozen = frozen_hashes()
    source_hashes = production_source_hashes()
    baseline = {
        "schema_version": "phase15c-baseline-manifest-v1",
        "phase": "15C",
        "status": "G0_PASS" if d20["candidate_parity"] and d20["top5_parity"] and d20["metrics_parity"] and pytest_result["passed"] and identity_pass(identity) else "STOP",
        "created_at_epoch": time.time(),
        "d20_replay": d20,
        "pytest": pytest_result,
        "runtime_composition": runtime_composition(settings),
        "active_identity": identity,
        "projection_hash": PROJECTION_HASH,
        "production_source_hashes": source_hashes,
        "historical_frozen_hashes": frozen,
        "reference": {
            "phase15b1_dir": str(REFERENCE_DIR.relative_to(ROOT)).replace("\\", "/"),
            "candidate_arm": "D30",
            "final_top_k": FINAL_TOP_K,
        },
    }
    write_json(OUTPUT_DIR / "baseline_manifest.json", baseline)
    return baseline


def write_promotion_audit(settings: Settings) -> None:
    """记录 candidate K 来源、重复位置及本阶段明确边界。"""

    text = f"""# Phase 15C Promotion Audit

## CANONICAL_CANDIDATE_K_SOURCE

`Settings.reranker_input_k` 是正式 candidate depth 的唯一运行时来源；当前值为 `{settings.reranker_input_k}`。
`runtime.py` 将该值同时传给 `DenseRetriever.top_k`、`LearnedReranker.max_candidates`、
`DenseRerankerRetriever.candidate_pool_k` 和 `reranker_input_k`。

## DUPLICATED_K_LOCATIONS

- `src/evidenceflow/config.py`: `Settings.reranker_input_k` 默认值 `{settings.reranker_input_k}`，本阶段唯一需要晋升的 canonical default。
- `src/evidenceflow/retrieval/reranker.py`: 通用 `DenseRerankerRetriever` 构造参数默认值 `20`；正式 Runtime 总是显式传入 Settings，不修改该通用 fallback，避免扩大本阶段行为范围。
- `experiments/` 与历史 `evals/` 中的 `20` 是冻结实验口径，不属于当前 Production source，不修改。

## FILES_TO_MODIFY

- `src/evidenceflow/config.py`
- `src/evidenceflow/runtime.py`
- `tests/test_config.py`
- `tests/test_runtime.py`
- `PROJECT_SPEC.md`
- `README.md`
- `experiments/run_phase15c_dense30_promotion.py`

## FILES_TO_TEST

- `tests/test_config.py`
- `tests/test_runtime.py`
- `tests/test_retriever.py`
- `tests/test_reranker.py`
- `tests/test_architecture_boundaries.py`
- `python -m pytest -q`

## FILES_NOT_TO_TOUCH

- `src/evidenceflow/retrieval/reranker.py` 的排序、tie-break 和评分实现
- Embedding、Chroma、Chunk、ContextBuilder、Generator、Citation、RAGService
- active Chroma、source PDFs、Gold、qrels、mapping、projection
- Phase 13F、15A、15B、15B.1、15B.5 冻结 artifacts
- `experiments/archive/` 与历史 Phase 15B.1 记录

## REQUIRED_CHANGE

只将正式 Runtime candidate depth 从 `20` 改为 `30`；Reranker 输入同步为 30，最终输出继续为 5。
"""
    (OUTPUT_DIR / "promotion_audit.md").parent.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "promotion_audit.md").write_text(text, encoding="utf-8")


def build_replay() -> dict[str, Any]:
    """用 Production 同一 Dense/Reranker primitive 执行 40Q D30 replay。"""

    configure_offline_cache()
    settings = Settings()
    if settings.reranker_input_k != CANDIDATE_K:
        raise RuntimeError(f"Production candidate K is {settings.reranker_input_k}, expected {CANDIDATE_K}")
    if settings.reranker_final_k != FINAL_TOP_K:
        raise RuntimeError("Production final Top-K changed during promotion")
    questions_v1 = read_json(QUESTIONS_V1_PATH)
    questions_v2 = read_json(QUESTIONS_V2_PATH)
    question_ids = [question["question_id"] for question in questions_v1["questions"]]
    if len(question_ids) != QUESTION_COUNT or question_ids != [q["question_id"] for q in questions_v2["questions"]]:
        raise RuntimeError("frozen 40Q question order mismatch")
    store = ChromaVectorStore(settings.vector_db_path, settings.collection_name)
    identity = active_identity(settings, store)
    if not identity_pass(identity):
        raise RuntimeError("active index identity gate failed")
    chunks = list(store.iter_all_chunks())
    projection = project_active_relevance(chunks, questions_v2)
    if projection["active_relevance_projection_hash"] != PROJECTION_HASH:
        raise RuntimeError("active relevance projection hash changed")

    embedding = BgeM3EmbeddingProvider(
        settings.embedding_model, device=settings.model_device
    )
    dense = DenseRetriever(embedding, store, top_k=CANDIDATE_K)
    load_started = time.perf_counter()
    scorer = TransformersCrossEncoderScorer(
        resolve_local_reranker_model(settings),
        device=settings.model_device,
        dtype=settings.reranker_dtype,
        max_length=settings.reranker_max_length,
        inference_mode=False,
    )
    model_load_ms = (time.perf_counter() - load_started) * 1000.0
    reranker = LearnedReranker(scorer, batch_size=settings.reranker_batch_size, max_candidates=CANDIDATE_K)

    questions_by_id = {question["question_id"]: question for question in questions_v1["questions"]}
    pools: dict[str, list[Any]] = {}
    final_results: dict[str, list[Any]] = {}
    scored_results: dict[str, list[Any]] = {}
    candidate_run: dict[str, list[str]] = {}
    final_run: dict[str, list[str]] = {}
    per_query: dict[str, dict[str, Any]] = {}
    latency_rows: list[dict[str, Any]] = []
    for question_id in question_ids:
        query = questions_by_id[question_id]["english_query"]
        total_started = time.perf_counter()
        dense_started = time.perf_counter()
        candidates = dense.retrieve(query, top_k=CANDIDATE_K)
        dense_ms = (time.perf_counter() - dense_started) * 1000.0
        if len(candidates) != CANDIDATE_K:
            raise RuntimeError(f"D30 returned {len(candidates)} candidates for {question_id}")
        pools[question_id] = candidates
        candidate_run[question_id] = [item.chunk.chunk_id for item in candidates]
        rerank_started = time.perf_counter()
        all_results = reranker.rerank(query, [item.chunk for item in candidates], top_k=None)
        reranker_e2e_ms = (time.perf_counter() - rerank_started) * 1000.0
        results = all_results[:FINAL_TOP_K]
        scored_results[question_id] = all_results
        final_results[question_id] = results
        final_run[question_id] = [item.chunk.chunk_id for item in results]
        candidate_rank = {item.chunk.chunk_id: item.rank for item in candidates}
        per_query[question_id] = {
            "question_id": question_id,
            "query": query,
            "candidate_count": len(candidates),
            "candidate_chunk_ids": candidate_run[question_id],
            "top5": [
                {
                    "rank": item.rank,
                    "chunk_id": item.chunk.chunk_id,
                    "document_id": item.chunk.document_id,
                    "candidate_rank": candidate_rank[item.chunk.chunk_id],
                    "reranker_score": item.relevance_score,
                }
                for item in results
            ],
        }
        latency_rows.append(
            {
                "question_id": question_id,
                "dense_search_ms": dense_ms,
                "reranker_pair_count": len(candidates),
                "reranker_model_ms": float(scorer.last_timing_seconds.get("model_forward", 0.0)) * 1000.0,
                "reranker_e2e_ms": reranker_e2e_ms,
                "retrieval_total_ms": (time.perf_counter() - total_started) * 1000.0,
                "truncated_pair_count": reranker.last_truncated_pair_count,
            }
        )

    qrels = projection["qrels"]
    groups = projection["groups_by_question"]
    candidate_pools = {question_id: [item.chunk for item in rows] for question_id, rows in pools.items()}
    candidate_metrics_payload = candidate_metrics(candidate_pools, groups, question_ids, CANDIDATE_K)
    ir = evaluate_ir_run(qrels, final_run, depths=(1, 3, 5, 10, 20))
    failures = classify_failures(
        {"30": candidate_pools},
        {"30": final_results},
        groups,
        question_ids,
    )
    latency = {
        "schema_version": "phase15c-latency-v1",
        "model_load_ms": model_load_ms,
        "arms": {
            "30": {
                "per_query": latency_rows,
                "aggregate": latency_summary(latency_rows),
            }
        },
    }
    write_json(OUTPUT_DIR / "production_replay.json", {
        "schema_version": "phase15c-production-replay-v1",
        "status": "PASS",
        "pipeline": "production Dense Top30 -> BAAI/bge-reranker-base -> Top5",
        "collection_name": settings.collection_name,
        "corpus_signature": CORPUS_SIGNATURE,
        "index_signature": INDEX_SIGNATURE,
        "active_relevance_projection_hash": PROJECTION_HASH,
        "candidate_k": CANDIDATE_K,
        "reranker_input_k": settings.reranker_input_k,
        "final_top_k": FINAL_TOP_K,
        "document_count": DOCUMENT_COUNT,
        "chunk_count": CHUNK_COUNT,
        "query_count": QUESTION_COUNT,
        "question_ids": question_ids,
        "candidate_run": candidate_run,
        "final_run": final_run,
        "per_query": per_query,
        "summary_at_5": standard_summary(ir),
        "standard_ir": ir,
        "failure_attribution": failures,
        "candidate_metrics": candidate_metrics_payload,
        "reranker": {
            "model": settings.reranker_model,
            "checkpoint": "BAAI/bge-reranker-base",
            "max_length": settings.reranker_max_length,
            "batch_size": settings.reranker_batch_size,
            "device": scorer.device,
            "dtype": settings.reranker_dtype,
            "final_top_k": FINAL_TOP_K,
        },
    })
    write_json(OUTPUT_DIR / "latency_metrics.json", latency)
    write_json(OUTPUT_DIR / "failure_attribution.json", failures)
    write_json(OUTPUT_DIR / "candidate_metrics.json", {
        "schema_version": "phase15c-candidate-metrics-v1",
        "active_relevance_projection_hash": PROJECTION_HASH,
        "arms": {"30": candidate_metrics_payload},
    })
    write_json(OUTPUT_DIR / "final_ir_metrics.json", {
        "schema_version": "phase15c-final-ir-v1",
        "active_relevance_projection_hash": PROJECTION_HASH,
        "arms": {"30": ir},
        "summary_at_5": {"30": standard_summary(ir)},
    })
    write_json(OUTPUT_DIR / "candidates/dense30.json", candidate_artifact("30", {qid: pools[qid] for qid in question_ids}, qrels, question_ids))
    write_json(OUTPUT_DIR / "reranker/dense30_top5.json", reranker_artifact("30", final_results, candidate_pools, question_ids))
    return {
        "settings": settings,
        "identity": identity,
        "projection": projection,
        "question_ids": question_ids,
        "candidate_run": candidate_run,
        "final_run": final_run,
        "pools": candidate_pools,
        "final_results": final_results,
        "groups": groups,
        "scored_results": scored_results,
        "qrels": qrels,
        "failures": failures,
        "ir": ir,
        "latency": latency,
    }


def latency_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """对每题 latency 做 mean/p50/p95/max 聚合，不混入模型加载时间。"""

    def percentile(values: list[float], p: float) -> float:
        ordered = sorted(values)
        if not ordered:
            return 0.0
        position = (len(ordered) - 1) * p / 100.0
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return ordered[lower]
        return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)

    output: dict[str, Any] = {}
    for key in ("dense_search_ms", "reranker_pair_count", "reranker_model_ms", "reranker_e2e_ms", "retrieval_total_ms"):
        values = [float(row[key]) for row in rows]
        output[key] = {
            "mean": sum(values) / len(values) if values else 0.0,
            "p50": percentile(values, 50),
            "p95": percentile(values, 95),
            "max": max(values) if values else 0.0,
        }
        if key == "reranker_pair_count":
            output[key]["total"] = sum(values)
    return output


def compare_promotion_replay(run: Mapping[str, Any]) -> dict[str, Any]:
    """以 Phase 15B.1 D30 frozen artifact 做 exact Candidate/Top5/metric Gate。"""

    frozen_candidate = read_json(REFERENCE_DIR / "candidates/dense30.json")
    frozen_final = read_json(REFERENCE_DIR / "reranker/dense30_top5.json")
    frozen_metrics = read_json(REFERENCE_DIR / "final_ir_metrics.json")
    frozen_failures = read_json(REFERENCE_DIR / "failure_attribution.json")
    expected_candidates = {row["question_id"]: row["chunk_ids"] for row in frozen_candidate["per_query"]}
    expected_final = {
        row["question_id"]: [item["chunk_id"] for item in row["rows"]]
        for row in frozen_final["per_query"]
    }
    actual_summary = standard_summary(run["ir"])
    expected_summary = frozen_metrics["summary_at_5"]["30"]
    metric_keys = ("P@5", "Recall@5", "MRR@5", "MAP@5", "nDCG@5")
    candidate_mismatches = [qid for qid in run["question_ids"] if run["candidate_run"].get(qid) != expected_candidates.get(qid)]
    top5_mismatches = [qid for qid in run["question_ids"] if run["final_run"].get(qid) != expected_final.get(qid)]
    metric_mismatches = [key for key in metric_keys if actual_summary.get(key) != expected_summary.get(key)]
    expected_failure_rows = {row["question_id"]: row for row in frozen_failures["arms"]["30"]["per_query"]}
    actual_failure_rows = {row["question_id"]: row for row in run["failures"]["arms"]["30"]["per_query"]}
    failure_mismatches = [qid for qid in run["question_ids"] if actual_failure_rows.get(qid) != expected_failure_rows.get(qid)]
    return {
        "schema_version": "phase15c-parity-result-v1",
        "reference": str(REFERENCE_DIR.relative_to(ROOT)).replace("\\", "/"),
        "projection_hash_match": run["projection"]["active_relevance_projection_hash"] == PROJECTION_HASH,
        "candidate_parity": not candidate_mismatches,
        "top5_parity": not top5_mismatches,
        "metrics_parity": not metric_mismatches,
        "failure_attribution_parity": not failure_mismatches,
        "candidate_mismatches": candidate_mismatches,
        "top5_mismatches": top5_mismatches,
        "metric_mismatches": metric_mismatches,
        "failure_mismatches": failure_mismatches,
        "actual_summary_at_5": actual_summary,
        "expected_summary_at_5": expected_summary,
        "parity": not (candidate_mismatches or top5_mismatches or metric_mismatches or failure_mismatches),
    }


def build_reranker_miss_seed(run: Mapping[str, Any]) -> dict[str, Any]:
    """只提取 Gold 已进 D30、但未进 Top5 的诊断数据，不改变评分算法。"""

    rows = []
    by_question = {row["question_id"]: row for row in run["failures"]["arms"]["30"]["per_query"]}
    for question_id in run["question_ids"]:
        failure = by_question[question_id]
        if failure["status"] != "RERANKER_MISS":
            continue
        candidates = run["pools"][question_id]
        all_scored = run["scored_results"][question_id]
        # 通过一次独立、确定性的分数回放取得 Gold 分数；不重新调用模型，
        # 因此诊断不会改变 Promotion ranking 或推理预算。
        candidate_rank = {item.chunk.chunk_id: item.rank for item in candidates}
        score_by_id = {item.chunk.chunk_id: item.relevance_score for item in all_scored}
        top5 = [item.chunk.chunk_id for item in run["final_results"][question_id]]
        gold_ids = list(run["qrels"].get(question_id, []))
        gold_candidates = [cid for cid in gold_ids if cid in candidate_rank]
        if not gold_candidates or any(cid in top5 for cid in gold_candidates):
            continue
        chunk_by_id = {item.chunk.chunk_id: item.chunk for item in candidates}
        rows.append(
            {
                "question_id": question_id,
                "gold_chunk_ids": gold_ids,
                "gold_candidate_rank": {cid: candidate_rank[cid] for cid in gold_candidates},
                "gold_reranker_score": {cid: score_by_id.get(cid) for cid in gold_candidates},
                "top5_chunk_ids": top5,
                "top5_reranker_scores": {
                    item.chunk.chunk_id: item.relevance_score for item in run["final_results"][question_id]
                },
                "document_ids": {
                    cid: chunk_by_id[cid].document_id for cid in gold_candidates if cid in chunk_by_id
                },
                "chunk_metadata": {
                    cid: chunk_by_id[cid].metadata for cid in gold_candidates if cid in chunk_by_id
                },
                "element_metadata": {
                    cid: {"element_ids": chunk_by_id[cid].metadata.get("element_ids", [])}
                    for cid in gold_candidates
                    if cid in chunk_by_id
                },
            }
        )
    return {
        "schema_version": "phase15c-reranker-miss-seed-v1",
        "candidate_k": CANDIDATE_K,
        "final_top_k": FINAL_TOP_K,
        "rows": rows,
    }


def remove_regenerable_caches() -> dict[str, Any]:
    """按显式 allowlist 清理测试缓存，不触碰模型、索引和评测证据。"""

    removed_files: list[str] = []
    removed_dirs: list[str] = []
    # 从项目根扫描这些明确名称，覆盖 pytest 在根目录生成的缓存，同时不触碰
    # active Chroma、模型缓存、source PDF 或任何正式 artifact。
    for root in (ROOT,):
        for path in sorted(root.rglob("*"), reverse=True):
            if path.is_file() and (path.name.endswith(".pyc") or path.name.endswith(".pyo")):
                removed_files.append(str(path.relative_to(ROOT)).replace("\\", "/"))
                path.unlink()
            elif path.is_dir() and path.name in {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}:
                removed_dirs.append(str(path.relative_to(ROOT)).replace("\\", "/"))
                shutil.rmtree(path)
    remaining = [
        str(path.relative_to(ROOT)).replace("\\", "/")
        for path in ROOT.rglob("*")
        if path.is_file() and (path.name.endswith(".pyc") or path.name.endswith(".pyo"))
    ]
    return {
        "schema_version": "phase15c-cleanup-manifest-v1",
        "status": "PASS" if not remaining else "FAIL",
        "allowlist": ["__pycache__", "*.pyc", "*.pyo", ".pytest_cache", ".mypy_cache", ".ruff_cache"],
        "removed_files": removed_files,
        "removed_directories": removed_dirs,
        "remaining_regenerable_files": remaining,
        "protected_roots": [str(path.relative_to(ROOT)).replace("\\", "/") for path in FROZEN_ROOTS],
    }


def promote(output_dir: Path = OUTPUT_DIR) -> dict[str, Any]:
    """执行 D30 replay、全部 Gate、轻量 cleanup，并生成最终报告。"""

    global OUTPUT_DIR
    OUTPUT_DIR = output_dir
    baseline = read_json(OUTPUT_DIR / "baseline_manifest.json")
    if baseline.get("status") != "G0_PASS":
        raise RuntimeError("G0 baseline did not pass; promotion is stopped")
    audit_path = OUTPUT_DIR / "promotion_audit.md"
    if not audit_path.is_file():
        raise RuntimeError("promotion_audit.md is missing")
    before_sources = baseline["production_source_hashes"]
    before_frozen = baseline["historical_frozen_hashes"]
    run = build_replay()
    parity = compare_promotion_replay(run)
    write_json(OUTPUT_DIR / "parity_result.json", parity)
    if not parity["parity"]:
        raise RuntimeError("D30 parity failed; rollback required before continuing")
    current_identity = active_identity(run["settings"], ChromaVectorStore(run["settings"].vector_db_path, run["settings"].collection_name))
    after_sources = production_source_hashes()
    after_frozen = frozen_hashes()
    changed_sources = sorted(path for path in before_sources if before_sources[path] != after_sources[path])
    forbidden_source_changes = sorted(path for path in before_sources if path not in {"src/evidenceflow/config.py", "src/evidenceflow/runtime.py"} and before_sources[path] != after_sources[path])
    full_pytest = run_pytest()
    write_json(OUTPUT_DIR / "reranker_miss_seed.json", build_reranker_miss_seed(run))
    execution = {
        "schema_version": "phase15c-execution-result-v1",
        "phase": "15C",
        "status": "PASS",
        "gates": {
            "G0_baseline": baseline["status"] == "G0_PASS",
            "G1_minimal_change": changed_sources == ["src/evidenceflow/config.py", "src/evidenceflow/runtime.py"] and not forbidden_source_changes,
            "G2_d30_replay_parity": parity["parity"],
            "G3_active_index_identity": identity_pass(current_identity),
            "G4_frozen_evidence": before_frozen == after_frozen,
            "G5_full_pytest": full_pytest["passed"],
            "G6_cleanup": False,
        },
        "production_behavior_change": "candidate depth 20 -> 30; final Top5 and all downstream components unchanged",
        "changed_production_sources": changed_sources,
        "forbidden_source_changes": forbidden_source_changes,
        "active_identity": current_identity,
        "full_pytest": full_pytest,
        "parity": parity,
        "failure_counts": run["failures"]["arms"]["30"]["counts"],
        "historical_frozen_hashes_unchanged": before_frozen == after_frozen,
        "completed_at_epoch": time.time(),
    }
    write_json(OUTPUT_DIR / "execution_result.json", execution)
    cleanup = remove_regenerable_caches()
    write_json(OUTPUT_DIR / "cleanup_manifest.json", cleanup)
    execution["gates"]["G6_cleanup"] = cleanup["status"] == "PASS"
    execution["status"] = "PASS" if all(execution["gates"].values()) else "FAIL"
    write_json(OUTPUT_DIR / "execution_result.json", execution)
    summary = standard_summary(run["ir"])
    counts = run["failures"]["arms"]["30"]["counts"]
    report = f"""# EvidenceFlow Phase 15C — Dense30 Runtime Promotion

## Result

**{execution['status']}** — Production candidate depth 已从 20 晋升为 30，最终输出仍为 Top5。

## Gates

| Gate | Result |
|---|---|
| G0 D20 baseline replay | {'PASS' if execution['gates']['G0_baseline'] else 'FAIL'} |
| G1 minimal source change | {'PASS' if execution['gates']['G1_minimal_change'] else 'FAIL'} |
| G2 D30 exact Candidate/Top5/metrics parity | {'PASS' if execution['gates']['G2_d30_replay_parity'] else 'FAIL'} |
| G3 active index identity | {'PASS' if execution['gates']['G3_active_index_identity'] else 'FAIL'} |
| G4 historical frozen evidence | {'PASS' if execution['gates']['G4_frozen_evidence'] else 'FAIL'} |
| G5 full pytest | {'PASS' if execution['gates']['G5_full_pytest'] else 'FAIL'} |
| G6 cleanup | {'PASS' if execution['gates']['G6_cleanup'] else 'FAIL'} |

## D30 exact reference

| Metric | Value |
|---|---:|
| P@5 | {summary['P@5']:.16g} |
| Recall@5 | {summary['Recall@5']:.16g} |
| MRR@5 | {summary['MRR@5']:.16g} |
| MAP@5 | {summary['MAP@5']:.16g} |
| nDCG@5 | {summary['nDCG@5']:.16g} |

Candidate coverage 与 D40 相同；D30 的 final ranking 指标优于 D40，且 Reranker pair count 为 1200，低于 D40 的 1600。

## Failure attribution

`UNINDEXABLE={counts['UNINDEXABLE']}`, `CANDIDATE_MISS={counts['CANDIDATE_MISS']}`, `RERANKER_MISS={counts['RERANKER_MISS']}`, `SUCCESS={counts['SUCCESS']}`。

详细诊断见 `reranker_miss_seed.json`；本阶段未更换模型、未校准分数、未修改 source diversity 或检索算法。

## Latency

`latency_metrics.json` 保存逐题 dense search、Reranker pair count、model/e2e 与 retrieval total 的 mean/p50/p95/max；模型加载时间单独记录。

## Stop boundary

Phase 15C 完成后停止，不自动进入 Reranker 优化、Parser 修复、FastAPI、Frontend、LangGraph 或 Agent 阶段。
"""
    (OUTPUT_DIR / "report.md").write_text(report, encoding="utf-8")
    return execution


def record_stop(output_dir: Path = OUTPUT_DIR) -> dict[str, Any]:
    """记录 parity 失败后的 STOP 与回滚状态，不重新运行或修改 active index。"""

    parity = read_json(output_dir / "parity_result.json")
    baseline = read_json(output_dir / "baseline_manifest.json")
    settings = Settings()
    store = ChromaVectorStore(settings.vector_db_path, settings.collection_name)
    identity = active_identity(settings, store)
    current_sources = production_source_hashes()
    rollback_complete = (
        settings.reranker_input_k == 20
        and current_sources == baseline["production_source_hashes"]
        and identity_pass(identity)
    )
    cleanup = {
        "schema_version": "phase15c-cleanup-manifest-v1",
        "status": "SKIPPED_STOP_AFTER_PARITY_FAILURE",
        "reason": "Hard Gate G2 failed; promotion was rolled back and the phase was stopped before final cleanup.",
        "removed_files": [],
        "removed_directories": [],
        "protected_roots": [str(path.relative_to(ROOT)).replace("\\", "/") for path in FROZEN_ROOTS],
    }
    execution = {
        "schema_version": "phase15c-execution-result-v1",
        "phase": "15C",
        "status": "STOP",
        "stop_reason": "D30 exact Candidate/Top5 parity failed; production rolled back to D20",
        "gates": {
            "G0_baseline": baseline["status"] == "G0_PASS",
            "G2_d30_replay_parity": parity["parity"],
            "rollback_to_d20": rollback_complete,
            "G3_active_index_identity": identity_pass(identity),
            "G4_frozen_evidence": baseline["historical_frozen_hashes"] == frozen_hashes(),
        },
        "parity": parity,
        "rollback": {
            "candidate_k": settings.reranker_input_k,
            "final_top_k": settings.reranker_final_k,
            "production_sources_restored": current_sources == baseline["production_source_hashes"],
        },
        "not_executed_after_stop": [
            "production latency replay",
            "failure attribution refresh",
            "reranker_miss_seed",
            "full regression after promotion",
            "current-contract Top30 documentation update",
        ],
        "completed_at_epoch": time.time(),
    }
    write_json(output_dir / "cleanup_manifest.json", cleanup)
    write_json(output_dir / "execution_result.json", execution)
    report = f"""# EvidenceFlow Phase 15C — STOP

## Result

**STOP**。D30 promotion 未通过 exact parity，Production 已回滚为 Dense Top20 → Learned Reranker → Top5。

## Failed hard gate

- Candidate Top30 mismatch: `{", ".join(parity["candidate_mismatches"])}`
- Final Top5 mismatch: `{", ".join(parity["top5_mismatches"])}`
- IR metrics parity: `{"PASS" if parity["metrics_parity"] else "FAIL"}`
- Active relevance projection parity: `{"PASS" if parity["projection_hash_match"] else "FAIL"}`

指标相同不能替代逐题 Candidate/Top5 identity；因此没有继续执行后续 latency、failure attribution
或 reranker miss diagnostic seed。

只读复现诊断显示：同一模型下 `embed_documents` 与 `embed_query` 的向量逐元素一致；
但 active Chroma HNSW（`ef_search=100`）在不同新进程的 D30 边界查询上出现候选波动，
已观察到 ARR-032/ARR-033 的候选差异。该现象是 exact parity 的实际阻塞因素，
不能通过放宽比较规则消除；在重新申请 Promotion 前必须先建立并冻结确定性的 ANN/replay contract。

本次 `latency_metrics.json` 中的耗时是失败 D30 replay 随流程产生的诊断数据，
不是 parity 通过后按计划执行的独立 latency gate；后续独立 latency、failure attribution refresh
和 reranker miss diagnostic seed 均未被接受为 Phase 15C 结果。

## Rollback

当前 Settings candidate K 为 `{settings.reranker_input_k}`，final K 为 `{settings.reranker_final_k}`；
active Collection 仍为 `{settings.collection_name}`，index identity 未改变。历史 frozen artifact
Hash 未改变。

## Retained evidence

`baseline_manifest.json`、`promotion_audit.md`、`production_replay.json`、`parity_result.json`、
`latency_metrics.json`、`candidate_metrics.json`、`final_ir_metrics.json`、`failure_attribution.json`
以及 Candidate/Reranker 明细保留用于人工复核。`cleanup_manifest.json` 标明 STOP 后未执行最终 cleanup。
"""
    (output_dir / "report.md").write_text(report, encoding="utf-8")
    return execution


def main() -> int:
    """提供 baseline-only 与 promote 两个显式阶段，避免跳过 G0。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument("--d20-replay", type=Path)
    parser.add_argument("--promote", action="store_true")
    parser.add_argument("--record-stop", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    if args.baseline_only:
        if args.d20_replay is None:
            parser.error("--baseline-only requires --d20-replay")
        settings = Settings()
        result = baseline_manifest(args.d20_replay)
        write_promotion_audit(settings)
        print(json.dumps(result, ensure_ascii=False))
        return 0 if result["status"] == "G0_PASS" else 1
    if args.promote:
        result = promote(args.output_dir)
        print(json.dumps(result, ensure_ascii=False))
        return 0 if result["status"] == "PASS" else 1
    if args.record_stop:
        result = record_stop(args.output_dir)
        print(json.dumps(result, ensure_ascii=False))
        return 0
    parser.error("choose --baseline-only or --promote")


if __name__ == "__main__":
    raise SystemExit(main())
