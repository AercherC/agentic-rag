"""执行 Phase 15C.1 的 exact Dense 架构验证与受控离线评测。

本脚本只建立进程内派生 ExactDenseIndex，不修改 Production Runtime、active Chroma、
Gold、qrels 或任何历史冻结 artifact。数值与评测逻辑集中复用项目 canonical primitive，
脚本自身只负责阶段编排、门禁和 artifact 汇总。
"""

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

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evidenceflow.config import Settings
from evidenceflow.domain.chunk import DocumentChunk
from evidenceflow.embedding.bge_m3 import BgeM3EmbeddingProvider
from evidenceflow.eval.ir_metrics import evaluate_ir_run
from evidenceflow.retrieval.exact_dense import ExactDenseIndex
from evidenceflow.retrieval.reranker import (
    LearnedReranker,
    RerankedChunk,
    TransformersCrossEncoderScorer,
    rank_scored_candidates,
)
from evidenceflow.storage.chroma import ChromaVectorStore
from experiments.phase15b_active_relevance import project_active_relevance
from experiments.run_phase15b1_dense_candidate_depth import (
    CANDIDATE_KS,
    CHUNK_COUNT,
    CHUNK_ID_HASH,
    COLLECTION_NAME,
    DOCUMENT_COUNT,
    EXPECTED_REPLAY_PATH,
    FINAL_TOP_K,
    INDEX_SIGNATURE,
    PROJECTION_HASH,
    QUESTIONS_V1_PATH,
    QUESTIONS_V2_PATH,
    candidate_artifact,
    candidate_metrics,
    classify_failures,
    compare_d20_to_replay,
    compare_replay,
    configure_offline_model_cache,
    frozen_tree_hashes,
    latency_summary,
    load_frozen_inputs,
    resolve_local_reranker_model,
    sha256_file,
    standard_summary,
    validate_active_index,
    write_json,
)


OUTPUT_DIR = ROOT / "evals" / "evaluation_v3" / "phase15c1_exact_dense_architecture_v1"
SNAPSHOT_PATH = ROOT / "artifacts" / "phase15a" / "baseline_snapshot.json"
WORKER_COUNT = 10
WORKER_CANDIDATE_K = 30


class Phase15C1IdentityError(RuntimeError):
    """表示 active identity、冻结输入或回放门禁不满足阶段合同。"""


def read_json(path: Path) -> Any:
    """读取本地 UTF-8 JSON artifact。"""

    return json.loads(path.read_text(encoding="utf-8"))


def sorted_chunk_id_hash(chunks: Sequence[DocumentChunk]) -> str:
    """按既有 Phase 15A 合同计算稳定 Chunk ID Hash。"""

    serialized = json.dumps(
        sorted(chunk.chunk_id for chunk in chunks),
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def source_hashes() -> dict[str, str]:
    """记录本阶段可能触及的源码 Hash，证明只改变了必要接口。"""

    paths = (
        ROOT / "src" / "evidenceflow" / "retrieval" / "exact_dense.py",
        ROOT / "src" / "evidenceflow" / "retrieval" / "retriever.py",
        ROOT / "src" / "evidenceflow" / "storage" / "base.py",
        ROOT / "src" / "evidenceflow" / "storage" / "chroma.py",
        ROOT / "src" / "evidenceflow" / "runtime.py",
        ROOT / "src" / "evidenceflow" / "config.py",
    )
    return {str(path.relative_to(ROOT)).replace("\\", "/"): sha256_file(path) for path in paths}


def phase_frozen_hashes() -> dict[str, str]:
    """冻结本阶段要求保护的历史目录；不包含本阶段输出目录。"""

    roots = (
        ROOT / "artifacts" / "phase15a",
        ROOT / "evals" / "evaluation_v3" / "phase13f_hybrid_runtime_promotion",
        ROOT / "evals" / "evaluation_v3" / "phase15b1_dense_candidate_depth_v1",
        ROOT / "evals" / "evaluation_v3" / "phase15b_hybrid_candidate_v2",
        ROOT / "evals" / "evaluation_v3" / "phase15b5_project_consolidation_v1",
        ROOT / "evals" / "evaluation_v3" / "phase15c_dense30_promotion_v1",
    )
    result: dict[str, str] = {}
    for root in roots:
        if not root.is_dir():
            raise Phase15C1IdentityError(f"frozen artifact root is missing: {root}")
        for path in sorted(path for path in root.rglob("*") if path.is_file()):
            result[str(path.relative_to(ROOT)).replace("\\", "/")] = sha256_file(path)
    return result


def architecture_audit_text() -> str:
    """生成只读消费者审计结论，不把历史脚本重新接入 Runtime。"""

    return """# EvidenceFlow Phase 15C.1 Architecture Audit

## Scope

只读检查 `src/evidenceflow/`、`tests/`、`experiments/`、`scripts/` 与 `evals/`。
当前 Production 保持 `DenseRetriever -> ChromaVectorStore.search -> LearnedReranker`。

## Consumer classification

| Area | Classification | Finding |
|---|---|---|
| `src/evidenceflow/runtime.py` | PRODUCTION | Composition Root 仍只装配 HNSW Dense Top20 -> Reranker Top5。 |
| `src/evidenceflow/retrieval/retriever.py` | PRODUCTION / CANONICAL_REUSABLE | `DenseRetriever` 只依赖最小 `search()` 合同，可消费 Chroma 或 ExactDenseIndex。 |
| `src/evidenceflow/storage/chroma.py` | PRODUCTION / CANONICAL_REUSABLE | Chroma 继续负责持久化、metadata、lifecycle 和既有 HNSW `search()`；新增只读 embedding iterator。 |
| `src/evidenceflow/retrieval/exact_dense.py` | CANONICAL_REUSABLE | 唯一 exact candidate primitive；生产、评测和 HNSW audit 共用。 |
| `experiments/run_phase15b1_dense_candidate_depth.py` | EXPERIMENT_ONLY | 复用评测指标与 artifact 适配器，不再实现 exact search。 |
| `experiments/run_phase15c_dense30_promotion.py` | HISTORICAL | Phase 15C STOP 证据与回滚脚本，未进入 Production import graph。 |
| `experiments/archive/run_phase7_reranker_validation.py` / `run_phase8_runtime_readiness.py` | HISTORICAL | 旧 exhaustive cosine helper 只作历史证据；当前 canonical primitive 已覆盖其 executable 职责。 |
| `evals/evaluation_v3/*` | HISTORICAL_EVIDENCE | Gold、qrels、mapping、projection 与 frozen reports 只读保护。 |

## Answers

1. Production DenseRetriever 过去直接绑定完整 VectorStore；本阶段收窄为只读 `search()` 依赖，生产调用语义不变。
2. Chroma 原先没有稳定的 embedding iterator；本阶段增加 `iter_all_chunks_with_embeddings()`，按 `chunk_id ASC` 返回完整 Domain Chunk 与普通 float 列表。
3. Phase 7/8 有历史重复 exhaustive cosine；它们不在当前 Runtime，Phase 15C.1 不删除历史证据，ExactDenseIndex 成为唯一新的 executable primitive。
4. Phase 7/8 helper 不再被当前 Production/Evaluation 调用；历史 artifact 保留以满足可追溯性。
5. HNSW-specific Settings/code 仍由 Chroma 的持久化 Collection 与 Production fallback 使用；本阶段不删除 Chroma query capability，也不增加 backend 开关。

## Boundary

FAISS/NumPy ExactDenseIndex 是进程内派生索引，不写 `.faiss`、`.index` 或第二份持久化向量真源；本阶段不修改 `runtime.py` 的 Production path。
"""


def validate_records(
    records: Sequence[tuple[DocumentChunk, list[float]]],
    chunks: Sequence[DocumentChunk],
) -> dict[str, Any]:
    """验证 Chroma embedding 读取结果与 active identity 的一一对应关系。"""

    record_chunks = [chunk for chunk, _ in records]
    vectors = [vector for _, vector in records]
    ids = [chunk.chunk_id for chunk in record_chunks]
    dimensions = sorted({len(vector) for vector in vectors})
    norms = [float(np.linalg.norm(np.asarray(vector, dtype=np.float64))) for vector in vectors]
    return {
        "record_count": len(records),
        "chunk_count_matches": len(record_chunks) == len(chunks),
        "unique_chunk_ids": len(ids) == len(set(ids)),
        "chunk_id_hash": sorted_chunk_id_hash(record_chunks) if record_chunks else None,
        "expected_chunk_id_hash": CHUNK_ID_HASH,
        "metadata_coverage": sum(bool(chunk.file_name and chunk.content_hash) for chunk in record_chunks) / len(record_chunks) if record_chunks else 0.0,
        "embedding_dimensions": dimensions,
        "embedding_dtype": "float32 at ExactDenseIndex boundary",
        "l2_normalized": bool(norms) and all(math.isclose(norm, 1.0, abs_tol=1e-4) for norm in norms),
        "min_l2_norm": min(norms) if norms else None,
        "max_l2_norm": max(norms) if norms else None,
    }


def numpy_reference(
    chunks: Sequence[DocumentChunk],
    embeddings: Sequence[Sequence[float]],
    query: Sequence[float],
    top_k: int,
) -> list[tuple[str, float]]:
    """用独立 NumPy 全量矩阵内积计算 exact reference。"""

    ordered = sorted(zip(chunks, embeddings), key=lambda item: item[0].chunk_id)
    matrix = np.asarray([embedding for _, embedding in ordered], dtype=np.float32)
    vector = np.asarray(query, dtype=np.float32)
    scores = matrix @ vector
    ranked = sorted(
        ((chunk.chunk_id, float(score)) for (chunk, _), score in zip(ordered, scores.tolist())),
        key=lambda item: (-item[1], item[0]),
    )
    return ranked[: min(top_k, len(ranked))]


def semantic_conformance(
    index: ExactDenseIndex,
    chunks: Sequence[DocumentChunk],
    embeddings: Sequence[Sequence[float]],
    query_vectors: Sequence[Sequence[float]],
) -> dict[str, Any]:
    """逐题比较 ExactDenseIndex 与独立 NumPy reference 的 IDs 和 float32 scores。"""

    mismatches: list[dict[str, Any]] = []
    max_abs_score_error = 0.0
    for query_index, vector in enumerate(query_vectors):
        actual = index.search(vector, top_k=min(WORKER_CANDIDATE_K, len(chunks)))
        expected = numpy_reference(chunks, embeddings, vector, WORKER_CANDIDATE_K)
        actual_pairs = [(item.chunk.chunk_id, item.similarity_score) for item in actual]
        if [item[0] for item in actual_pairs] != [item[0] for item in expected]:
            mismatches.append({"query_index": query_index, "kind": "ids"})
            continue
        for (_, actual_score), (_, expected_score) in zip(actual_pairs, expected):
            max_abs_score_error = max(max_abs_score_error, abs(actual_score - expected_score))
            if not math.isclose(actual_score, expected_score, rel_tol=0.0, abs_tol=1e-6):
                mismatches.append({"query_index": query_index, "kind": "score"})
                break
    return {
        "reference": "independent NumPy float32 matrix @ query with stable chunk_id tie-break",
        "query_count": len(query_vectors),
        "ids_and_scores_match": not mismatches,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
        "max_abs_score_error": max_abs_score_error,
        "backend": index.backend,
        "faiss_numpy_conformance": index.backend == "faiss.IndexFlatIP" and not mismatches,
    }


def load_exact_index(
    store: ChromaVectorStore,
    chunks: Sequence[DocumentChunk],
) -> tuple[ExactDenseIndex, list[DocumentChunk], list[list[float]], float, dict[str, Any]]:
    """从 active Chroma 读取完整 embedding 快照并构建进程内 ExactDenseIndex。"""

    started = time.perf_counter()
    records = list(store.iter_all_chunks_with_embeddings())
    record_validation = validate_records(records, chunks)
    if record_validation["record_count"] != CHUNK_COUNT:
        raise Phase15C1IdentityError("active Chroma embedding record count is not 817")
    if not record_validation["unique_chunk_ids"] or record_validation["chunk_id_hash"] != CHUNK_ID_HASH:
        raise Phase15C1IdentityError("active Chroma embedding IDs differ from frozen identity")
    if record_validation["metadata_coverage"] != 1.0:
        raise Phase15C1IdentityError("active Chroma metadata coverage is not 100%")
    if not record_validation["l2_normalized"]:
        raise Phase15C1IdentityError("active Chroma embeddings are not L2-normalized")
    index_chunks = [chunk for chunk, _ in records]
    embeddings = [embedding for _, embedding in records]
    index = ExactDenseIndex(index_chunks, embeddings)
    build_ms = (time.perf_counter() - started) * 1000.0
    record_validation["index_backend"] = index.backend
    record_validation["index_dimension"] = index.dimension
    record_validation["index_matrix_nbytes"] = index.matrix_nbytes
    return index, index_chunks, embeddings, build_ms, record_validation


def run_hnsw_audit(
    store: ChromaVectorStore,
    exact_pools: Mapping[str, Mapping[str, Sequence[Any]]],
    query_vectors: Mapping[str, Sequence[float]],
    question_ids: Sequence[str],
) -> dict[str, Any]:
    """以 Exact 为 ground truth，记录 HNSW Top20/30/40 的逐题召回和边界漂移。"""

    aggregate: dict[str, Any] = {}
    details: dict[str, Any] = {}
    for depth in CANDIDATE_KS:
        arm = str(depth)
        total_recall = 0.0
        per_query: dict[str, Any] = {}
        for qid in question_ids:
            exact_results = list(exact_pools[arm][qid])
            ann_results = store.search(query_vectors[qid], top_k=depth)
            exact_ids = [item.chunk.chunk_id for item in exact_results]
            ann_ids = [item.chunk.chunk_id for item in ann_results]
            exact_rank = {cid: rank for rank, cid in enumerate(exact_ids, 1)}
            ann_rank = {cid: rank for rank, cid in enumerate(ann_ids, 1)}
            exact_score = {item.chunk.chunk_id: item.similarity_score for item in exact_results}
            ann_score = {item.chunk.chunk_id: item.similarity_score for item in ann_results}
            rows = []
            for cid in sorted(set(exact_ids) | set(ann_ids)):
                rows.append(
                    {
                        "chunk_id": cid,
                        "exact_rank": exact_rank.get(cid),
                        "ann_rank": ann_rank.get(cid),
                        "entered": cid in ann_rank and cid not in exact_rank,
                        "dropped": cid in exact_rank and cid not in ann_rank,
                        "score_margin": (exact_score[cid] - ann_score[cid]) if cid in exact_score and cid in ann_score else None,
                    }
                )
            overlap = len(set(exact_ids) & set(ann_ids))
            recall = overlap / depth if depth else 0.0
            total_recall += recall
            per_query[qid] = {
                "exact_ids": exact_ids,
                "ann_ids": ann_ids,
                "ann_recall": recall,
                "rank_drift": rows,
            }
        aggregate[arm] = {
            "depth": depth,
            "mean_ann_recall": total_recall / len(question_ids),
            "query_count": len(question_ids),
            "mismatched_queries": [qid for qid, row in per_query.items() if row["ann_recall"] < 1.0],
        }
        details[arm] = per_query
    focus = {
        qid: {
            arm: details[arm].get(qid)
            for arm in ("20", "30", "40")
        }
        for qid in ("ARR-032", "ARR-033")
        if qid in question_ids
    }
    return {
        "schema_version": "phase15c1-hnsw-vs-exact-v1",
        "ground_truth": "ExactDenseIndex",
        "aggregate": aggregate,
        "focus_queries": focus,
        "per_query": details,
    }


def run_cross_process_worker(output_path: Path) -> int:
    """在一个 fresh process 中重建 Exact index、编码 query 并执行同一 Reranker。"""

    try:
        configure_offline_model_cache()
        settings = Settings()
        inputs = load_frozen_inputs()
        questions = inputs["questions_v1"]["questions"]
        store = ChromaVectorStore(settings.vector_db_path, settings.collection_name)
        chunks = validate_active_index(settings, inputs, store)
        index, _, _, _, _ = load_exact_index(store, chunks)
        embedding = BgeM3EmbeddingProvider(
            settings.embedding_model, device=settings.model_device
        )
        scorer = TransformersCrossEncoderScorer(
            resolve_local_reranker_model(settings),
            device=settings.model_device,
            dtype=settings.reranker_dtype,
            max_length=settings.reranker_max_length,
            inference_mode=False,
        )
        reranker = LearnedReranker(
            scorer,
            batch_size=settings.reranker_batch_size,
            max_candidates=WORKER_CANDIDATE_K,
        )
        payload: dict[str, Any] = {"status": "PASS", "candidate": {}, "top5": {}}
        for question in questions:
            qid = question["question_id"]
            query = question["english_query"]
            vector = embedding.embed_query(query)
            candidates = index.search(vector, top_k=WORKER_CANDIDATE_K)
            reranked = reranker.rerank(query, [item.chunk for item in candidates], top_k=FINAL_TOP_K)
            payload["candidate"][qid] = {
                "ids": [item.chunk.chunk_id for item in candidates],
                "scores": [item.similarity_score for item in candidates],
            }
            payload["top5"][qid] = [item.chunk.chunk_id for item in reranked]
        write_json(output_path, payload)
        return 0
    except Exception as exc:  # worker 输出结构化错误，主进程负责最终门禁判断
        write_json(
            output_path,
            {"status": "FAIL", "error_type": type(exc).__name__, "error": str(exc)},
        )
        return 1


def run_cross_process_replay(
    output_dir: Path,
    qrels: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    """并发启动 10 个 fresh process，比较 Candidate Top30、Top5 与 IR 指标。"""

    worker_dir = output_dir / "details" / "_cross_process"
    worker_dir.mkdir(parents=True, exist_ok=True)
    processes: list[tuple[subprocess.Popen[bytes], Path, Path]] = []
    for index in range(WORKER_COUNT):
        result_path = worker_dir / f"worker_{index:02d}.json"
        log_path = worker_dir / f"worker_{index:02d}.log"
        log_handle = log_path.open("wb")
        process = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "--worker", str(result_path)],
            cwd=str(ROOT),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            env={**os.environ, "TOKENIZERS_PARALLELISM": "false"},
        )
        processes.append((process, result_path, log_path))
        log_handle.close()
    exit_codes = [process.wait() for process, _, _ in processes]
    payloads = [read_json(result_path) for _, result_path, _ in processes if result_path.is_file()]
    successful = [payload for payload in payloads if payload.get("status") == "PASS"]
    errors = [payload for payload in payloads if payload.get("status") != "PASS"]
    candidate_order_equal = False
    candidate_scores_equal = False
    top5_equal = False
    metrics_equal = False
    if successful:
        reference = successful[0]
        candidate_order_equal = all(payload["candidate"] == reference["candidate"] for payload in successful)
        candidate_scores_equal = all(
            all(
                np.allclose(
                    payload["candidate"][qid]["scores"],
                    reference["candidate"][qid]["scores"],
                    rtol=0.0,
                    atol=1e-6,
                )
                for qid in reference["candidate"]
            )
            for payload in successful
        )
        top5_equal = all(payload["top5"] == reference["top5"] for payload in successful)
        metrics = [evaluate_ir_run(qrels, payload["top5"], depths=(1, 3, 5, 10, 20)) for payload in successful]
        metrics_equal = all(item == metrics[0] for item in metrics[1:])
    result = {
        "schema_version": "phase15c1-deterministic-replay-v1",
        "worker_count": WORKER_COUNT,
        "successful_workers": len(successful),
        "exit_codes": exit_codes,
        "candidate_top30_ids_equal": candidate_order_equal,
        "candidate_top30_scores_equal": candidate_scores_equal,
        "top5_ids_equal": top5_equal,
        "ir_metrics_equal": metrics_equal,
        "candidate_order_gate": len(successful) == WORKER_COUNT and candidate_order_equal,
        "top5_gate": len(successful) == WORKER_COUNT and top5_equal,
        "metrics_gate": len(successful) == WORKER_COUNT and metrics_equal,
        "errors": errors,
    }
    if successful:
        result["reference_candidate"] = successful[0]["candidate"]
        result["reference_top5"] = successful[0]["top5"]
    # worker 日志和逐进程 JSON 只是 scratch；主 artifact 已保留完整汇总和 mismatch 状态。
    shutil.rmtree(worker_dir, ignore_errors=True)
    return result


def choose_exact_k(
    candidate_rows: Mapping[str, Mapping[str, Any]],
    ir_summaries: Mapping[str, Mapping[str, Any]],
) -> tuple[int | None, str]:
    """按 Phase 15C.1 的 Case A/B/C 规则选择最小 exact candidate K。"""

    d30 = candidate_rows["30"]["INDEXABLE_ONLY"]
    d40 = candidate_rows["40"]["INDEXABLE_ONLY"]
    d20 = candidate_rows["20"]["INDEXABLE_ONLY"]
    same_30_40 = all(
        math.isclose(float(d30[f"{name}@30"]), float(d40[f"{name}@40"]), abs_tol=1e-12)
        for name in ("HitRate", "EvidenceAnchorRecall", "AllRequired")
    )
    d30_not_worse = all(
        float(ir_summaries["30"][name]) >= float(ir_summaries["40"][name])
        for name in ("Recall@5", "nDCG@5")
    )
    if same_30_40 and d30_not_worse:
        return 30, "Case A: E30 与 E40 coverage 相同，且 E30 主指标不低于 E40。"
    same_20_30 = all(
        math.isclose(float(d20[f"{name}@20"]), float(d30[f"{name}@30"]), abs_tol=1e-12)
        for name in ("HitRate", "EvidenceAnchorRecall", "AllRequired")
    )
    if same_20_30:
        return 20, "Case B: E20 已达到 E30 coverage。"
    if not same_30_40:
        return None, "Case C: E40 仍新增 meaningful coverage；本阶段不自动 sweep E50。"
    return None, "E20/E30/E40 未满足预注册的最小 K 决策条件。"


def cleanup_regenerable_files() -> dict[str, Any]:
    """只清理项目内可再生 bytecode/cache，不触碰模型、Chroma 或正式证据。"""

    protected = {
        (ROOT / ".cache").resolve(),
        (ROOT / "data" / "chroma").resolve(),
    }

    def is_protected(path: Path) -> bool:
        resolved = path.resolve()
        return any(resolved == root or root in resolved.parents for root in protected)

    removed_files: list[str] = []
    removed_dirs: list[str] = []
    all_paths = list(ROOT.rglob("*"))
    for path in sorted(all_paths, key=lambda item: len(item.parts), reverse=True):
        if is_protected(path):
            continue
        if path.is_file() and path.suffix in {".pyc", ".pyo"}:
            removed_files.append(str(path.relative_to(ROOT)).replace("\\", "/"))
            path.unlink()
        elif path.is_dir() and path.name in {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}:
            removed_dirs.append(str(path.relative_to(ROOT)).replace("\\", "/"))
            shutil.rmtree(path, ignore_errors=True)
    remaining = [
        str(path.relative_to(ROOT)).replace("\\", "/")
        for path in ROOT.rglob("*")
        if not is_protected(path) and path.is_file() and path.suffix in {".pyc", ".pyo"}
    ]
    orphan_indexes = [
        str(path.relative_to(ROOT)).replace("\\", "/")
        for path in ROOT.rglob("*")
        if not is_protected(path) and path.is_file() and path.suffix.lower() in {".faiss", ".index"}
    ]
    return {
        "schema_version": "phase15c1-cleanup-manifest-v1",
        "status": "PASS" if not remaining and not orphan_indexes else "FAIL",
        "removed_files": removed_files,
        "removed_directories": removed_dirs,
        "remaining_bytecode": remaining,
        "orphan_faiss_or_index_files": orphan_indexes,
        "protected_roots": [".cache", "data/chroma"],
    }


def render_report(payload: Mapping[str, Any]) -> str:
    """生成阶段结果、门禁和生产边界报告。"""

    gates = payload["gates"]
    exact = payload["exact_index"]
    lines = [
        "# EvidenceFlow Phase 15C.1 — Deterministic Exact Dense Architecture",
        "",
        f"**STATUS = {payload['status']}**",
        "",
        f"`EXACT_ARCHITECTURE_RECOMMENDED = {payload['exact_architecture_recommended']}`",
        f"`RECOMMENDED_EXACT_K = {payload['recommended_exact_k']}`",
        "",
        "## Architecture",
        "",
        f"- Chroma remains the only persistent source of truth: `{COLLECTION_NAME}` / {CHUNK_COUNT} chunks.",
        f"- Exact backend: `{exact['backend']}`; index is in-memory and not written to disk.",
        "- Production remains HNSW Dense Top20 → Learned Reranker → Top5; no runtime promotion was performed.",
        "",
        "## Gates",
        "",
        "| Gate | Result |",
        "|---|---|",
    ]
    for name, value in gates.items():
        lines.append(f"| {name} | {'PASS' if value else 'FAIL'} |")
    lines += [
        "",
        "## Exact metrics",
        "",
        "| Arm | P@5 | Recall@5 | MRR@5 | MAP@5 | nDCG@5 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for arm in ("20", "30", "40"):
        row = payload["final_ir_metrics"]["summary_at_5"][arm]
        lines.append(
            f"| E{arm} | {row['P@5']:.6f} | {row['Recall@5']:.6f} | {row['MRR@5']:.6f} | {row['MAP@5']:.6f} | {row['nDCG@5']:.6f} |"
        )
    lines += [
        "",
        "## HNSW audit",
        "",
        "Exact is the ground truth. See `hnsw_vs_exact.json` and `details/hnsw_rankings.json` for per-query rank drift and ARR-032/ARR-033 focus rows.",
        "",
        "## Determinism",
        "",
        f"Fresh processes: {payload['deterministic_replay']['successful_workers']}/{payload['deterministic_replay']['worker_count']}; candidate order gate = `{payload['deterministic_replay']['candidate_order_gate']}`; Top5 gate = `{payload['deterministic_replay']['top5_gate']}`.",
        "",
        "## Stop boundary",
        "",
        "This phase does not change Production defaults. If the result is STOP, the artifact is diagnostic evidence only; Phase 15C-R is not started.",
        "",
    ]
    if payload.get("stop_reason"):
        lines += [f"**Stop reason:** {payload['stop_reason']}", ""]
    return "\n".join(lines)


def run_phase(output_dir: Path, baseline_replay_path: Path) -> dict[str, Any]:
    """按 G0–G9 顺序执行 Phase 15C.1，并在硬门禁失败后保留完整诊断。"""

    configure_offline_model_cache()
    started = time.perf_counter()
    output_dir = output_dir if output_dir.is_absolute() else ROOT / output_dir
    baseline_replay_path = baseline_replay_path if baseline_replay_path.is_absolute() else ROOT / baseline_replay_path
    output_dir.mkdir(parents=True, exist_ok=True)
    inputs = load_frozen_inputs()
    settings = Settings()
    store = ChromaVectorStore(settings.vector_db_path, settings.collection_name)
    chunks = validate_active_index(settings, inputs, store)
    question_ids = [question["question_id"] for question in inputs["questions_v1"]["questions"]]
    questions_by_id = {question["question_id"]: question for question in inputs["questions_v1"]["questions"]}
    if not baseline_replay_path.is_file():
        raise Phase15C1IdentityError(f"G0 replay is missing: {baseline_replay_path}")
    actual_replay = read_json(baseline_replay_path)
    expected_replay = read_json(EXPECTED_REPLAY_PATH)
    projection = project_active_relevance(chunks, inputs["questions_v2"])
    replay_parity = compare_replay(expected_replay, actual_replay, PROJECTION_HASH, question_ids)
    before_frozen = phase_frozen_hashes()
    before_sources = source_hashes()
    baseline = {
        "schema_version": "phase15c1-baseline-manifest-v1",
        "phase": "15C.1",
        "status": "G0_PASS" if replay_parity["parity"] else "STOP",
        "current_production": "Chroma HNSW Dense Top20 -> BGE Learned Reranker -> Top5",
        "active_identity": {
            "collection_name": settings.collection_name,
            "document_count": len({chunk.document_id for chunk in chunks}),
            "chunk_count": len(chunks),
            "collection_metadata": store.metadata,
            "corpus_signature": inputs["active_manifest"]["corpus_signature"],
            "index_signature": inputs["active_manifest"]["index"]["index_signature"],
            "chunk_id_hash": sorted_chunk_id_hash(chunks),
            "active_relevance_projection_hash": projection["active_relevance_projection_hash"],
        },
        "d20_replay": {**replay_parity, "source": str(baseline_replay_path.relative_to(ROOT)).replace("\\", "/")},
        "gold_qrels_mapping": {
            "gold_mapping_hash": inputs["snapshot"]["frozen_artifacts"]["gold_mapping_hash"],
            "qrels_hash": inputs["snapshot"]["frozen_artifacts"]["qrels_hash"],
            "projection_hash": projection["active_relevance_projection_hash"],
        },
        "production_source_hashes_at_phase_start": before_sources,
        "historical_frozen_hashes": before_frozen,
        "query_count": len(question_ids),
    }
    write_json(output_dir / "baseline_manifest.json", baseline)
    (output_dir / "architecture_audit.md").write_text(architecture_audit_text(), encoding="utf-8")
    if not replay_parity["parity"]:
        result = {
            "schema_version": "phase15c1-execution-result-v1",
            "phase": "15C.1",
            "status": "STOP",
            "stop_reason": "G0 current Production D20 replay parity failed",
            "gates": {"G0_current_production_baseline": False},
            "production_mutated": False,
            "completed_at_epoch": time.time(),
        }
        write_json(output_dir / "execution_result.json", result)
        (output_dir / "report.md").write_text("# EvidenceFlow Phase 15C.1\n\n**STOP**\n\nG0 current Production replay parity failed.\n", encoding="utf-8")
        return result

    exact_index, index_chunks, embeddings, build_ms, record_validation = load_exact_index(store, chunks)
    write_json(
        output_dir / "exact_index_manifest.json",
        {
            "schema_version": "phase15c1-exact-index-manifest-v1",
            "backend": exact_index.backend,
            "persistent": False,
            "source_of_truth": "active Chroma collection",
            "chunk_count": exact_index.size,
            "dimension": exact_index.dimension,
            "chunk_id_order": "ASC",
            "similarity": "float32 inner product over L2-normalized vectors",
            "stable_order": "similarity_score DESC, chunk_id ASC",
            "build_ms": build_ms,
            "embedding_matrix_nbytes": exact_index.matrix_nbytes,
            "record_validation": record_validation,
        },
    )
    embedding_provider = BgeM3EmbeddingProvider(
        settings.embedding_model, device=settings.model_device
    )
    query_vectors_list = embedding_provider.embed_documents(
        [questions_by_id[qid]["english_query"] for qid in question_ids]
    )
    query_vectors = dict(zip(question_ids, query_vectors_list))
    conformance = semantic_conformance(exact_index, index_chunks, embeddings, query_vectors_list)
    write_json(output_dir / "details" / "exact_reference_conformance.json", conformance)

    pool_results: dict[str, dict[str, list[Any]]] = {str(k): {} for k in CANDIDATE_KS}
    pools: dict[str, dict[str, list[DocumentChunk]]] = {str(k): {} for k in CANDIDATE_KS}
    exact_rankings: dict[str, Any] = {}
    exact_latency: dict[str, dict[str, float]] = {str(k): {} for k in CANDIDATE_KS}
    for depth in CANDIDATE_KS:
        arm = str(depth)
        for qid in question_ids:
            started_search = time.perf_counter()
            results = exact_index.search(query_vectors[qid], top_k=depth)
            exact_latency[arm][qid] = (time.perf_counter() - started_search) * 1000.0
            if len(results) != depth:
                raise Phase15C1IdentityError(f"E{depth} returned {len(results)} candidates for {qid}")
            pool_results[arm][qid] = results
            pools[arm][qid] = [result.chunk for result in results]
        exact_rankings[arm] = {
            qid: [
                {"rank": result.rank, "chunk_id": result.chunk.chunk_id, "similarity_score": result.similarity_score}
                for result in pool_results[arm][qid]
            ]
            for qid in question_ids
        }
        write_json(output_dir / "candidates" / f"exact{arm}.json", candidate_artifact(arm, pool_results[arm], projection["qrels"], question_ids))
    write_json(output_dir / "details" / "exact_rankings.json", exact_rankings)

    candidate_metric_rows = {
        arm: candidate_metrics(pools[arm], projection["groups_by_question"], question_ids, int(arm))
        for arm in ("20", "30", "40")
    }
    write_json(
        output_dir / "candidate_metrics.json",
        {
            "schema_version": "phase15c1-candidate-metrics-v1",
            "active_relevance_projection_hash": PROJECTION_HASH,
            "search_backend": exact_index.backend,
            "arms": candidate_metric_rows,
        },
    )

    model_load_started = time.perf_counter()
    scorer = TransformersCrossEncoderScorer(
        resolve_local_reranker_model(settings),
        device=settings.model_device,
        dtype=settings.reranker_dtype,
        max_length=settings.reranker_max_length,
        inference_mode=False,
    )
    model_load_ms = (time.perf_counter() - model_load_started) * 1000.0
    reranker = LearnedReranker(scorer, batch_size=settings.reranker_batch_size, max_candidates=40)
    final_results: dict[str, dict[str, list[RerankedChunk]]] = {str(k): {} for k in CANDIDATE_KS}
    reranker_scores: dict[str, dict[str, dict[str, float]]] = {str(k): {} for k in CANDIDATE_KS}
    latency_rows: dict[str, list[dict[str, Any]]] = {str(k): [] for k in CANDIDATE_KS}
    for depth in CANDIDATE_KS:
        arm = str(depth)
        for qid in question_ids:
            started_rerank = time.perf_counter()
            all_results = reranker.rerank(
                questions_by_id[qid]["english_query"],
                pools[arm][qid],
                top_k=None,
            )
            rerank_e2e_ms = (time.perf_counter() - started_rerank) * 1000.0
            final_results[arm][qid] = all_results[:FINAL_TOP_K]
            reranker_scores[arm][qid] = {item.chunk.chunk_id: item.relevance_score for item in all_results}
            latency_rows[arm].append(
                {
                    "question_id": qid,
                    "exact_search_ms": exact_latency[arm][qid],
                    # 兼容既有 latency_summary 的聚合字段，同时保留更准确的 Exact 命名。
                    "dense_search_ms": exact_latency[arm][qid],
                    "reranker_pair_count": len(pools[arm][qid]),
                    "reranker_model_ms": float(reranker.last_timing_seconds.get("model_forward", 0.0)) * 1000.0,
                    "reranker_end_to_end_ms": rerank_e2e_ms,
                    "retrieval_total_ms": exact_latency[arm][qid] + rerank_e2e_ms,
                    "truncated_pair_count": reranker.last_truncated_pair_count,
                }
            )
        write_json(
            output_dir / "reranker" / f"exact{arm}_top5.json",
            {
                "schema_version": "phase15c1-reranker-top5-v1",
                "arm": arm,
                "per_query": [
                    {
                        "question_id": qid,
                        "candidate_count": len(pools[arm][qid]),
                        "rows": [
                            {
                                "rank": result.rank,
                                "chunk_id": result.chunk.chunk_id,
                                "document_id": result.chunk.document_id,
                                "candidate_rank": next(
                                    rank for rank, chunk in enumerate(pools[arm][qid], 1) if chunk.chunk_id == result.chunk.chunk_id
                                ),
                                "reranker_score": result.relevance_score,
                            }
                            for result in final_results[arm][qid]
                        ],
                    }
                    for qid in question_ids
                ],
            },
        )

    runs = {
        arm: {qid: [result.chunk.chunk_id for result in final_results[arm][qid]] for qid in question_ids}
        for arm in ("20", "30", "40")
    }
    ir_results = {
        arm: evaluate_ir_run(projection["qrels"], runs[arm], depths=(1, 3, 5, 10, 20))
        for arm in ("20", "30", "40")
    }
    ir_summaries = {arm: standard_summary(ir_results[arm]) for arm in ("20", "30", "40")}
    write_json(
        output_dir / "final_ir_metrics.json",
        {
            "schema_version": "phase15c1-final-ir-v1",
            "active_relevance_projection_hash": PROJECTION_HASH,
            "search_backend": exact_index.backend,
            "arms": ir_results,
            "summary_at_5": ir_summaries,
        },
    )
    failures = classify_failures(pools, final_results, projection["groups_by_question"], question_ids)
    write_json(output_dir / "failure_attribution.json", failures)
    latency_payload = {
        "schema_version": "phase15c1-latency-v1",
        "index_build_ms": build_ms,
        "embedding_matrix_nbytes": exact_index.matrix_nbytes,
        "model_load_ms": model_load_ms,
        "arms": {
            arm: {"per_query": latency_rows[arm], "aggregate": latency_summary(latency_rows[arm])}
            for arm in ("20", "30", "40")
        },
    }
    write_json(output_dir / "latency_metrics.json", latency_payload)

    hnsw_audit = run_hnsw_audit(store, pool_results, query_vectors, question_ids)
    write_json(output_dir / "hnsw_vs_exact.json", hnsw_audit)
    write_json(output_dir / "details" / "hnsw_rankings.json", hnsw_audit["per_query"])
    exact_d20_run = runs["20"]
    production_d20_parity = compare_d20_to_replay(actual_replay, exact_d20_run, ir_summaries["20"], question_ids)
    cross_process = run_cross_process_replay(output_dir, projection["qrels"])
    recommended_k, recommendation_reason = choose_exact_k(candidate_metric_rows, ir_summaries)
    recommended_k_value: int | str = (
        recommended_k if recommended_k is not None else "NEEDS_FURTHER_K_TEST"
    )

    frozen_after = phase_frozen_hashes()
    source_after = source_hashes()
    cleanup = cleanup_regenerable_files()
    write_json(output_dir / "cleanup_manifest.json", cleanup)
    gates = {
        "G0_current_production_baseline": replay_parity["parity"],
        "G1_exact_semantic_conformance": bool(conformance["faiss_numpy_conformance"]),
        "G2_cross_process_determinism": bool(cross_process["candidate_order_gate"] and cross_process["top5_gate"] and cross_process["metrics_gate"]),
        "G3_exact_quality_recomputed": all(len(ir_results[arm]["per_query"]) == len(question_ids) for arm in ("20", "30", "40")),
        "G4_candidate_k_decision": recommended_k in {20, 30},
        "G5_latency_measurement": all(math.isfinite(float(latency_payload["arms"][arm]["aggregate"]["retrieval_total_ms"]["p95"])) for arm in ("20", "30", "40")),
        "G6_active_index_identity": baseline["active_identity"]["index_signature"] == INDEX_SIGNATURE and baseline["active_identity"]["chunk_count"] == CHUNK_COUNT,
        "G7_frozen_evidence": before_frozen == frozen_after,
        "G8_production_behavior_replay": replay_parity["parity"],
        "G9_cleanup": cleanup["status"] == "PASS",
    }
    all_pass = all(gates.values())
    stop_reason = None
    if not conformance["faiss_numpy_conformance"]:
        stop_reason = "FAISS IndexFlatIP unavailable or did not match independent NumPy reference; NumPy exact fallback was evaluated but G1 is not PASS."
    elif not gates["G2_cross_process_determinism"]:
        stop_reason = "cross-process Exact Top30/Top5/metrics determinism failed"
    elif not all_pass:
        stop_reason = "one or more Phase 15C.1 hard gates failed"
    result = {
        "schema_version": "phase15c1-execution-result-v1",
        "phase": "15C.1",
        "status": "PASS" if all_pass else "STOP",
        "stop_reason": stop_reason,
        "duration_seconds": time.perf_counter() - started,
        "gates": gates,
        "exact_architecture_recommended": all_pass,
        "recommended_exact_k": recommended_k_value,
        "recommendation_reason": recommendation_reason,
        "production_mutated": False,
        "production_d20_replay": production_d20_parity,
        "deterministic_replay": cross_process,
        "frozen_hashes_unchanged": before_frozen == frozen_after,
        "source_hashes_after": source_after,
        "artifacts": [
            "baseline_manifest.json",
            "architecture_audit.md",
            "exact_index_manifest.json",
            "deterministic_replay.json",
            "hnsw_vs_exact.json",
            "candidate_metrics.json",
            "final_ir_metrics.json",
            "failure_attribution.json",
            "latency_metrics.json",
            "cleanup_manifest.json",
            "execution_result.json",
            "report.md",
        ],
        "completed_at_epoch": time.time(),
    }
    write_json(output_dir / "deterministic_replay.json", cross_process)
    # 将完整 payload 聚合到 report，同时避免 execution_result 复制大型逐题排名。
    report_payload = {
        **result,
        "exact_index": read_json(output_dir / "exact_index_manifest.json"),
        "final_ir_metrics": {"summary_at_5": ir_summaries},
    }
    (output_dir / "report.md").write_text(render_report(report_payload), encoding="utf-8")
    write_json(output_dir / "execution_result.json", result)
    return result


def parse_args() -> argparse.Namespace:
    """解析阶段目录、baseline replay 或 worker 模式。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--baseline-replay", type=Path, default=OUTPUT_DIR / "baseline_d20_replay.json")
    parser.add_argument("--worker", type=Path)
    return parser.parse_args()


def main() -> int:
    """运行主阶段或单个 fresh-process worker。"""

    args = parse_args()
    if args.worker is not None:
        return run_cross_process_worker(args.worker)
    try:
        result = run_phase(args.output_dir, args.baseline_replay)
    except Exception as exc:
        result = {
            "schema_version": "phase15c1-execution-result-v1",
            "phase": "15C.1",
            "status": "BLOCKED",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "production_mutated": False,
        }
        output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        write_json(output_dir / "execution_result.json", result)
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
