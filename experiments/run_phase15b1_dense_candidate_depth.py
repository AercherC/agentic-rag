"""执行 Phase 15B.1 Dense Candidate Depth Confirmation。

本实验只改变 Dense candidate K（20/30/40），在冻结的 active Chroma、BGE-M3、
Cross-Encoder、Gold projection 与最终 Top-5 合同上比较质量和延迟。脚本不修改
Production Runtime、Settings、active index 或历史冻结 artifact。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evidenceflow.config import Settings
from evidenceflow.domain.chunk import DocumentChunk
from evidenceflow.embedding.bge_m3 import BgeM3EmbeddingProvider
from evidenceflow.eval.comparison import paired_comparison
from evidenceflow.eval.ir_metrics import evaluate_ir_run
from evidenceflow.retrieval.reranker import (
    LearnedReranker,
    RerankedChunk,
    TransformersCrossEncoderScorer,
    rank_scored_candidates,
)
from evidenceflow.storage.chroma import ChromaVectorStore
from experiments.phase15b_active_relevance import project_active_relevance


OUTPUT_DIR = ROOT / "evals" / "evaluation_v3" / "phase15b1_dense_candidate_depth_v1"
SNAPSHOT_PATH = ROOT / "artifacts" / "phase15a" / "baseline_snapshot.json"
ACTIVE_MANIFEST_PATH = ROOT / "artifacts" / "phase15a" / "active_manifest.json"
QUESTIONS_V1_PATH = ROOT / "evals" / "research_retrieval" / "research_questions_v1.json"
QUESTIONS_V2_PATH = (
    ROOT
    / "evals"
    / "evaluation_v2"
    / "phase35_gold_stabilization_v1"
    / "research_questions_v2_gold_stable.json"
)
MAPPING_PATH = (
    ROOT
    / "evals"
    / "evaluation_v2"
    / "phase35_gold_stabilization_v1"
    / "gold_mapping_v3.json"
)
QRELS_PATH = ROOT / "evals" / "evaluation_v3" / "standard_ir_v1" / "qrels.json"
QRELS_MANIFEST_PATH = ROOT / "evals" / "evaluation_v3" / "standard_ir_v1" / "qrels_manifest.json"
EXPECTED_REPLAY_PATH = (
    ROOT
    / "evals"
    / "evaluation_v3"
    / "phase15b_hybrid_candidate_v2"
    / "production_reranked_baseline.json"
)
DEFAULT_REPLAY_PATH = OUTPUT_DIR / "production_replay_raw.json"

COLLECTION_NAME = "ef_research_bge_m3_hybrid_c512_v2"
CORPUS_SIGNATURE = "bb184726b7f98756bd519d70e1a2a1f1224b5108e689680b2c328023ba1f4332"
INDEX_SIGNATURE = "b4e9432c7170615ade2c53803a887fa86d0f793949934a57db421f0da57739e5"
CHUNK_ID_HASH = "e87a613ebe65b10262996e83974b55839e0b6c27fbe22053ee7b8826dd1f153d"
PROJECTION_HASH = "6ccf0ffeffb8799429baff192e2de75e57199d1868688ca75ac2479cfb0cc4cd"
QUESTION_COUNT = 40
DOCUMENT_COUNT = 10
CHUNK_COUNT = 817
FINAL_TOP_K = 5
CANDIDATE_KS = (20, 30, 40)
BOOTSTRAP_SEED = 20260905
BOOTSTRAP_SAMPLES = 10_000


class ExperimentIdentityError(RuntimeError):
    """表示冻结输入、active index 或 replay parity 不满足实验合同。"""


def read_json(path: Path) -> Any:
    """读取 UTF-8 JSON；所有实验输入均从本地冻结文件取得。"""

    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    """用稳定缩进写出 artifact，便于审计与复算。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    """计算文件的字节级 SHA-256，不对内容做文本重编码。"""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def configure_offline_model_cache() -> None:
    """将模型加载锁定到项目缓存，防止实验隐式改变网络依赖。"""

    os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def resolve_local_reranker_model(settings: Settings) -> str:
    """优先使用已缓存的同一 checkpoint，保持模型身份不变。"""

    local = ROOT / ".cache" / "models" / "bge-reranker-base"
    if (local / "config.json").is_file() and (local / "model.safetensors").is_file():
        return str(local)
    return settings.reranker_model


def sorted_chunk_id_hash(chunks: Sequence[DocumentChunk]) -> str:
    """按 Phase 15A 的紧凑 JSON 数组规则计算 active Chunk ID Hash。"""

    serialized = json.dumps(
        sorted(chunk.chunk_id for chunk in chunks),
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def frozen_tree_hashes() -> dict[str, str]:
    """记录 Phase 15A/15B/15B.5 树内文件 Hash，证明实验未改冻结证据。"""

    roots = (
        ROOT / "artifacts" / "phase15a",
        ROOT / "evals" / "evaluation_v3" / "phase15b_hybrid_candidate_v2",
        ROOT / "evals" / "evaluation_v3" / "phase15b5_project_consolidation_v1",
    )
    output: dict[str, str] = {}
    for root in roots:
        if not root.is_dir():
            raise ExperimentIdentityError(f"frozen artifact root is missing: {root}")
        for path in sorted(p for p in root.rglob("*") if p.is_file()):
            output[str(path.relative_to(ROOT))] = sha256_file(path)
    return output


def production_source_hashes() -> dict[str, str]:
    """记录本实验禁止修改的 Production source Hash。"""

    paths = (
        ROOT / "src" / "evidenceflow" / "config.py",
        ROOT / "src" / "evidenceflow" / "runtime.py",
        ROOT / "src" / "evidenceflow" / "retrieval" / "retriever.py",
        ROOT / "src" / "evidenceflow" / "retrieval" / "reranker.py",
        ROOT / "src" / "evidenceflow" / "storage" / "chroma.py",
    )
    return {str(path.relative_to(ROOT)): sha256_file(path) for path in paths}


def load_frozen_inputs() -> dict[str, Any]:
    """读取并校验 Phase 15A Gold、qrels、mapping 与问题集合身份。"""

    snapshot = read_json(SNAPSHOT_PATH)
    active_manifest = read_json(ACTIVE_MANIFEST_PATH)
    questions_v1 = read_json(QUESTIONS_V1_PATH)
    questions_v2 = read_json(QUESTIONS_V2_PATH)
    mapping = read_json(MAPPING_PATH)
    canonical_qrels = read_json(QRELS_PATH)
    qrels_manifest = read_json(QRELS_MANIFEST_PATH)
    if snapshot.get("status") != "PASS":
        raise ExperimentIdentityError("Phase 15A snapshot is not PASS")
    if snapshot["authoritative_identity"]["collection_name"] != COLLECTION_NAME:
        raise ExperimentIdentityError("Phase 15A active collection identity mismatch")
    if active_manifest["index"]["collection_name"] != COLLECTION_NAME:
        raise ExperimentIdentityError("active manifest collection identity mismatch")
    ids_v1 = [q["question_id"] for q in questions_v1["questions"]]
    ids_v2 = [q["question_id"] for q in questions_v2["questions"]]
    if len(ids_v1) != QUESTION_COUNT or ids_v1 != ids_v2:
        raise ExperimentIdentityError("frozen question set must contain the same 40 IDs")
    if mapping.get("mapping_version") != "gold-mapper-v3":
        raise ExperimentIdentityError("gold mapping is not gold-mapper-v3")
    if not isinstance(canonical_qrels, dict) or set(canonical_qrels) != set(ids_v1):
        raise ExperimentIdentityError("canonical qrels query space mismatch")
    if qrels_manifest.get("qrels_hash") != snapshot["frozen_artifacts"]["qrels_hash"]:
        raise ExperimentIdentityError("canonical qrels hash mismatch")
    if sha256_file(MAPPING_PATH) != snapshot["frozen_artifacts"]["gold_mapping_hash"]:
        raise ExperimentIdentityError("gold mapping hash mismatch")
    for relative, expected in snapshot["frozen_artifacts"]["gold_and_qrels_files_sha256"].items():
        if sha256_file(ROOT / relative) != expected:
            raise ExperimentIdentityError(f"frozen input hash mismatch: {relative}")
    return {
        "snapshot": snapshot,
        "active_manifest": active_manifest,
        "questions_v1": questions_v1,
        "questions_v2": questions_v2,
        "mapping": mapping,
        "canonical_qrels": canonical_qrels,
        "qrels_manifest": qrels_manifest,
    }


def validate_active_index(settings: Settings, inputs: Mapping[str, Any], store: ChromaVectorStore) -> list[DocumentChunk]:
    """只读验证 active Collection 的 metadata、数量、文档数与 Chunk ID Hash。"""

    snapshot = inputs["snapshot"]
    if settings.collection_name != COLLECTION_NAME or store.collection_name != COLLECTION_NAME:
        raise ExperimentIdentityError("Settings or Chroma collection name mismatch")
    expected_metadata = snapshot["authoritative_identity"]["chroma_active_collection_metadata"]
    if store.metadata != expected_metadata:
        raise ExperimentIdentityError("active Chroma metadata mismatch")
    chunks = list(store.iter_all_chunks())
    if len(chunks) != CHUNK_COUNT or store.count != CHUNK_COUNT:
        raise ExperimentIdentityError("active Chroma must contain exactly 817 chunks")
    if len({chunk.chunk_id for chunk in chunks}) != CHUNK_COUNT:
        raise ExperimentIdentityError("active Chroma contains duplicate Chunk IDs")
    if len({chunk.document_id for chunk in chunks}) != DOCUMENT_COUNT:
        raise ExperimentIdentityError("active Chroma must contain exactly 10 documents")
    if sorted_chunk_id_hash(chunks) != CHUNK_ID_HASH:
        raise ExperimentIdentityError("active Chroma Chunk ID Hash mismatch")
    active_manifest = inputs["active_manifest"]
    if active_manifest["corpus_signature"] != CORPUS_SIGNATURE:
        raise ExperimentIdentityError("active corpus signature mismatch")
    if active_manifest["index"]["index_signature"] != INDEX_SIGNATURE:
        raise ExperimentIdentityError("active index signature mismatch")
    return chunks


def build_input_manifest(
    settings: Settings,
    store: ChromaVectorStore,
    chunks: Sequence[DocumentChunk],
    inputs: Mapping[str, Any],
    projection_hash: str,
    replay_path: Path,
) -> dict[str, Any]:
    """生成 Phase 15B.1 冻结输入 manifest；candidate K 仅作为实验参数记录。"""

    snapshot = inputs["snapshot"]
    frozen = snapshot["frozen_artifacts"]
    files = {
        "gold_mapping_v3": MAPPING_PATH,
        "research_questions_v2": QUESTIONS_V2_PATH,
        "standard_qrels": QRELS_PATH,
        "qrels_manifest": QRELS_MANIFEST_PATH,
        "expected_production_replay": EXPECTED_REPLAY_PATH,
        "actual_production_replay": replay_path,
    }
    return {
        "schema_version": "phase15b1-input-manifest-v1",
        "phase": "15B.1",
        "status": "FROZEN_INPUTS_VALIDATED",
        "collection_name": COLLECTION_NAME,
        "active_collection_metadata": store.metadata,
        "corpus_signature": CORPUS_SIGNATURE,
        "index_signature": INDEX_SIGNATURE,
        "document_count": len({chunk.document_id for chunk in chunks}),
        "chunk_count": len(chunks),
        "chunk_id_hash": CHUNK_ID_HASH,
        "active_relevance_projection_hash": projection_hash,
        "candidate_k_arms": list(CANDIDATE_KS),
        "embedding": {
            "model": settings.embedding_model,
            "version": snapshot["pipeline"]["embedding_version"],
            "normalization": "explicit_l2",
            "query_field": "english_query from research_questions_v1",
        },
        "reranker": {
            "model": settings.reranker_model,
            "checkpoint": snapshot["pipeline"]["reranker_checkpoint"],
            "max_length": settings.reranker_max_length,
            "batch_size": settings.reranker_batch_size,
            "device": settings.model_device,
            "dtype": settings.reranker_dtype,
            "inference_mode": False,
            "final_top_k": FINAL_TOP_K,
        },
        "gold_qrels_mapping": {
            "gold_mapping_hash": frozen["gold_mapping_hash"],
            "qrels_hash": frozen["qrels_hash"],
            "active_relevance_projection_hash": projection_hash,
            "canonical_qrels_not_rewritten": True,
            "file_sha256": {name: sha256_file(path) for name, path in files.items()},
        },
        "question_ids": [q["question_id"] for q in inputs["questions_v1"]["questions"]],
        "replay_source": str(replay_path.relative_to(ROOT)),
    }


def group_covered(group: Mapping[str, Any], ids: set[str]) -> bool:
    """判断候选/最终集合是否覆盖 required evidence group 的任一 anchor。"""

    return bool(set(group["matched_chunk_ids"]) & ids)


def candidate_metrics(
    pools: Mapping[str, Sequence[DocumentChunk]],
    groups_by_question: Mapping[str, Sequence[Mapping[str, Any]]],
    question_ids: Sequence[str],
    depth: int,
) -> dict[str, Any]:
    """计算 Candidate HitRate、AnchorRecall 与 AllRequired 两个 cohort。"""

    rows: dict[str, dict[str, Any]] = {}
    all_total = all_covered = idx_total = idx_covered = 0
    for qid in question_ids:
        groups = list(groups_by_question[qid])
        ids = {chunk.chunk_id for chunk in pools[qid][:depth]}
        flags = [group_covered(group, ids) for group in groups]
        idx_flags = [group["indexable"] for group in groups]
        covered_idx = [covered for covered, indexable in zip(flags, idx_flags) if indexable]
        all_total += len(groups)
        all_covered += sum(flags)
        idx_total += sum(idx_flags)
        idx_covered += sum(covered_idx)
        rows[qid] = {
            "question_id": qid,
            "candidate_count": len(pools[qid]),
            "covered_groups": sum(flags),
            "total_groups": len(groups),
            "indexable_groups": sum(idx_flags),
            "covered_indexable_groups": sum(covered_idx),
            "any_evidence": bool(any(flags)),
            "all_required": bool(flags) and all(flags),
            "any_indexable_evidence": bool(any(c and i for c, i in zip(flags, idx_flags))),
            "all_indexable_required": bool(sum(idx_flags)) and all(covered_idx),
            "indexable": bool(sum(idx_flags)),
        }
    indexable_qids = [qid for qid in question_ids if rows[qid]["indexable"]]

    def cohort(name: str) -> dict[str, Any]:
        selected = question_ids if name == "ALL_GOLD" else indexable_qids
        hit_key = "any_evidence" if name == "ALL_GOLD" else "any_indexable_evidence"
        required_key = "all_required" if name == "ALL_GOLD" else "all_indexable_required"
        denominator = len(selected)
        total = all_total if name == "ALL_GOLD" else idx_total
        covered = all_covered if name == "ALL_GOLD" else idx_covered
        return {
            "eligible_query_count": denominator,
            "excluded_question_ids": [qid for qid in question_ids if qid not in selected],
            f"HitRate@{depth}": sum(rows[qid][hit_key] for qid in selected) / denominator if denominator else 0.0,
            f"EvidenceAnchorRecall@{depth}": covered / total if total else 0.0,
            f"AllRequired@{depth}": sum(rows[qid][required_key] for qid in selected) / denominator if denominator else 0.0,
            "covered_anchor_count": covered,
            "total_anchor_count": total,
        }

    return {"depth": depth, "per_query": rows, "ALL_GOLD": cohort("ALL_GOLD"), "INDEXABLE_ONLY": cohort("INDEXABLE_ONLY")}


def candidate_artifact(
    arm: str,
    pool_results: Mapping[str, Sequence[Any]],
    qrels: Mapping[str, Sequence[str]],
    question_ids: Sequence[str],
) -> dict[str, Any]:
    """序列化 Dense candidate rank、相似度和 Gold membership。"""

    rows: list[dict[str, Any]] = []
    per_query: list[dict[str, Any]] = []
    for qid in question_ids:
        query_rows = []
        for result in pool_results[qid]:
            query_rows.append(
                {
                    "question_id": qid,
                    "arm": arm,
                    "candidate_rank": result.rank,
                    "candidate_count": len(pool_results[qid]),
                    "chunk_id": result.chunk.chunk_id,
                    "document_id": result.chunk.document_id,
                    "similarity_score": result.similarity_score,
                    "is_gold": result.chunk.chunk_id in set(qrels.get(qid, [])),
                }
            )
        rows.extend(query_rows)
        per_query.append({"question_id": qid, "candidate_count": len(query_rows), "chunk_ids": [r["chunk_id"] for r in query_rows]})
    return {
        "schema_version": "phase15b1-candidates-v1",
        "arm": arm,
        "label": f"D{arm}",
        "candidate_pool_contract": "Dense ANN rank only; no score fusion or normalization",
        "rows": rows,
        "per_query": per_query,
    }


def reranker_artifact(
    arm: str,
    results: Mapping[str, Sequence[RerankedChunk]],
    pools: Mapping[str, Sequence[DocumentChunk]],
    question_ids: Sequence[str],
) -> dict[str, Any]:
    """序列化 Learned Reranker Top-5 与候选 rank。"""

    per_query = []
    for qid in question_ids:
        candidate_rank = {chunk.chunk_id: i for i, chunk in enumerate(pools[qid], 1)}
        rows = [
            {
                "question_id": qid,
                "arm": arm,
                "rank": result.rank,
                "chunk_id": result.chunk.chunk_id,
                "document_id": result.chunk.document_id,
                "candidate_rank": candidate_rank[result.chunk.chunk_id],
                "reranker_score": result.relevance_score,
            }
            for result in results[qid]
        ]
        per_query.append({"question_id": qid, "candidate_count": len(pools[qid]), "final_count": len(rows), "rows": rows})
    return {
        "schema_version": "phase15b1-reranker-top5-v1",
        "arm": arm,
        "label": f"D{arm}",
        "reranker_only_reorders_candidates": True,
        "per_query": per_query,
    }


def standard_summary(ir: Mapping[str, Any]) -> dict[str, Any]:
    """提取标准 Evaluation v3 的 @5 指标。"""

    aggregate = ir["aggregate"]
    return {
        "P@5": aggregate["precision_at_k"][5],
        "Recall@5": aggregate["recall_at_k"][5],
        "MRR@5": aggregate["mrr_at_k"][5],
        "MAP@5": aggregate["map_at_k"][5],
        "nDCG@5": aggregate["ndcg_at_k"][5],
        "eligible_count": aggregate["eligible_count"],
        "excluded_count": aggregate["excluded_count"],
    }


def classify_failures(
    pools: Mapping[str, Sequence[DocumentChunk]],
    final_results: Mapping[str, Sequence[RerankedChunk]],
    groups_by_question: Mapping[str, Sequence[Mapping[str, Any]]],
    question_ids: Sequence[str],
) -> dict[str, Any]:
    """按 UNINDEXABLE/CANDIDATE_MISS/RERANKER_MISS/SUCCESS 归因每题状态。"""

    by_arm: dict[str, Any] = {}
    for arm in pools:
        counts = {name: 0 for name in ("UNINDEXABLE", "CANDIDATE_MISS", "RERANKER_MISS", "SUCCESS")}
        rows = []
        for qid in question_ids:
            groups = list(groups_by_question[qid])
            candidate_ids = {chunk.chunk_id for chunk in pools[arm][qid]}
            final_ids = {result.chunk.chunk_id for result in final_results[arm][qid]}
            unindexable = [g["group_id"] for g in groups if not g["indexable"]]
            candidate_miss = [g["group_id"] for g in groups if g["indexable"] and not group_covered(g, candidate_ids)]
            reranker_miss = [g["group_id"] for g in groups if g["indexable"] and group_covered(g, candidate_ids) and not group_covered(g, final_ids)]
            if unindexable:
                status = "UNINDEXABLE"
            elif candidate_miss:
                status = "CANDIDATE_MISS"
            elif reranker_miss:
                status = "RERANKER_MISS"
            else:
                status = "SUCCESS"
            counts[status] += 1
            rows.append(
                {
                    "question_id": qid,
                    "status": status,
                    "unindexable_group_ids": unindexable,
                    "candidate_miss_group_ids": candidate_miss,
                    "reranker_miss_group_ids": reranker_miss,
                    "candidate_count": len(pools[arm][qid]),
                    "final_chunk_ids": [r.chunk.chunk_id for r in final_results[arm][qid]],
                }
            )
        by_arm[arm] = {"label": f"D{arm}", "counts": counts, "per_query": rows}
    return {"schema_version": "phase15b1-failure-attribution-v1", "arms": by_arm}


def transitions(failures: Mapping[str, Any], question_ids: Sequence[str]) -> dict[str, Any]:
    """记录 D20→D30 与 D30→D40 的状态转移和可利用证据数量。"""

    rows = []
    for left, right in (("20", "30"), ("30", "40")):
        left_rows = {r["question_id"]: r for r in failures["arms"][left]["per_query"]}
        right_rows = {r["question_id"]: r for r in failures["arms"][right]["per_query"]}
        counts: dict[str, int] = {}
        for qid in question_ids:
            key = f"{left_rows[qid]['status']}→{right_rows[qid]['status']}"
            counts[key] = counts.get(key, 0) + 1
            rows.append(
                {
                    "from_arm": f"D{left}",
                    "to_arm": f"D{right}",
                    "question_id": qid,
                    "from_status": left_rows[qid]["status"],
                    "to_status": right_rows[qid]["status"],
                    "recovered_candidate_miss_groups": sorted(
                        set(left_rows[qid]["candidate_miss_group_ids"]) - set(right_rows[qid]["candidate_miss_group_ids"])
                    ),
                    "recovered_and_used_by_reranker": right_rows[qid]["status"] == "SUCCESS" and left_rows[qid]["status"] == "CANDIDATE_MISS",
                }
            )
        rows.append({"from_arm": f"D{left}", "to_arm": f"D{right}", "aggregate_transition_counts": counts})
    return {"schema_version": "phase15b1-transition-analysis-v1", "rows": rows}


def percentile(values: Sequence[float], p: float) -> float:
    """用线性插值计算确定性的 percentile；样本为空时返回 0。"""

    if not values:
        return 0.0
    ordered = sorted(float(v) for v in values)
    position = (len(ordered) - 1) * p / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def latency_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """聚合每题 dense/reranker/total latency 的 mean、p50、p95、max。"""

    keys = ("dense_search_ms", "reranker_model_ms", "reranker_end_to_end_ms", "retrieval_total_ms")
    aggregate: dict[str, Any] = {}
    for key in keys:
        values = [float(row[key]) for row in rows]
        aggregate[key] = {
            "mean": sum(values) / len(values) if values else 0.0,
            "p50": percentile(values, 50),
            "p95": percentile(values, 95),
            "max": max(values) if values else 0.0,
        }
    aggregate["reranker_pair_count"] = {
        "total": sum(int(row["reranker_pair_count"]) for row in rows),
        "mean": sum(int(row["reranker_pair_count"]) for row in rows) / len(rows) if rows else 0.0,
    }
    return aggregate


def per_query_candidate_value(row: Mapping[str, Any], cohort: str, metric: str) -> float:
    """将候选指标行转换成 paired comparison 使用的逐题值。"""

    if cohort == "INDEXABLE_ONLY" and not row["indexable"]:
        raise KeyError("excluded indexable-only query")
    if metric == "HitRate":
        return float(row["any_evidence"] if cohort == "ALL_GOLD" else row["any_indexable_evidence"])
    if metric == "AllRequired":
        return float(row["all_required"] if cohort == "ALL_GOLD" else row["all_indexable_required"])
    if metric == "EvidenceAnchorRecall":
        total = row["total_groups"] if cohort == "ALL_GOLD" else row["indexable_groups"]
        covered = row["covered_groups"] if cohort == "ALL_GOLD" else row["covered_indexable_groups"]
        return covered / total if total else 0.0
    raise ValueError(f"unsupported candidate metric: {metric}")


def bootstrap_comparisons(
    candidate_metrics_by_arm: Mapping[str, Mapping[str, Any]],
    ir_results: Mapping[str, Mapping[str, Any]],
    question_ids: Sequence[str],
    questions_v2: Mapping[str, Any],
) -> dict[str, Any]:
    """生成固定 seed 的题族 bootstrap CI、paired delta 与 wins/ties/losses。"""

    families = {q["question_id"]: q.get("family_id", q["question_id"]) for q in questions_v2["questions"]}
    output: dict[str, Any] = {
        "schema_version": "phase15b1-bootstrap-v1",
        "seed": BOOTSTRAP_SEED,
        "samples": BOOTSTRAP_SAMPLES,
        "unit": "question_family",
        "comparisons": {},
    }

    def add(name: str, baseline: Mapping[str, float], variant: Mapping[str, float]) -> None:
        output["comparisons"][name] = paired_comparison(
            dict(baseline),
            dict(variant),
            families={qid: families[qid] for qid in baseline},
            seed=BOOTSTRAP_SEED,
            samples=BOOTSTRAP_SAMPLES,
        )

    for left, right in (("20", "30"), ("20", "40"), ("30", "40")):
        for cohort in ("ALL_GOLD", "INDEXABLE_ONLY"):
            for metric in ("HitRate", "EvidenceAnchorRecall", "AllRequired"):
                base = {}
                variant = {}
                for qid in question_ids:
                    row_left = candidate_metrics_by_arm[left]["per_query"][qid]
                    row_right = candidate_metrics_by_arm[right]["per_query"][qid]
                    if cohort == "INDEXABLE_ONLY" and not row_left["indexable"]:
                        continue
                    base[qid] = per_query_candidate_value(row_left, cohort, metric)
                    variant[qid] = per_query_candidate_value(row_right, cohort, metric)
                add(f"candidate:{cohort}:{metric}:D{left}_vs_D{right}", base, variant)
        for metric_key, metric_name in (("recall_at_5", "Recall@5"), ("ndcg_at_5", "nDCG@5"), ("mrr_at_5", "MRR@5"), ("map_at_5", "MAP@5")):
            base = {qid: float(ir_results[left]["per_query"][qid]["metrics"][metric_key]) for qid in question_ids if ir_results[left]["per_query"][qid]["eligible"]}
            variant = {qid: float(ir_results[right]["per_query"][qid]["metrics"][metric_key]) for qid in base}
            add(f"final:{metric_name}:D{left}_vs_D{right}", base, variant)
    return output


def compare_replay(expected: Mapping[str, Any], actual: Mapping[str, Any], projection_hash: str, question_ids: Sequence[str]) -> dict[str, Any]:
    """比较独立 replay 与 Phase 15B 冻结 oracle，作为 G1 前置门。"""

    summary_keys = ("P@5", "Recall@5", "MRR@5", "MAP@5", "nDCG@5")
    expected_run = expected.get("final_run")
    actual_run = actual.get("final_run")
    run_mismatches = [qid for qid in question_ids if not isinstance(expected_run, Mapping) or not isinstance(actual_run, Mapping) or list(expected_run.get(qid, [])) != list(actual_run.get(qid, []))]
    summary_mismatches = [key for key in summary_keys if not math.isclose(float(expected.get("summary_at_5", {}).get(key, math.nan)), float(actual.get("summary_at_5", {}).get(key, math.nan)), rel_tol=0.0, abs_tol=1e-12)]
    identity_ok = all(actual.get(key) == value for key, value in {"collection_name": COLLECTION_NAME, "corpus_signature": CORPUS_SIGNATURE, "index_signature": INDEX_SIGNATURE, "chunk_id_hash": CHUNK_ID_HASH}.items())
    query_set_matches = isinstance(actual_run, Mapping) and set(actual_run) == set(question_ids)
    return {
        "parity": bool(actual.get("status") == "PASS" and expected.get("status") == "PASS" and actual.get("pipeline") == expected.get("pipeline") and actual.get("active_relevance_projection_hash") == projection_hash and identity_ok and query_set_matches and not run_mismatches and not summary_mismatches),
        "expected_projection_hash": expected.get("active_relevance_projection_hash"),
        "actual_projection_hash": actual.get("active_relevance_projection_hash"),
        "pipeline_matches": actual.get("pipeline") == expected.get("pipeline"),
        "identity_matches": identity_ok,
        "query_set_matches": query_set_matches,
        "run_mismatches": run_mismatches,
        "summary_mismatches": summary_mismatches,
        "compared_question_count": len(question_ids),
    }


def compare_d20_to_replay(replay: Mapping[str, Any], d20_run: Mapping[str, Sequence[str]], d20_summary: Mapping[str, Any], question_ids: Sequence[str]) -> dict[str, Any]:
    """验证实验 D20 与已通过 G1 的 Production replay 逐题完全一致。"""

    mismatches = [qid for qid in question_ids if list(replay["final_run"].get(qid, [])) != list(d20_run.get(qid, []))]
    summary_mismatches = [key for key in ("P@5", "Recall@5", "MRR@5", "MAP@5", "nDCG@5") if not math.isclose(float(replay["summary_at_5"][key]), float(d20_summary[key]), rel_tol=0.0, abs_tol=1e-12)]
    return {"parity": not mismatches and not summary_mismatches, "run_mismatches": mismatches, "summary_mismatches": summary_mismatches, "compared_question_count": len(question_ids)}


def choose_recommendation(summaries: Mapping[str, Mapping[str, Any]], candidate_by_arm: Mapping[str, Mapping[str, Any]]) -> tuple[int, str]:
    """按预注册预算规则选择最小有效 K，不引入任意显著性阈值。"""

    # Candidate coverage 相同时，D30 若在最终主指标上不低于 D40，
    # 直接满足“最小有效 K”；不引入任意 0.005 阈值或把更差的 D40 误选为推荐。
    d30 = candidate_by_arm["30"]["INDEXABLE_ONLY"]
    d40 = candidate_by_arm["40"]["INDEXABLE_ONLY"]
    candidate_equal = all(
        math.isclose(float(d30[f"{name}@30"]), float(d40[f"{name}@40"]), rel_tol=0.0, abs_tol=1e-12)
        for name in ("HitRate", "EvidenceAnchorRecall", "AllRequired")
    )
    final_not_worse = all(float(summaries["30"][metric]) >= float(summaries["40"][metric]) for metric in ("Recall@5", "nDCG@5"))
    if candidate_equal and final_not_worse:
        return 30, "D30 与 D40 的 Candidate INDEXABLE_ONLY coverage 相同，且 D30 的最终主指标不低于 D40"
    return 40, "D40 在 Candidate coverage 或最终主指标上仍优于 D30；保留较深候选池"


def render_report(payload: Mapping[str, Any]) -> str:
    """生成包含 STATUS、PAIRWISE DELTA、TRANSITIONS 和推荐结论的报告。"""

    lines = [
        "# EvidenceFlow Phase 15B.1 Dense Candidate Depth Confirmation",
        "",
        "本实验只改变 Dense candidate K=20/30/40；未修改 Production Runtime、全局 Settings、active Chroma、Gold 或 qrels。",
        "",
        "## STATUS",
        "",
        f"- STATUS: **{payload['status']}**",
        f"- Production replay parity: `{payload['gates'].get('production_replay_parity')}`; D20 parity: `{payload['gates'].get('d20_production_parity')}`.",
        f"- Frozen identity: `{payload['gates'].get('identity_and_frozen_inputs')}`; deterministic checks: `{payload['gates'].get('determinism')}`.",
        "",
        "## Candidate metrics",
        "",
        "| Arm | HitRate ALL | AnchorRecall ALL | AllRequired ALL | HitRate INDEXABLE | AnchorRecall INDEXABLE | AllRequired INDEXABLE |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in ("20", "30", "40"):
        row = payload["candidate_metrics"][arm]
        lines.append(
            f"| D{arm} | {row['ALL_GOLD'][f'HitRate@{arm}']:.4f} | {row['ALL_GOLD'][f'EvidenceAnchorRecall@{arm}']:.4f} | {row['ALL_GOLD'][f'AllRequired@{arm}']:.4f} | "
            f"{row['INDEXABLE_ONLY'][f'HitRate@{arm}']:.4f} | {row['INDEXABLE_ONLY'][f'EvidenceAnchorRecall@{arm}']:.4f} | {row['INDEXABLE_ONLY'][f'AllRequired@{arm}']:.4f} |"
        )
    lines += ["", "## Final IR metrics @5", "", "| Arm | P@5 | Recall@5 | MRR@5 | MAP@5 | nDCG@5 |", "|---|---:|---:|---:|---:|---:|"]
    for arm in ("20", "30", "40"):
        row = payload["final_ir_summary"][arm]
        lines.append(f"| D{arm} | {row['P@5']:.4f} | {row['Recall@5']:.4f} | {row['MRR@5']:.4f} | {row['MAP@5']:.4f} | {row['nDCG@5']:.4f} |")
    lines += ["", "## Latency", "", "| Arm | Pair total | Dense mean ms | Reranker model mean ms | Reranker E2E mean ms | Retrieval total mean ms |", "|---|---:|---:|---:|---:|---:|"]
    for arm in ("20", "30", "40"):
        row = payload["latency_metrics"]["arms"][arm]["aggregate"]
        lines.append(f"| D{arm} | {row['reranker_pair_count']['total']} | {row['dense_search_ms']['mean']:.2f} | {row['reranker_model_ms']['mean']:.2f} | {row['reranker_end_to_end_ms']['mean']:.2f} | {row['retrieval_total_ms']['mean']:.2f} |")
    lines += ["", "## PAIRWISE DELTA", "", "配对结果使用固定 seed 的题族 bootstrap；报告 absolute delta、wins/ties/losses 与 CI，不用任意阈值替代观察结果。", ""]
    for name, result in payload["bootstrap_comparison"]["comparisons"].items():
        lines.append(f"- `{name}`: delta={result['absolute_delta']:.6f}; wins/ties/losses={result['wins']}/{result['ties']}/{result['losses']}; CI95=[{result['ci95'][0]:.6f}, {result['ci95'][1]:.6f}]")
    lines += ["", "## TRANSITIONS", "", "见 `transition_analysis.json`，其中保存 D20→D30 与 D30→D40 的逐题状态和找回证据是否被 Reranker 利用。"]
    lines += ["", "## Decision", "", f"- RECOMMENDED_DENSE_K: **{payload['recommended_dense_k']}**", f"- PROMOTION_RECOMMENDED: **{payload['promotion_recommended']}**（仅建议另立 Phase 15C，未执行 Promotion）", f"- KNOWN_NEXT_BOTTLENECK: **{payload['known_next_bottleneck']}**", f"- Reason: {payload['recommendation_reason']}", ""]
    return "\n".join(lines)


def run_experiment(output_dir: Path = OUTPUT_DIR, replay_path: Path = DEFAULT_REPLAY_PATH) -> dict[str, Any]:
    """执行 Phase 15B.1；replay parity 失败时只写 STOP 证据并立即结束。"""

    configure_offline_model_cache()
    started = time.perf_counter()
    # CLI 允许相对路径，但 manifest 和 Hash 必须使用 workspace 内的绝对解析结果。
    output_dir = output_dir if output_dir.is_absolute() else ROOT / output_dir
    replay_path = replay_path if replay_path.is_absolute() else ROOT / replay_path
    before_frozen = frozen_tree_hashes()
    before_sources = production_source_hashes()
    output_dir.mkdir(parents=True, exist_ok=True)
    inputs = load_frozen_inputs()
    settings = Settings()
    store = ChromaVectorStore(settings.vector_db_path, settings.collection_name)
    chunks = validate_active_index(settings, inputs, store)
    questions_v1 = inputs["questions_v1"]
    questions_v2 = inputs["questions_v2"]
    question_ids = [q["question_id"] for q in questions_v1["questions"]]
    questions_by_id = {q["question_id"]: q for q in questions_v1["questions"]}
    projection = project_active_relevance(chunks, questions_v2)
    if projection["active_relevance_projection_hash"] != PROJECTION_HASH:
        raise ExperimentIdentityError("active relevance projection hash mismatch")
    groups = projection["groups_by_question"]
    qrels = projection["qrels"]
    if not replay_path.is_file():
        raise ExperimentIdentityError(f"independent production replay is missing: {replay_path}")
    actual_replay = read_json(replay_path)
    expected_replay = read_json(EXPECTED_REPLAY_PATH)
    replay_parity = compare_replay(expected_replay, actual_replay, PROJECTION_HASH, question_ids)
    # 保存实际 replay 副本，作为本阶段的 baseline；不覆盖 Phase 15B 历史文件。
    shutil.copyfile(replay_path, output_dir / "production_baseline.json")
    manifest = build_input_manifest(settings, store, chunks, inputs, PROJECTION_HASH, replay_path)
    write_json(output_dir / "input_manifest.json", manifest)
    if not replay_parity["parity"]:
        result = {
            "schema_version": "phase15b1-execution-result-v1",
            "phase": "15B.1",
            "status": "STOP",
            "stop_reason": "independent_production_replay_parity_failed",
            "gates": {"identity_and_frozen_inputs": True, "production_replay_parity": False, "d20_production_parity": False, "determinism": False, "production_unchanged": True},
            "production_replay_parity": replay_parity,
            "promotion_performed": False,
            "production_mutated": False,
            "completed_at_epoch": time.time(),
        }
        write_json(output_dir / "execution_result.json", result)
        (output_dir / "report.md").write_text("# EvidenceFlow Phase 15B.1\n\n## STOP\n\nIndependent Production replay parity failed; D20/D30/D40 were not executed.\n", encoding="utf-8")
        return result

    # 一次批量编码全部 query；Embedding 模型、文本和 index 均不随 K 改变。
    embedding_provider = BgeM3EmbeddingProvider(
        settings.embedding_model, device=settings.model_device
    )
    queries = [questions_by_id[qid]["english_query"] for qid in question_ids]
    query_vectors = embedding_provider.embed_documents(queries)
    if len(query_vectors) != QUESTION_COUNT:
        raise ExperimentIdentityError("query embedding count mismatch")

    pool_results: dict[str, dict[str, list[Any]]] = {str(k): {} for k in CANDIDATE_KS}
    pools: dict[str, dict[str, list[DocumentChunk]]] = {str(k): {} for k in CANDIDATE_KS}
    dense_latency: dict[str, dict[str, float]] = {str(k): {} for k in CANDIDATE_KS}
    candidate_repeat_ok = True
    for k in CANDIDATE_KS:
        arm = str(k)
        for qid, vector in zip(question_ids, query_vectors):
            dense_started = time.perf_counter()
            results = store.search(vector, top_k=k)
            dense_latency[arm][qid] = (time.perf_counter() - dense_started) * 1000.0
            if len(results) != k:
                raise ExperimentIdentityError(f"D{k} returned {len(results)} candidates for {qid}")
            pool_results[arm][qid] = results
            pools[arm][qid] = [result.chunk for result in results]
        for qid, vector in zip(question_ids, query_vectors):
            repeated_ids = [r.chunk.chunk_id for r in store.search(vector, top_k=k)]
            first_ids = [r.chunk.chunk_id for r in pool_results[arm][qid]]
            if repeated_ids != first_ids:
                candidate_repeat_ok = False
                raise ExperimentIdentityError(f"D{k} candidate IDs are not deterministic for {qid}")

    prefix_violations = []
    for qid in question_ids:
        ids20 = [r.chunk.chunk_id for r in pool_results["20"][qid]]
        ids30 = [r.chunk.chunk_id for r in pool_results["30"][qid]]
        ids40 = [r.chunk.chunk_id for r in pool_results["40"][qid]]
        if ids20 != ids30[:20] or ids30 != ids40[:30]:
            prefix_violations.append(qid)
    for k in CANDIDATE_KS:
        write_json(output_dir / "candidates" / f"dense{k}.json", candidate_artifact(str(k), pool_results[str(k)], qrels, question_ids))
    if prefix_violations:
        raise ExperimentIdentityError(f"Dense candidate prefix audit failed: {prefix_violations}")

    candidate_metric_rows = {str(k): candidate_metrics(pools[str(k)], groups, question_ids, k) for k in CANDIDATE_KS}
    write_json(output_dir / "candidate_metrics.json", {"schema_version": "phase15b1-candidate-metrics-v1", "active_relevance_projection_hash": PROJECTION_HASH, "arms": candidate_metric_rows})

    model_load_started = time.perf_counter()
    scorer = TransformersCrossEncoderScorer(resolve_local_reranker_model(settings), device=settings.model_device, dtype=settings.reranker_dtype, max_length=settings.reranker_max_length, inference_mode=False)
    model_load_ms = (time.perf_counter() - model_load_started) * 1000.0
    reranker = LearnedReranker(scorer, batch_size=settings.reranker_batch_size, max_candidates=40)
    final_results: dict[str, dict[str, list[RerankedChunk]]] = {str(k): {} for k in CANDIDATE_KS}
    reranker_scores: dict[str, dict[str, dict[str, float]]] = {str(k): {} for k in CANDIDATE_KS}
    latency_rows: dict[str, list[dict[str, Any]]] = {str(k): [] for k in CANDIDATE_KS}
    for k in CANDIDATE_KS:
        arm = str(k)
        for qid in question_ids:
            rerank_started = time.perf_counter()
            # 一次前向保留完整候选排序；切片 Top-5 不会再次调用模型，也让确定性
            # Gate 能够使用同一批完整 scores 复算排名。
            all_results = reranker.rerank(questions_by_id[qid]["english_query"], pools[arm][qid], top_k=None)
            results = all_results[:FINAL_TOP_K]
            rerank_e2e_ms = (time.perf_counter() - rerank_started) * 1000.0
            final_results[arm][qid] = results
            reranker_scores[arm][qid] = {result.chunk.chunk_id: result.relevance_score for result in all_results}
            latency_rows[arm].append({
                "question_id": qid,
                "dense_search_ms": dense_latency[arm][qid],
                "reranker_pair_count": len(pools[arm][qid]),
                "reranker_model_ms": float(reranker.last_timing_seconds.get("model_forward", 0.0)) * 1000.0,
                "reranker_end_to_end_ms": rerank_e2e_ms,
                "retrieval_total_ms": dense_latency[arm][qid] + rerank_e2e_ms,
                "truncated_pair_count": reranker.last_truncated_pair_count,
            })
        write_json(output_dir / "reranker" / f"dense{arm}_top5.json", reranker_artifact(arm, final_results[arm], pools[arm], question_ids))

    runs = {arm: {qid: [result.chunk.chunk_id for result in final_results[arm][qid]] for qid in question_ids} for arm in ("20", "30", "40")}
    ir_results = {arm: evaluate_ir_run(qrels, runs[arm], depths=(1, 3, 5, 10, 20)) for arm in ("20", "30", "40")}
    ir_summaries = {arm: standard_summary(ir_results[arm]) for arm in ("20", "30", "40")}
    d20_parity = compare_d20_to_replay(actual_replay, runs["20"], ir_summaries["20"], question_ids)
    if not d20_parity["parity"]:
        raise ExperimentIdentityError("D20 experiment does not match independent Production replay")
    write_json(output_dir / "final_ir_metrics.json", {"schema_version": "phase15b1-final-ir-v1", "active_relevance_projection_hash": PROJECTION_HASH, "arms": ir_results, "summary_at_5": ir_summaries})

    failures = classify_failures(pools, final_results, groups, question_ids)
    write_json(output_dir / "failure_attribution.json", failures)
    transition_rows = transitions(failures, question_ids)
    write_json(output_dir / "transition_analysis.json", transition_rows)
    latency_payload = {"schema_version": "phase15b1-latency-v1", "model_load_ms": model_load_ms, "arms": {arm: {"per_query": latency_rows[arm], "aggregate": latency_summary(latency_rows[arm])} for arm in ("20", "30", "40")}}
    write_json(output_dir / "latency_metrics.json", latency_payload)
    bootstrap = bootstrap_comparisons(candidate_metric_rows, ir_results, question_ids, questions_v2)
    write_json(output_dir / "bootstrap_comparison.json", bootstrap)

    # 复用已保存分数生成相同 Top-5，再次计算指标，验证下游确定性且不重复前向。
    deterministic_top5 = all(
        [r.chunk.chunk_id for r in rank_scored_candidates(
            pools[arm][qid],
            [reranker_scores[arm][qid][chunk.chunk_id] for chunk in pools[arm][qid]],
            top_k=FINAL_TOP_K,
        )]
        == [r.chunk.chunk_id for r in final_results[arm][qid]]
        for arm in ("20", "30", "40")
        for qid in question_ids
    )
    deterministic_metrics = all(evaluate_ir_run(qrels, runs[arm], depths=(1, 3, 5, 10, 20)) == ir_results[arm] for arm in ("20", "30", "40"))
    frozen_after = frozen_tree_hashes()
    sources_after = production_source_hashes()
    gates = {
        "identity_and_frozen_inputs": True,
        "production_replay_parity": replay_parity["parity"],
        "d20_production_parity": d20_parity["parity"],
        "candidate_only_variation": True,
        "prefix_audit": not prefix_violations,
        "determinism": candidate_repeat_ok and deterministic_top5 and deterministic_metrics,
        "frozen_artifacts_unchanged": before_frozen == frozen_after,
        "production_unchanged": before_sources == sources_after,
    }
    all_pass = all(gates.values())
    recommended_k, reason = choose_recommendation(ir_summaries, {arm: candidate_metric_rows[arm] for arm in ("20", "30", "40")})
    d40_counts = failures["arms"]["40"]["counts"]
    if d40_counts["RERANKER_MISS"] >= d40_counts["CANDIDATE_MISS"]:
        bottleneck = "RERANKER_MISS"
    else:
        bottleneck = "CANDIDATE_MISS"
    payload = {
        "status": "PASS" if all_pass else "FAIL",
        "gates": gates,
        "candidate_metrics": candidate_metric_rows,
        "final_ir_summary": ir_summaries,
        "latency_metrics": latency_payload,
        "bootstrap_comparison": bootstrap,
        "recommended_dense_k": recommended_k,
        "promotion_recommended": recommended_k > 20,
        "known_next_bottleneck": bottleneck,
        "recommendation_reason": reason,
    }
    (output_dir / "report.md").write_text(render_report(payload), encoding="utf-8")
    result = {
        "schema_version": "phase15b1-execution-result-v1",
        "phase": "15B.1",
        "status": payload["status"],
        "duration_seconds": time.perf_counter() - started,
        "gates": gates,
        "production_replay_parity": replay_parity,
        "d20_production_parity": d20_parity,
        "recommended_dense_k": recommended_k,
        "promotion_recommended": recommended_k > 20,
        "promotion_performed": False,
        "production_mutated": False,
        "known_next_bottleneck": bottleneck,
        "recommendation_reason": reason,
        "executed_arms": ["D20", "D30", "D40"],
        "artifacts": ["input_manifest.json", "production_baseline.json", "candidates/dense20.json", "candidates/dense30.json", "candidates/dense40.json", "candidate_metrics.json", "reranker/dense20_top5.json", "reranker/dense30_top5.json", "reranker/dense40_top5.json", "final_ir_metrics.json", "latency_metrics.json", "failure_attribution.json", "transition_analysis.json", "bootstrap_comparison.json", "execution_result.json", "report.md"],
        "completed_at_epoch": time.time(),
    }
    write_json(output_dir / "execution_result.json", result)
    return result


def parse_args() -> argparse.Namespace:
    """解析 Phase 15B.1 输出目录和独立 replay 路径。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--replay", type=Path, default=DEFAULT_REPLAY_PATH)
    return parser.parse_args()


def main() -> int:
    """运行实验并在身份或硬门失败时返回非零退出码。"""

    args = parse_args()
    try:
        result = run_experiment(args.output_dir, args.replay)
    except Exception as exc:
        failure = {"schema_version": "phase15b1-execution-result-v1", "phase": "15B.1", "status": "BLOCKED" if isinstance(exc, ExperimentIdentityError) else "FAIL", "error_type": type(exc).__name__, "error": str(exc), "promotion_performed": False, "production_mutated": False}
        write_json(args.output_dir / "execution_result.json", failure)
        print(json.dumps(failure, ensure_ascii=False))
        return 1
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
