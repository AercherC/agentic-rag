"""已冻结的 Phase 15B Hybrid Candidate Retrieval Benchmark 历史 runner。

本脚本只读取 Phase 15A 冻结的 active Chroma Collection，并复用现有的
BM25、BGE-M3 Dense、Learned Reranker 与 Evaluation v3 实现。它不修改 Runtime
配置、生产 Collection、Gold、qrels 或 mapping；所有输出均写入本阶段专用目录。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
import unicodedata
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evidenceflow.config import Settings
from evidenceflow.domain.chunk import DocumentChunk
from evidenceflow.embedding.bge_m3 import BgeM3EmbeddingProvider
from evidenceflow.eval.comparison import paired_comparison
from evidenceflow.eval.ir_metrics import evaluate_ir_run
from evidenceflow.ingestion.tokenizer import BgeM3Tokenizer
from evidenceflow.retrieval.bm25 import BM25Index
from evidenceflow.retrieval.reranker import (
    LearnedReranker,
    TransformersCrossEncoderScorer,
    rank_scored_candidates,
)
from evidenceflow.storage.chroma import ChromaVectorStore
from experiments.phase15b_active_relevance import project_active_relevance


OUTPUT_DIR = ROOT / "evals" / "evaluation_v3" / "phase15b_hybrid_candidate_v2"
V1_OUTPUT_DIR = ROOT / "evals" / "evaluation_v3" / "phase15b_hybrid_candidate_v1"
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
QRELS_MANIFEST_PATH = (
    ROOT / "evals" / "evaluation_v3" / "standard_ir_v1" / "qrels_manifest.json"
)

COLLECTION_NAME = "ef_research_bge_m3_hybrid_c512_v2"
CORPUS_SIGNATURE = "bb184726b7f98756bd519d70e1a2a1f1224b5108e689680b2c328023ba1f4332"
INDEX_SIGNATURE = "b4e9432c7170615ade2c53803a887fa86d0f793949934a57db421f0da57739e5"
CHUNK_ID_HASH = "e87a613ebe65b10262996e83974b55839e0b6c27fbe22053ee7b8826dd1f153d"
EXPECTED_DOCUMENT_COUNT = 10
EXPECTED_CHUNK_COUNT = 817
RERANKER_FINAL_K = 5
BOOTSTRAP_SEED = 20260905
BOOTSTRAP_SAMPLES = 10000

ARM_ORDER = ("dense20", "dense40", "bm25_20", "dense20_union_bm25_20")
ARM_LABELS = {
    "dense20": "B0 Dense20",
    "dense40": "B1 Dense40",
    "bm25_20": "B2 BM25-20",
    "dense20_union_bm25_20": "B3 Dense20 union BM25-20",
}


class BenchmarkIdentityError(RuntimeError):
    """表示 frozen input 与当前 active production representation 不一致。"""


def read_json(path: Path) -> Any:
    """以 UTF-8 读取 JSON，所有输入文件先在本地完成身份校验。"""

    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    """计算文件字节 Hash；Hash 只用于审计，不参与检索排序。"""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    """以稳定缩进写出 benchmark artifact，避免字段顺序和空白漂移。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def normalize_text(value: str) -> str:
    """按 Phase 13F 已验证的规则规范化 anchor 与 Chunk 文本。"""

    # NFKC 统一兼容字符，casefold 处理大小写，空白折叠处理 PDF 换行断裂。
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return " ".join(normalized.split())


def sorted_chunk_id_hash(chunks: Sequence[DocumentChunk]) -> str:
    """对排序后的 chunk_id 做确定性序列化 Hash，验证 active index 身份。"""

    # Phase 15A 的冻结合同采用紧凑 JSON 数组序列化，而不是换行文本；保持同一
    # 字节级规则才能让 benchmark 与既有 active_manifest 身份完全一致。
    serialized = json.dumps(
        sorted(chunk.chunk_id for chunk in chunks),
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def configure_offline_model_cache() -> None:
    """将模型加载锁定到仓库缓存，避免 benchmark 隐式发起网络请求。"""

    cache_dir = ROOT / ".cache" / "huggingface"
    os.environ.setdefault("HF_HOME", str(cache_dir))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def resolve_local_reranker_model(settings: Settings) -> str:
    """优先使用仓库已缓存的同一 checkpoint，避免 benchmark 隐式联网下载。"""

    local_model = ROOT / ".cache" / "models" / "bge-reranker-base"
    if (local_model / "config.json").is_file() and (
        (local_model / "model.safetensors").is_file()
        or (local_model / "pytorch_model.bin").is_file()
    ):
        # manifest 仍记录 settings.reranker_model；本地路径只是加载位置，不改变模型身份。
        return str(local_model)
    return settings.reranker_model


def load_frozen_inputs() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """读取 Phase 15A snapshot、Gold、mapping 与 canonical qrels。"""

    snapshot = read_json(SNAPSHOT_PATH)
    active_manifest = read_json(ACTIVE_MANIFEST_PATH)
    questions_v1 = read_json(QUESTIONS_V1_PATH)
    questions_v2 = read_json(QUESTIONS_V2_PATH)
    mapping = read_json(MAPPING_PATH)
    canonical_qrels = read_json(QRELS_PATH)
    qrels_manifest = read_json(QRELS_MANIFEST_PATH)

    if snapshot.get("status") != "PASS":
        raise BenchmarkIdentityError("Phase 15A baseline snapshot is not PASS")
    identity = snapshot["authoritative_identity"]
    if identity["collection_name"] != COLLECTION_NAME:
        raise BenchmarkIdentityError("Phase 15A collection identity is unexpected")
    if active_manifest["index"]["collection_name"] != COLLECTION_NAME:
        raise BenchmarkIdentityError("Phase 15A active manifest collection mismatches")
    if len(questions_v1["questions"]) != 40 or len(questions_v2["questions"]) != 40:
        raise BenchmarkIdentityError("Phase 15B requires the frozen 40-question set")
    ids_v1 = [item["question_id"] for item in questions_v1["questions"]]
    ids_v2 = [item["question_id"] for item in questions_v2["questions"]]
    if ids_v1 != ids_v2:
        raise BenchmarkIdentityError("v1 and v2 question IDs are not identical")
    if mapping.get("mapping_version") != "gold-mapper-v3":
        raise BenchmarkIdentityError("gold mapping is not gold-mapper-v3")
    if qrels_manifest.get("qrels_hash") != snapshot["frozen_artifacts"]["qrels_hash"]:
        raise BenchmarkIdentityError("canonical qrels semantic hash mismatches snapshot")
    if sha256_file(MAPPING_PATH) != snapshot["frozen_artifacts"]["gold_mapping_hash"]:
        raise BenchmarkIdentityError("gold mapping file hash mismatches snapshot")
    expected_file_hashes = snapshot["frozen_artifacts"]["gold_and_qrels_files_sha256"]
    for path, expected in expected_file_hashes.items():
        if sha256_file(ROOT / path) != expected:
            raise BenchmarkIdentityError(f"frozen input file hash mismatch: {path}")

    # canonical qrels 仍作为身份输入保留；当前 hybrid Chunk IDs 的 qrels 通过 anchor
    # 投影得到，不能把旧 fixed index 的 chunk_id 直接当作 active hybrid 的 Gold。
    if not isinstance(canonical_qrels, dict) or set(canonical_qrels) != set(ids_v1):
        raise BenchmarkIdentityError("canonical qrels query space mismatches questions")
    return snapshot, active_manifest, questions_v1, questions_v2, mapping


def validate_active_index(
    settings: Settings,
    snapshot: Mapping[str, Any],
    store: ChromaVectorStore,
) -> list[DocumentChunk]:
    """只读验证 active Collection 的 frozen identity，并返回全部正式 Chunk。"""

    if settings.collection_name != COLLECTION_NAME:
        raise BenchmarkIdentityError("Settings collection is not the Phase 15A active collection")
    if store.collection_name != COLLECTION_NAME:
        raise BenchmarkIdentityError("opened Chroma collection has an unexpected name")
    chunks = list(store.iter_all_chunks())
    metadata = store.metadata
    expected_metadata = snapshot["authoritative_identity"]["chroma_active_collection_metadata"]
    if metadata != expected_metadata:
        raise BenchmarkIdentityError("active Chroma metadata mismatches frozen identity")
    if store.count != EXPECTED_CHUNK_COUNT or len(chunks) != EXPECTED_CHUNK_COUNT:
        raise BenchmarkIdentityError("active Chroma chunk count is not 817")
    if len({chunk.chunk_id for chunk in chunks}) != EXPECTED_CHUNK_COUNT:
        raise BenchmarkIdentityError("active Chroma contains duplicate chunk IDs")
    if len({chunk.document_id for chunk in chunks}) != EXPECTED_DOCUMENT_COUNT:
        raise BenchmarkIdentityError("active Chroma document count is not 10")
    if sorted_chunk_id_hash(chunks) != CHUNK_ID_HASH:
        raise BenchmarkIdentityError("active Chroma sorted chunk_id hash mismatches snapshot")
    index = snapshot["index"]
    if index["corpus_signature"] != CORPUS_SIGNATURE or index["index_signature"] != INDEX_SIGNATURE:
        raise BenchmarkIdentityError("snapshot index signatures are unexpected")
    return chunks


def project_current_groups(
    chunks: Sequence[DocumentChunk],
    questions_v2: Mapping[str, Any],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[str]], str]:
    """复用独立 replay 的 active relevance projection，并返回其稳定 Hash。"""

    projection = project_active_relevance(chunks, questions_v2)
    return (
        projection["groups_by_question"],
        projection["qrels"],
        projection["active_relevance_projection_hash"],
    )


def build_manifest(
    snapshot: Mapping[str, Any],
    active_manifest: Mapping[str, Any],
    settings: Settings,
    store: ChromaVectorStore,
    chunks: Sequence[DocumentChunk],
    question_ids: Sequence[str],
    active_relevance_projection_hash: str,
) -> dict[str, Any]:
    """建立 Phase 15B 输入合同，记录模型、索引和 Gold/qrels/mapping 身份。"""

    frozen = snapshot["frozen_artifacts"]
    files = {
        "gold_mapping_v3": MAPPING_PATH,
        "research_questions_v2": QUESTIONS_V2_PATH,
        "standard_qrels": QRELS_PATH,
        "qrels_manifest": QRELS_MANIFEST_PATH,
    }
    return {
        "schema_version": "phase15b-input-manifest-v2",
        "phase": "15B",
        "status": "FROZEN_INPUTS_VALIDATED",
        "collection_name": COLLECTION_NAME,
        "active_collection_metadata": store.metadata,
        "corpus_signature": CORPUS_SIGNATURE,
        "index_signature": INDEX_SIGNATURE,
        "document_count": len({chunk.document_id for chunk in chunks}),
        "chunk_count": len(chunks),
        "chunk_id_hash": CHUNK_ID_HASH,
        "active_relevance_projection_hash": active_relevance_projection_hash,
        "embedding": {
            "model": settings.embedding_model,
            "version": snapshot["pipeline"]["embedding_version"],
            "normalization": "explicit_l2",
            "query_field": "english_query from research_questions_v1",
            "chunk_strategy": settings.chunk_strategy,
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
        },
        "reranker": {
            "model": settings.reranker_model,
            "checkpoint": snapshot["pipeline"]["reranker_checkpoint"],
            "max_length": settings.reranker_max_length,
            "batch_size": settings.reranker_batch_size,
            "device": settings.reranker_device,
            "dtype": settings.reranker_dtype,
            "inference_mode": False,
            "final_top_k": RERANKER_FINAL_K,
        },
        "bm25": {
            "tokenizer_model": settings.embedding_model,
            "k1": 1.5,
            "b": 0.75,
            "corpus": "active collection iter_all_chunks() content, 817 chunks",
        },
        "gold_qrels_mapping": {
            "gold_mapping_hash": frozen["gold_mapping_hash"],
            "mapping_hash": frozen["gold_mapping_hash"],
            "qrels_hash": frozen["qrels_hash"],
            "file_sha256": {name: sha256_file(path) for name, path in files.items()},
            "canonical_qrels_not_rewritten": True,
            "current_index_relevance": "read-only evidence_anchor projection",
            "active_relevance_projection_hash": active_relevance_projection_hash,
        },
        "question_ids": list(question_ids),
        "active_manifest_identity": {
            "corpus_signature": active_manifest["corpus_signature"],
            "index_signature": active_manifest["index"]["index_signature"],
        },
    }


def make_union_candidates(
    dense20: Sequence[Any],
    bm25_20: Sequence[Any],
) -> list[DocumentChunk]:
    """按 Dense20 后 BM25-only 的顺序去重，形成不带融合分数的 B3 候选池。"""

    result: list[DocumentChunk] = []
    seen: set[str] = set()
    for item in dense20:
        chunk = item.chunk
        if chunk.chunk_id not in seen:
            result.append(chunk)
            seen.add(chunk.chunk_id)
    for item in bm25_20:
        chunk = item.chunk
        if chunk.chunk_id not in seen:
            result.append(chunk)
            seen.add(chunk.chunk_id)
    if len(result) > 40:
        raise AssertionError("B3 union exceeds the required maximum of 40 candidates")
    return result


def build_pools(
    dense20: Mapping[str, Sequence[Any]],
    dense40: Mapping[str, Sequence[Any]],
    bm25_20: Mapping[str, Sequence[Any]],
    question_ids: Sequence[str],
) -> dict[str, dict[str, list[DocumentChunk]]]:
    """构造四个 benchmark arm，并验证 B3 只执行 set-union/dedupe。"""

    pools: dict[str, dict[str, list[DocumentChunk]]] = {arm: {} for arm in ARM_ORDER}
    for question_id in question_ids:
        pools["dense20"][question_id] = [item.chunk for item in dense20[question_id]]
        pools["dense40"][question_id] = [item.chunk for item in dense40[question_id]]
        pools["bm25_20"][question_id] = [item.chunk for item in bm25_20[question_id]]
        pools["dense20_union_bm25_20"][question_id] = make_union_candidates(
            dense20[question_id], bm25_20[question_id]
        )
        for arm in ARM_ORDER:
            ids = [chunk.chunk_id for chunk in pools[arm][question_id]]
            if len(ids) != len(set(ids)):
                raise AssertionError(f"{arm} has duplicate candidates for {question_id}")
    return pools


def candidate_artifact(
    arm: str,
    pools: Mapping[str, Sequence[DocumentChunk]],
    dense20: Mapping[str, Sequence[Any]],
    dense40: Mapping[str, Sequence[Any]],
    bm25_20: Mapping[str, Sequence[Any]],
    qrels: Mapping[str, Sequence[str]],
    question_ids: Sequence[str],
) -> dict[str, Any]:
    """序列化候选池字段，保留各分支原始 rank 但不写入任何 fusion score。"""

    dense_ranks = {
        qid: {item.chunk.chunk_id: item.rank for item in dense40[qid]} for qid in question_ids
    }
    bm25_ranks = {
        qid: {item.chunk.chunk_id: item.rank for item in bm25_20[qid]} for qid in question_ids
    }
    dense20_ids = {
        qid: {item.chunk.chunk_id for item in dense20[qid]} for qid in question_ids
    }
    rows: list[dict[str, Any]] = []
    per_query: list[dict[str, Any]] = []
    for question_id in question_ids:
        query_rows: list[dict[str, Any]] = []
        for chunk in pools[question_id]:
            cid = chunk.chunk_id
            sources: list[str] = []
            if cid in dense20_ids[question_id]:
                sources.append("dense")
            if cid in bm25_ranks[question_id]:
                sources.append("bm25")
            row = {
                "question_id": question_id,
                "arm": arm,
                "candidate_count": len(pools[question_id]),
                "chunk_id": cid,
                "document_id": chunk.document_id,
                "dense_rank": dense_ranks[question_id].get(cid),
                "bm25_rank": bm25_ranks[question_id].get(cid),
                "candidate_sources": sources,
                "is_gold": cid in set(qrels.get(question_id, [])),
            }
            query_rows.append(row)
            rows.append(row)
        per_query.append(
            {
                "question_id": question_id,
                "candidate_count": len(query_rows),
                "chunk_ids": [row["chunk_id"] for row in query_rows],
            }
        )
    return {
        "schema_version": "phase15b-candidates-v1",
        "arm": arm,
        "label": ARM_LABELS[arm],
        "candidate_pool_contract": "IDs only for B3 union; no score normalization or fusion",
        "rows": rows,
        "per_query": per_query,
    }


def group_covered(group: Mapping[str, Any], candidate_ids: set[str]) -> bool:
    """判断候选池是否覆盖 required evidence group 的任一 Gold anchor。"""

    # 一个 group 的 alternatives 是 OR 关系；其匹配 chunk 集合已经是 active index 投影。
    return bool(set(group["matched_chunk_ids"]) & candidate_ids)


def candidate_metric_rows(
    pools: Mapping[str, Sequence[DocumentChunk]],
    groups_by_question: Mapping[str, Sequence[Mapping[str, Any]]],
    question_ids: Sequence[str],
    depth: int,
) -> dict[str, Any]:
    """计算 Candidate 层三项指标，并同时保留 ALL_GOLD 与 INDEXABLE_ONLY。"""

    per_query: dict[str, dict[str, Any]] = {}
    all_group_total = 0
    all_group_covered = 0
    indexable_group_total = 0
    indexable_group_covered = 0
    for question_id in question_ids:
        groups = list(groups_by_question[question_id])
        candidate_ids = {chunk.chunk_id for chunk in list(pools[question_id])[:depth]}
        covered_flags = [group_covered(group, candidate_ids) for group in groups]
        indexable_flags = [group["indexable"] for group in groups]
        all_group_total += len(groups)
        all_group_covered += sum(covered_flags)
        indexable_group_total += sum(indexable_flags)
        indexable_group_covered += sum(
            covered for covered, indexable in zip(covered_flags, indexable_flags) if indexable
        )
        per_query[question_id] = {
            "question_id": question_id,
            "candidate_count": len(pools[question_id]),
            "covered_groups": sum(covered_flags),
            "total_groups": len(groups),
            "indexable_groups": sum(indexable_flags),
            "covered_indexable_groups": sum(
                covered for covered, indexable in zip(covered_flags, indexable_flags) if indexable
            ),
            "any_evidence": bool(any(covered_flags)),
            "all_required": bool(covered_flags) and all(covered_flags),
            "any_indexable_evidence": bool(
                any(covered and indexable for covered, indexable in zip(covered_flags, indexable_flags))
            ),
            "all_indexable_required": bool(sum(indexable_flags)) and all(
                covered for covered, indexable in zip(covered_flags, indexable_flags) if indexable
            ),
            "indexable": bool(sum(indexable_flags)),
        }

    indexable_questions = [
        question_id for question_id in question_ids if per_query[question_id]["indexable"]
    ]
    all_count = len(question_ids)
    indexable_count = len(indexable_questions)

    def cohort_metrics(cohort: str) -> dict[str, Any]:
        if cohort == "ALL_GOLD":
            selected = list(question_ids)
            hit_key = "any_evidence"
            required_key = "all_required"
            group_total = all_group_total
            group_covered = all_group_covered
            denominator = all_count
        else:
            selected = indexable_questions
            hit_key = "any_indexable_evidence"
            required_key = "all_indexable_required"
            group_total = indexable_group_total
            group_covered = indexable_group_covered
            denominator = indexable_count
        return {
            "eligible_query_count": denominator,
            "excluded_question_ids": [qid for qid in question_ids if qid not in selected],
            f"HitRate@{depth}": sum(per_query[qid][hit_key] for qid in selected) / denominator
            if denominator
            else 0.0,
            f"EvidenceAnchorRecall@{depth}": group_covered / group_total if group_total else 0.0,
            f"AllRequired@{depth}": sum(per_query[qid][required_key] for qid in selected) / denominator
            if denominator
            else 0.0,
            "covered_anchor_count": group_covered,
            "total_anchor_count": group_total,
        }

    return {
        "depth": depth,
        "per_query": per_query,
        "ALL_GOLD": cohort_metrics("ALL_GOLD"),
        "INDEXABLE_ONLY": cohort_metrics("INDEXABLE_ONLY"),
    }


def reranker_artifact(
    arm: str,
    results_by_question: Mapping[str, Sequence[Any]],
    pools: Mapping[str, Sequence[DocumentChunk]],
    question_ids: Sequence[str],
) -> dict[str, Any]:
    """序列化同一 Learned Reranker 的 Top-5 输出及审计字段。"""

    per_query: list[dict[str, Any]] = []
    for question_id in question_ids:
        pool = list(pools[question_id])
        candidate_rank = {chunk.chunk_id: index for index, chunk in enumerate(pool, start=1)}
        rows = []
        for result in results_by_question[question_id]:
            rows.append(
                {
                    "question_id": question_id,
                    "arm": arm,
                    "rank": result.rank,
                    "chunk_id": result.chunk.chunk_id,
                    "document_id": result.chunk.document_id,
                    "candidate_rank": candidate_rank[result.chunk.chunk_id],
                    "reranker_score": result.relevance_score,
                }
            )
        per_query.append(
            {
                "question_id": question_id,
                "candidate_count": len(pool),
                "final_count": len(rows),
                "rows": rows,
            }
        )
    return {
        "schema_version": "phase15b-reranker-top5-v1",
        "arm": arm,
        "label": ARM_LABELS[arm],
        "reranker_only_reorders_candidates": True,
        "per_query": per_query,
    }


def standard_ir_summary(ir_result: Mapping[str, Any]) -> dict[str, Any]:
    """提取报告需要的 @5 指标，同时保留 canonical Evaluation v3 结果。"""

    aggregate = ir_result["aggregate"]
    return {
        "P@5": aggregate["precision_at_k"][5],
        "Recall@5": aggregate["recall_at_k"][5],
        "MRR@5": aggregate["mrr_at_k"][5],
        "MAP@5": aggregate["map_at_k"][5],
        "nDCG@5": aggregate["ndcg_at_k"][5],
        "eligible_count": aggregate["eligible_count"],
        "excluded_count": aggregate["excluded_count"],
    }


def attribute_failures(
    pools: Mapping[str, Sequence[DocumentChunk]],
    reranked: Mapping[str, Sequence[Any]],
    groups_by_question: Mapping[str, Sequence[Mapping[str, Any]]],
    question_ids: Sequence[str],
) -> dict[str, Any]:
    """按候选池与 Top-5 两个可观测边界归因前三类失败。"""

    statuses = ("UNINDEXABLE", "CANDIDATE_MISS", "RERANKER_MISS", "CONTEXT_MISS", "GENERATION_MISS", "SUCCESS")
    by_arm: dict[str, Any] = {}
    for arm in ARM_ORDER:
        counts = {status: 0 for status in statuses}
        rows: list[dict[str, Any]] = []
        for question_id in question_ids:
            groups = list(groups_by_question[question_id])
            candidate_ids = {chunk.chunk_id for chunk in pools[arm][question_id]}
            final_ids = {result.chunk.chunk_id for result in reranked[arm][question_id]}
            unindexable = [group["group_id"] for group in groups if not group["indexable"]]
            candidate_miss = [
                group["group_id"]
                for group in groups
                if group["indexable"] and not group_covered(group, candidate_ids)
            ]
            reranker_miss = [
                group["group_id"]
                for group in groups
                if group["indexable"]
                and group_covered(group, candidate_ids)
                and not group_covered(group, final_ids)
            ]
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
                    "question_id": question_id,
                    "status": status,
                    "unindexable_group_ids": unindexable,
                    "candidate_miss_group_ids": candidate_miss,
                    "reranker_miss_group_ids": reranker_miss,
                    "candidate_count": len(pools[arm][question_id]),
                    "final_chunk_ids": [result.chunk.chunk_id for result in reranked[arm][question_id]],
                }
            )
        by_arm[arm] = {"label": ARM_LABELS[arm], "counts": counts, "per_query": rows}
    return {"schema_version": "phase15b-failure-attribution-v1", "arms": by_arm}


def complementarity_analysis(
    dense20: Mapping[str, Sequence[Any]],
    dense40: Mapping[str, Sequence[Any]],
    bm25_20: Mapping[str, Sequence[Any]],
    pools: Mapping[str, Mapping[str, Sequence[DocumentChunk]]],
    groups_by_question: Mapping[str, Sequence[Mapping[str, Any]]],
    question_ids: Sequence[str],
) -> dict[str, Any]:
    """统计 Dense/BM25 的互补命中，并列出 BM25 恢复 Dense20 漏召回的 anchors。"""

    counts = {"dense_only": 0, "bm25_only": 0, "both": 0, "neither": 0}
    recovered: list[dict[str, Any]] = []
    for question_id in question_ids:
        d20 = {item.chunk.chunk_id: item.rank for item in dense20[question_id]}
        d40 = {item.chunk.chunk_id: item.rank for item in dense40[question_id]}
        bm = {item.chunk.chunk_id: item.rank for item in bm25_20[question_id]}
        hybrid_ids = {chunk.chunk_id for chunk in pools["dense20_union_bm25_20"][question_id]}
        for group in groups_by_question[question_id]:
            if not group["indexable"]:
                continue
            gold_ids = set(group["matched_chunk_ids"])
            dense20_hit = bool(gold_ids & set(d20))
            bm25_hit = bool(gold_ids & set(bm))
            if dense20_hit and bm25_hit:
                category = "both"
            elif dense20_hit:
                category = "dense_only"
            elif bm25_hit:
                category = "bm25_only"
            else:
                category = "neither"
            counts[category] += 1
            if not dense20_hit and bm25_hit:
                best_id = min(
                    (cid for cid in gold_ids if cid in bm),
                    key=lambda cid: (bm[cid], cid),
                )
                anchors = [
                    alternative["anchor"]
                    for alternative in group["alternatives"]
                    if alternative["matched_chunk_ids"]
                ]
                recovered.append(
                    {
                        "classification": "BM25_RECOVERED_FROM_DENSE20_MISS",
                        "question_id": question_id,
                        "group_id": group["group_id"],
                        "gold_anchor": anchors,
                        "gold_chunk_id": best_id,
                        "dense20_rank": None,
                        "dense20_status": "MISS",
                        "bm25_rank": bm[best_id],
                        "dense40_rank": d40.get(best_id),
                        "dense40_status": "HIT" if best_id in d40 else "MISS",
                        "hybrid_membership": best_id in hybrid_ids,
                    }
                )
    return {
        "schema_version": "phase15b-complementarity-v1",
        "scope": "indexable evidence groups only",
        "counts": counts,
        "total_indexable_groups": sum(counts.values()),
        "bm25_recovered_from_dense20_miss": recovered,
    }


def context_diversity(
    reranked: Mapping[str, Sequence[Any]], question_ids: Sequence[str]
) -> dict[str, Any]:
    """记录 Top-5 的文档多样性，作为上下文审计信息而非选择条件。"""

    per_arm: dict[str, Any] = {}
    for arm in ARM_ORDER:
        rows = []
        for question_id in question_ids:
            results = list(reranked[arm][question_id])
            docs = [result.chunk.document_id for result in results]
            unique_docs = len(set(docs))
            rows.append(
                {
                    "question_id": question_id,
                    "top5_count": len(results),
                    "unique_document_count": unique_docs,
                    "max_chunks_from_one_document": max((docs.count(doc) for doc in set(docs)), default=0),
                    "duplicate_document_fraction": (len(docs) - unique_docs) / len(docs) if docs else 0.0,
                }
            )
        per_arm[arm] = rows
    return {"schema_version": "phase15b-context-diversity-v1", "arms": per_arm}


def per_query_metric_values(
    candidate_metrics: Mapping[str, Mapping[str, Any]],
    question_ids: Sequence[str],
    cohort: str,
    metric: str,
) -> dict[str, float]:
    """为 paired bootstrap 生成逐题候选指标值。"""

    values: dict[str, float] = {}
    for question_id in question_ids:
        row = candidate_metrics[question_id]
        indexable = row["indexable"]
        if cohort == "INDEXABLE_ONLY" and not indexable:
            continue
        if metric == "HitRate":
            values[question_id] = float(row["any_evidence"] if cohort == "ALL_GOLD" else row["any_indexable_evidence"])
        elif metric == "EvidenceAnchorRecall":
            if cohort == "ALL_GOLD":
                denominator = row["total_groups"]
                numerator = row["covered_groups"]
            else:
                denominator = row["indexable_groups"]
                numerator = row["covered_indexable_groups"]
            values[question_id] = numerator / denominator if denominator else 0.0
        elif metric == "AllRequired":
            values[question_id] = float(row["all_required"] if cohort == "ALL_GOLD" else row["all_indexable_required"])
        else:
            raise ValueError(f"unsupported candidate metric: {metric}")
    return values


def bootstrap_comparisons(
    candidate_per_query: Mapping[str, Mapping[str, Mapping[str, Any]]],
    ir_results: Mapping[str, Mapping[str, Any]],
    questions_v2: Mapping[str, Any],
    question_ids: Sequence[str],
) -> dict[str, Any]:
    """执行固定 seed 的逐题/题族配对 bootstrap 与随机化比较。"""

    families = {
        question["question_id"]: question.get("family_id", question["question_id"])
        for question in questions_v2["questions"]
    }
    output: dict[str, Any] = {
        "schema_version": "phase15b-bootstrap-v1",
        "seed": BOOTSTRAP_SEED,
        "samples": BOOTSTRAP_SAMPLES,
        "comparison_baseline": "dense20",
        "comparisons": {},
    }

    def add_comparison(name: str, baseline_values: Mapping[str, float], variant_values: Mapping[str, float]) -> None:
        if baseline_values.keys() != variant_values.keys():
            raise AssertionError(f"bootstrap query keys mismatch for {name}")
        scoped_families = {qid: families[qid] for qid in baseline_values}
        output["comparisons"][name] = paired_comparison(
            dict(baseline_values),
            dict(variant_values),
            families=scoped_families,
            seed=BOOTSTRAP_SEED,
            samples=BOOTSTRAP_SAMPLES,
        )

    for arm in ("dense40", "bm25_20", "dense20_union_bm25_20"):
        for cohort in ("ALL_GOLD", "INDEXABLE_ONLY"):
            for metric in ("HitRate", "EvidenceAnchorRecall", "AllRequired"):
                baseline_values = per_query_metric_values(
                    candidate_per_query["dense20"][cohort],
                    question_ids,
                    cohort,
                    metric,
                )
                variant_values = per_query_metric_values(
                    candidate_per_query[arm][cohort],
                    question_ids,
                    cohort,
                    metric,
                )
                add_comparison(f"candidate:{cohort}:{metric}:dense20_vs_{arm}", baseline_values, variant_values)
        for metric_key, metric_name in (
            ("recall_at_5", "Recall@5"),
            ("ndcg_at_5", "nDCG@5"),
            ("mrr_at_5", "MRR@5"),
            ("map_at_5", "MAP@5"),
        ):
            baseline_values = {
                qid: float(ir_results["dense20"]["per_query"][qid]["metrics"][metric_key])
                for qid in question_ids
                if ir_results["dense20"]["per_query"][qid]["eligible"]
            }
            variant_values = {
                qid: float(ir_results[arm]["per_query"][qid]["metrics"][metric_key])
                for qid in baseline_values
            }
            add_comparison(f"final:{metric_name}:dense20_vs_{arm}", baseline_values, variant_values)

    # 直接回答“Dense40 vs Hybrid”的配对比较，避免只提供相对 B0 的结果。
    for metric_key, metric_name in (
        ("recall_at_5", "Recall@5"),
        ("ndcg_at_5", "nDCG@5"),
        ("mrr_at_5", "MRR@5"),
        ("map_at_5", "MAP@5"),
    ):
        baseline_values = {
            qid: float(ir_results["dense40"]["per_query"][qid]["metrics"][metric_key])
            for qid in question_ids
            if ir_results["dense40"]["per_query"][qid]["eligible"]
        }
        variant_values = {
            qid: float(ir_results["dense20_union_bm25_20"]["per_query"][qid]["metrics"][metric_key])
            for qid in baseline_values
        }
        add_comparison(f"final:{metric_name}:dense40_vs_dense20_union_bm25_20", baseline_values, variant_values)
    return output


def _candidate_metric_value(metrics: Mapping[str, Any], cohort: str, name: str) -> float:
    """读取候选指标的 cohort 值，用于硬门禁。"""

    return float(metrics[cohort][name])


def check_baseline(summary: Mapping[str, Any]) -> bool:
    """检查 raw Dense20 是否复现历史 Phase13F 参考值；该值不是 B0 Gate。"""

    expected = {
        "P@5": 0.23157894736842108,
        "Recall@5": 0.5123862463645745,
        "MRR@5": 0.5521929824561403,
        "MAP@5": 0.38551310316790194,
        "nDCG@5": 0.49734103517021605,
    }
    return all(math.isclose(float(summary[key]), value, rel_tol=0.0, abs_tol=1e-12) for key, value in expected.items())


def baseline_reproduction_diagnosis(
    raw_dense_summary: Mapping[str, Any],
    reranked_summary: Mapping[str, Any],
    replay_parity: Mapping[str, Any],
) -> dict[str, Any]:
    """区分历史 raw 参考值，并记录正式 Production replay parity 结果。"""

    expected = {
        "P@5": 0.23157894736842108,
        "Recall@5": 0.5123862463645745,
        "MRR@5": 0.5521929824561403,
        "MAP@5": 0.38551310316790194,
        "nDCG@5": 0.49734103517021605,
    }
    return {
        "formal_b0_pipeline": "Production Dense Top20 -> BAAI/bge-reranker-base -> Top5",
        "historical_reference_classification": "RAW_DENSE20_TOP5 / PRE_RERANK_REFERENCE",
        "historical_reference_values": expected,
        "raw_dense20_top5": {key: raw_dense_summary[key] for key in expected},
        "reranked_b0_top5": {key: reranked_summary[key] for key in expected},
        "reference_matches_raw_dense20": check_baseline(raw_dense_summary),
        "reference_is_not_post_reranker_gate": True,
        "production_replay_parity": dict(replay_parity),
        "diagnosis": (
            "Phase13F values are retained only as RAW_DENSE20_TOP5 / PRE_RERANK_REFERENCE; "
            "formal B0 is decided by independent production replay parity."
        ),
    }


def compare_production_replay(
    baseline: Mapping[str, Any],
    projection_hash: str,
    b0_run: Mapping[str, Sequence[str]],
    b0_summary: Mapping[str, Any],
    question_ids: Sequence[str],
) -> dict[str, Any]:
    """比较独立 Production replay 与 benchmark B0，作为 B1/B2/B3 的前置 Gate。"""

    expected_run = baseline.get("final_run")
    actual_run = {question_id: list(b0_run[question_id]) for question_id in question_ids}
    query_set_matches = isinstance(expected_run, Mapping) and set(expected_run) == set(question_ids)
    run_mismatches = [
        question_id
        for question_id in question_ids
        if not query_set_matches
        or not isinstance(expected_run, Mapping)
        or list(expected_run.get(question_id, [])) != actual_run[question_id]
    ]
    expected_summary = baseline.get("summary_at_5", {})
    summary_keys = ("P@5", "Recall@5", "MRR@5", "MAP@5", "nDCG@5")
    summary_mismatches = [
        key
        for key in summary_keys
        if key not in expected_summary
        or not math.isclose(
            float(expected_summary[key]), float(b0_summary[key]), rel_tol=0.0, abs_tol=1e-12
        )
    ]
    projection_matches = baseline.get("active_relevance_projection_hash") == projection_hash
    pipeline_matches = baseline.get("pipeline") == (
        "production Dense Top20 -> BAAI/bge-reranker-base -> Top5"
    )
    status_matches = baseline.get("status") == "PASS"
    return {
        "parity": bool(
            status_matches
            and pipeline_matches
            and projection_matches
            and query_set_matches
            and not run_mismatches
            and not summary_mismatches
        ),
        "status_matches": status_matches,
        "pipeline_matches": pipeline_matches,
        "projection_hash_matches": projection_matches,
        "query_set_matches": query_set_matches,
        "expected_projection_hash": baseline.get("active_relevance_projection_hash"),
        "actual_projection_hash": projection_hash,
        "run_mismatches": run_mismatches,
        "summary_mismatches": summary_mismatches,
        "compared_question_count": len(question_ids),
    }


def dense40_clear_superiority(
    ir_summaries: Mapping[str, Mapping[str, Any]], margin: float = 0.005
) -> bool:
    """用预注册绝对 margin 判断 Dense40 是否明显优于 B0 与 B3。"""

    dense40 = ir_summaries["dense40"]
    comparison_arms = ("dense20", "dense20_union_bm25_20")
    return all(
        float(dense40[metric])
        >= max(float(ir_summaries[arm][metric]) for arm in comparison_arms) + margin
        for metric in ("Recall@5", "nDCG@5")
    )


def source_hashes_before_after() -> tuple[dict[str, str], dict[str, str]]:
    """对生产检索源文件做只读前后 Hash，证明 benchmark 未改生产代码。"""

    paths = (
        ROOT / "src" / "evidenceflow" / "config.py",
        ROOT / "src" / "evidenceflow" / "runtime.py",
        ROOT / "src" / "evidenceflow" / "retrieval" / "retriever.py",
        ROOT / "src" / "evidenceflow" / "retrieval" / "bm25.py",
        ROOT / "src" / "evidenceflow" / "retrieval" / "reranker.py",
    )
    before = {str(path.relative_to(ROOT)): sha256_file(path) for path in paths}
    return before, before.copy()


def render_report(
    manifest: Mapping[str, Any],
    candidate_metrics: Mapping[str, Mapping[str, Any]],
    ir_summaries: Mapping[str, Mapping[str, Any]],
    baseline_diagnosis: Mapping[str, Any],
    complementarity: Mapping[str, Any],
    failures: Mapping[str, Any],
    gates: Mapping[str, Any],
    recommendation: str,
) -> str:
    """生成面向审计的简短 Markdown 报告，不把探索性比较写成晋升结论。"""

    lines = [
        "# EvidenceFlow Phase 15B Hybrid Candidate Retrieval Benchmark",
        "",
        "本报告只描述离线候选池实验；未修改 Production Runtime、active Chroma、Gold、qrels 或 mapping。",
        "",
        "## Frozen inputs",
        "",
        f"- Collection: `{manifest['collection_name']}`; documents/chunks: `{manifest['document_count']}/{manifest['chunk_count']}`.",
        f"- Corpus signature: `{manifest['corpus_signature']}`.",
        f"- Index signature: `{manifest['index_signature']}`; chunk ID hash: `{manifest['chunk_id_hash']}`.",
        f"- Active relevance projection hash: `{manifest['active_relevance_projection_hash']}`.",
        f"- Gold mapping hash: `{manifest['gold_qrels_mapping']['mapping_hash']}`; qrels hash: `{manifest['gold_qrels_mapping']['qrels_hash']}`.",
        "- Current active qrels are a read-only evidence-anchor projection because the canonical qrels contain frozen fixed-index chunk IDs.",
        "",
        "## Candidate metrics",
        "",
        "| Arm | HitRate@pool ALL | AnchorRecall@pool ALL | AllRequired@pool ALL | HitRate@pool INDEXABLE | AnchorRecall@pool INDEXABLE | AllRequired@pool INDEXABLE |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in ARM_ORDER:
        depth = candidate_metrics[arm]["depth"]
        all_metrics = candidate_metrics[arm]["ALL_GOLD"]
        indexable_metrics = candidate_metrics[arm]["INDEXABLE_ONLY"]
        lines.append(
            f"| {ARM_LABELS[arm]} | {all_metrics[f'HitRate@{depth}']:.4f} | "
            f"{all_metrics[f'EvidenceAnchorRecall@{depth}']:.4f} | {all_metrics[f'AllRequired@{depth}']:.4f} | "
            f"{indexable_metrics[f'HitRate@{depth}']:.4f} | {indexable_metrics[f'EvidenceAnchorRecall@{depth}']:.4f} | "
            f"{indexable_metrics[f'AllRequired@{depth}']:.4f} |"
        )
    lines += [
        "",
        "## Final Learned Reranker IR @5",
        "",
        "| Arm | P@5 | Recall@5 | MRR@5 | MAP@5 | nDCG@5 | eligible | excluded |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in ARM_ORDER:
        row = ir_summaries[arm]
        lines.append(
            f"| {ARM_LABELS[arm]} | {row['P@5']:.4f} | {row['Recall@5']:.4f} | {row['MRR@5']:.4f} | "
            f"{row['MAP@5']:.4f} | {row['nDCG@5']:.4f} | {row['eligible_count']} | {row['excluded_count']} |"
        )
    lines += [
        "",
        "## Baseline contract and replay parity",
        "",
        f"- Formal B0: `{baseline_diagnosis['formal_b0_pipeline']}`.",
        f"- Phase13F numeric classification: `{baseline_diagnosis['historical_reference_classification']}`.",
        f"- Phase13F values match raw Dense20 Top5: `{baseline_diagnosis['reference_matches_raw_dense20']}` (informational only).",
        f"- Active relevance projection hash: `{manifest['active_relevance_projection_hash']}`.",
        f"- Independent Production replay parity: `{baseline_diagnosis['production_replay_parity']['parity']}`.",
        f"- Diagnosis: {baseline_diagnosis['diagnosis']}",
        "- B0 parity is a prerequisite for B1/B2/B3; no Runtime promotion or Phase 15C action is part of this run.",
        "",
        "## Complementarity",
        "",
        f"Indexable evidence groups: `{complementarity['total_indexable_groups']}`; "
        f"dense-only `{complementarity['counts']['dense_only']}`, "
        f"BM25-only `{complementarity['counts']['bm25_only']}`, "
        f"both `{complementarity['counts']['both']}`, "
        f"neither `{complementarity['counts']['neither']}`.",
        f"BM25 recovered `{len(complementarity['bm25_recovered_from_dense20_miss'])}` indexable Dense20 misses.",
        "",
        "## Failure attribution",
        "",
        "Failure labels are assigned from frozen-index membership, candidate membership and final Top-5 membership; no answer generation was run.",
        "",
        "| Arm | UNINDEXABLE | CANDIDATE_MISS | RERANKER_MISS | SUCCESS |",
        "|---|---:|---:|---:|---:|",
    ]
    for arm in ARM_ORDER:
        counts = failures["arms"][arm]["counts"]
        lines.append(
            f"| {ARM_LABELS[arm]} | {counts['UNINDEXABLE']} | {counts['CANDIDATE_MISS']} | "
            f"{counts['RERANKER_MISS']} | {counts['SUCCESS']} |"
        )
    lines += [
        "",
        "## Gates and decision",
        "",
        f"- All required Phase 15B gates: **{'PASS' if all(gates.values()) else 'FAIL'}**.",
        f"- Recommendation: **{recommendation}**.",
        "- This recommendation is offline evidence only; no Runtime promotion was performed.",
        "",
    ]
    return "\n".join(lines)


def render_b0_stop_report(
    manifest: Mapping[str, Any],
    parity: Mapping[str, Any],
    baseline_diagnosis: Mapping[str, Any],
) -> str:
    """生成 B0 parity 失败时的最小 STOP 记录，不创建后续 arm 产物。"""

    return "\n".join(
        [
            "# EvidenceFlow Phase 15B Hybrid Candidate Retrieval Benchmark v2",
            "",
            "## STOP: production reranked B0 parity failed",
            "",
            f"- Formal B0: `{baseline_diagnosis['formal_b0_pipeline']}`.",
            f"- Active relevance projection hash: `{manifest['active_relevance_projection_hash']}`.",
            f"- Replay parity: `{parity['parity']}`.",
            f"- Run mismatches: `{parity['run_mismatches']}`.",
            f"- Summary mismatches: `{parity['summary_mismatches']}`.",
            "- B1/B2/B3 were not executed. Production Runtime and Phase 15C were untouched.",
            "",
        ]
    )


def run_benchmark(output_dir: Path = OUTPUT_DIR) -> dict[str, Any]:
    """按 B0 parity 前置顺序执行完整 Phase 15B v2 离线 benchmark。"""

    configure_offline_model_cache()
    started = time.perf_counter()
    before_hashes, _ = source_hashes_before_after()
    snapshot, active_manifest, questions_v1, questions_v2, mapping = load_frozen_inputs()
    settings = Settings()
    store = ChromaVectorStore(settings.vector_db_path, settings.collection_name)
    chunks = validate_active_index(settings, snapshot, store)
    question_ids = [item["question_id"] for item in questions_v1["questions"]]
    q_by_id = {item["question_id"]: item for item in questions_v1["questions"]}

    # 独立 replay 与 benchmark 必须使用同一个只读 projection；Hash 是两者的身份锚点。
    groups_by_question, qrels, projection_hash = project_current_groups(chunks, questions_v2)
    indexable_group_count = sum(
        group["indexable"] for groups in groups_by_question.values() for group in groups
    )
    if indexable_group_count != 42:
        raise BenchmarkIdentityError(
            f"current active anchor projection expected 42 indexable groups, got {indexable_group_count}"
        )
    if groups_by_question["ARR-003"][0]["indexable"] or groups_by_question["ARR-033"][0]["indexable"]:
        raise BenchmarkIdentityError("ARR-003 and ARR-033 must remain UNINDEXABLE")

    manifest = build_manifest(
        snapshot, active_manifest, settings, store, chunks, question_ids, projection_hash
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "phase15b_input_manifest.json", manifest)

    # production_reranked_baseline.json 必须来自独立 replay，benchmark runner 只读它。
    baseline_path = output_dir / "production_reranked_baseline.json"
    if not baseline_path.is_file():
        raise BenchmarkIdentityError(
            "independent production_reranked_baseline.json is missing; STOP before B1/B2/B3"
        )
    production_baseline = read_json(baseline_path)
    baseline_identity = {
        "collection_name": COLLECTION_NAME,
        "corpus_signature": CORPUS_SIGNATURE,
        "index_signature": INDEX_SIGNATURE,
        "chunk_id_hash": CHUNK_ID_HASH,
    }
    for field, expected in baseline_identity.items():
        if production_baseline.get(field) != expected:
            raise BenchmarkIdentityError(
                f"production replay baseline identity mismatch: {field}"
            )

    queries = [q_by_id[question_id]["english_query"] for question_id in question_ids]
    embedding_provider = BgeM3EmbeddingProvider(settings.embedding_model)
    query_vectors = embedding_provider.embed_documents(queries)
    if len(query_vectors) != len(question_ids):
        raise AssertionError("query embedding batch count mismatch")

    # B0 Gate 之前只生成 Dense20；因此 parity 失败时不会触发 Dense40、BM25 或 B3。
    dense20: dict[str, list[Any]] = {}
    for question_id, vector in zip(question_ids, query_vectors):
        dense20[question_id] = store.search(vector, top_k=20)

    scorer = TransformersCrossEncoderScorer(
        resolve_local_reranker_model(settings),
        device=settings.reranker_device,
        dtype=settings.reranker_dtype,
        max_length=settings.reranker_max_length,
        inference_mode=False,
    )
    reranker = LearnedReranker(
        scorer,
        batch_size=settings.reranker_batch_size,
        # B0 实际只传入 Production 的 20 个候选；100 仅允许后续离线候选比较。
        max_candidates=100,
    )
    b0_reranked: dict[str, list[Any]] = {}
    b0_scores: dict[str, dict[str, float]] = {}
    reranker_timing: dict[str, float] = {"model_forward": 0.0, "pair_count": 0.0}
    for question_id in question_ids:
        pool = [item.chunk for item in dense20[question_id]]
        scored = reranker.rerank(q_by_id[question_id]["english_query"], pool, top_k=None)
        score_by_id = {result.chunk.chunk_id: result.relevance_score for result in scored}
        b0_scores[question_id] = score_by_id
        b0_reranked[question_id] = rank_scored_candidates(
            pool, [score_by_id[chunk.chunk_id] for chunk in pool], top_k=RERANKER_FINAL_K
        )
        reranker_timing["model_forward"] += float(
            reranker.last_timing_seconds.get("model_forward", 0.0)
        )
        reranker_timing["pair_count"] += len(pool)

    b0_run = {
        question_id: [result.chunk.chunk_id for result in b0_reranked[question_id]]
        for question_id in question_ids
    }
    b0_ir = evaluate_ir_run(qrels, b0_run, depths=(1, 3, 5, 10, 20))
    b0_summary = standard_ir_summary(b0_ir)
    parity = compare_production_replay(
        production_baseline, projection_hash, b0_run, b0_summary, question_ids
    )
    raw_dense_ir = evaluate_ir_run(
        qrels,
        {
            question_id: [item.chunk.chunk_id for item in dense20[question_id]]
            for question_id in question_ids
        },
        depths=(1, 3, 5, 10, 20),
    )
    baseline_diagnosis = baseline_reproduction_diagnosis(
        standard_ir_summary(raw_dense_ir), b0_summary, parity
    )

    if not parity["parity"]:
        # STOP 分支只保留 B0 的审计信息；绝不计算或写入 B1/B2/B3 结果。
        b0_pool_map = {question_id: [item.chunk for item in dense20[question_id]] for question_id in question_ids}
        b0_artifact = reranker_artifact("dense20", b0_reranked, b0_pool_map, question_ids)
        b0_artifact["active_relevance_projection_hash"] = projection_hash
        write_json(output_dir / "reranker" / "dense20_top5.json", b0_artifact)
        write_json(
            output_dir / "final_ir_metrics.json",
            {
                "schema_version": "phase15b-final-ir-v2-stop",
                "active_relevance_projection_hash": projection_hash,
                "qrels_source": "read-only current active anchor projection",
                "arms": {"dense20": b0_ir},
                "summary_at_5": {"dense20": b0_summary},
            },
        )
        (output_dir / "report.md").write_text(
            render_b0_stop_report(manifest, parity, baseline_diagnosis), encoding="utf-8"
        )
        after_hashes = {name: sha256_file(ROOT / name) for name in before_hashes}
        stop_result = {
            "schema_version": "phase15b-execution-result-v2",
            "phase": "15B",
            "status": "STOP",
            "stop_reason": "production_reranked_b0_replay_parity_failed",
            "duration_seconds": time.perf_counter() - started,
            "active_relevance_projection_hash": projection_hash,
            "gates": {
                "production_replay_parity": False,
                "b0_baseline_exact": False,
                "b1_b2_b3_executed": False,
                "production_sources_unchanged": before_hashes == after_hashes,
            },
            "baseline_reproduction": baseline_diagnosis,
            "executed_arms": ["dense20"],
            "skipped_arms": ["dense40", "bm25_20", "dense20_union_bm25_20"],
            "recommendation": "STOP",
            "promotion_performed": False,
            "production_mutated": False,
            "artifacts": [
                "production_reranked_baseline.json",
                "phase15b_input_manifest.json",
                "reranker/dense20_top5.json",
                "final_ir_metrics.json",
                "report.md",
            ],
            "completed_at_epoch": time.time(),
        }
        write_json(output_dir / "execution_result.json", stop_result)
        return stop_result

    # 只有 B0 parity 通过，才允许生成后续三条候选分支。
    dense40: dict[str, list[Any]] = {}
    for question_id, vector in zip(question_ids, query_vectors):
        dense40[question_id] = store.search(vector, top_k=40)
        ids20 = [item.chunk.chunk_id for item in dense20[question_id]]
        ids40 = [item.chunk.chunk_id for item in dense40[question_id]]
        if ids20 != ids40[: len(ids20)]:
            raise AssertionError(f"Dense20 is not a prefix of Dense40 for {question_id}; ANN path changed")

    tokenizer = BgeM3Tokenizer(settings.embedding_model)
    bm25_index = BM25Index(chunks, tokenizer, k1=1.5, b=0.75)
    bm25_20 = {
        question_id: bm25_index.search(q_by_id[question_id]["english_query"], top_k=20)
        for question_id in question_ids
    }
    pools = build_pools(dense20, dense40, bm25_20, question_ids)
    candidate_metrics: dict[str, dict[str, Any]] = {}
    candidate_per_query: dict[str, dict[str, Any]] = {}
    for arm in ARM_ORDER:
        # B3 的每题并集长度可小于 40；指标深度仍按预注册 arm 合同计算。
        depth = 40 if arm in ("dense40", "dense20_union_bm25_20") else 20
        candidate_metrics[arm] = candidate_metric_rows(
            pools[arm], groups_by_question, question_ids, depth
        )
        candidate_per_query[arm] = {
            "ALL_GOLD": candidate_metrics[arm]["per_query"],
            "INDEXABLE_ONLY": candidate_metrics[arm]["per_query"],
        }
        artifact = candidate_artifact(
            arm, pools[arm], dense20, dense40, bm25_20, qrels, question_ids
        )
        artifact["schema_version"] = "phase15b-candidates-v2"
        artifact["active_relevance_projection_hash"] = projection_hash
        write_json(output_dir / "candidates" / f"{arm}.json", artifact)
    write_json(
        output_dir / "candidate_metrics.json",
        {
            "schema_version": "phase15b-candidate-metrics-v2",
            "active_relevance_projection_hash": projection_hash,
            "arms": candidate_metrics,
        },
    )

    reranked: dict[str, dict[str, list[Any]]] = {arm: {} for arm in ARM_ORDER}
    reranker_scores: dict[str, dict[str, dict[str, float]]] = {arm: {} for arm in ARM_ORDER}
    reranked["dense20"] = b0_reranked
    reranker_scores["dense20"] = b0_scores
    for question_id in question_ids:
        # B0 的 20 个分数已经由前置 Gate 生成；这里只对新增候选做一次前向，
        # 既保持 B0 replay 结果不变，也避免在通过 Gate 后重复计算完整 Dense20。
        dense20_ids = {chunk.chunk_id for chunk in pools["dense20"][question_id]}
        scoring_pool: list[DocumentChunk] = []
        seen: set[str] = set(dense20_ids)
        for chunk in pools["dense40"][question_id] + pools["bm25_20"][question_id]:
            if chunk.chunk_id not in seen:
                scoring_pool.append(chunk)
                seen.add(chunk.chunk_id)
        scored = reranker.rerank(
            q_by_id[question_id]["english_query"], scoring_pool, top_k=None
        ) if scoring_pool else []
        score_by_id = dict(b0_scores[question_id])
        score_by_id.update({result.chunk.chunk_id: result.relevance_score for result in scored})
        reranker_timing["model_forward"] += float(
            reranker.last_timing_seconds.get("model_forward", 0.0)
        ) if scoring_pool else 0.0
        reranker_timing["pair_count"] += len(scoring_pool)
        for arm in ("dense40", "bm25_20", "dense20_union_bm25_20"):
            pool = pools[arm][question_id]
            scores = [score_by_id[chunk.chunk_id] for chunk in pool]
            reranker_scores[arm][question_id] = {
                chunk.chunk_id: score for chunk, score in zip(pool, scores)
            }
            ranked = rank_scored_candidates(pool, scores, top_k=RERANKER_FINAL_K)
            if not {result.chunk.chunk_id for result in ranked}.issubset(
                {chunk.chunk_id for chunk in pool}
            ):
                raise AssertionError(f"Reranker changed candidate identity for {question_id}/{arm}")
            reranked[arm][question_id] = ranked
    for arm in ARM_ORDER:
        artifact = reranker_artifact(arm, reranked[arm], pools[arm], question_ids)
        artifact["schema_version"] = "phase15b-reranker-top5-v2"
        artifact["active_relevance_projection_hash"] = projection_hash
        write_json(output_dir / "reranker" / f"{arm}_top5.json", artifact)

    runs = {
        arm: {
            question_id: [result.chunk.chunk_id for result in reranked[arm][question_id]]
            for question_id in question_ids
        }
        for arm in ARM_ORDER
    }
    ir_results = {
        arm: evaluate_ir_run(qrels, runs[arm], depths=(1, 3, 5, 10, 20))
        for arm in ARM_ORDER
    }
    ir_summaries = {arm: standard_ir_summary(ir_results[arm]) for arm in ARM_ORDER}
    write_json(
        output_dir / "final_ir_metrics.json",
        {
            "schema_version": "phase15b-final-ir-v2",
            "active_relevance_projection_hash": projection_hash,
            "production_b0_replay_parity": parity,
            "qrels_source": "read-only current active anchor projection",
            "arms": ir_results,
            "summary_at_5": ir_summaries,
        },
    )
    failures = attribute_failures(pools, reranked, groups_by_question, question_ids)
    failures["active_relevance_projection_hash"] = projection_hash
    write_json(output_dir / "failure_attribution.json", failures)
    complementarity = complementarity_analysis(
        dense20, dense40, bm25_20, pools, groups_by_question, question_ids
    )
    complementarity["active_relevance_projection_hash"] = projection_hash
    write_json(output_dir / "complementarity_analysis.json", complementarity)
    write_json(
        output_dir / "per_query_candidate_results.json",
        {
            "schema_version": "phase15b-per-query-candidate-v2",
            "active_relevance_projection_hash": projection_hash,
            "arms": candidate_metrics,
            "failure_scope": "candidate and reranker only; context/generation not executed",
        },
    )
    diversity = context_diversity(reranked, question_ids)
    diversity["active_relevance_projection_hash"] = projection_hash
    write_json(output_dir / "context_diversity.json", diversity)

    bootstrap = bootstrap_comparisons(candidate_per_query, ir_results, questions_v2, question_ids)
    bootstrap["active_relevance_projection_hash"] = projection_hash
    write_json(output_dir / "bootstrap_comparison.json", bootstrap)

    candidate_gate_names = ("HitRate", "EvidenceAnchorRecall", "AllRequired")
    candidate_monotonicity = all(
        _candidate_metric_value(
            candidate_metrics["dense20_union_bm25_20"],
            cohort,
            f"{name}@{candidate_metrics['dense20_union_bm25_20']['depth']}",
        )
        >= _candidate_metric_value(
            candidate_metrics["dense20"], cohort, f"{name}@{candidate_metrics['dense20']['depth']}"
        )
        - 1e-12
        for cohort in ("ALL_GOLD", "INDEXABLE_ONLY")
        for name in candidate_gate_names
    )
    hybrid_invariants = all(
        len(pools["dense20_union_bm25_20"][qid]) <= 40
        and [chunk.chunk_id for chunk in pools["dense20_union_bm25_20"][qid]][: len(pools["dense20"][qid])]
        == [chunk.chunk_id for chunk in pools["dense20"][qid]]
        and set(chunk.chunk_id for chunk in pools["dense20_union_bm25_20"][qid])
        == set(chunk.chunk_id for chunk in pools["dense20"][qid])
        | set(chunk.chunk_id for chunk in pools["bm25_20"][qid])
        for qid in question_ids
    )
    deterministic_bm25 = all(
        [item.chunk.chunk_id for item in bm25_index.search(q_by_id[qid]["english_query"], top_k=20)]
        == [item.chunk.chunk_id for item in bm25_20[qid]]
        for qid in question_ids
    )
    deterministic_reranker = all(
        [
            result.chunk.chunk_id
            for result in rank_scored_candidates(
                pools[arm][qid],
                [reranker_scores[arm][qid][chunk.chunk_id] for chunk in pools[arm][qid]],
                top_k=RERANKER_FINAL_K,
            )
        ]
        == [result.chunk.chunk_id for result in reranked[arm][qid]]
        for arm in ARM_ORDER
        for qid in question_ids
    )
    # 复算 Evaluation v3 纯函数；该门只验证输出确定性，不改变任何 artifact 输入。
    deterministic_metrics = all(
        evaluate_ir_run(qrels, runs[arm], depths=(1, 3, 5, 10, 20)) == ir_results[arm]
        for arm in ARM_ORDER
    )
    after_hashes = {name: sha256_file(ROOT / name) for name in before_hashes}
    production_unchanged = before_hashes == after_hashes
    superiority = dense40_clear_superiority(ir_summaries)
    gates = {
        "identity_and_frozen_inputs": True,
        "production_replay_parity": parity["parity"],
        # 该字段沿用 B0 名称，但语义已改为独立 replay exact，不再比较 Phase13F 数值。
        "b0_baseline_exact": parity["parity"],
        "dense20_subset_dense40": True,
        "b3_exact_set_union_and_max40": hybrid_invariants,
        "candidate_monotonicity_b3_ge_b0": candidate_monotonicity,
        "no_rrf_or_score_fusion": True,
        "reranker_only_reorders": True,
        "deterministic_bm25": deterministic_bm25,
        "deterministic_reranker": deterministic_reranker,
        "deterministic_metrics": deterministic_metrics,
        "dense40_clear_superiority_over_dense20_and_b3": superiority,
        "production_sources_unchanged": production_unchanged,
    }
    all_gates_pass = all(gates.values())
    recommendation = "DO_NOT_PROMOTE" if all_gates_pass and superiority else "NEEDS_REVIEW"
    report = render_report(
        manifest,
        candidate_metrics,
        ir_summaries,
        baseline_diagnosis,
        complementarity,
        failures,
        gates,
        recommendation,
    )
    (output_dir / "report.md").write_text(report, encoding="utf-8")

    execution_result = {
        "schema_version": "phase15b-execution-result-v2",
        "phase": "15B",
        "status": "PASS" if all_gates_pass else "FAIL",
        "duration_seconds": time.perf_counter() - started,
        "active_relevance_projection_hash": projection_hash,
        "gates": gates,
        "baseline_reproduction": baseline_diagnosis,
        "formal_result": {
            "dense40_clear_superiority_over_dense20_and_b3": superiority,
            "clear_superiority_margin": 0.005,
            "primary_metrics": ["Recall@5", "nDCG@5"],
        },
        "recommendation": recommendation,
        "executed_arms": list(ARM_ORDER),
        "skipped_arms": [],
        "promotion_performed": False,
        "production_mutated": False,
        "reranker_runtime": {
            "model": settings.reranker_model,
            "pair_count": int(reranker_timing["pair_count"]),
            "model_forward_seconds": reranker_timing["model_forward"],
        },
        "unindexable_question_ids": [
            qid
            for qid in question_ids
            if not any(group["indexable"] for group in groups_by_question[qid])
        ],
        "artifacts": [
            "architecture_audit.md",
            "production_reranked_baseline.json",
            "phase15b_input_manifest.json",
            "candidates/*.json",
            "candidate_metrics.json",
            "complementarity_analysis.json",
            "per_query_candidate_results.json",
            "reranker/*_top5.json",
            "final_ir_metrics.json",
            "failure_attribution.json",
            "bootstrap_comparison.json",
            "context_diversity.json",
            "report.md",
        ],
        "completed_at_epoch": time.time(),
    }
    write_json(output_dir / "execution_result.json", execution_result)
    return execution_result


def parse_args() -> argparse.Namespace:
    """解析可选输出目录，默认只写入 Phase 15B 专用目录。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def main() -> int:
    """运行 benchmark；失败时返回非零退出码并保留失败原因。"""

    args = parse_args()
    try:
        result = run_benchmark(args.output_dir)
    except Exception as exc:
        failure = {
            "schema_version": "phase15b-execution-result-v1",
            "phase": "15B",
            "status": "BLOCKED" if isinstance(exc, BenchmarkIdentityError) else "FAIL",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "promotion_performed": False,
        }
        write_json(args.output_dir / "execution_result.json", failure)
        print(json.dumps(failure, ensure_ascii=False))
        return 1
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
