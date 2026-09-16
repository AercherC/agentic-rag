#!/usr/bin/env python
"""Phase 11A：Standard IR Evaluation Migration —— 冻结 ranking replay 重算脚本。

输入（全部冻结，只读）：
  - evals/evaluation_v2/phase35_gold_stabilization_v1/gold_mapping_v3.json  (Gold Mapping)
  - evals/evaluation_v2/phase6_retrieval_baselines_v1/b0_original_dense.json
  - evals/evaluation_v2/phase7_reranker_validation_v1/reranker_results.json (c0_original_dense)
  - evals/evaluation_v2/phase9_runtime_promotion_v1/runtime_after.json
  - evals/evaluation_v2/phase10_reranker_performance_v1/baseline.json

输出：
  evals/evaluation_v3/standard_ir_v1/ 下全部 v3 artifact。

强制边界：
  - 不修改 Runtime / Gold / Chunk / Embedding / Chroma；
  - 只 replay 冻结排名，不重跑检索或 Reranker；
  - 不覆盖任何 Phase 6–10 历史 JSON（新文件独立落盘，排他创建）。
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, "src")

from pydantic import TypeAdapter

from evidenceflow.domain.evaluation_v2 import MappingArtifact
from evidenceflow.eval.v3 import (
    _sha256_file,
    build_qrels,
    evaluate_v3,
    qrels_hash,
    save_artifact,
)

EV2 = Path("evals/evaluation_v2")
OUT = Path("evals/evaluation_v3/standard_ir_v1")
MAPPING_PATH = EV2 / "phase35_gold_stabilization_v1" / "gold_mapping_v3.json"
FROZEN_DATASET_HASH = "be1e8354babef3927163c19e32c614f7639323095900c487d559ade8249e7357"

# 标准截断集合（对应 Step 3-7）。
FULL_DEPTHS = (1, 3, 5, 10, 20)
RUNTIME_DEPTHS = (1, 3, 5, 10)  # 当前 Runtime 冻结排名仅 Top-10，@20 不可重建


def load(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def extract_run(*, source: str) -> dict[str, list[str]]:
    """从各历史 artifact 抽取 question_id -> 排名后 chunk_id 列表。"""

    data = load(EV2 / source)
    if source.endswith("b0_original_dense.json"):
        return {r["question_id"]: list(r["ranked_chunk_ids"]) for r in data["per_question"]}
    if source.endswith("reranker_results.json"):
        baseline = data["c0_original_dense"]["20"]["baseline"]
        return {r["question_id"]: list(r["ranked_chunk_ids"]) for r in baseline["per_question"]}
    if source.endswith("runtime_after.json") or source.endswith("baseline.json"):
        return {qid: list(ids) for qid, ids in data["rankings"].items()}
    raise ValueError(f"unknown run source: {source}")


def recompute(source_phase: str, source_rel: str, run: dict, mapping: MappingArtifact,
              qrels: dict, depths: tuple[int, ...], configuration: dict) -> dict:
    """对一次冻结排名 replay 计算 v3 结果，并封装溯源元数据。"""

    source_path = EV2 / source_rel
    result = evaluate_v3(qrels, run, mapping, metric_depths=depths)
    payload = {
        "artifact_type": "phase11a_standard_ir_recomputed",
        "source_phase": source_phase,
        "source_artifact": str(source_path),
        "source_hash": _sha256_file(source_path),
        "qrels_hash": qrels_hash(qrels),
        "dataset_hash": FROZEN_DATASET_HASH,
        "evaluation_version": "3.0",
        "metric_semantics": "standard_ir_trec_beir_binary",
        "dataset_id": mapping.dataset_id,
        "corpus_version": mapping.corpus_version,
        "index_version": mapping.index_version,
        "configuration": configuration,
        "depths": list(depths),
        "depth20_status": (
            "non_reconstructable_frozen_ranking_depth_10"
            if 20 not in depths
            else "full"
        ),
        "aggregate": result["aggregate"],
        "per_query": result["per_query"],
        "excluded_queries": result["excluded_queries"],
    }
    return payload


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    mapping = TypeAdapter(MappingArtifact).validate_python(load(MAPPING_PATH))
    qrels = build_qrels(mapping)
    qh = qrels_hash(qrels)

    # 抽取四个历史 run（只读，不改任何源）。
    runs = {
        "phase6": extract_run(source="phase6_retrieval_baselines_v1/b0_original_dense.json"),
        "phase7": extract_run(source="phase7_reranker_validation_v1/reranker_results.json"),
        "phase9": extract_run(source="phase9_runtime_promotion_v1/runtime_after.json"),
        "phase10": extract_run(source="phase10_reranker_performance_v1/baseline.json"),
    }

    # 校验 query 对齐。
    for name, run in runs.items():
        missing = set(qrels) - set(run)
        extra = set(run) - set(qrels)
        assert not missing, f"{name}: qrels 中 {len(missing)} 个 query 无 ranking"
        assert not extra, f"{name}: ranking 含 {len(extra)} 个 qrels 外 query（{list(extra)[:3]}）"

    # 1) qrels + manifest
    qrels_out = {qid: cids for qid, cids in qrels.items()}
    save_artifact(qrels_out, OUT / "qrels.json")
    save_artifact(
        {
            "qrels_hash": qh,
            "dataset_id": mapping.dataset_id,
            "corpus_version": mapping.corpus_version,
            "index_version": mapping.index_version,
            "retrieval_unit": "chunk_id",
            "query_unit": "question_id",
            "relevance": "binary (relevance=1)",
            "included_status": ["FULLY_MAPPED", "FRAGMENTED_ACROSS_CHUNKS"],
            "excluded_status": ["UNMAPPED", "PARTIALLY_MAPPED", "SOURCE_EXTRACTION_FAILURE",
                                 "ANCHOR_NORMALIZATION_FAILURE", "AMBIGUOUS_MAPPING"],
            "num_questions": len(qrels),
            "num_relevant_chunks": sum(len(v) for v in qrels.values()),
            "derived_from": str(MAPPING_PATH),
            "derived_from_hash": _sha256_file(MAPPING_PATH),
            "note": "qrels 不修改 Canonical Gold，仅从冻结 Gold Mapping 派生。",
        },
        OUT / "qrels_manifest.json",
    )

    # 2) 四个历史 artifact 重算（不覆盖原文件）。
    configs = {
        "phase6": ("phase6_retrieval_baselines_v1/b0_original_dense.json", FULL_DEPTHS,
                   {"variant": "original_dense", "reranker": None}),
        "phase7": ("phase7_reranker_validation_v1/reranker_results.json", FULL_DEPTHS,
                   {"variant": "c0_original_dense_baseline", "reranker": None}),
        "phase9": ("phase9_runtime_promotion_v1/runtime_after.json", RUNTIME_DEPTHS,
                   {"strategy": "dense_reranker", "reranker": "BGE Reranker", "final_k": 5}),
        "phase10": ("phase10_reranker_performance_v1/baseline.json", RUNTIME_DEPTHS,
                    {"strategy": "dense_reranker", "reranker": "BGE Reranker", "final_k": 5}),
    }
    recomputed = {}
    for name, (rel, depths, cfg) in configs.items():
        payload = recompute(name, rel, runs[name], mapping, qrels, depths, cfg)
        recomputed[name] = payload
        save_artifact(payload, OUT / f"{name}_recomputed.json")

    # 3) 标准基线对比：Dense Baseline (phase6) vs Current Runtime (phase9)。
    def agg(name):
        return recomputed[name]["aggregate"]

    dense = agg("phase6")
    rt = agg("phase9")
    metric_keys = [
        ("precision_at_k", "Precision@5", 5),
        ("recall_at_k", "Recall@5", 5),
        ("recall_at_k", "Recall@10", 10),
        ("recall_at_k", "Recall@20", 20),
        ("mrr_at_k", "MRR@5", 5),
        ("map_at_k", "MAP@5", 5),
        ("ndcg_at_k", "nDCG@5", 5),
    ]
    table = {}
    for prefix, label, k in metric_keys:
        db = dense[prefix][k]
        cr = rt[prefix].get(k)
        table[label] = {
            "dense_baseline": db,
            "current_runtime": cr,
            "delta": (cr - db) if cr is not None else None,
            "current_runtime_note": (
                None if cr is not None
                else "N/A: 当前 Runtime 冻结排名仅 Top-10，标准 @20 不可重建"
            ),
        }

    # wins / ties / losses（逐题标准指标配对比较，@20 跳过）。
    dense_pq = recomputed["phase6"]["per_query"]
    rt_pq = recomputed["phase9"]["per_query"]
    compare_metrics = [
        ("precision_at_k", 5), ("recall_at_k", 5), ("recall_at_k", 10),
        ("mrr_at_k", 5), ("map_at_k", 5), ("ndcg_at_k", 5),
    ]
    wins = ties = losses = 0
    per_query_deltas = {}
    for qid in dense_pq:
        if not (dense_pq[qid]["eligible"] and rt_pq[qid]["eligible"]):
            continue
        dm = dense_pq[qid]["standard_metrics"]
        rm = rt_pq[qid]["standard_metrics"]
        qwins = qties = qlosses = 0
        for prefix, k in compare_metrics:
            pq_key = prefix.replace("_at_k", f"_at_{k}")
            d = dm[pq_key]
            r = rm[pq_key]
            if r > d + 1e-9:
                qwins += 1
            elif r < d - 1e-9:
                qlosses += 1
            else:
                qties += 1
        per_query_deltas[qid] = {"wins": qwins, "ties": qties, "losses": qlosses}
        if qwins > qlosses:
            wins += 1
        elif qlosses > qwins:
            losses += 1
        else:
            ties += 1

    save_artifact(
        {
            "evaluation_version": "3.0",
            "dataset_hash": FROZEN_DATASET_HASH,
            "qrels_hash": qh,
            "dense_baseline_source": "phase6_retrieval_baselines_v1/b0_original_dense.json",
            "current_runtime_source": "phase9_runtime_promotion_v1/runtime_after.json",
            "current_runtime_depth20": "non_reconstructable_frozen_ranking_depth_10",
            "metrics": table,
            "wins_ties_losses": {"wins": wins, "ties": ties, "losses": losses,
                                  "evaluated_questions": wins + ties + losses},
            "per_query_deltas": per_query_deltas,
        },
        OUT / "standard_baseline_comparison.json",
    )

    # 4) 逐题结果（四个配置合并）。
    save_artifact(
        {name: recomputed[name]["per_query"] for name in recomputed},
        OUT / "per_query_results.json",
    )

    # 5) resume_metrics（仅标准指标；当前 Runtime @20 标 null）。
    def std_block(agg_):
        return {
            "Precision@1": agg_["precision_at_k"][1],
            "Precision@5": agg_["precision_at_k"][5],
            "Recall@5": agg_["recall_at_k"][5],
            "Recall@10": agg_["recall_at_k"][10],
            "Recall@20": agg_["recall_at_k"].get(20),
            "MRR@5": agg_["mrr_at_k"][5],
            "MAP@5": agg_["map_at_k"][5],
            "nDCG@5": agg_["ndcg_at_k"][5],
            "nDCG@10": agg_["ndcg_at_k"][10],
        }

    db_block = std_block(dense)
    rt_block = std_block(rt)
    resume = {
        "evaluation_version": "3.0",
        "dataset_hash": FROZEN_DATASET_HASH,
        "qrels_hash": qh,
        "dense_baseline": db_block,
        "current_runtime": rt_block,
        "delta": {k: (rt_block[k] - db_block[k] if (rt_block[k] is not None and db_block[k] is not None) else None)
                   for k in db_block},
        "excluded_custom_metrics": [
            "CompleteEvidenceRR", "custom EvidenceGroup nDCG", "custom accuracy"
        ],
        "note": "resume_metrics 只含标准 IR 指标；当前 Runtime Recall@20 为 null（冻结排名仅 Top-10）。",
    }
    save_artifact(resume, OUT / "resume_metrics.json")

    print("Dense Baseline nDCG@5 =", round(dense["ndcg_at_k"][5], 4))
    print("Current Runtime nDCG@5 =", round(rt["ndcg_at_k"][5], 4))
    print("Reranker delta nDCG@5 =", round(rt["ndcg_at_k"][5] - dense["ndcg_at_k"][5], 4))
    print("Wins/Ties/Losses =", wins, ties, losses)
    print("Artifacts written under", OUT)


if __name__ == "__main__":
    main()
