"""Evaluation v3：标准 IR 指标 + 多证据诊断，统一编排。

职责边界（对应 Phase 11A 强制边界）：
- 只做评估计算，绝不调用检索器、Reranker、Embedding 或 Chroma；
- 标准 qrels 从冻结的 Gold Mapping Artifact 派生，不修改 Canonical Gold；
- run（排名）由调用方从冻结历史 artifact replay 提供，本模块不重跑 Runtime；
- 自定义指标不再作为主指标：仅 evidence_group_recall / all_required_evidence /
  gold_mappability 作为 diagnostic-only 保留，且明确标记 custom=true。
"""

import hashlib
import json
from pathlib import Path
from typing import Any

from evidenceflow.domain.evaluation_v2 import EvaluationDatasetV2, MappingArtifact
from evidenceflow.eval.ir_metrics import STANDARD_DEPTHS, evaluate_ir_run


def _sha256_file(path: Path) -> str:
    """返回文件 SHA-256 十六进制摘要，用于 artifact 溯源与可审计性。"""

    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 16), b""):
            h.update(block)
    return h.hexdigest()


def build_qrels(mapping: MappingArtifact) -> dict[str, list[str]]:
    """从冻结 MappingArtifact 派生标准 IR 二元 qrels。

    规则：
    - 只有状态为 FULLY_MAPPED / FRAGMENTED_ACROSS_CHUNKS 的组进入 qrels；
    - UNMAPPED / mapping failure 不假装 relevant（不进入 qrels）；
    - 每个已验证 Gold chunk（含 FRAGMENTED 跨块集合中的每一块）作为 rel=1 的相关单元；
    - 不把未标注 chunk 自动创造为 relevant。
    """

    qrels: dict[str, list[str]] = {}
    for group in mapping.groups:
        mapped = group.status in {"FULLY_MAPPED", "FRAGMENTED_ACROSS_CHUNKS"}
        if not mapped:
            continue
        relevant = qrels.setdefault(group.question_id, [])
        for bundle in group.chunk_sets:
            for cid in bundle:
                if cid not in relevant:
                    relevant.append(cid)
    return qrels


def qrels_hash(qrels: dict[str, list[str]]) -> str:
    """对 qrels 做确定性序列化后取 SHA-256，便于跨 artifact 比对。"""

    payload = json.dumps(
        {qid: sorted(cids) for qid, cids in sorted(qrels.items())},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _diagnostic_block(
    ranked_ids: list[str], groups_chunk_sets: list[list[list[str]]], depths: tuple[int, ...]
) -> dict[str, float]:
    """复用 v2.ranking_metrics 计算多证据诊断，只抽取保持语义一致的两个字段。

    若组不可解（理论不会发生，因为 qrels 已保证只纳入已映射组），回退为 0。
    """

    from evidenceflow.eval.v2 import ranking_metrics

    # 去重保留首次出现顺序，避免历史排名中偶发重复破坏严格校验。
    seen: set[str] = set()
    unique = [c for c in ranked_ids if not (c in seen or seen.add(c))]
    try:
        full = ranking_metrics(unique, groups_chunk_sets, depths=depths)
    except ValueError:
        return {
            f"evidence_group_recall_at_{k}": 0.0 for k in depths
        } | {f"all_required_evidence_at_{k}": 0.0 for k in depths}
    return {
        f"evidence_group_recall_at_{k}": full[f"evidence_group_recall_at_{k}"]
        for k in depths
    } | {
        f"all_required_evidence_at_{k}": full[f"all_required_evidence_at_{k}"]
        for k in depths
    }


def evaluate_v3(
    qrels: dict[str, list[str]],
    run: dict[str, list[str]],
    mapping: MappingArtifact,
    *,
    metric_depths: tuple[int, ...] = STANDARD_DEPTHS,
    diagnostic_depths: tuple[int, ...] = (5, 10, 20),
) -> dict:
    """对一次冻结排名 replay 计算 v3 标准指标 + 多证据诊断。"""

    ir = evaluate_ir_run(qrels, run, depths=metric_depths)

    # 按 question_id 归集每题的 chunk_sets（用于多证据诊断）。
    by_question: dict[str, list[list[list[str]]]] = {}
    for group in mapping.groups:
        by_question.setdefault(group.question_id, []).append(group.chunk_sets)

    per_query = {}
    for qid, row in ir["per_query"].items():
        diag = _diagnostic_block(
            list(run.get(qid, [])), by_question.get(qid, []), diagnostic_depths
        )
        per_query[qid] = {
            "question_id": qid,
            "eligible": row["eligible"],
            "num_relevant": row["num_relevant"],
            "standard_metrics": row["metrics"],
            "diagnostics": diag,
        }

    # 聚合诊断（仅 eligible query）。
    eligible = [row for row in per_query.values() if row["eligible"]]
    diag_agg: dict[str, float] = {}
    for k in diagnostic_depths:
        for name in ("evidence_group_recall_at_", "all_required_evidence_at_"):
            key = f"{name}{k}"
            diag_agg[key] = sum(r["diagnostics"][key] for r in eligible) / len(eligible)

    total_groups = len(mapping.groups)
    mapped_groups = sum(
        1 for g in mapping.groups if g.status in {"FULLY_MAPPED", "FRAGMENTED_ACROSS_CHUNKS"}
    )
    gold_mappability_rate = mapped_groups / total_groups if total_groups else None

    return {
        "evaluation_version": "3.0",
        "depths": list(metric_depths),
        "diagnostic_depths": list(diagnostic_depths),
        "aggregate": {
            "question_count": ir["aggregate"]["question_count"],
            "eligible_count": ir["aggregate"]["eligible_count"],
            "precision_at_k": ir["aggregate"]["precision_at_k"],
            "recall_at_k": ir["aggregate"]["recall_at_k"],
            "mrr_at_k": ir["aggregate"]["mrr_at_k"],
            "map_at_k": ir["aggregate"]["map_at_k"],
            "ndcg_at_k": ir["aggregate"]["ndcg_at_k"],
            "diagnostics": diag_agg,
            "gold_mappability_rate": gold_mappability_rate,
        },
        "per_query": per_query,
        "excluded_queries": ir["excluded_queries"],
    }


def save_artifact(payload: dict, path: Path) -> None:
    """写入版本化 JSON。仅向 evals/evaluation_v3/standard_ir_v1/ 等新目录落盘，
    绝不向 evals/evaluation_v2/ 等历史 artifact 目录写入（对应 Step 13 不覆盖规则）。
    """

    text = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        output.write(text)
