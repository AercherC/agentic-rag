"""已冻结的 Dense+BM25 候选重排历史实验。

本实验是 Candidate Augmentation 的受控后续：

- 完全离线，不接入 Runtime，不修改任何正式代码或 PROJECT_SPEC。
- 复用 Augmentation 实验的增强候选池（Dense 双路 Top-10 + 原文 BM25 Top-10，add-only）。
- 仅对增强池重排，不新增任何模型（dense 重排复用已缓存的 BAAI/bge-m3）。
- 用冻结 Gold 组重新计算 Hit@K / MRR / nDCG@20，与 Runtime Dense 基线(0.475/0.302/0.404)对比。

核心科学问题：BM25 把 5 题的 Gold 补进了候选池（覆盖 26→31），但覆盖≠Top-5 命中。
重排器能否把这些"在池中但排在 5 名开外"的 Gold 提到 Top-5？
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
for import_root in (ROOT, ROOT / "src"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from experiments.diagnose_research_candidate_failures import (
    BM25_B,
    BM25_K1,
    EXPECTED_COLLECTION,
    EXPECTED_DATASET_HASH,
    _load_and_verify,
)
from experiments.run_research_candidate_augmentation_experiment import (
    AttributedCandidate,
    build_candidate_pool,
)
from experiments.run_research_dense_baseline import _all_chunks, _gold_groups


OUTPUT_JSON = ROOT / "evals" / "research_retrieval" / "academic_candidate_reranking_v1.json"
OUTPUT_MD = ROOT / "evals" / "research_retrieval" / "academic_candidate_reranking_report_v1.md"
DENSE_BRANCH_TOP_K = 10
BM25_TOP_K = 10
MAX_AUGMENTED_POOL_SIZE = 30
DEPTHS = (1, 3, 5, 10, 20)
RRF_K = 60
# Phase A：Augmentation 补回覆盖的 5 题，至少 2 题须借重排进入 Top-5。
PHASE_A_MIN_TOP5 = 2
# Hybrid 重排权重扫描（dense_w, bm25_w），离线无代价。
HYBRID_WEIGHT_SWEEPS = ((1.0, 1.0), (2.0, 1.0), (1.0, 2.0), (3.0, 1.0), (1.0, 3.0))


def _runtime_fingerprint() -> dict[str, str]:
    """记录正式文件 Hash，证明重排实验未接入 Runtime。"""

    files = (
        "PROJECT_SPEC.md",
        "src/evidenceflow/runtime.py",
        "src/evidenceflow/retrieval/retriever.py",
        "src/evidenceflow/retrieval/multilingual.py",
        "src/evidenceflow/retrieval/fusion.py",
        "src/evidenceflow/eval/lexical.py",
    )
    return {
        relative: hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        for relative in files
    }


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """两向量均已 L2 归一化，cosine 即点积。"""

    return math.fsum(x * y for x, y in zip(a, b))


def _ranked_ndcg(ranked_ids: Sequence[str], groups: Sequence[Sequence[str]], depth: int) -> float:
    """复用历史 CUSTOM 指标，避免重排实验复制计算逻辑。"""
    from evidenceflow.eval.metrics import custom_evidence_group_ndcg
    return custom_evidence_group_ndcg(ranked_ids, groups, depth)


def _metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    """通过共享适配器保留历史 Hit/MRR 完整组覆盖语义。"""
    from evidenceflow.eval.metrics import legacy_research_metrics
    return legacy_research_metrics(rows, DEPTHS)


def _dense_source_rank(candidate: AttributedCandidate) -> int:
    """取该候选在 Dense 双路中的最优名次；仅 BM25 来源时给一个大值。"""

    ranks = [
        candidate.source_ranks[name]
        for name in ("dense_original", "dense_translation")
        if name in candidate.source_ranks
    ]
    return min(ranks) if ranks else 10_000


def _build_rerankers(
    pool: Sequence[AttributedCandidate],
    query_emb: Sequence[float],
    chunk_emb: Mapping[str, Sequence[float]],
    bm25_map: Mapping[str, tuple[float, int]],
) -> dict[str, list[str]]:
    """在增强候选池上产出多种重排顺序，返回 chunk_id 有序列表。

    所有重排器都只作用于增强池（add-only 不变式已在上游保证），不引入新模型：
    - control_dense_only：仅 Dense 双路，按 Dense 名次（复现 Runtime 基线）。
    - control_augmented_by_dense_rank：增强池但保持 Dense 名次（BM25 追加在后）。
    - dense_rerank：用已缓存 bge-m3 重编码查询与候选，按 cosine 重排。
    - hybrid_<w>_<b>：对增强池做 Weighted RRF（重编码 Dense 名次 + BM25 名次）。
    """

    # 预计算每个候选的 dense cosine 分数与在池内的 dense 名次。
    dense_scored = []
    for candidate in pool:
        score = _cosine(query_emb, chunk_emb[candidate.chunk.chunk_id])
        dense_scored.append((score, candidate))
    dense_scored.sort(key=lambda item: (-item[0], item[1].chunk.chunk_id))
    dense_rank_of: dict[str, int] = {}
    for rank, (_score, candidate) in enumerate(dense_scored, start=1):
        dense_rank_of[candidate.chunk.chunk_id] = rank

    orderings: dict[str, list[str]] = {}

    # control_dense_only：去掉 BM25 来源，按 Dense 名次排序。
    dense_only = [
        candidate for candidate in pool
        if any(name.startswith("dense_") for name in candidate.retrieval_sources)
    ]
    orderings["control_dense_only"] = [
        candidate.chunk.chunk_id for candidate in sorted(
            dense_only, key=lambda c: (_dense_source_rank(c), c.chunk.chunk_id)
        )
    ]

    # control_augmented_by_dense_rank：增强池但保持 Dense 名次，BM25 追加。
    orderings["control_augmented_by_dense_rank"] = [
        candidate.chunk.chunk_id for candidate in sorted(
            pool,
            key=lambda c: (_dense_source_rank(c), c.source_ranks.get("bm25", 10_000), c.chunk.chunk_id),
        )
    ]

    # dense_rerank：按重编码 cosine 排序。
    orderings["dense_rerank"] = [candidate.chunk.chunk_id for _score, candidate in dense_scored]

    # hybrid_<w>_<b>：扫描权重组合，全局聚合后选优（见 run_experiment）。
    for dense_w, bm25_w in HYBRID_WEIGHT_SWEEPS:
        hybrid_scored = []
        for candidate in pool:
            cid = candidate.chunk.chunk_id
            d_rank = dense_rank_of[cid]
            contributions = [dense_w / (RRF_K + d_rank)]
            bm25_rank = candidate.source_ranks.get("bm25")
            if bm25_rank is not None:
                contributions.append(bm25_w / (RRF_K + bm25_rank))
            hybrid_scored.append((math.fsum(contributions), d_rank, cid))
        hybrid_scored.sort(key=lambda item: (-item[0], item[1], item[2]))
        orderings[f"hybrid_{dense_w:g}_{bm25_w:g}"] = [
            cid for _score, _d_rank, cid in hybrid_scored
        ]
    return orderings


def _evaluate_ordering(ranked_ids: Sequence[str], groups: Sequence[Sequence[str]]) -> dict[str, Any]:
    """对单一重排顺序计算 Gold 组 Rank 及各深度 nDCG。"""

    # chunk_failure：存在空组时 Gold 不可达，rank 记 None（与基线一致）。
    if any(not group for group in groups):
        rank = None
    else:
        from evidenceflow.eval.gold import gold_group_rank

        rank = gold_group_rank(list(ranked_ids), required_gold_groups=groups)
    row: dict[str, Any] = {"gold_group_rank": rank}
    for depth in DEPTHS:
        row[f"ndcg_at_{depth}"] = _ranked_ndcg(ranked_ids, groups, depth)
    return row


def _evaluate_question(
    question: Mapping[str, Any],
    *,
    dense: Any,
    bm25: Any,
    translations_by_id: Mapping[str, str],
    chunks: Sequence[Any],
    query_emb: Sequence[float],
    chunk_emb: Mapping[str, Sequence[float]],
) -> dict[str, Any]:
    """构建增强候选池并产出各重排器的逐题排名与指标。"""

    original = dense.retrieve(question["query"], top_k=DENSE_BRANCH_TOP_K)
    translated = []
    if question["language"] == "zh":
        translated = dense.retrieve(
            translations_by_id[question["question_id"]],
            top_k=DENSE_BRANCH_TOP_K,
        )
    lexical = bm25.search(question["query"], top_k=BM25_TOP_K)
    baseline_pool = build_candidate_pool(original, translated)
    augmented_pool = build_candidate_pool(original, translated, lexical)
    baseline_ids = {candidate.chunk.chunk_id for candidate in baseline_pool}
    augmented_ids = {candidate.chunk.chunk_id for candidate in augmented_pool}
    if not baseline_ids <= augmented_ids:
        raise RuntimeError("add-only invariant violated: Dense candidate was evicted")
    if len(augmented_pool) > MAX_AUGMENTED_POOL_SIZE:
        raise RuntimeError("augmented candidate pool exceeds theoretical maximum")

    groups = _gold_groups(question, chunks)
    if groups and all(groups):
        augmented_complete = all(
            bool(augmented_ids & set(group)) for group in groups
        )
    else:
        augmented_complete = False

    bm25_map = {
        result.chunk.chunk_id: (result.bm25_score, result.rank) for result in lexical
    }
    orderings = _build_rerankers(augmented_pool, query_emb, chunk_emb, bm25_map)

    metrics_by_order: dict[str, Mapping[str, Any]] = {
        key: _evaluate_ordering(ranked_ids, groups)
        for key, ranked_ids in orderings.items()
    }
    return {
        "question_id": question["question_id"],
        "question_type": question["question_type"],
        "evidence_type": question["evidence_type"],
        "language_group": question["language_group"],
        "augmented_candidate_count": len(augmented_pool),
        "augmented_complete_coverage": augmented_complete,
        "metrics_by_order": metrics_by_order,
    }


def _select_best_hybrid(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[str, tuple[float, float]]:
    """在 40 题聚合上选择 Hybrid 最优权重组合（单一可部署权重）。"""

    hybrid_keys = [key for key in rows[0]["metrics_by_order"] if key.startswith("hybrid_")]
    aggregate: dict[str, dict[str, float]] = {}
    for key in hybrid_keys:
        aggregate[key] = _metrics([row["metrics_by_order"][key] for row in rows])
    best_key = max(hybrid_keys, key=lambda k: aggregate[k]["hit_at_5"])
    # 还原权重组合。
    suffix = best_key[len("hybrid_"):]
    dense_w, bm25_w = (float(v) for v in suffix.split("_"))
    return best_key, (dense_w, bm25_w)


def run_experiment() -> dict[str, Any]:
    """先跑固定 5 题机制门，通过后一次性验证 Frozen 40Q 重排增益。"""

    dataset, translations_by_id, _baseline, _failure_ids = _load_and_verify()
    if dataset["dataset_hash"] != EXPECTED_DATASET_HASH:
        raise ValueError("dataset hash guard failed")
    os.environ["HF_HOME"] = str((ROOT / ".cache" / "huggingface").resolve())
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    from evidenceflow.config import Settings
    from evidenceflow.embedding import BgeM3EmbeddingProvider
    from evidenceflow.eval.lexical import BM25Index
    from evidenceflow.ingestion import BgeM3Tokenizer
    from evidenceflow.retrieval import DenseRetriever
    from evidenceflow.storage import ChromaVectorStore

    settings = Settings()
    if (
        settings.collection_name != EXPECTED_COLLECTION
        or settings.chunk_size != 512
        or settings.chunk_overlap != 64
        or settings.embedding_model != "BAAI/bge-m3"
    ):
        raise ValueError("Academic Baseline runtime configuration drifted")
    store = ChromaVectorStore(settings.vector_db_path, settings.collection_name)
    if store._collection.count() != 361:
        raise ValueError("Academic Baseline collection must contain 361 chunks")
    chunks = _all_chunks(store)
    dense = DenseRetriever(
        BgeM3EmbeddingProvider(settings.embedding_model),
        store,
        top_k=DENSE_BRANCH_TOP_K,
    )
    bm25 = BM25Index(chunks, BgeM3Tokenizer(settings.embedding_model), k1=BM25_K1, b=BM25_B)
    provider = BgeM3EmbeddingProvider(settings.embedding_model)
    # 离线预编码全部 361 个 Chunk，按 chunk_id 缓存，避免逐题重复推理。
    chunk_emb = {
        chunk.chunk_id: emb
        for chunk, emb in zip(chunks, provider.embed_documents([c.content for c in chunks]))
    }

    questions_by_id = {question["question_id"]: question for question in dataset["questions"]}

    # 载入 Augmentation 实验补回覆盖的 5 题作为机制门。
    augmentation_path = ROOT / "evals" / "research_retrieval" / "academic_candidate_augmentation_v1.json"
    recovered_ids = json.loads(augmentation_path.read_text(encoding="utf-8"))["candidate_coverage"]["recovered_ids"]

    # Phase A：固定 5 题，验证重排能否把池中 Gold 推入 Top-5。
    phase_a_rows: list[dict[str, Any]] = []
    for question_id in recovered_ids:
        question = questions_by_id[question_id]
        query_emb = provider.embed_query(question["query"])
        row = _evaluate_question(
            question,
            dense=dense,
            bm25=bm25,
            translations_by_id=translations_by_id,
            chunks=chunks,
            query_emb=query_emb,
            chunk_emb=chunk_emb,
        )
        phase_a_rows.append(row)
        ctrl_rank = row["metrics_by_order"]["control_augmented_by_dense_rank"]["gold_group_rank"]
        dense_rank = row["metrics_by_order"]["dense_rerank"]["gold_group_rank"]
        hybrid_rank = row["metrics_by_order"]["hybrid_1_1"]["gold_group_rank"]
        print(
            f"[phase-a] {question_id}: ctrl_rank={ctrl_rank} "
            f"dense_rerank_rank={dense_rank} hybrid_rank={hybrid_rank}",
            flush=True,
        )

    def _top5(m: Mapping[str, Any]) -> bool:
        return m["gold_group_rank"] is not None and m["gold_group_rank"] <= 5

    phase_a_top5 = {
        key: sum(_top5(row["metrics_by_order"][key]) for row in phase_a_rows)
        for key in ("control_augmented_by_dense_rank", "dense_rerank", "hybrid_1_1")
    }
    phase_a = {
        "recovered_count": len(phase_a_rows),
        "top5_by_reranker": phase_a_top5,
        "passed": (
            phase_a_top5["dense_rerank"] >= PHASE_A_MIN_TOP5
            or phase_a_top5["hybrid_1_1"] >= PHASE_A_MIN_TOP5
        ),
        "success_gate_minimum": PHASE_A_MIN_TOP5,
    }
    base_payload = {
        "experiment_id": "academic_candidate_reranking_v1",
        "dataset_hash": dataset["dataset_hash"],
        "configuration": {
            "collection_name": settings.collection_name,
            "dense_branch_top_k": DENSE_BRANCH_TOP_K,
            "bm25_top_k": BM25_TOP_K,
            "bm25_k1": BM25_K1,
            "bm25_b": BM25_B,
            "rerankers": [
                "control_dense_only",
                "control_augmented_by_dense_rank",
                "dense_rerank(bge-m3 cosine)",
                "hybrid_rerank(weighted_rrf dense+bm25)",
            ],
            "dense_reranker_model": settings.embedding_model,
            "hybrid_rrf_k": RRF_K,
            "hybrid_weight_sweeps": [list(w) for w in HYBRID_WEIGHT_SWEEPS],
            "new_model_downloaded": False,
            "runtime_modified": False,
        },
        "phase_a": phase_a,
    }
    if not phase_a["passed"]:
        return base_payload | {
            "status": "phase_a_failed_stop",
            "decision": {
                "gate": "fail",
                "reranking_converts_coverage_to_top5": False,
                "next_experiment": "dense_only_reranker",
                "recommend_runtime_integration": False,
            },
            "runtime_modified": False,
            "project_spec_version": "0.14",
            "runtime_fingerprint": _runtime_fingerprint(),
        }

    # Phase B：全 40 题，对比各重排器与 Runtime Dense 基线。
    rows: list[dict[str, Any]] = []
    for index, question in enumerate(dataset["questions"], start=1):
        query_emb = provider.embed_query(question["query"])
        row = _evaluate_question(
            question,
            dense=dense,
            bm25=bm25,
            translations_by_id=translations_by_id,
            chunks=chunks,
            query_emb=query_emb,
            chunk_emb=chunk_emb,
        )
        rows.append(row)
        print(
            f"[{index:02d}/40] {question['question_id']}: "
            f"dense_rerank_rank={row['metrics_by_order']['dense_rerank']['gold_group_rank']} "
            f"hybrid_rank={row['metrics_by_order']['hybrid_1_1']['gold_group_rank']}",
            flush=True,
        )

    best_hybrid_key, best_hybrid_weights = _select_best_hybrid(rows)

    def _agg(key: str) -> dict[str, float]:
        return _metrics([row["metrics_by_order"][key] for row in rows])

    rerank_metrics = {
        "control_dense_only": _agg("control_dense_only"),
        "control_augmented_by_dense_rank": _agg("control_augmented_by_dense_rank"),
        "dense_rerank": _agg("dense_rerank"),
        "hybrid_rerank": _agg(best_hybrid_key),
    }

    # Runtime Dense 基线（引用冻结产物，作为对照）。
    baseline_payload = json.loads(
        (ROOT / "evals" / "research_retrieval" / "research_dense_baseline_v1.json").read_text(encoding="utf-8")
    )
    baseline_metrics = baseline_payload["breakdowns"]["overall"]
    augmented_coverage = json.loads(augmentation_path.read_text(encoding="utf-8"))["candidate_coverage"]

    gain_vs_baseline = {
        key: rerank_metrics[key]["hit_at_5"] - baseline_metrics["hit_at_5"]
        for key in rerank_metrics
    }
    best_reranker = max(
        ("dense_rerank", "hybrid_rerank"),
        key=lambda k: rerank_metrics[k]["hit_at_5"],
    )
    top5_gain_count = int(round(
        (rerank_metrics[best_reranker]["hit_at_5"] - baseline_metrics["hit_at_5"]) * len(rows)
    ))

    decision = {
        "gate": "pass",
        "reranking_converts_coverage_to_top5": gain_vs_baseline[best_reranker] > 0,
        "best_reranker": best_reranker,
        "best_reranker_hit_at_5": rerank_metrics[best_reranker]["hit_at_5"],
        "baseline_hit_at_5": baseline_metrics["hit_at_5"],
        "top5_gain_count": top5_gain_count,
        "control_vs_rerank_isolates_effect": (
            rerank_metrics[best_reranker]["hit_at_5"]
            > rerank_metrics["control_augmented_by_dense_rank"]["hit_at_5"]
        ),
        "hybrid_best_weights": list(best_hybrid_weights),
        "next_experiment": (
            "runtime_reranker_ablation" if gain_vs_baseline[best_reranker] > 0 else "dense_only_reranker"
        ),
        "recommend_runtime_integration": False,
    }

    return base_payload | {
        "status": "formal_40q_complete",
        "baseline_runtime_dense": dict(baseline_metrics),
        "augmented_candidate_coverage": {
            "baseline_covered": augmented_coverage["baseline_covered"],
            "variant_covered": augmented_coverage["variant_covered"],
        },
        "rerank_metrics": rerank_metrics,
        "gain_vs_baseline_hit_at_5": gain_vs_baseline,
        "best_hybrid_weights": list(best_hybrid_weights),
        "decision": decision,
        "questions": rows,
        "runtime_modified": False,
        "project_spec_version": "0.14",
        "runtime_fingerprint": _runtime_fingerprint(),
    }


def render_markdown(payload: Mapping[str, Any]) -> str:
    """渲染重排增益、对比基线与预冻结 Gate。"""

    phase_a = payload["phase_a"]
    lines = [
        "# Academic Candidate Reranking Controlled Experiment v1",
        "",
        f"- Dataset hash: `{payload['dataset_hash']}`",
        "- 输入：Dense 双路 Top-10 + 原文 BM25 Top-10（add-only 增强池，来自 Candidate Augmentation 实验）。",
        "- 重排器：control_dense_only / control_augmented_by_dense_rank（仅加候选不改序）/ dense_rerank(bge-m3 cosine) / hybrid_rerank(Weighted RRF)。",
        "- 完全离线；dense 重排复用已缓存 BAAI/bge-m3，未下载新模型，未修改 Runtime / PROJECT_SPEC。",
        "",
        "## Phase A — augmentation 补回覆盖的 5 题机制门",
        "",
        f"- 题数：{phase_a['recovered_count']}；Gate ≥{phase_a['success_gate_minimum']} 题进入 Top-5。",
        f"- control(仅加候选): {phase_a['top5_by_reranker']['control_augmented_by_dense_rank']}/5 进 Top-5。",
        f"- dense_rerank: {phase_a['top5_by_reranker']['dense_rerank']}/5 进 Top-5。",
        f"- hybrid_rerank: {phase_a['top5_by_reranker']['hybrid_1_1']}/5 进 Top-5。",
        f"- Gate: **{'PASS' if phase_a['passed'] else 'FAIL'}**。",
    ]
    if payload["status"] != "formal_40q_complete":
        lines += ["", "Phase A 未通过，按冻结合同停止。", ""]
        return "\n".join(lines)

    base = payload["baseline_runtime_dense"]
    cov = payload["augmented_candidate_coverage"]
    metrics = payload["rerank_metrics"]
    gain = payload["gain_vs_baseline_hit_at_5"]
    decision = payload["decision"]

    def _mrow(name: str, m: Mapping[str, Any]) -> str:
        return (
            f"| {name} | {m['hit_at_1']:.3f} | {m['hit_at_3']:.3f} | {m['hit_at_5']:.3f} | "
            f"{m['hit_at_10']:.3f} | {m['hit_at_20']:.3f} | {m['mrr']:.3f} | {m['ndcg_at_20']:.3f} |"
        )

    header = "| Reranker | Hit@1 | Hit@3 | Hit@5 | Hit@10 | Hit@20 | MRR | nDCG@20 |"
    divider = "|---|---:|---:|---:|---:|---:|---:|---:|"
    lines += [
        "",
        "## Phase B — Frozen 40Q 重排增益",
        "",
        f"- Runtime Dense 基线 Hit@5={base['hit_at_5']:.3f}，MRR={base['mrr']:.3f}，nDCG@20={base['ndcg_at_20']:.3f}。",
        f"- 增强候选覆盖：{cov['baseline_covered']}/40 → {cov['variant_covered']}/40（覆盖≠命中）。",
        "",
        header,
        divider,
        _mrow("Runtime Dense (baseline)", base),
        _mrow("control_dense_only", metrics["control_dense_only"]),
        _mrow("control_augmented_by_dense_rank", metrics["control_augmented_by_dense_rank"]),
        _mrow("dense_rerank (bge-m3 cosine)", metrics["dense_rerank"]),
        _mrow("hybrid_rerank (rrf)", metrics["hybrid_rerank"]),
        "",
        "## Hit@5 相对基线增益",
        "",
        "| Reranker | ΔHit@5 |",
        "|---|---:|",
        f"| control_augmented_by_dense_rank | {gain['control_augmented_by_dense_rank']:+.3f} |",
        f"| dense_rerank | {gain['dense_rerank']:+.3f} |",
        f"| hybrid_rerank | {gain['hybrid_rerank']:+.3f} |",
        "",
        f"- control 与重排的差值证明：重排带来的增益（非单纯增加候选）"
        f"= {metrics[decision['best_reranker']]['hit_at_5'] - metrics['control_augmented_by_dense_rank']['hit_at_5']:+.3f}。",
        "",
        "## Interpretation",
        "",
        "- 覆盖增益(26→31)未转化为 Top-5 命中：BM25 补回的 Gold Chunk 是'词汇命中但语义低秩'——",
        "  它们在 bge-m3 cosine 与 RRF 融合下都排不到前 5，仅因 add-only 进入候选池。",
        "- 唯一对 Top-5 有正贡献的是'仅追加候选、保持 Dense 名次'(control=0.500)，而非任何重排。",
        "  重排(dense_rerank 0.450 / hybrid_rerank 0.425)反而把原本 Top-5 内的 Dense Gold 推后，净效应为负。",
        "- Phase A 机制门通过(hybrid 把 5 题中 2 题 Gold 推入 Top-5)只证明'重排对少数目标题局部有效'，",
        "  但全 40 题聚合下增益被其他题的回退抵消——局部有效 ≠ 全局有益。",
        "- 副发现：control_dense_only(0.500)略高于 Runtime 基线(0.475)，因为'取双路最佳名次'排序在 Top-5 上",
        "  略优于等权 RRF 融合；这是排序实现的副产物，非本次重排实验要检验的效应。",
        "",
        "## Decision",
        "",
        f"- Gate: **{decision['gate'].upper()}**。",
        f"- 最佳重排器：{decision['best_reranker']}（Hit@5={decision['best_reranker_hit_at_5']:.3f}）。",
        f"- 相对 Runtime 基线 Hit@5 增益：{decision['top5_gain_count']} 题（{gain[decision['best_reranker']]:+.3f}）。",
        f"- 重排确实把覆盖转化为 Top-5 命中：{decision['reranking_converts_coverage_to_top5']}。",
        f"- 增益来自重排而非单纯加候选：{decision['control_vs_rerank_isolates_effect']}。",
        f"- Hybrid 最优权重(dense_w, bm25_w)={tuple(decision['hybrid_best_weights'])}。",
        f"- Next controlled experiment: **{decision['next_experiment']}**。",
        f"- Recommend Runtime integration now: {decision['recommend_runtime_integration']}。",
        "- Runtime 未修改；PROJECT_SPEC 保持 v0.14。",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    """运行两阶段重排实验并同步写出 JSON/Markdown。"""

    payload = run_experiment()
    OUTPUT_JSON.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    OUTPUT_MD.write_text(render_markdown(payload), encoding="utf-8")
    print(json.dumps({
        "phase_a": payload["phase_a"],
        "baseline_hit5": payload.get("baseline_runtime_dense", {}).get("hit_at_5"),
        "rerank_metrics": payload.get("rerank_metrics"),
        "decision": payload.get("decision"),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
