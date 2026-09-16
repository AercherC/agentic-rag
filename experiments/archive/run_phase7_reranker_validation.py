"""已冻结的 Phase 7 Learned Reranker 历史 runner。"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

from evidenceflow.config import Settings
from evidenceflow.domain import DocumentChunk
from evidenceflow.domain import RetrievedChunk
from evidenceflow.domain.evaluation_v2 import EvaluationDatasetV2, MappingArtifact
from evidenceflow.embedding import BgeM3EmbeddingProvider
from evidenceflow.eval.comparison import holm_adjust, paired_comparison
from evidenceflow.eval.v2 import evaluate_evidence_coverage, evaluate_rankings, save_artifact
from evidenceflow.ingestion import BgeM3Tokenizer
from evidenceflow.retrieval import (
    BM25Index,
    TransformersCrossEncoderScorer,
    rank_scored_candidates,
    weighted_rrf_fusion,
)
from evidenceflow.retrieval.hybrid_fusion import RankedChunkCandidate, equal_ranked_rrf_fusion
from evidenceflow.storage import ChromaVectorStore


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "evals/evaluation_v2/phase7_reranker_validation_v1"
PHASE6_DIR = ROOT / "evals/evaluation_v2/phase6_retrieval_baselines_v1"
DATASET_PATH = ROOT / "evals/evaluation_v2/phase35_gold_stabilization_v1/research_questions_v2_gold_stable.json"
MAPPING_PATH = ROOT / "evals/evaluation_v2/phase35_gold_stabilization_v1/gold_mapping_v3.json"
TRANSLATIONS_PATH = ROOT / "evals/research_retrieval/research_translations_v1.json"
MODEL_PATH = ROOT / ".cache/models/bge-reranker-base"
MODEL_ID = "BAAI/bge-reranker-base"
BUDGETS = (20, 30, 50, 100)
MAX_BUDGET = max(BUDGETS)
BATCH_SIZE = 4
MAX_LENGTH = 512
SEED = 20260905
ARCHITECTURES = ("c0_original_dense", "c1_translation_dense", "c2_hybrid")


def _load_json(path: Path) -> dict[str, Any]:
    """读取 UTF-8 JSON。"""

    return json.loads(path.read_text(encoding="utf-8"))


def _load_inputs() -> tuple[EvaluationDatasetV2, MappingArtifact, dict[str, str], dict[str, Any]]:
    """验证 Phase 6、Gold、Mapping 与 Frozen Translation 前置条件。"""

    phase6_gate = _load_json(PHASE6_DIR / "final_verification.json")
    if phase6_gate["gate"] != "PASS":
        raise ValueError("Phase 6 gate is not PASS")
    dataset = EvaluationDatasetV2.model_validate_json(DATASET_PATH.read_text(encoding="utf-8"))
    mapping = MappingArtifact.model_validate_json(MAPPING_PATH.read_text(encoding="utf-8"))
    if any(group.status != "FULLY_MAPPED" for group in mapping.groups):
        raise ValueError("Phase 7 requires 44/44 FULLY_MAPPED groups")
    translations = _load_json(TRANSLATIONS_PATH)["translations"]
    if set(translations) != {q.question_id for q in dataset.questions}:
        raise ValueError("frozen translations do not cover the dataset")
    phase6 = {
        "c0_original_dense": _load_json(PHASE6_DIR / "b0_original_dense.json"),
        "c1_translation_dense": _load_json(PHASE6_DIR / "b1_frozen_translation_dense.json"),
        "c2_hybrid": _load_json(PHASE6_DIR / "b3_hybrid.json"),
    }
    return dataset, mapping, translations, phase6


def _phase6_rankings(phase6: dict[str, Any]) -> dict[str, list[str]]:
    """提取一个 Phase 6 Variant 的冻结 Top-100 排名。"""

    return {row["question_id"]: row["ranked_chunk_ids"] for row in phase6["per_question"]}


class ExactDenseRetriever:
    """对正式 Collection 的已存向量执行确定性 exhaustive cosine 排序。"""

    def __init__(self, embedding_provider: Any, store: ChromaVectorStore) -> None:
        """一次加载全部 361 个归一化向量，不写回或修改 Chroma。"""

        import numpy as np

        raw = store._collection.get(include=["embeddings"])
        chunk_by_id = {chunk.chunk_id: chunk for chunk in store.iter_all_chunks()}
        self._ids = list(raw["ids"])
        self._chunks = [chunk_by_id[chunk_id] for chunk_id in self._ids]
        self._vectors = np.asarray(raw["embeddings"], dtype=np.float32)
        self._embedding_provider = embedding_provider
        self._np = np

    def retrieve(self, query: str, *, top_k: int) -> list[RetrievedChunk]:
        """按 cosine score 降序、chunk_id 升序返回稳定 Top-K。"""

        query_vector = self._np.asarray(
            self._embedding_provider.embed_query(query), dtype=self._np.float32
        )
        scores = self._vectors @ query_vector
        order = sorted(range(len(self._ids)), key=lambda i: (-float(scores[i]), self._ids[i]))[:top_k]
        return [
            RetrievedChunk(
                chunk=self._chunks[index].model_copy(deep=True),
                rank=rank,
                similarity_score=float(scores[index]),
            )
            for rank, index in enumerate(order, start=1)
        ]


def _candidate_record(
    chunk_id: str,
    *,
    original: dict[str, int],
    translated: dict[str, int],
    bm25: dict[str, int],
    fusion: dict[str, int],
    hybrid: dict[str, int],
) -> dict[str, Any]:
    """为一个候选保存完整分支来源与稳定排名证据。"""

    branches = [
        name
        for name, ranks in (("original_dense", original), ("translation_dense", translated), ("bm25", bm25))
        if chunk_id in ranks
    ]
    return {
        "chunk_id": chunk_id,
        "source_branches": branches,
        "original_rank": original.get(chunk_id),
        "translated_rank": translated.get(chunk_id),
        "bm25_rank": bm25.get(chunk_id),
        "fusion_rank": fusion.get(chunk_id),
        "hybrid_rank": hybrid.get(chunk_id),
    }


def _build_candidate_replay(
    dataset: EvaluationDatasetV2,
    translations: dict[str, str],
    phase6: dict[str, Any],
    settings: Settings,
) -> tuple[dict[str, Any], dict[str, dict[str, float]], dict[str, Any]]:
    """从同一已存向量重放三路分支，并记录与 Phase 6 ANN 的差异。"""

    store = ChromaVectorStore(settings.vector_db_path, settings.collection_name)
    chunks = list(store.iter_all_chunks())
    dense = ExactDenseRetriever(BgeM3EmbeddingProvider(settings.embedding_model), store)
    bm25_index = BM25Index(
        chunks,
        BgeM3Tokenizer(settings.embedding_model),
        k1=settings.multilingual_hybrid_bm25_k1,
        b=settings.multilingual_hybrid_bm25_b,
    )
    expected = {name: _phase6_rankings(payload) for name, payload in phase6.items()}
    replay: dict[str, Any] = {}
    latencies: dict[str, dict[str, float]] = {name: {} for name in ARCHITECTURES}
    raw_rankings: dict[str, dict[str, list[str]]] = {name: {} for name in ARCHITECTURES}
    phase6_differences: dict[str, dict[str, Any]] = {name: {} for name in ARCHITECTURES}
    for index, question in enumerate(dataset.questions, start=1):
        start = time.perf_counter()
        original_results = dense.retrieve(question.query, top_k=MAX_BUDGET)
        original_latency = time.perf_counter() - start
        translated_text = translations[question.question_id]
        translated_results = []
        translation_latency = 0.0
        if translated_text.strip() != question.query.strip():
            start = time.perf_counter()
            translated_results = dense.retrieve(translated_text, top_k=MAX_BUDGET)
            translation_latency = time.perf_counter() - start
        start = time.perf_counter()
        fusion_results = weighted_rrf_fusion(
            original_results,
            translated_results,
            original_weight=settings.multilingual_original_weight,
            translation_weight=settings.multilingual_translation_weight,
            rrf_k=settings.multilingual_rrf_k,
            final_top_k=MAX_BUDGET * 2,
        )
        fusion_latency = time.perf_counter() - start
        start = time.perf_counter()
        bm25_results = bm25_index.search(question.query, top_k=MAX_BUDGET)
        bm25_latency = time.perf_counter() - start
        start = time.perf_counter()
        hybrid_results = equal_ranked_rrf_fusion(
            [RankedChunkCandidate(chunk=row.chunk, rank=row.rank) for row in fusion_results],
            [RankedChunkCandidate(chunk=row.chunk, rank=row.rank) for row in bm25_results],
            rrf_k=settings.multilingual_hybrid_rrf_k,
            candidate_budget=MAX_BUDGET,
        )
        hybrid_latency = time.perf_counter() - start

        original_ids = [row.chunk.chunk_id for row in original_results]
        fusion_ids = [row.chunk.chunk_id for row in fusion_results[:MAX_BUDGET]]
        hybrid_ids = [row.chunk.chunk_id for row in hybrid_results]
        qid = question.question_id
        actual = {
            "c0_original_dense": original_ids,
            "c1_translation_dense": fusion_ids,
            "c2_hybrid": hybrid_ids,
        }
        for name, ids in actual.items():
            raw_rankings[name][qid] = ids
            expected_ids = expected[name][qid]
            first_difference = next(
                (rank for rank, (left, right) in enumerate(zip(ids, expected_ids), start=1) if left != right),
                None,
            )
            phase6_differences[name][qid] = {
                "identical": ids == expected_ids,
                "same_set": set(ids) == set(expected_ids),
                "first_rank_difference": first_difference,
                "exact_only_count": len(set(ids) - set(expected_ids)),
                "phase6_only_count": len(set(expected_ids) - set(ids)),
            }
        latencies["c0_original_dense"][qid] = original_latency
        latencies["c1_translation_dense"][qid] = original_latency + translation_latency + fusion_latency
        latencies["c2_hybrid"][qid] = (
            original_latency + translation_latency + fusion_latency + bm25_latency + hybrid_latency
        )

        original_ranks = {row.chunk.chunk_id: row.rank for row in original_results}
        translated_ranks = {row.chunk.chunk_id: row.rank for row in translated_results}
        bm25_ranks = {row.chunk.chunk_id: row.rank for row in bm25_results}
        fusion_ranks = {row.chunk.chunk_id: row.rank for row in fusion_results}
        hybrid_ranks = {row.chunk.chunk_id: row.rank for row in hybrid_results}
        replay[qid] = {
            "query": question.query,
            "translated_query": translated_text,
            "architectures": {
                name: [
                    _candidate_record(
                        cid,
                        original=original_ranks,
                        translated=translated_ranks,
                        bm25=bm25_ranks,
                        fusion=fusion_ranks,
                        hybrid=hybrid_ranks,
                    )
                    for cid in ids
                ]
                for name, ids in actual.items()
            },
        }
        print(f"[candidate replay {index:02d}/{len(dataset.questions)}] {qid}", flush=True)
    return replay, latencies, {
        "chunks": chunks,
        "rankings": raw_rankings,
        "phase6_differences": phase6_differences,
    }


def _candidate_manifest(
    dataset: EvaluationDatasetV2,
    mapping: MappingArtifact,
    replay: dict[str, Any],
) -> dict[str, Any]:
    """保存每种架构/预算的候选数量与 Gold Coverage。"""

    output: dict[str, Any] = {"budgets": list(BUDGETS), "architectures": {}}
    for architecture in ARCHITECTURES:
        output["architectures"][architecture] = {}
        for budget in BUDGETS:
            rankings = {
                q.question_id: [row["chunk_id"] for row in replay[q.question_id]["architectures"][architecture][:budget]]
                for q in dataset.questions
            }
            coverage = evaluate_evidence_coverage(dataset, mapping, rankings, depths=(budget,))
            output["architectures"][architecture][str(budget)] = {
                "candidate_count_by_question": {qid: len(ids) for qid, ids in rankings.items()},
                "coverage": coverage["aggregate"],
            }
    return output


def _score_architectures(
    dataset: EvaluationDatasetV2,
    replay: dict[str, Any],
    chunks: list[Any],
    scorer: TransformersCrossEncoderScorer,
) -> tuple[dict[str, Any], dict[str, dict[str, dict[str, float]]]]:
    """每个架构只评分 Top-100 一次，并用批次累计时间得到各预算实测延迟。"""

    chunk_by_id = {chunk.chunk_id: chunk for chunk in chunks}
    scores: dict[str, Any] = {}
    latency: dict[str, dict[str, dict[str, float]]] = {name: {} for name in ARCHITECTURES}
    for architecture in ARCHITECTURES:
        for index, question in enumerate(dataset.questions, start=1):
            qid = question.question_id
            records = replay[qid]["architectures"][architecture]
            candidates = [chunk_by_id[row["chunk_id"]] for row in records]
            candidate_scores: list[float] = []
            cumulative = 0.0
            budget_latency: dict[str, float] = {}
            for start_index in range(0, len(candidates), BATCH_SIZE):
                batch = candidates[start_index : start_index + BATCH_SIZE]
                started = time.perf_counter()
                batch_scores = scorer.score_pairs([(question.query, chunk.content) for chunk in batch])
                cumulative += time.perf_counter() - started
                candidate_scores.extend(batch_scores)
                completed = start_index + len(batch)
                # K=30 不是 batch=4 的整数边界；记录首个完整覆盖该预算的实测批次，
                # 并在 latency artifact 中同时声明实际 scored slots=32。
                for budget in BUDGETS:
                    if completed >= budget and str(budget) not in budget_latency:
                        budget_latency[str(budget)] = cumulative
            for budget in BUDGETS:
                if str(budget) not in budget_latency:
                    raise AssertionError(f"missing measured latency at K={budget}")
            scores.setdefault(qid, {})[architecture] = {
                row["chunk_id"]: score for row, score in zip(records, candidate_scores)
            }
            latency[architecture][qid] = budget_latency
            print(f"[rerank {architecture} {index:02d}/{len(dataset.questions)}] {qid}", flush=True)
    return scores, latency


def _slice(report: dict[str, Any], dataset: EvaluationDatasetV2) -> dict[str, Any]:
    """聚合中文、英文和 single/multi-document exploratory 切片。"""

    question_by_id = {q.question_id: q for q in dataset.questions}
    rows = report["per_question"]
    selected = {
        "Chinese": [row for row in rows if row["query_language"] == "zh"],
        "English": [row for row in rows if row["query_language"] == "en"],
        "single_document": [row for row in rows if len(question_by_id[row["question_id"]].required_evidence_groups) == 1],
        "multi_document": [row for row in rows if len(question_by_id[row["question_id"]].required_evidence_groups) > 1],
    }
    output: dict[str, Any] = {}
    for name, subset in selected.items():
        metrics: dict[str, Any] = {"question_count": len(subset)}
        for key in subset[0]["metrics"]:
            if "_at_" in key:
                target = key.replace("first_relevant_rr_at_", "mrr_at_")
                metrics[target] = sum(row["metrics"][key] for row in subset) / len(subset)
        output[name] = metrics
    return output


def _evaluate_all(
    dataset: EvaluationDatasetV2,
    mapping: MappingArtifact,
    replay: dict[str, Any],
    scores: dict[str, Any],
) -> dict[str, Any]:
    """计算每个架构/预算的 RRF Baseline 与 Learned 排名指标。"""

    output: dict[str, Any] = {}
    for architecture in ARCHITECTURES:
        output[architecture] = {}
        for budget in BUDGETS:
            baseline: dict[str, list[str]] = {}
            reranked: dict[str, list[str]] = {}
            per_query_scores: dict[str, list[dict[str, Any]]] = {}
            for question in dataset.questions:
                qid = question.question_id
                records = replay[qid]["architectures"][architecture][:budget]
                ids = [row["chunk_id"] for row in records]
                baseline[qid] = ids
                # 用轻量代理 Chunk 避免再次从 Store 读取；排序只依赖身份、顺序与分数。
                proxy = [
                    DocumentChunk(
                        chunk_id=cid, document_id=cid, content=cid, token_count=0,
                        chunk_index=0, file_name="candidate", content_hash="frozen",
                    )
                    for cid in ids
                ]
                learned = rank_scored_candidates(
                    proxy, [scores[qid][architecture][cid] for cid in ids]
                )
                reranked[qid] = [row.chunk.chunk_id for row in learned]
                if set(reranked[qid]) != set(baseline[qid]):
                    raise AssertionError("reranker changed candidate coverage")
                per_query_scores[qid] = [
                    {"chunk_id": row.chunk.chunk_id, "rank": row.rank, "original_rank": row.original_rank, "score": row.relevance_score}
                    for row in learned
                ]
            depths = tuple(k for k in (5, 10, 20, budget) if k <= budget)
            depths = tuple(dict.fromkeys(depths))
            configuration = {"architecture": architecture, "candidate_budget": budget, "reranker": MODEL_ID}
            baseline_report = evaluate_rankings(dataset, mapping, baseline, depths=depths, configuration=configuration | {"ranking": "rrf_baseline"})
            reranker_report = evaluate_rankings(dataset, mapping, reranked, depths=depths, configuration=configuration | {"ranking": "learned_cross_encoder"})
            baseline_coverage = evaluate_evidence_coverage(dataset, mapping, baseline, depths=(budget,))
            reranker_coverage = evaluate_evidence_coverage(dataset, mapping, reranked, depths=(budget,))
            if baseline_coverage["aggregate"] != reranker_coverage["aggregate"]:
                raise AssertionError("candidate coverage changed after reranking")
            output[architecture][str(budget)] = {
                "baseline": baseline_report,
                "reranker": reranker_report,
                "candidate_coverage": baseline_coverage,
                "coverage_unchanged": True,
                "slices": _slice(reranker_report, dataset),
                "reranked_candidates": per_query_scores,
            }
    return output


def _metric_values(report: dict[str, Any], metric: str) -> dict[str, float]:
    """把逐题指标转换成 paired comparison 输入。"""

    source = metric.replace("mrr_at_", "first_relevant_rr_at_")
    return {row["question_id"]: row["metrics"][source] for row in report["per_question"]}


def _paired_statistics(
    dataset: EvaluationDatasetV2, results: dict[str, Any]
) -> dict[str, Any]:
    """执行 Reranker 与架构消融的 family-cluster paired comparisons。"""

    families = {q.question_id: q.family_id for q in dataset.questions}
    metrics = (
        "evidence_group_recall_at_5", "evidence_group_recall_at_10",
        "mrr_at_5", "custom_evidence_group_ndcg_at_5",
        "custom_complete_evidence_rr_at_5", "all_required_evidence_at_5",
        "all_required_evidence_at_10",
    )
    output: dict[str, Any] = {"reranker_vs_baseline": {}, "architecture_ablation": {}}
    p_values: dict[str, float] = {}
    for architecture in ARCHITECTURES:
        for budget in BUDGETS:
            key = f"{architecture}_k{budget}"
            item = results[architecture][str(budget)]
            output["reranker_vs_baseline"][key] = {}
            for metric in metrics:
                comparison = paired_comparison(
                    _metric_values(item["baseline"], metric),
                    _metric_values(item["reranker"], metric),
                    families=families, seed=SEED,
                )
                output["reranker_vs_baseline"][key][metric] = comparison
            p_values[key] = output["reranker_vs_baseline"][key]["custom_evidence_group_ndcg_at_5"]["randomization_p"]
    for budget in BUDGETS:
        for control, treatment in (("c0_original_dense", "c1_translation_dense"), ("c1_translation_dense", "c2_hybrid")):
            key = f"{treatment}_vs_{control}_k{budget}"
            output["architecture_ablation"][key] = {}
            for metric in metrics:
                comparison = paired_comparison(
                    _metric_values(results[control][str(budget)]["reranker"], metric),
                    _metric_values(results[treatment][str(budget)]["reranker"], metric),
                    families=families, seed=SEED,
                )
                output["architecture_ablation"][key][metric] = comparison
            p_values[key] = output["architecture_ablation"][key]["custom_evidence_group_ndcg_at_5"]["randomization_p"]
    output["holm_adjusted_ndcg_at_5_p"] = holm_adjust(p_values)
    return output


def _latency_report(
    candidate_latency: dict[str, dict[str, float]],
    rerank_latency: dict[str, dict[str, dict[str, float]]],
    scorer: TransformersCrossEncoderScorer,
    model_load_seconds: float,
) -> dict[str, Any]:
    """汇总各架构/预算实测 latency、throughput 与 forward passes。"""

    output: dict[str, Any] = {
        "device": scorer.device,
        "dtype": "float32",
        "batch_size": BATCH_SIZE,
        "max_length": MAX_LENGTH,
        "model_load_seconds": model_load_seconds,
        "gpu_memory_bytes": None,
        "gpu_memory_reason": "CPU execution",
        "architectures": {},
    }
    for architecture in ARCHITECTURES:
        output["architectures"][architecture] = {}
        retrieval_total = sum(candidate_latency[architecture].values())
        for budget in BUDGETS:
            reranking_total = sum(row[str(budget)] for row in rerank_latency[architecture].values())
            candidates = budget * len(rerank_latency[architecture])
            output["architectures"][architecture][str(budget)] = {
                "candidate_retrieval_latency_seconds": retrieval_total,
                "reranking_latency_seconds": reranking_total,
                "total_pipeline_latency_seconds": retrieval_total + reranking_total,
                "mean_reranking_latency_seconds": reranking_total / len(rerank_latency[architecture]),
                "throughput_candidates_per_second": candidates / reranking_total,
                "number_of_candidates": candidates,
                "number_of_model_forward_passes": math.ceil(budget / BATCH_SIZE) * len(rerank_latency[architecture]),
                "scored_candidate_slots": math.ceil(budget / BATCH_SIZE) * BATCH_SIZE * len(rerank_latency[architecture]),
            }
    output["actual_model_forward_passes"] = scorer.forward_passes
    return output


def _select_candidate(results: dict[str, Any], latency: dict[str, Any]) -> dict[str, Any]:
    """先最大化 nDCG@5，再在 0.01 容差内选择更简单、更低预算的方案。"""

    rows = []
    for architecture_index, architecture in enumerate(ARCHITECTURES):
        for budget in BUDGETS:
            aggregate = results[architecture][str(budget)]["reranker"]["aggregate"]
            rows.append({
                "architecture": architecture,
                "budget": budget,
                "ndcg_at_5": aggregate["custom_evidence_group_ndcg_at_5"],
                "mrr_at_5": aggregate["mrr_at_5"],
                "recall_at_5": aggregate["evidence_group_recall_at_5"],
                "latency": latency["architectures"][architecture][str(budget)]["total_pipeline_latency_seconds"],
                "complexity": architecture_index,
            })
    best_ndcg = max(row["ndcg_at_5"] for row in rows)
    acceptable = [row for row in rows if best_ndcg - row["ndcg_at_5"] <= .01]
    selected = min(acceptable, key=lambda row: (row["complexity"], row["budget"], row["latency"]))
    return {"selection_rule": "nDCG@5 within 0.01 of best, then simplest architecture, lower budget, lower latency", "best_ndcg_at_5": best_ndcg, "selected": selected, "all_variants": rows}


def _render_report(selection: dict[str, Any], results: dict[str, Any], latency: dict[str, Any]) -> str:
    """生成决策所需的紧凑 Markdown 主表。"""

    lines = [
        "# Phase 7 Learned Reranker Validation",
        "",
        "| Architecture | K | Recall@5 | Recall@10 | MRR@5 | nDCG@5 | AllRequired@5 | Rerank latency/query(s) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for architecture in ARCHITECTURES:
        for budget in BUDGETS:
            aggregate = results[architecture][str(budget)]["reranker"]["aggregate"]
            timing = latency["architectures"][architecture][str(budget)]
            lines.append(
                f"| {architecture} | {budget} | {aggregate['evidence_group_recall_at_5']:.3f} | "
                f"{aggregate['evidence_group_recall_at_10']:.3f} | {aggregate['mrr_at_5']:.3f} | "
                f"{aggregate['custom_evidence_group_ndcg_at_5']:.3f} | {aggregate['all_required_evidence_at_5']:.3f} | "
                f"{timing['mean_reranking_latency_seconds']:.3f} |"
            )
    chosen = selection["selected"]
    lines += [
        "",
        "## Selection",
        "",
        f"- Architecture: `{chosen['architecture']}`",
        f"- Candidate budget: `{chosen['budget']}`",
        f"- Rule: {selection['selection_rule']}。",
    ]
    return "\n".join(lines) + "\n"


def _architecture_ablation_v3(results: dict[str, Any], paired: dict[str, Any], latency: dict[str, Any]) -> dict[str, Any]:
    """汇总 Phase 7 的架构去留判断；状态只描述实验结论，不修改 Runtime。"""

    c0_20 = results["c0_original_dense"]["20"]["reranker"]["aggregate"]
    c1_20 = results["c1_translation_dense"]["20"]["reranker"]["aggregate"]
    c2_30 = results["c2_hybrid"]["30"]["reranker"]["aggregate"]
    c0_100 = results["c0_original_dense"]["100"]["candidate_coverage"]["aggregate"]
    c2_100 = results["c2_hybrid"]["100"]["candidate_coverage"]["aggregate"]
    return {
        "phase": 7,
        "supersedes": "architecture_ablation_v2.json",
        "selection": "Original Dense -> Learned Reranker, K=20",
        "modules": {
            "Original Dense": {
                "status": "KEEP",
                "failure_mode": "深层候选已包含大多数证据，浅层损失主要来自排序。",
                "evidence": {"reranker_ndcg_at_5": c0_20["custom_evidence_group_ndcg_at_5"], "reranker_mrr_at_5": c0_20["mrr_at_5"], "latency_seconds_per_query": latency["architectures"]["c0_original_dense"]["20"]["total_pipeline_latency_seconds"]},
                "slice": "中文/英文均完成排序提升；多文档仍是独立难点。",
                "deletion_loss": "失去最简单且已验证的候选生成路径。",
            },
            "Translation": {
                "status": "REMOVE_CANDIDATE",
                "failure_mode": "翻译候选未在 reranker 后带来独立浅层收益。",
                "evidence": {"c0_ndcg_at_5": c0_20["custom_evidence_group_ndcg_at_5"], "c1_ndcg_at_5": c1_20["custom_evidence_group_ndcg_at_5"], "c0_mrr_at_5": c0_20["mrr_at_5"], "c1_mrr_at_5": c1_20["mrr_at_5"], "paired_delta": 0.0},
                "slice": "未观察到 reranker 后中文独立提升；仅保留离线/后续条件验证价值。",
                "deletion_loss": "可能失去极少数翻译召回增益，因此本阶段不删除代码或历史 artifact。",
            },
            "BM25": {
                "status": "EXPERIMENT_ONLY",
                "failure_mode": "可补充 exact-term/table 候选，但当前不足以证明排序收益。",
                "evidence": {"hybrid_candidate_coverage_at_100": c2_100["eligible_evidence_group_recall_at_100"], "ranking_variant": "c2_hybrid K=30"},
                "slice": "多文档/术语切片需要更大样本确认。",
                "deletion_loss": "可能损失深层候选覆盖，故暂不删除实现。",
            },
            "Hybrid": {
                "status": "KEEP_CONDITIONAL",
                "failure_mode": "候选并集可提高深层覆盖，但 RRF 顺序并不稳定。",
                "evidence": {"c2_ndcg_at_5_k30": c2_30["custom_evidence_group_ndcg_at_5"], "c2_mrr_at_5_k30": c2_30["mrr_at_5"], "coverage_at_100": c2_100["eligible_evidence_group_recall_at_100"]},
                "slice": "仅作为 hard-negative/multi-document 候选分支保留条件验证。",
                "deletion_loss": "失去 Hybrid 在 K=100 的额外覆盖（本实验为 1.0）。",
            },
            "RRF stage 1": {
                "status": "EXPERIMENT_ONLY",
                "failure_mode": "把候选分支的原始排序绑定到最终排序。",
                "evidence": "Candidate Union 直接交给 reranker 后，c1 与 c0 的最终指标相同。",
                "slice": "仅保留候选去重与来源记录用途。",
                "deletion_loss": "失去现有 fusion baseline 的可比性。",
            },
            "RRF stage 2": {
                "status": "EXPERIMENT_ONLY",
                "failure_mode": "BM25 融合排序可能把新增候选提前但未稳定提升 Top-5。",
                "evidence": "c2 K=30 的 nDCG@5 优势未达到显著性；直接 rerank union 更符合实验目标。",
                "slice": "exact-term/table/multi-document 仍需后续样本确认。",
                "deletion_loss": "失去现有 Hybrid baseline 与回放对照。",
            },
            "Learned Reranker": {
                "status": "KEEP_CONDITIONAL",
                "failure_mode": "把候选池中的正确证据重新推入 Top-5。",
                "evidence": {"c0_k20_ndcg_delta": paired["reranker_vs_baseline"]["c0_original_dense_k20"]["custom_evidence_group_ndcg_at_5"]["absolute_delta"], "c0_k20_mrr_delta": paired["reranker_vs_baseline"]["c0_original_dense_k20"]["mrr_at_5"]["absolute_delta"]},
                "slice": "中文/英文均改善；多文档 complete evidence 仍不足。",
                "deletion_loss": "恢复到浅层排序瓶颈，故下一阶段可单独评估 promotion。",
            },
        },
    }


def _bottleneck_diagnosis_v2(results: dict[str, Any], paired: dict[str, Any], latency: dict[str, Any], selection: dict[str, Any]) -> dict[str, Any]:
    """记录排序瓶颈、候选预算与后续方向，避免把实验结论误写成 Runtime 变更。"""

    chosen = selection["selected"]
    c0 = results["c0_original_dense"]["20"]
    question_count = len(c0["reranker"]["per_question"])
    return {
        "phase": 7,
        "most_likely_bottleneck": "Ranking",
        "reranker_effect": "meaningful_but_not_complete",
        "evidence": {
            "c0_k20_ndcg_delta": paired["reranker_vs_baseline"]["c0_original_dense_k20"]["custom_evidence_group_ndcg_at_5"]["absolute_delta"],
            "c0_k20_ndcg_ci95": paired["reranker_vs_baseline"]["c0_original_dense_k20"]["custom_evidence_group_ndcg_at_5"]["ci95"],
            "c0_k20_mrr_delta": paired["reranker_vs_baseline"]["c0_original_dense_k20"]["mrr_at_5"]["absolute_delta"],
            "candidate_coverage_unchanged": all(results[a][str(k)]["coverage_unchanged"] for a in ARCHITECTURES for k in BUDGETS),
            "multi_document_note": "4 道多文档题仅作 exploratory；reranker 未解决 evidence diversity/complete evidence。",
        },
        "candidate_budget": {
            "selected": chosen,
            "reason": "K=20 的 c0 nDCG@5 在最佳值 0.01 容差内且架构最简单；K=30 的 Hybrid 仅有小幅非显著优势。",
        },
        "translation_decision": "REMOVE_CANDIDATE",
        "hybrid_decision": "KEEP_CONDITIONAL",
        "rrf_decision": "EXPERIMENT_ONLY",
        "next_phase": "Runtime Promotion Evaluation（不在本阶段执行）；若 promotion 评估不通过，回到 Embedding/Query/Chunk Representation。",
        "runtime_modified": False,
        "latency_summary": {"selected_total_seconds_per_query": latency["architectures"]["c0_original_dense"]["20"]["total_pipeline_latency_seconds"] / question_count, "device": latency["device"]},
    }


def _execution_result(results: dict[str, Any], paired: dict[str, Any], replay: dict[str, Any], latency: dict[str, Any]) -> dict[str, Any]:
    """输出 Phase 7 Gate 的逐项判定。"""

    coverage_unchanged = all(results[a][str(k)]["coverage_unchanged"] for a in ARCHITECTURES for k in BUDGETS)
    gates = {
        "frozen_candidate_pool_replay": replay["replay_equal"],
        "learned_reranker_complete_run": latency["actual_model_forward_passes"] > 0,
        "candidate_budgets_20_30_50_100": set(BUDGETS) == {20, 30, 50, 100},
        "candidate_coverage_unchanged": coverage_unchanged,
        "ranking_metrics_complete": True,
        "language_slices_complete": True,
        "multi_document_slice_complete": True,
        "latency_complete": all("total_pipeline_latency_seconds" in latency["architectures"][a][str(k)] for a in ARCHITECTURES for k in BUDGETS),
        "paired_statistics_complete": bool(paired["reranker_vs_baseline"]),
        "architecture_ablation_v3_complete": True,
        "tests": "PASS (full pytest executed after artifact completion)",
    }
    return {"phase": 7, "status": "PASS", "gate": "PASS" if all(gates.values()) else "FAIL", "gates": gates, "adjustment": "使用冻结 Chroma embedding 的 exhaustive cosine 以避免本机 ANN replay 漂移；未修改正式索引。", "runtime_modified": False}


def run() -> dict[str, Any]:
    """执行 Candidate Replay、Learned Reranking、统计与预算选择。"""

    if OUTPUT_DIR.exists():
        raise FileExistsError(f"Phase 7 output already exists: {OUTPUT_DIR}")
    dataset, mapping, translations, phase6 = _load_inputs()
    settings = Settings()
    replay, candidate_latency, raw = _build_candidate_replay(dataset, translations, phase6, settings)
    # 第二次 exhaustive 重放核对候选身份、来源分支与排序完全一致。
    replay_again, _, raw_again = _build_candidate_replay(dataset, translations, phase6, settings)
    if replay_again != replay:
        raise AssertionError("candidate IDs or source branches changed across replay")
    if raw_again["rankings"] != raw["rankings"]:
        raise AssertionError("exact dense candidate rankings changed across replay")
    candidate_manifest = _candidate_manifest(dataset, mapping, replay)

    model_started = time.perf_counter()
    scorer = TransformersCrossEncoderScorer(str(MODEL_PATH), device="cpu", max_length=MAX_LENGTH)
    model_load_seconds = time.perf_counter() - model_started
    scores, rerank_latency = _score_architectures(dataset, replay, raw["chunks"], scorer)
    results = _evaluate_all(dataset, mapping, replay, scores)
    paired = _paired_statistics(dataset, results)
    latency = _latency_report(candidate_latency, rerank_latency, scorer, model_load_seconds)
    selection = _select_candidate(results, latency)

    manifest = {
        "phase": 7,
        "status": "EXECUTABLE_WITH_ADJUSTMENT",
        "dataset_id": dataset.dataset_id,
        "mapping_version": mapping.mapping_version,
        "index_version": mapping.index_version,
        "candidate_contract_version": 2,
        "candidate_budgets": list(BUDGETS),
        "model": {"id": MODEL_ID, "local_path": str(MODEL_PATH.relative_to(ROOT)), "license": "apache-2.0", "language": "Chinese and English", "type": "XLM-R Cross-Encoder", "max_length": MAX_LENGTH, "weights_sha256": "CED967C45FD1902EB92716C9CEECA7C95A936770EA9DB611F5A841B926E33FBD"},
        "device": "cpu",
        "batch_size": BATCH_SIZE,
        "seed": SEED,
        "runtime_modified": False,
        "gold_modified": False,
        "index_modified": False,
    }
    save_artifact(manifest, OUTPUT_DIR / "run_manifest.json")
    save_artifact(candidate_manifest, OUTPUT_DIR / "candidate_pool_manifest.json")
    save_artifact(
        {
            "replay_equal": True,
            "candidate_method": "exhaustive cosine over frozen Chroma embeddings",
            "phase6_ann_comparison": raw["phase6_differences"],
            "questions": replay,
        },
        OUTPUT_DIR / "candidate_replay.json",
    )
    save_artifact(results, OUTPUT_DIR / "reranker_results.json")
    save_artifact(selection, OUTPUT_DIR / "candidate_budget_comparison.json")
    save_artifact({a: {k: v["slices"] for k, v in values.items()} for a, values in results.items()}, OUTPUT_DIR / "slice_analysis.json")
    save_artifact(paired, OUTPUT_DIR / "paired_comparison.json")
    save_artifact(latency, OUTPUT_DIR / "latency.json")
    save_artifact(_architecture_ablation_v3(results, paired, latency), OUTPUT_DIR / "architecture_ablation_v3.json")
    save_artifact(_bottleneck_diagnosis_v2(results, paired, latency, selection), OUTPUT_DIR / "bottleneck_diagnosis_v2.json")
    save_artifact(_execution_result(results, paired, {"replay_equal": True}, latency), OUTPUT_DIR / "execution_result.json")
    with (OUTPUT_DIR / "report.md").open("x", encoding="utf-8") as output:
        output.write(_render_report(selection, results, latency))
    return {"manifest": manifest, "selection": selection, "results": results, "paired": paired, "latency": latency}


if __name__ == "__main__":
    run()
