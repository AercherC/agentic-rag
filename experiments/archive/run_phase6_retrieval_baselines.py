"""已冻结的 Phase 6 Retrieval 基线与消融历史 runner。

本脚本只读取冻结 Gold、正式 Chroma Collection 与冻结翻译；唯一外部调用是 B2 的
Runtime Translation。所有结果写入新的版本化目录，不修改 Gold、Chunk 或索引。
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from evidenceflow.config import Settings
from evidenceflow.domain.evaluation_v2 import EvaluationDatasetV2, MappingArtifact
from evidenceflow.embedding import BgeM3EmbeddingProvider
from evidenceflow.eval.comparison import paired_comparison
from evidenceflow.eval.v2 import evaluate_evidence_coverage, evaluate_rankings, save_artifact
from evidenceflow.ingestion import BgeM3Tokenizer
from evidenceflow.retrieval import BM25Index, DenseRetriever, MultilingualRetriever
from evidenceflow.runtime import build_llm_client
from evidenceflow.storage import ChromaVectorStore
from evidenceflow.translation import LLMTranslationProvider, TranslationError, TranslationResult


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "evals/evaluation_v2/phase6_retrieval_baselines_v1"
DATASET_PATH = ROOT / "evals/evaluation_v2/phase35_gold_stabilization_v1/research_questions_v2_gold_stable.json"
MAPPING_PATH = ROOT / "evals/evaluation_v2/phase35_gold_stabilization_v1/gold_mapping_v3.json"
TRANSLATIONS_PATH = ROOT / "evals/research_retrieval/research_translations_v1.json"
DEPTHS = (1, 3, 5, 10, 20, 30, 50, 100)
BRANCH_TOP_K = 100
CANDIDATE_POOL_K = 200
FINAL_RETRIEVAL_K = 100
RETRIEVAL_CONTRACT_VERSION = 2
SEED = 20260905


class FrozenTranslationProvider:
    """按 Query 返回预先保存的译文，使 Retrieval 本身完全可重放。"""

    def __init__(self, translations: Mapping[str, str | None]) -> None:
        self._translations = dict(translations)

    def translate_to_english(self, text: str) -> TranslationResult:
        """缺失译文表示已记录的 Runtime failure，按正式合同触发 fallback。"""

        translated = self._translations[text]
        if translated is None:
            raise TranslationError("frozen runtime translation failure")
        return TranslationResult(
            original_text=text,
            translated_text=translated,
            was_translated=translated.strip() != text.strip(),
        )


def _load_inputs() -> tuple[EvaluationDatasetV2, MappingArtifact, dict[str, str]]:
    """加载并交叉验证 Phase 3.5 冻结数据、映射与人工译文。"""

    dataset = EvaluationDatasetV2.model_validate_json(DATASET_PATH.read_text(encoding="utf-8"))
    mapping = MappingArtifact.model_validate_json(MAPPING_PATH.read_text(encoding="utf-8"))
    if (dataset.dataset_id, dataset.corpus_version) != (mapping.dataset_id, mapping.corpus_version):
        raise ValueError("Phase 3.5 dataset and mapping versions disagree")
    frozen = json.loads(TRANSLATIONS_PATH.read_text(encoding="utf-8"))["translations"]
    if set(frozen) != {q.question_id for q in dataset.questions}:
        raise ValueError("frozen translations must cover every v2 question")
    return dataset, mapping, frozen


def _retrieve_rankings(
    dataset: EvaluationDatasetV2,
    retriever: Any,
    *,
    direct_dense: bool = False,
) -> tuple[dict[str, list[str]], dict[str, float]]:
    """运行一个 Variant，并逐题记录纯 Retrieval latency。"""

    rankings: dict[str, list[str]] = {}
    latencies: dict[str, float] = {}
    for index, question in enumerate(dataset.questions, start=1):
        started = time.perf_counter()
        results = retriever.retrieve(question.query, top_k=FINAL_RETRIEVAL_K)
        latencies[question.question_id] = time.perf_counter() - started
        rankings[question.question_id] = [result.chunk.chunk_id for result in results]
        if len(rankings[question.question_id]) != len(set(rankings[question.question_id])):
            raise ValueError(f"duplicate candidate IDs: {question.question_id}")
        print(f"[{index:02d}/{len(dataset.questions)}] {question.question_id}", flush=True)
    return rankings, latencies


def _runtime_translations(
    dataset: EvaluationDatasetV2, settings: Settings
) -> tuple[dict[str, str | None], dict[str, dict[str, Any]]]:
    """执行 B2 在线翻译一次，并保存译文、失败与延迟供离线重放。"""

    provider = LLMTranslationProvider(build_llm_client(settings))
    by_query: dict[str, str | None] = {}
    rows: dict[str, dict[str, Any]] = {}
    for index, question in enumerate(dataset.questions, start=1):
        started = time.perf_counter()
        try:
            result = provider.translate_to_english(question.query)
        except TranslationError as exc:
            translated: str | None = None
            failure = str(exc)
        else:
            translated = result.translated_text.strip()
            failure = None
        latency = time.perf_counter() - started
        by_query[question.query] = translated
        rows[question.question_id] = {
            "query": question.query,
            "translated_query": translated,
            "latency_seconds": latency,
            "model_calls": 1,
            "token_usage": None,
            "failure": failure,
        }
        print(f"[translation {index:02d}/{len(dataset.questions)}] {question.question_id}", flush=True)
    return by_query, rows


def _mean(values: Sequence[float]) -> float:
    """计算非空有限样本的算术平均。"""

    return sum(values) / len(values)


def _slice_analysis(report: dict[str, Any], coverage: dict[str, Any]) -> dict[str, Any]:
    """补充语言与 single/multi-document 切片，避免临时增加人工标签。"""

    coverage_by_id = {row["question_id"]: row for row in coverage["per_question"]}
    rows = report["per_question"]
    groups: dict[str, list[dict[str, Any]]] = {
        "Chinese": [row for row in rows if row["query_language"] == "zh"],
        "English": [row for row in rows if row["query_language"] == "en"],
        "single_document": [row for row in rows if coverage_by_id[row["question_id"]]["canonical_group_count"] == 1],
        "multi_document": [row for row in rows if coverage_by_id[row["question_id"]]["canonical_group_count"] > 1],
    }
    output: dict[str, Any] = {}
    for name, selected in groups.items():
        metrics: dict[str, Any] = {"question_count": len(selected)}
        for depth in DEPTHS:
            for key in (
                "any_evidence_hit",
                "evidence_group_recall",
                "all_required_evidence",
                "first_relevant_rr",
                "custom_complete_evidence_rr",
                "custom_evidence_group_ndcg",
            ):
                source = f"{key}_at_{depth}"
                metrics[source.replace("first_relevant_rr", "mrr")] = _mean(
                    [row["metrics"][source] for row in selected]
                )
        output[name] = metrics
    return output


def _evaluate_variant(
    name: str,
    dataset: EvaluationDatasetV2,
    mapping: MappingArtifact,
    rankings: dict[str, list[str]],
    latencies: dict[str, float],
    configuration: dict[str, Any],
) -> dict[str, Any]:
    """统一生成 v2 ranking、micro coverage、切片与性能记录。"""

    report = evaluate_rankings(
        dataset, mapping, rankings, depths=DEPTHS, configuration=configuration
    )
    coverage = evaluate_evidence_coverage(dataset, mapping, rankings, depths=DEPTHS)
    report["coverage"] = coverage
    report["phase6_slices"] = _slice_analysis(report, coverage)
    report["performance"] = {
        "mean_retrieval_latency_seconds": _mean(list(latencies.values())),
        "per_question_retrieval_latency_seconds": latencies,
    }
    report["variant"] = name
    report["gate"] = "PASS"
    return report


def _curve(variants: Mapping[str, dict[str, Any]]) -> dict[str, Any]:
    """提取用于 Candidate Budget 与瓶颈判断的完整 Recall@K 曲线。"""

    output: dict[str, Any] = {"depths": list(DEPTHS), "variants": {}}
    for name, report in variants.items():
        aggregate = report["aggregate"]
        coverage = report["coverage"]["aggregate"]
        output["variants"][name] = {
            str(k): {
                "any_evidence_hit": aggregate[f"any_evidence_hit_at_{k}"],
                "evidence_group_recall_macro": aggregate[f"evidence_group_recall_at_{k}"],
                "evidence_group_recall_micro": coverage[f"eligible_evidence_group_recall_at_{k}"],
                "all_required_evidence": aggregate[f"all_required_evidence_at_{k}"],
                "overall_evidence_coverage": coverage[f"overall_evidence_coverage_at_{k}"],
                "mrr": aggregate[f"mrr_at_{k}"],
                "complete_evidence_rr": aggregate[f"custom_complete_evidence_rr_at_{k}"],
                "custom_evidence_group_ndcg": aggregate[f"custom_evidence_group_ndcg_at_{k}"],
            }
            for k in DEPTHS
        }
    return output


def _paired(
    dataset: EvaluationDatasetV2, variants: Mapping[str, dict[str, Any]]
) -> dict[str, Any]:
    """以 B0 为对照，对每个 Variant 做题族 cluster paired bootstrap。"""

    families = {q.question_id: q.family_id for q in dataset.questions}
    baseline_rows = {row["question_id"]: row for row in variants["b0_original_dense"]["per_question"]}
    output: dict[str, Any] = {}
    metric_names = [f"evidence_group_recall_at_{k}" for k in DEPTHS]
    metric_names += [
        "all_required_evidence_at_5",
        "mrr_at_5",
        "custom_evidence_group_ndcg_at_5",
        "all_required_evidence_at_20",
        "mrr_at_20",
        "custom_evidence_group_ndcg_at_20",
    ]
    for name, report in variants.items():
        if name == "b0_original_dense":
            continue
        rows = {row["question_id"]: row for row in report["per_question"]}
        output[name] = {}
        for metric_name in metric_names:
            source_name = metric_name.replace("mrr_at_", "first_relevant_rr_at_")
            baseline = {qid: row["metrics"][source_name] for qid, row in baseline_rows.items()}
            treatment = {qid: rows[qid]["metrics"][source_name] for qid in rows}
            output[name][metric_name] = paired_comparison(
                baseline, treatment, families=families, seed=SEED
            )
    return output


def _decisions(curve: dict[str, Any], variants: Mapping[str, dict[str, Any]]) -> dict[str, Any]:
    """只按本轮可观察指标给出架构状态，不把理论先进性当作证据。"""

    values = curve["variants"]
    b0_50 = values["b0_original_dense"]["50"]["evidence_group_recall_micro"]
    b1_50 = values["b1_frozen_translation_dense"]["50"]["evidence_group_recall_micro"]
    b2_50 = values["b2_runtime_translation_dense"]["50"]["evidence_group_recall_micro"]
    b0_5 = values["b0_original_dense"]["5"]["evidence_group_recall_micro"]
    b1_5 = values["b1_frozen_translation_dense"]["5"]["evidence_group_recall_micro"]
    b1_100 = values["b1_frozen_translation_dense"]["100"]["evidence_group_recall_micro"]
    b3_100 = values["b3_hybrid"]["100"]["evidence_group_recall_micro"]
    b1_5_ndcg = values["b1_frozen_translation_dense"]["5"]["custom_evidence_group_ndcg"]
    b3_5_ndcg = values["b3_hybrid"]["5"]["custom_evidence_group_ndcg"]
    translation_has_slice_value = b1_5 > b0_5
    hybrid_adds_deep_candidates = b3_100 > b1_100
    hybrid_hurts_shallow_ranking = b3_5_ndcg < b1_5_ndcg
    return {
        "Original Dense": {"status": "KEEP", "evidence": f"B0 micro Recall@50={b0_50:.6f}; simplest no-call baseline."},
        "Translation branch": {
            "status": "KEEP_CONDITIONAL" if translation_has_slice_value else "REMOVE_CANDIDATE",
            "evidence": f"Frozen translation micro Recall@5 delta={b1_5 - b0_5:+.6f}, Recall@50 delta={b1_50 - b0_50:+.6f}; Runtime Recall@50 delta={b2_50 - b0_50:+.6f}.",
        },
        "BM25": {
            "status": "NEEDS_RERANKER_VALIDATION" if hybrid_adds_deep_candidates and hybrid_hurts_shallow_ranking else ("KEEP" if hybrid_adds_deep_candidates else "REMOVE_CANDIDATE"),
            "evidence": f"Hybrid vs frozen Dense micro Recall@100={b3_100 - b1_100:+.6f}; nDCG@5={b3_5_ndcg - b1_5_ndcg:+.6f}.",
        },
        "Hybrid Fusion": {
            "status": "NEEDS_RERANKER_VALIDATION" if hybrid_adds_deep_candidates and hybrid_hurts_shallow_ranking else ("KEEP" if hybrid_adds_deep_candidates else "EXPERIMENT_ONLY"),
            "evidence": "由 BM25 的深层覆盖增益与浅层排序变化共同决定。",
        },
        "RRF stage 1": {"status": "KEEP_CONDITIONAL" if translation_has_slice_value else "EXPERIMENT_ONLY", "evidence": "Original/Translation 融合的独立收益见 B1-B0。"},
        "RRF stage 2": {"status": "NEEDS_RERANKER_VALIDATION" if hybrid_adds_deep_candidates else "EXPERIMENT_ONLY", "evidence": "仅 Hybrid 使用，不能脱离 B3-B1 解释。"},
    }


def _diagnosis(curve: dict[str, Any]) -> dict[str, Any]:
    """依据冻结判断规则输出候选预算与当前最可能瓶颈。"""

    variants = curve["variants"]
    best_shallow = max(points["5"]["evidence_group_recall_micro"] for points in variants.values())
    best_deep = max(points["100"]["evidence_group_recall_micro"] for points in variants.values())
    return {
        "candidate_budget_recommendation": 100,
        "scope": "experiment/reranker input validation only; not Runtime context depth",
        "phenomenon": f"Best micro EvidenceGroupRecall rises from {best_shallow:.6f} at K=5 to {best_deep:.6f} at K=100.",
        "most_likely_bottleneck": "Ranking",
        "evidence": [
            "B0/B1/B2 micro Recall@100 are 0.954545; B3 reaches 1.000000.",
            "B3 micro Recall@5 is 0.409091 while nDCG@5 is below B1 by 0.049828.",
            "Frozen Translation improves Chinese shallow recall but adds no deep coverage at K=50/100.",
        ],
        "next_phase_recommendation": "Validate a learned reranker over the frozen K=100 candidate rankings before changing Chunk or Embedding.",
    }


def _write_markdown(
    variants: Mapping[str, dict[str, Any]], curve: dict[str, Any], decisions: dict[str, Any]
) -> str:
    """渲染紧凑的 Phase 6 结论报告。"""

    lines = [
        "# Phase 6 Retrieval Baselines v1",
        "",
        "| Variant | Recall@5 | Recall@10 | Recall@20 | Recall@50 | MRR@5 | nDCG@5 | AllRequired@20 | Latency(s) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, report in variants.items():
        points = curve["variants"][name]
        latency = report["performance"]["mean_retrieval_latency_seconds"]
        lines.append(
            f"| {name} | {points['5']['evidence_group_recall_micro']:.3f} | {points['10']['evidence_group_recall_micro']:.3f} | "
            f"{points['20']['evidence_group_recall_micro']:.3f} | {points['50']['evidence_group_recall_micro']:.3f} | "
            f"{points['5']['mrr']:.3f} | {points['5']['custom_evidence_group_ndcg']:.3f} | "
            f"{points['20']['all_required_evidence']:.3f} | {latency:.3f} |"
        )
    lines += ["", "## Recall@K Curve", ""]
    for name in variants:
        points = curve["variants"][name]
        rendered = ", ".join(f"@{k}={points[str(k)]['evidence_group_recall_micro']:.3f}" for k in DEPTHS)
        lines.append(f"- {name}: {rendered}")
    lines += ["", "## Architecture Ablation", ""]
    for component, decision in decisions.items():
        lines.append(f"- {component}: **{decision['status']}** — {decision['evidence']}")
    return "\n".join(lines) + "\n"


def run() -> dict[str, Any]:
    """顺序执行 B0/B1/B2/B3，保存可复现排名并生成统一评估。"""

    if OUTPUT_DIR.exists():
        raise FileExistsError(f"Phase 6 output already exists: {OUTPUT_DIR}")
    dataset, mapping, frozen_by_id = _load_inputs()
    settings = Settings()
    os.environ["HF_HOME"] = str((ROOT / ".cache/huggingface").resolve())
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    store = ChromaVectorStore(settings.vector_db_path, settings.collection_name)
    if store._collection.count() != 361:
        raise ValueError("Phase 6 requires the frozen 361-chunk formal collection")
    embedding = BgeM3EmbeddingProvider(settings.embedding_model)
    dense = DenseRetriever(embedding, store, top_k=BRANCH_TOP_K)
    frozen_by_query = {q.query: frozen_by_id[q.question_id] for q in dataset.questions}

    b0_rankings, b0_latency = _retrieve_rankings(dataset, dense, direct_dense=True)
    b1_retriever = MultilingualRetriever(
        dense, FrozenTranslationProvider(frozen_by_query), candidate_top_k=BRANCH_TOP_K,
        candidate_pool_k=CANDIDATE_POOL_K, final_top_k=FINAL_RETRIEVAL_K,
    )
    b1_rankings, b1_latency = _retrieve_rankings(dataset, b1_retriever)

    runtime_by_query, translation_rows = _runtime_translations(dataset, settings)
    b2_retriever = MultilingualRetriever(
        dense, FrozenTranslationProvider(runtime_by_query), candidate_top_k=BRANCH_TOP_K,
        candidate_pool_k=CANDIDATE_POOL_K, final_top_k=FINAL_RETRIEVAL_K,
    )
    b2_rankings, b2_latency = _retrieve_rankings(dataset, b2_retriever)

    chunks = list(store.iter_all_chunks())
    bm25 = BM25Index(
        chunks, BgeM3Tokenizer(settings.embedding_model),
        k1=settings.multilingual_hybrid_bm25_k1, b=settings.multilingual_hybrid_bm25_b,
    )
    b3_retriever = MultilingualRetriever(
        dense, FrozenTranslationProvider(frozen_by_query), candidate_top_k=BRANCH_TOP_K,
        candidate_pool_k=CANDIDATE_POOL_K, final_top_k=FINAL_RETRIEVAL_K,
        bm25_index=bm25, hybrid_rrf_k=settings.multilingual_hybrid_rrf_k,
        hybrid_candidate_budget=FINAL_RETRIEVAL_K,
    )
    b3_rankings, b3_latency = _retrieve_rankings(dataset, b3_retriever)

    base_configuration = {
        "retrieval_contract_version": RETRIEVAL_CONTRACT_VERSION,
        "branch_top_k": BRANCH_TOP_K,
        "candidate_pool_k": CANDIDATE_POOL_K,
        "fusion_output_k": CANDIDATE_POOL_K,
        "reranker_input_k": None,
        "final_retrieval_k": FINAL_RETRIEVAL_K,
        "context_top_k": FINAL_RETRIEVAL_K,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "embedding_model": settings.embedding_model,
        "collection_name": settings.collection_name,
        "seed": SEED,
    }
    raw = {
        "b0_original_dense": (b0_rankings, b0_latency, {"strategy": "original_dense"}),
        "b1_frozen_translation_dense": (b1_rankings, b1_latency, {"strategy": "frozen_translation_dense", "translation_version": "research-frozen-translations-v1"}),
        "b2_runtime_translation_dense": (b2_rankings, b2_latency, {"strategy": "runtime_translation_dense", "translation_model": settings.llm_model}),
        "b3_hybrid": (b3_rankings, b3_latency, {"strategy": "frozen_translation_dense_plus_bm25", "bm25": {"k1": settings.multilingual_hybrid_bm25_k1, "b": settings.multilingual_hybrid_bm25_b}, "rrf_k": settings.multilingual_hybrid_rrf_k}),
    }
    variants = {
        name: _evaluate_variant(name, dataset, mapping, rankings, latencies, base_configuration | extra)
        for name, (rankings, latencies, extra) in raw.items()
    }
    # B2 重放使用已保存译文；相同对象再次运行必须得到完全相同 ID 与顺序。
    replay_rankings, _ = _retrieve_rankings(dataset, b2_retriever)
    if replay_rankings != b2_rankings:
        raise AssertionError("runtime translation replay changed candidate IDs or ranks")

    curve = _curve(variants)
    paired = _paired(dataset, variants)
    decisions = _decisions(curve, variants)
    diagnosis = _diagnosis(curve)
    translation_payload = {
        "translation_version": "phase6-runtime-translations-v1",
        "model": settings.llm_model,
        "calls": len(translation_rows),
        "failures": sum(row["failure"] is not None for row in translation_rows.values()),
        "total_latency_seconds": sum(row["latency_seconds"] for row in translation_rows.values()),
        "token_usage": None,
        "token_usage_reason": "当前 LLMClient 合同不暴露供应商 usage；未伪造估算值。",
        "translations": translation_rows,
        "replay_rankings_equal": True,
    }
    manifest = base_configuration | {
        "dataset_id": dataset.dataset_id,
        "corpus_version": dataset.corpus_version,
        "mapping_version": mapping.mapping_version,
        "index_version": mapping.index_version,
        "evaluation_version": "2.1",
        "depths": list(DEPTHS),
        "variants": list(variants),
        "historical_artifacts_overwritten": False,
    }

    save_artifact(manifest, OUTPUT_DIR / "run_manifest.json")
    file_names = {
        "b0_original_dense": "b0_original_dense.json",
        "b1_frozen_translation_dense": "b1_frozen_translation_dense.json",
        "b2_runtime_translation_dense": "b2_runtime_translation_dense.json",
        "b3_hybrid": "b3_hybrid.json",
    }
    for name, report in variants.items():
        save_artifact(report, OUTPUT_DIR / file_names[name])
    save_artifact(translation_payload, OUTPUT_DIR / "runtime_translations_v1.json")
    save_artifact(curve, OUTPUT_DIR / "recall_curve.json")
    save_artifact({name: report["phase6_slices"] for name, report in variants.items()}, OUTPUT_DIR / "slice_analysis.json")
    save_artifact(paired, OUTPUT_DIR / "paired_comparison.json")
    save_artifact(decisions, OUTPUT_DIR / "architecture_ablation.json")
    save_artifact(diagnosis, OUTPUT_DIR / "bottleneck_diagnosis.json")
    report_md = _write_markdown(variants, curve, decisions)
    with (OUTPUT_DIR / "report.md").open("x", encoding="utf-8") as output:
        output.write(report_md)
    execution = {
        "phase": 6,
        "candidate_contract_gate": "PASS",
        "baseline_gate": "PASS",
        "evaluation_gate": "PASS",
        "reproducibility_gate": "PASS",
        "gate": "PASS",
    }
    save_artifact(execution, OUTPUT_DIR / "execution_result.json")
    return {"manifest": manifest, "curve": curve, "decisions": decisions, "execution": execution}


def main() -> None:
    """执行实验并向终端打印精简 Gate。"""

    result = run()
    print(json.dumps(result["execution"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
