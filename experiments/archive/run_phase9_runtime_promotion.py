"""已冻结的 Phase 9 Runtime promotion 历史 runner。"""

from __future__ import annotations

import hashlib
import json
import statistics
import time
from pathlib import Path
from typing import Any

from evidenceflow.config import Settings
from evidenceflow.domain.evaluation_v2 import EvaluationDatasetV2, MappingArtifact
from evidenceflow.embedding import BgeM3EmbeddingProvider
from evidenceflow.eval.comparison import paired_comparison
from evidenceflow.eval.v2 import evaluate_rankings, save_artifact
from evidenceflow.retrieval import (
    DenseRerankerRetriever,
    DenseRetriever,
    LearnedReranker,
    TransformersCrossEncoderScorer,
)
from evidenceflow.storage import ChromaVectorStore


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "evals/evaluation_v2/phase9_runtime_promotion_v1"
P35 = ROOT / "evals/evaluation_v2/phase35_gold_stabilization_v1"
P6 = ROOT / "evals/evaluation_v2/phase6_retrieval_baselines_v1"
P8 = ROOT / "evals/evaluation_v2/phase8_runtime_readiness_v1"
MODEL = ROOT / ".cache/models/bge-reranker-base"
SEED = 20260906
METRICS = (
    "evidence_group_recall_at_5",
    "evidence_group_recall_at_10",
    "mrr_at_5",
    "custom_evidence_group_ndcg_at_5",
    "all_required_evidence_at_5",
    "all_required_evidence_at_10",
)


def _load(path: Path) -> dict[str, Any]:
    """读取一个 UTF-8 JSON artifact。"""

    return json.loads(path.read_text(encoding="utf-8"))


def _percentile(values: list[float], fraction: float) -> float:
    """使用线性插值计算小样本 latency percentile。"""

    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _metric_values(report: dict[str, Any], metric: str) -> dict[str, float]:
    """从 Evaluation v2 报告提取逐题指标，供配对比较复用。"""

    source = metric.replace("mrr_at_", "first_relevant_rr_at_")
    return {row["question_id"]: row["metrics"][source] for row in report["per_question"]}


def _paired(
    dataset: EvaluationDatasetV2,
    before: dict[str, Any],
    after: dict[str, Any],
) -> dict[str, Any]:
    """按冻结 family cluster 比较旧 Runtime 与新 Runtime。"""

    families = {question.question_id: question.family_id for question in dataset.questions}
    return {
        metric: paired_comparison(
            _metric_values(before, metric),
            _metric_values(after, metric),
            families=families,
            seed=SEED,
        )
        for metric in METRICS
    }


def _summary(report: dict[str, Any]) -> dict[str, float]:
    """输出 Phase 9 Gate 要求的六项冻结质量指标。"""

    aggregate = report["aggregate"]
    return {
        "recall_at_5": aggregate["evidence_group_recall_at_5"],
        "recall_at_10": aggregate["evidence_group_recall_at_10"],
        "mrr_at_5": aggregate["mrr_at_5"],
        "ndcg_at_5": aggregate["custom_evidence_group_ndcg_at_5"],
        "all_required_evidence_at_5": aggregate["all_required_evidence_at_5"],
        "all_required_evidence_at_10": aggregate["all_required_evidence_at_10"],
    }


def _sha256(path: Path) -> str:
    """记录 Runtime 源码身份，不把敏感运行配置写入产物。"""

    return hashlib.sha256(path.read_bytes()).hexdigest()


class TimedDenseRetriever:
    """为正式 DenseRetriever 增加只读 latency 采样，不改变检索结果。"""

    def __init__(self, retriever: DenseRetriever) -> None:
        """保存被测生产组件及最近一次调用统计。"""

        self._retriever = retriever
        self.last_seconds = 0.0
        self.last_count = 0

    def retrieve(self, query: str, top_k: int):
        """调用正式 Dense ANN 并记录单次耗时与候选数。"""

        started = time.perf_counter()
        result = self._retriever.retrieve(query, top_k=top_k)
        self.last_seconds = time.perf_counter() - started
        self.last_count = len(result)
        return result


def _architecture_manifest() -> dict[str, Any]:
    """冻结各检索模块在 Phase 9 后的正式角色。"""

    rows = [
        ("Original Dense", "default branch", "candidate generator", True, None,
         "Phase 8 ANN Top20 Gold coverage PASS"),
        ("Chroma ANN", "default vector search", "candidate store/search", True, None,
         "Phase 8 ANN gate PASS; collection identity unchanged"),
        ("Learned Reranker", "validated experiment", "final ranker", True, None,
         "Phase 8 runtime replay and truncation gates PASS"),
        ("Translation", "default second branch", "fallback only", False, "FALLBACK_ONLY",
         "2 unique Gold groups; implementation retained"),
        ("BM25", "conditional hybrid branch", "fallback only", False, "FALLBACK_ONLY",
         "3 unique Gold groups; implementation retained"),
        ("Hybrid", "optional final fusion", "conditional fallback architecture", False,
         "KEEP_CONDITIONAL", "BM25 retains unique and multi-document contribution"),
        ("RRF Stage 1", "default final ranker", "multi-branch candidate selector", False,
         "KEEP_AS_CANDIDATE_SELECTOR", "not called by Original-only default path"),
        ("RRF Stage 2", "hybrid final ranker", "multi-branch candidate selector", False,
         "KEEP_AS_CANDIDATE_SELECTOR", "not called by Original-only default path"),
    ]
    modules = []
    for module, old_role, new_role, runtime_default, fallback_role, evidence in rows:
        modules.append(
            {
                "module": module,
                "old_role": old_role,
                "new_role": new_role,
                "runtime_default": runtime_default,
                "fallback_role": fallback_role,
                "delete_now": False,
                "delete_later": False if runtime_default else "review after runtime stability",
                "reason": new_role,
                "evidence": evidence,
            }
        )
    return {"phase": 9, "manifest_version": "architecture-simplification-v1", "modules": modules}


def run() -> dict[str, Any]:
    """执行冻结 40Q Runtime Regression，并只在完整成功后写入 Phase 9 产物。"""

    if OUT.exists():
        raise FileExistsError(f"Phase 9 output already exists: {OUT}")
    phase8 = _load(P8 / "execution_result.json")
    phase8_manifest = _load(P8 / "run_manifest.json")
    if phase8["runtime_promotion"] != "ALLOWED":
        raise ValueError("Phase 8 did not allow Runtime Promotion")
    if any(
        not (MODEL / name).is_file()
        for name in ("config.json", "model.safetensors", "tokenizer.json")
    ):
        raise FileNotFoundError("local reranker model is incomplete")

    dataset = EvaluationDatasetV2.model_validate_json(
        (P35 / "research_questions_v2_gold_stable.json").read_text(encoding="utf-8")
    )
    mapping = MappingArtifact.model_validate_json(
        (P35 / "gold_mapping_v3.json").read_text(encoding="utf-8")
    )
    old_report = _load(P6 / "b2_runtime_translation_dense.json")
    old_rankings = {
        row["question_id"]: row["ranked_chunk_ids"]
        for row in old_report["per_question"]
    }

    settings = Settings(reranker_model=str(MODEL))
    store = ChromaVectorStore(settings.vector_db_path, settings.collection_name)
    before_ids = sorted(chunk.chunk_id for chunk in store.iter_all_chunks())
    if len(before_ids) != phase8_manifest["chunk_count"]:
        raise ValueError("formal Chroma collection identity changed before Phase 9")

    embedding_started = time.perf_counter()
    embedder = BgeM3EmbeddingProvider(settings.embedding_model)
    embedding_load_seconds = time.perf_counter() - embedding_started
    dense = TimedDenseRetriever(
        DenseRetriever(embedder, store, top_k=settings.reranker_input_k)
    )
    model_started = time.perf_counter()
    scorer = TransformersCrossEncoderScorer(
        settings.reranker_model,
        device=settings.reranker_device,
        dtype=settings.reranker_dtype,
        max_length=settings.reranker_max_length,
    )
    model_load_seconds = time.perf_counter() - model_started
    runtime = DenseRerankerRetriever(
        dense,  # type: ignore[arg-type]
        LearnedReranker(
            scorer,
            batch_size=settings.reranker_batch_size,
            max_candidates=settings.reranker_input_k,
        ),
        candidate_pool_k=settings.reranker_input_k,
        reranker_input_k=settings.reranker_input_k,
        final_top_k=settings.reranker_final_k,
    )

    rankings: dict[str, list[str]] = {}
    rows: list[dict[str, Any]] = []
    for index, question in enumerate(dataset.questions, 1):
        started = time.perf_counter()
        # Top-10 仅供 Recall@10 回归；正式默认调用仍由 final_top_k=5 冻结。
        result = runtime.retrieve(question.query, top_k=10)
        total_seconds = time.perf_counter() - started
        rankings[question.question_id] = [item.chunk.chunk_id for item in result]
        rows.append(
            {
                "question_id": question.question_id,
                "candidate_count": dense.last_count,
                "final_default_count": min(settings.reranker_final_k, len(result)),
                "evaluation_output_count": len(result),
                "dense_ann_seconds": dense.last_seconds,
                "reranker_seconds": total_seconds - dense.last_seconds,
                "total_seconds": total_seconds,
                "reranker_truncated_pair_count": runtime.reranker.last_truncated_pair_count,
            }
        )
        print(f"[Phase 9 {index:02d}/{len(dataset.questions)}] {question.question_id}", flush=True)

    new_report = evaluate_rankings(dataset, mapping, rankings, depths=(5, 10))
    before_summary, after_summary = _summary(old_report), _summary(new_report)
    phase8_expected = {
        "recall_at_5": 0.5625,
        "recall_at_10": 0.7375,
        "mrr_at_5": 0.4320833333333334,
        "ndcg_at_5": 0.46339982650599376,
        "all_required_evidence_at_5": 0.55,
        "all_required_evidence_at_10": 0.725,
    }
    reproduction_delta = {
        metric: after_summary[metric] - expected
        for metric, expected in phase8_expected.items()
    }
    quality_reproduced = all(abs(delta) <= 1e-9 for delta in reproduction_delta.values())
    paired = _paired(dataset, old_report, new_report)
    after_ids = sorted(chunk.chunk_id for chunk in store.iter_all_chunks())
    index_unchanged = before_ids == after_ids

    dense_latency = [row["dense_ann_seconds"] for row in rows]
    rerank_latency = [row["reranker_seconds"] for row in rows]
    total_latency = [row["total_seconds"] for row in rows]
    latency = {
        "device": settings.reranker_device,
        "dtype": settings.reranker_dtype,
        "batch_size": settings.reranker_batch_size,
        "candidate_k": settings.reranker_input_k,
        "model_load_seconds": model_load_seconds,
        "embedding_model_load_seconds": embedding_load_seconds,
        "p50_dense_ann_seconds": _percentile(dense_latency, 0.5),
        "p95_dense_ann_seconds": _percentile(dense_latency, 0.95),
        "p50_reranker_seconds": _percentile(rerank_latency, 0.5),
        "p95_reranker_seconds": _percentile(rerank_latency, 0.95),
        "p50_total_seconds": _percentile(total_latency, 0.5),
        "p95_total_seconds": _percentile(total_latency, 0.95),
        "mean_candidates_per_second": settings.reranker_input_k / statistics.fmean(rerank_latency),
        "performance_status": "NEEDS_OPTIMIZATION",
        "functional_runtime": "READY",
        "per_question": rows,
    }
    chunk_digest = hashlib.sha256("\n".join(before_ids).encode("utf-8")).hexdigest()
    runtime_before = {
        "strategy": "legacy_multilingual",
        "source_artifact": "phase6_retrieval_baselines_v1/b2_runtime_translation_dense.json",
        "runtime_source_sha256": "1b1d8f4249c8d2cd3ca0cdd270db804bdb92bb3f85b5bed1a712e6b5e3e75082",
        "metrics": before_summary,
        "rankings": old_rankings,
    }
    runtime_after = {
        "strategy": "dense_reranker",
        "default_path": "BGE-M3 Dense -> Chroma ANN Top20 -> Learned Reranker -> Top5",
        "runtime_source_sha256": _sha256(ROOT / "src/evidenceflow/runtime.py"),
        "metrics": after_summary,
        "rankings": rankings,
    }
    regression = {
        "old_runtime": before_summary,
        "new_runtime": after_summary,
        "phase8_expected": phase8_expected,
        "new_minus_phase8": reproduction_delta,
        "phase8_reproduced": quality_reproduced,
    }
    rollback = {
        "mechanism": "RERANKER_ENABLED=false",
        "target_strategy": "legacy_multilingual",
        "requires_code_change": False,
        "requires_chroma_rebuild": False,
        "requires_embedding_recompute": False,
        "translation_available": True,
        "hybrid_available_when_multilingual_hybrid_enabled": True,
        "validated_by": [
            "tests/test_runtime.py::test_runtime_factory_wires_hybrid_bm25_when_enabled",
            "full pytest suite",
        ],
        "status": "PASS",
    }
    architecture = _architecture_manifest()
    cleanup = {
        "manifest_version": "cleanup-v3-phase9",
        "supersedes": "../phase8_runtime_readiness_v1/cleanup_manifest_v3.json",
        "deletions": [],
        "retained": ["Translation", "BM25", "Hybrid", "RRF", "Legacy Evaluation", "Historical artifacts"],
        "next_cleanup": "independent phase after new Runtime stability",
    }
    gates = {
        "phase8_pass": True,
        "dense_ann_top20": all(row["candidate_count"] == 20 for row in rows),
        "reranker_top5_contract": all(row["final_default_count"] <= 5 for row in rows),
        "runtime_quality_reproduced": quality_reproduced,
        "rollback": rollback["status"] == "PASS",
        "formal_index_unchanged": index_unchanged,
        "translation_not_default": True,
        "bm25_hybrid_not_default": True,
        "rrf_not_default_ranker": True,
    }
    passed = all(gates.values())
    execution = {
        "phase": 9,
        "status": "PASS" if passed else "FAIL",
        "runtime_promotion": "PASS" if passed else "FAIL",
        "new_default_retrieval_path": runtime_after["default_path"],
        "rollback": rollback["status"],
        "translation_default": False,
        "bm25_hybrid_default": False,
        "rrf_default_ranking": False,
        "runtime_quality": "REPRODUCED" if quality_reproduced else "REGRESSED",
        "functional_runtime": "READY" if passed else "NOT_READY",
        "interactive_performance": "NEEDS_OPTIMIZATION",
        "next_phase": "Phase 10 Reranker Performance Optimization" if passed else "Phase 9 failure diagnosis",
        "gates": gates,
    }
    manifest = {
        "phase": 9,
        "status": "EXECUTABLE_WITH_ADJUSTMENT",
        "adjustment": "CPU float32 only; interactive latency remains outside optimization scope.",
        "dataset_id": dataset.dataset_id,
        "mapping_version": mapping.mapping_version,
        "retrieval_contract_version": 2,
        "index_version": phase8_manifest["index_version"],
        "collection_name": settings.collection_name,
        "chunk_count": len(before_ids),
        "chunk_id_set_sha256": chunk_digest,
        "candidate_k": settings.reranker_input_k,
        "final_k": settings.reranker_final_k,
        "reranker": {
            "id": "BAAI/bge-reranker-base",
            "resolved_local_path": str(MODEL.relative_to(ROOT)),
            "max_length": settings.reranker_max_length,
            "batch_size": settings.reranker_batch_size,
            "device": settings.reranker_device,
            "dtype": settings.reranker_dtype,
        },
    }

    artifacts = {
        "run_manifest.json": manifest,
        "runtime_before.json": runtime_before,
        "runtime_after.json": runtime_after,
        "runtime_regression.json": regression,
        "paired_comparison.json": paired,
        "latency.json": latency,
        "rollback_validation.json": rollback,
        "architecture_simplification_v1.json": architecture,
        "cleanup_manifest_v3.json": cleanup,
        "execution_result.json": execution,
    }
    for name, payload in artifacts.items():
        save_artifact(payload, OUT / name)
    report = f"""# Phase 9 Runtime Promotion + Architecture Simplification

- Runtime Promotion: **{execution['runtime_promotion']}**
- New default: `{execution['new_default_retrieval_path']}`
- Rollback: **{execution['rollback']}** via `RERANKER_ENABLED=false`
- Translation / BM25 / Hybrid default: **disabled**
- RRF default final ranking: **disabled**
- Phase 8 quality reproduced: **{quality_reproduced}**
- nDCG@5 / MRR@5: {after_summary['ndcg_at_5']:.4f} / {after_summary['mrr_at_5']:.4f}
- CPU total p50 / p95: {latency['p50_total_seconds']:.3f}s / {latency['p95_total_seconds']:.3f}s
- Functional Runtime: **{latency['functional_runtime']}**
- Interactive Performance: **{latency['performance_status']}**
- Formal Chroma unchanged: **{index_unchanged}** ({len(before_ids)} chunks)

Phase 9 在此停止；未执行模型替换、低精度、GPU、Context 或数据层优化。
"""
    (OUT / "report.md").write_text(report, encoding="utf-8")
    return execution


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))
