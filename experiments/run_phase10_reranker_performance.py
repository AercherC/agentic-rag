"""执行 Phase 10 的冻结 Candidate Replay Reranker 性能验证。"""

from __future__ import annotations

import hashlib
import json
import platform
import statistics
import time
from pathlib import Path
from typing import Any

import torch

from evidenceflow.config import Settings
from evidenceflow.domain.evaluation_v2 import EvaluationDatasetV2, MappingArtifact
from evidenceflow.embedding import BgeM3EmbeddingProvider
from evidenceflow.eval.v2 import evaluate_rankings, save_artifact
from evidenceflow.retrieval import DenseRetriever, LearnedReranker, TransformersCrossEncoderScorer
from evidenceflow.storage import ChromaVectorStore


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "evals/evaluation_v2/phase10_reranker_performance_v1"
P9 = ROOT / "evals/evaluation_v2/phase9_runtime_promotion_v1"
P35 = ROOT / "evals/evaluation_v2/phase35_gold_stabilization_v1"
MODEL = ROOT / ".cache/models/bge-reranker-base"
BATCH_SIZES = (1, 2, 4, 8, 16)
THREAD_COUNTS = (4, 8, 16)
WARMUP_RUNS = 1
MEASURED_RUNS = 2
EXPECTED = {
    "recall_at_5": 0.5625,
    "recall_at_10": 0.7375,
    "mrr_at_5": 0.4320833333333334,
    "ndcg_at_5": 0.46339982650599376,
    "all_required_evidence_at_5": 0.55,
    "all_required_evidence_at_10": 0.725,
}


def _load(path: Path) -> dict[str, Any]:
    """读取 UTF-8 JSON artifact。"""

    return json.loads(path.read_text(encoding="utf-8"))


def _percentile(values: list[float], fraction: float) -> float:
    """以线性插值计算小样本分位数。"""

    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _summary(report: dict[str, Any]) -> dict[str, float]:
    """提取 Phase 10 Gate 固定的六项质量指标。"""

    aggregate = report["aggregate"]
    return {
        "recall_at_5": aggregate["evidence_group_recall_at_5"],
        "recall_at_10": aggregate["evidence_group_recall_at_10"],
        "mrr_at_5": aggregate["mrr_at_5"],
        "ndcg_at_5": aggregate["custom_evidence_group_ndcg_at_5"],
        "all_required_evidence_at_5": aggregate["all_required_evidence_at_5"],
        "all_required_evidence_at_10": aggregate["all_required_evidence_at_10"],
    }


def _environment() -> dict[str, Any]:
    """记录可影响推理性能的本机环境事实。"""

    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "initial_intra_op_threads": torch.get_num_threads(),
        "initial_inter_op_threads": torch.get_num_interop_threads(),
    }


def _build_replay(dataset: EvaluationDatasetV2, settings: Settings) -> dict[str, list[Any]]:
    """只读地冻结 40Q 的 Dense ANN Top20 候选，禁止 benchmark 重查 Chroma。"""

    store = ChromaVectorStore(settings.vector_db_path, settings.collection_name)
    dense = DenseRetriever(
        BgeM3EmbeddingProvider(settings.embedding_model, device="cpu"), store, top_k=20
    )
    replay: dict[str, list[Any]] = {}
    for question in dataset.questions:
        candidates = dense.retrieve(question.query, top_k=20)
        if len(candidates) != 20:
            raise RuntimeError(f"candidate count anomaly for {question.question_id}: {len(candidates)}")
        ids = [candidate.chunk.chunk_id for candidate in candidates]
        if len(ids) != len(set(ids)):
            raise RuntimeError(f"duplicate candidate ID for {question.question_id}")
        replay[question.question_id] = candidates
    return replay


def _run_variant(
    *,
    name: str,
    dataset: EvaluationDatasetV2,
    mapping: MappingArtifact,
    replay: dict[str, list[Any]],
    batch_size: int,
    cpu_threads: int,
    inference_mode: bool,
) -> dict[str, Any]:
    """在同一回放候选上测试唯一一项推理配置，并校验排序与质量。"""

    torch.set_num_threads(cpu_threads)
    scorer = TransformersCrossEncoderScorer(
        str(MODEL), device="cpu", dtype="float32", max_length=512, inference_mode=inference_mode
    )
    reranker = LearnedReranker(scorer, batch_size=batch_size, max_candidates=20)

    # 预热只触发模型内核初始化；不混入稳态 latency 或质量统计。
    warmup_query = dataset.questions[0]
    for _ in range(WARMUP_RUNS):
        reranker.rerank(warmup_query.query, [row.chunk for row in replay[warmup_query.question_id]], top_k=10)

    rows: list[dict[str, Any]] = []
    rankings: dict[str, list[str]] = {}
    for repeat_index in range(MEASURED_RUNS):
        for question in dataset.questions:
            candidates = replay[question.question_id]
            candidate_ids = [row.chunk.chunk_id for row in candidates]
            started = time.perf_counter()
            ranked = reranker.rerank(question.query, [row.chunk for row in candidates], top_k=10)
            total = time.perf_counter() - started
            output_ids = [row.chunk.chunk_id for row in ranked]
            if not set(output_ids) <= set(candidate_ids) or len(output_ids) != 10:
                raise RuntimeError(f"reranker candidate identity anomaly for {question.question_id}")
            # 每个重复运行必须完全一致，避免把波动误判为性能优化。
            if repeat_index and output_ids != rankings[question.question_id]:
                raise RuntimeError(f"ranking drift for {question.question_id}")
            rankings[question.question_id] = output_ids
            rows.append(
                {
                    "repeat": repeat_index + 1,
                    "question_id": question.question_id,
                    "candidate_count": len(candidate_ids),
                    "candidate_id_sha256": hashlib.sha256("\n".join(candidate_ids).encode("utf-8")).hexdigest(),
                    "reranker_seconds": total,
                    "timing_seconds": reranker.last_timing_seconds,
                    "forward_passes": scorer.forward_passes,
                }
            )

    report = evaluate_rankings(dataset, mapping, rankings, depths=(5, 10))
    quality = _summary(report)
    if any(abs(quality[key] - value) > 1e-9 for key, value in EXPECTED.items()):
        raise RuntimeError(f"quality anomaly for {name}: {quality}")
    latencies = [row["reranker_seconds"] for row in rows]
    breakdown = {
        stage: statistics.fmean(row["timing_seconds"].get(stage, 0.0) for row in rows)
        for stage in ("untruncated_tokenization", "model_tokenization", "model_forward", "score_postprocess")
    }
    return {
        "name": name,
        "configuration": {
            "model": "BAAI/bge-reranker-base",
            "device": "cpu",
            "dtype": "float32",
            "batch_size": batch_size,
            "cpu_threads": cpu_threads,
            "max_length": 512,
            "candidate_k": 20,
            "final_k": 5,
            "inference_mode": inference_mode,
            "warmup_runs": WARMUP_RUNS,
            "measured_runs": MEASURED_RUNS,
        },
        "p50_seconds": _percentile(latencies, 0.5),
        "p95_seconds": _percentile(latencies, 0.95),
        "mean_seconds": statistics.fmean(latencies),
        "throughput_candidates_per_second": 20 / statistics.fmean(latencies),
        "memory": None,
        "quality_metrics": quality,
        "rankings": rankings,
        "latency_breakdown_mean_seconds": breakdown,
        "per_measurement": rows,
    }


def _write_anomaly(completed: str, exc: Exception, state: dict[str, Any]) -> None:
    """按 STOP-ON-ANOMALY 保存当前事实，禁止继续执行后续变体。"""

    OUT.mkdir(parents=True, exist_ok=True)
    checkpoint = {"status": "STOPPED_ON_ANOMALY", "completed_step": completed, "error": repr(exc), **state}
    save_artifact(checkpoint, OUT / "progress_checkpoint.json")
    report = f"""# Phase 10 Anomaly Report

## 停止位置

{completed}

## 原计划与实际结果

计划在冻结 40Q Candidate Replay 上仅测试 Reranker 推理配置。实际异常：`{exc!r}`。

## 已确认事实

- 未修改正式 Runtime、Chroma、Gold、Chunk 或 Embedding。
- 已完成状态和已生成 artifact 见 `progress_checkpoint.json`。

## 未确认假设

- 不对异常根因作推测。

## 建议下一步

检查 checkpoint 所记录的配置、候选与质量数据后，由用户决定是否启动独立诊断。
"""
    (OUT / "anomaly_report.md").write_text(report, encoding="utf-8")


def _pareto_front(variants: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """保留质量相同前提下 p50、p95 与吞吐均不被支配的真实折中方案。"""

    front: list[dict[str, Any]] = []
    for candidate in variants:
        dominated = any(
            other is not candidate
            and other["p50_seconds"] <= candidate["p50_seconds"]
            and other["p95_seconds"] <= candidate["p95_seconds"]
            and other["throughput_candidates_per_second"] >= candidate["throughput_candidates_per_second"]
            and (
                other["p50_seconds"] < candidate["p50_seconds"]
                or other["p95_seconds"] < candidate["p95_seconds"]
                or other["throughput_candidates_per_second"] > candidate["throughput_candidates_per_second"]
            )
            for other in variants
        )
        if not dominated:
            front.append(candidate)
    return front


def run() -> dict[str, Any]:
    """执行 Phase 10；任何异常保存报告并立即返回 STOPPED 状态。"""

    if OUT.exists():
        raise FileExistsError("Phase 10 output already exists; refusing to overwrite artifacts")
    state: dict[str, Any] = {"environment": _environment(), "variants_completed": []}
    try:
        phase9 = _load(P9 / "execution_result.json")
        if phase9["status"] != "PASS":
            raise RuntimeError("Phase 9 baseline is not PASS")
        if any(not (MODEL / file).is_file() for file in ("config.json", "model.safetensors", "tokenizer.json")):
            raise FileNotFoundError("local reranker model is incomplete")
        dataset = EvaluationDatasetV2.model_validate_json((P35 / "research_questions_v2_gold_stable.json").read_text(encoding="utf-8"))
        mapping = MappingArtifact.model_validate_json((P35 / "gold_mapping_v3.json").read_text(encoding="utf-8"))
        settings = Settings(reranker_model=str(MODEL))
        replay = _build_replay(dataset, settings)
        state["candidate_replay"] = {key: [row.chunk.chunk_id for row in value] for key, value in replay.items()}

        variants = [_run_variant(name="baseline_cpu_fp32_b4", dataset=dataset, mapping=mapping, replay=replay, batch_size=4, cpu_threads=16, inference_mode=False)]
        state["variants_completed"].append(variants[-1]["name"])
        for batch_size in BATCH_SIZES:
            if batch_size == 4:
                continue
            variants.append(_run_variant(name=f"cpu_fp32_b{batch_size}", dataset=dataset, mapping=mapping, replay=replay, batch_size=batch_size, cpu_threads=16, inference_mode=False))
            state["variants_completed"].append(variants[-1]["name"])
        variants.append(_run_variant(name="cpu_fp32_b4_inference_mode", dataset=dataset, mapping=mapping, replay=replay, batch_size=4, cpu_threads=16, inference_mode=True))
        state["variants_completed"].append(variants[-1]["name"])
        for threads in THREAD_COUNTS:
            if threads == 16:
                continue
            variants.append(_run_variant(name=f"cpu_fp32_b4_threads{threads}", dataset=dataset, mapping=mapping, replay=replay, batch_size=4, cpu_threads=threads, inference_mode=True))
            state["variants_completed"].append(variants[-1]["name"])

        baseline = variants[0]
        pareto_variants = _pareto_front(variants)
        pareto = [{key: row[key] for key in ("name", "p50_seconds", "p95_seconds", "throughput_candidates_per_second", "quality_metrics")} for row in pareto_variants]
        # Promotion 同时要求 p50 改善且 p95 不退化；不能只挑选更漂亮的中位数。
        promotable = [
            row for row in pareto_variants
            if row["p50_seconds"] < baseline["p50_seconds"]
            and row["p95_seconds"] <= baseline["p95_seconds"]
        ]
        promoted = min(promotable, key=lambda row: row["p50_seconds"]) if promotable else None
        manifest = {"phase": 10, "dataset_id": dataset.dataset_id, "candidate_replay_sha256": hashlib.sha256(json.dumps(state["candidate_replay"], ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest(), "environment": state["environment"], "gpu_benchmark": None}
        execution = {"phase": 10, "status": "PASS", "best_cpu_variant": promoted["name"] if promoted else None, "pareto_front": [row["name"] for row in pareto_variants], "runtime_modified": False, "runtime_observability_changed": True, "formal_chroma_modified": False, "gold_chunk_embedding_modified": False, "interactive_performance": "INTERACTIVE_ACCEPTABLE" if promoted and promoted["p95_seconds"] < 2.0 else "NEEDS_OPTIMIZATION"}
        OUT.mkdir(parents=True)
        save_artifact(manifest, OUT / "run_manifest.json")
        save_artifact(baseline, OUT / "baseline.json")
        save_artifact({"baseline": baseline["latency_breakdown_mean_seconds"], "promoted": promoted["latency_breakdown_mean_seconds"] if promoted else None}, OUT / "latency_breakdown.json")
        save_artifact([row for row in variants if "threads" not in row["name"]], OUT / "cpu_batch_benchmark.json")
        save_artifact([row for row in variants if "inference_mode" in row["name"] or "threads" in row["name"]], OUT / "cpu_runtime_benchmark.json")
        save_artifact(None, OUT / "gpu_benchmark.json")
        save_artifact([], OUT / "model_ablation.json")
        save_artifact(pareto, OUT / "quality_latency_pareto.json")
        save_artifact({"phase9_expected": EXPECTED, "pareto_quality": {row["name"]: row["quality_metrics"] for row in pareto_variants}, "runtime_config_promoted": promoted["name"] if promoted else None}, OUT / "runtime_regression.json")
        save_artifact(_load(P9 / "rollback_validation.json"), OUT / "rollback_validation.json")
        save_artifact(execution, OUT / "execution_result.json")
        report = f"# Phase 10 Reranker Performance Optimization\n\n- Baseline p50/p95: {baseline['p50_seconds']:.3f}s / {baseline['p95_seconds']:.3f}s\n- Pareto front: {', '.join(execution['pareto_front'])}\n- Runtime promotion: {execution['best_cpu_variant'] or 'none; no variant improved p50 without p95 regression'}\n- GPU: unavailable\n- Runtime configuration modified: no\n- Interactive performance: {execution['interactive_performance']}\n"
        (OUT / "report.md").write_text(report, encoding="utf-8")
        return execution
    except Exception as exc:
        _write_anomaly(state["variants_completed"][-1] if state["variants_completed"] else "Step 0/1", exc, state)
        return {"phase": 10, "status": "STOPPED_ON_ANOMALY", "error": repr(exc)}
    finally:
        torch.set_num_threads(state["environment"]["initial_intra_op_threads"])


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))
