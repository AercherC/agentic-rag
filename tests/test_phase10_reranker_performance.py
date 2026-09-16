"""验证 Phase 10 性能实验 artifact 的冻结合同。"""

import json
from pathlib import Path


PHASE10 = Path(__file__).parents[1] / "evals/evaluation_v2/phase10_reranker_performance_v1"


def _load(name: str):
    """读取 Phase 10 的确定性 JSON 产物。"""

    return json.loads((PHASE10 / name).read_text(encoding="utf-8"))


def test_phase10_artifacts_prove_frozen_quality_and_no_runtime_promotion() -> None:
    """性能实验必须保留 Phase 9 质量，且 p95 回归时不得晋升配置。"""

    required = {
        "run_manifest.json",
        "baseline.json",
        "latency_breakdown.json",
        "cpu_batch_benchmark.json",
        "cpu_runtime_benchmark.json",
        "gpu_benchmark.json",
        "model_ablation.json",
        "quality_latency_pareto.json",
        "runtime_regression.json",
        "rollback_validation.json",
        "execution_result.json",
        "report.md",
    }
    assert required <= {path.name for path in PHASE10.iterdir()}

    execution = _load("execution_result.json")
    baseline = _load("baseline.json")
    pareto = _load("quality_latency_pareto.json")
    regression = _load("runtime_regression.json")

    assert execution["status"] == "PASS"
    assert execution["best_cpu_variant"] is None
    assert execution["runtime_modified"] is False
    assert execution["formal_chroma_modified"] is False
    assert execution["gold_chunk_embedding_modified"] is False
    assert baseline["quality_metrics"] == regression["phase9_expected"]
    assert {row["name"] for row in pareto} == {
        "baseline_cpu_fp32_b4",
        "cpu_fp32_b4_inference_mode",
    }
    assert pareto[1]["p50_seconds"] < pareto[0]["p50_seconds"]
    assert pareto[1]["p95_seconds"] > pareto[0]["p95_seconds"]
