"""验证已生成的 Phase 7 冻结候选与 Gate 产物。"""

import json
from pathlib import Path


PHASE7 = Path(__file__).parents[1] / "evals" / "evaluation_v2" / "phase7_reranker_validation_v1"


def _load(name: str):
    """读取确定性实验 artifact，避免测试依赖外部模型或网络。"""

    return json.loads((PHASE7 / name).read_text(encoding="utf-8"))


def test_phase7_required_artifacts_and_gate_are_present() -> None:
    """计划要求的 11 个 JSON/Markdown 产物必须完整且 Gate 为 PASS。"""

    names = [
        "run_manifest.json", "candidate_pool_manifest.json", "candidate_replay.json",
        "reranker_results.json", "candidate_budget_comparison.json", "slice_analysis.json",
        "paired_comparison.json", "latency.json", "architecture_ablation_v3.json",
        "bottleneck_diagnosis_v2.json", "execution_result.json", "report.md",
    ]
    assert all((PHASE7 / name).is_file() for name in names)
    execution = _load("execution_result.json")
    assert execution["gate"] == "PASS"
    assert all(execution["gates"].values())


def test_candidate_replay_ids_and_sources_are_stable() -> None:
    """每道题的三个架构均保存 100 个唯一候选及来源分支。"""

    replay = _load("candidate_replay.json")
    assert replay["replay_equal"] is True
    assert set(replay["questions"]) == {f"ARR-{index:03d}" for index in range(1, 41)}
    for question in replay["questions"].values():
        for architecture in ("c0_original_dense", "c1_translation_dense", "c2_hybrid"):
            records = question["architectures"][architecture]
            ids = [record["chunk_id"] for record in records]
            assert len(records) == 100
            assert len(ids) == len(set(ids))
            assert all(record["source_branches"] for record in records)


def test_reranker_candidate_coverage_guardrail_is_unchanged() -> None:
    """预算和架构变化只能影响排序，不能改变候选覆盖率。"""

    results = _load("reranker_results.json")
    for architecture in ("c0_original_dense", "c1_translation_dense", "c2_hybrid"):
        for budget in ("20", "30", "50", "100"):
            assert results[architecture][budget]["coverage_unchanged"] is True
