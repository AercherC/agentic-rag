"""标准 IR 指标语义一致性单元测试（Phase 11C Step 4）。

环境未安装 pytrec_eval / BEIR 等参考实现，因此采用方案 B：
由 `experiments/run_phase11c_technical_debt_closure.py fixture` 按教科书定义独立实现一份
参考实现，生成冻结 TREC-style fixture；本测试再用该 fixture 校验
`evidenceflow.eval.ir_metrics` 的生产实现，确保语义未被实现细节改变。

fixture 覆盖：Precision / Recall / MRR / MAP / nDCG，深度 1/3/5/10；
含「无命中」「重复 chunk_id 去重」「无 qrels 需排除」三类边界。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evidenceflow.eval.ir_metrics import (
    average_precision_at_k,
    evaluate_ir_run,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank_at_k,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXTURE = (
    PROJECT_ROOT
    / "evals/evaluation_v3/phase11c_technical_debt_closure_v1/ir_metric_conformance_fixture.json"
)
TOLERANCE = 1e-12

PRODUCTION_FUNCTIONS = {
    "precision": precision_at_k,
    "recall": recall_at_k,
    "mrr": reciprocal_rank_at_k,
    "map": average_precision_at_k,
    "ndcg": ndcg_at_k,
}


@pytest.fixture(scope="module")
def fixture() -> dict:
    """读取冻结 fixture；缺失时直接失败，避免测试静默退化。"""

    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def _eligible_cases(fixture: dict):
    """展开 (question_id, family, depth, expected) 的全部有 qrels 组合。"""

    for qid, metrics in sorted(fixture["expected_per_query"].items()):
        if not fixture["qrels"].get(qid):
            continue
        for key, expected in sorted(metrics.items()):
            family, depth = key.rsplit("_at_", 1)
            yield qid, family, int(depth), expected


def test_frozen_fixture_matches_production_implementation(fixture: dict) -> None:
    """冻结值与生产实现在 1e-12 内一致。"""

    for qid, family, depth, expected in _eligible_cases(fixture):
        actual = PRODUCTION_FUNCTIONS[family](fixture["run"][qid], set(fixture["qrels"][qid]), depth)
        assert abs(actual - expected) <= TOLERANCE, f"{qid} {family}@{depth}: {actual} != {expected}"


@pytest.mark.parametrize(
    "metric",
    ["precision_at_5", "recall_at_5", "recall_at_10", "mrr_at_5", "map_at_5", "ndcg_at_5"],
)
def test_gate_metrics_have_frozen_reference_values(fixture: dict, metric: str) -> None:
    """Step 13 关注的六个 Gate 指标在 fixture 中都有冻结参考值。"""

    assert metric in fixture["expected_per_query"]["q1"]


def test_evaluate_ir_run_aggregate_matches_frozen(fixture: dict) -> None:
    """聚合路径与逐题路径给出同一组数字。"""

    result = evaluate_ir_run(fixture["qrels"], fixture["run"], depths=tuple(fixture["depths"]))
    assert result["excluded_queries"] == fixture["excluded_queries"]
    for qid, metrics in fixture["expected_per_query"].items():
        if not fixture["qrels"].get(qid):
            continue
        for key, expected in metrics.items():
            assert abs(result["per_query"][qid]["metrics"][key] - expected) <= TOLERANCE
    # JSON 的 key 恒为字符串，聚合结果以 int 深度为 key，比较前统一成 int。
    frozen_precision = {int(k): v for k, v in fixture["expected_aggregate"]["precision_at_k"].items()}
    assert result["aggregate"]["precision_at_k"] == pytest.approx(frozen_precision, abs=TOLERANCE)


def test_zero_hit_query_yields_zero_metrics(fixture: dict) -> None:
    """无命中 query 的全部指标为 0，避免 0/0 被算成 1。"""

    rel = set(fixture["qrels"]["q2"])
    ranked = fixture["run"]["q2"]
    for family, func in PRODUCTION_FUNCTIONS.items():
        assert func(ranked, rel, 5) == 0.0, family
