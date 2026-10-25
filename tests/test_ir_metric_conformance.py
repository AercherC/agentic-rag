"""Unit tests for semantic conformance of standard IR metrics (Phase 11C Step 4).

The environment has no reference implementation such as pytrec_eval / BEIR installed, so option B is
used: `experiments/run_phase11c_technical_debt_closure.py fixture` independently implements a reference
version from the textbook definitions and generates a frozen TREC-style fixture; this test then uses that
fixture to validate the production implementation in `evidenceflow.eval.ir_metrics`, confirming that the
semantics were not altered by implementation details.

The fixture covers Precision / Recall / MRR / MAP / nDCG at depths 1/3/5/10, including three boundary
classes: "no hit", "duplicate chunk_id deduplication", and "no qrels, must be excluded".
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
    """Read the frozen fixture; fail outright when it is missing, so the test cannot degrade silently."""

    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def _eligible_cases(fixture: dict):
    """Expand every (question_id, family, depth, expected) combination that has qrels."""

    for qid, metrics in sorted(fixture["expected_per_query"].items()):
        if not fixture["qrels"].get(qid):
            continue
        for key, expected in sorted(metrics.items()):
            family, depth = key.rsplit("_at_", 1)
            yield qid, family, int(depth), expected


def test_frozen_fixture_matches_production_implementation(fixture: dict) -> None:
    """The frozen values and the production implementation agree within 1e-12."""

    for qid, family, depth, expected in _eligible_cases(fixture):
        actual = PRODUCTION_FUNCTIONS[family](fixture["run"][qid], set(fixture["qrels"][qid]), depth)
        assert abs(actual - expected) <= TOLERANCE, f"{qid} {family}@{depth}: {actual} != {expected}"


@pytest.mark.parametrize(
    "metric",
    ["precision_at_5", "recall_at_5", "recall_at_10", "mrr_at_5", "map_at_5", "ndcg_at_5"],
)
def test_gate_metrics_have_frozen_reference_values(fixture: dict, metric: str) -> None:
    """The six Gate metrics of interest to Step 13 all have frozen reference values in the fixture."""

    assert metric in fixture["expected_per_query"]["q1"]


def test_evaluate_ir_run_aggregate_matches_frozen(fixture: dict) -> None:
    """The aggregate path and the per-query path produce the same set of numbers."""

    result = evaluate_ir_run(fixture["qrels"], fixture["run"], depths=tuple(fixture["depths"]))
    assert result["excluded_queries"] == fixture["excluded_queries"]
    for qid, metrics in fixture["expected_per_query"].items():
        if not fixture["qrels"].get(qid):
            continue
        for key, expected in metrics.items():
            assert abs(result["per_query"][qid]["metrics"][key] - expected) <= TOLERANCE
    # JSON keys are always strings while the aggregate result keys by int depth, so convert before
    # comparing.
    frozen_precision = {int(k): v for k, v in fixture["expected_aggregate"]["precision_at_k"].items()}
    assert result["aggregate"]["precision_at_k"] == pytest.approx(frozen_precision, abs=TOLERANCE)


def test_zero_hit_query_yields_zero_metrics(fixture: dict) -> None:
    """All metrics of a query with no hits are 0, preventing 0/0 from being computed as 1."""

    rel = set(fixture["qrels"]["q2"])
    ranked = fixture["run"]["q2"]
    for family, func in PRODUCTION_FUNCTIONS.items():
        assert func(ranked, rel, 5) == 0.0, family
