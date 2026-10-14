"""Phase 15B.1 实验纯函数的确定性测试。"""

from experiments.run_phase15b1_dense_candidate_depth import (
    choose_recommendation,
    percentile,
    transitions,
)


def test_percentile_uses_deterministic_linear_interpolation() -> None:
    """延迟聚合的 p95 应在固定样本上得到稳定值。"""

    assert percentile([0.0, 10.0], 50) == 5.0
    assert percentile([1.0, 2.0, 3.0, 4.0], 95) == 3.85


def test_choose_recommendation_prefers_d30_when_quality_is_not_lower() -> None:
    """Candidate coverage 相同且 D30 主指标不低时，应选择较省预算的 D30。"""

    summaries = {
        "30": {"Recall@5": 0.58, "nDCG@5": 0.52},
        "40": {"Recall@5": 0.57, "nDCG@5": 0.50},
    }
    candidate = {
        "30": {"INDEXABLE_ONLY": {"HitRate@30": 1.0, "EvidenceAnchorRecall@30": 1.0, "AllRequired@30": 1.0}},
        "40": {"INDEXABLE_ONLY": {"HitRate@40": 1.0, "EvidenceAnchorRecall@40": 1.0, "AllRequired@40": 1.0}},
    }

    recommended, _reason = choose_recommendation(summaries, candidate)

    assert recommended == 30


def test_transition_analysis_keeps_per_query_status_and_recovered_groups() -> None:
    """状态转移必须保留从候选 miss 到成功的逐题证据。"""

    failures = {
        "arms": {
            "20": {"per_query": [{"question_id": "Q1", "status": "CANDIDATE_MISS", "candidate_miss_group_ids": ["G1"], "unindexable_group_ids": [], "reranker_miss_group_ids": []}]},
            "30": {"per_query": [{"question_id": "Q1", "status": "SUCCESS", "candidate_miss_group_ids": [], "unindexable_group_ids": [], "reranker_miss_group_ids": []}]},
            "40": {"per_query": [{"question_id": "Q1", "status": "SUCCESS", "candidate_miss_group_ids": [], "unindexable_group_ids": [], "reranker_miss_group_ids": []}]},
        }
    }

    result = transitions(failures, ["Q1"])
    row = next(item for item in result["rows"] if item.get("question_id") == "Q1")

    assert row["from_arm"] == "D20"
    assert row["to_arm"] == "D30"
    assert row["recovered_candidate_miss_groups"] == ["G1"]
    assert row["recovered_and_used_by_reranker"] is True
