"""标准 IR 指标公式级确定性测试（对应 Phase 11A Step 11）。

所有用例均为手工 qrels/run fixture，不依赖外部检索、Embedding 或 Chroma。
覆盖：Precision / Recall / MRR / MAP / nDCG 的端点语义，以及截断 K、重复
chunk_id、空 qrels、无 eligible qrels、确定性输出等边界。
"""

import pytest

from evidenceflow.eval.ir_metrics import (
    average_precision_at_k,
    evaluate_ir_run,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank_at_k,
)


def test_precision_at_3_matches_spec() -> None:
    # qrels={A,B}, ranking=[A,X,B] => 前 3 命中 A,B => 2/3
    rel = {"A", "B"}
    assert precision_at_k(["A", "X", "B"], rel, 3) == pytest.approx(2 / 3)
    assert precision_at_k(["A", "X", "B"], rel, 1) == pytest.approx(1.0)
    assert precision_at_k(["A", "X", "B"], rel, 2) == pytest.approx(1 / 2)


def test_recall_at_3_full() -> None:
    # Recall@3 = 检索到的相关 / 相关总数 = 2/2 = 1
    rel = {"A", "B"}
    assert recall_at_k(["A", "X", "B"], rel, 3) == pytest.approx(1.0)
    # 只命中一个 => 1/2
    assert recall_at_k(["A", "X", "Y"], rel, 3) == pytest.approx(1 / 2)
    # 截断更小则遗漏第二个相关
    assert recall_at_k(["A", "X", "B"], rel, 1) == pytest.approx(1 / 2)


def test_recall_denominator_is_relevant_count_not_k() -> None:
    # 标准 Recall 分母是相关总数，而非 K；早期 HitRate 实现分母才是 K。
    rel = {"A", "B", "C"}
    assert recall_at_k(["A", "B", "C", "X", "Y"], rel, 5) == pytest.approx(1.0)
    assert recall_at_k(["A", "B", "X", "Y", "Z"], rel, 5) == pytest.approx(2 / 3)


def test_mrr_at_2_first_relevant_rank() -> None:
    rel = {"A", "B"}
    # ranking=[X,B,A] => 首个相关在 rank 2 => RR=1/2
    assert reciprocal_rank_at_k(["X", "B", "A"], rel, 5) == pytest.approx(1 / 2)
    # 首个相关在 rank 1
    assert reciprocal_rank_at_k(["A", "X", "B"], rel, 5) == pytest.approx(1.0)
    # Top-K 内无相关
    assert reciprocal_rank_at_k(["X", "Y", "Z"], rel, 3) == 0.0


def test_map_multiple_relevant_ranks() -> None:
    rel = {"A", "B"}
    # rank1=A (P=1/1=1), rank2=B (P=2/2=1) => AP=(1+1)/2=1
    assert average_precision_at_k(["A", "B", "X"], rel, 3) == pytest.approx(1.0)
    # rank1=A (1), rank3=B (2/3) => AP=(1+2/3)/2 = 5/6
    assert average_precision_at_k(["A", "X", "B"], rel, 3) == pytest.approx(5 / 6)
    # 仅命中一个且 rank1 => AP=1/2
    assert average_precision_at_k(["A", "X", "Y"], rel, 3) == pytest.approx(1 / 2)


def test_ndcg_perfect_shifted_and_none() -> None:
    rel = {"A", "B"}
    # 完美排序：rank1,rank2 相关 => DCG=1/log2(2)+1/log2(3); IDCG 相同 => 1
    assert ndcg_at_k(["A", "B", "X"], rel, 3) == pytest.approx(1.0)
    # 相关后移：rank2,rank3 => DCG < IDCG => < 1
    shifted = ndcg_at_k(["X", "A", "B"], rel, 3)
    assert 0.0 < shifted < 1.0
    # 无相关
    assert ndcg_at_k(["X", "Y", "Z"], rel, 3) == 0.0


def test_cutoff_k_limits_computation() -> None:
    rel = {"A", "B", "C"}
    # @1 只看到 A
    assert precision_at_k(["A", "B", "C"], rel, 1) == pytest.approx(1.0)
    assert recall_at_k(["A", "B", "C"], rel, 1) == pytest.approx(1 / 3)
    # @2 看到 A,B
    assert recall_at_k(["A", "B", "C"], rel, 2) == pytest.approx(2 / 3)
    # 截断到深度 2 时，rank3 的命中不计入
    assert recall_at_k(["X", "Y", "A"], rel, 2) == pytest.approx(0.0)


def test_duplicate_chunk_id_ignored() -> None:
    rel = {"A", "B"}
    # 重复 A 不制造新的有效名次；rank1=A 命中一次
    assert precision_at_k(["A", "A", "B"], rel, 3) == pytest.approx(2 / 3)
    assert recall_at_k(["A", "A", "B"], rel, 3) == pytest.approx(1.0)
    # 重复非相关不影响首个相关名次
    assert reciprocal_rank_at_k(["X", "X", "A"], rel, 5) == pytest.approx(1 / 3)


def test_empty_qrels_recall_and_ndcg_zero() -> None:
    assert recall_at_k(["A", "B"], set(), 5) == 0.0
    assert ndcg_at_k(["A", "B"], set(), 5) == 0.0
    assert average_precision_at_k(["A", "B"], set(), 5) == 0.0


def test_query_without_eligible_qrels_excluded_from_aggregate() -> None:
    qrels = {"q1": ["A", "B"]}
    run = {"q1": ["A", "X", "B"], "q2": ["Y", "Z"]}  # q2 无 eligible qrels
    result = evaluate_ir_run(qrels, run, depths=(1, 3, 5))
    # 聚合只覆盖 eligible 的 q1
    assert result["aggregate"]["question_count"] == 2
    assert result["aggregate"]["eligible_count"] == 1
    assert result["aggregate"]["excluded_count"] == 1
    assert result["excluded_queries"] == ["q2"]
    assert result["aggregate"]["precision_at_k"][3] == pytest.approx(2 / 3)
    assert result["per_query"]["q2"]["metrics"] is None


def test_deterministic_output() -> None:
    qrels = {"q1": ["A", "B"], "q2": ["C"]}
    run = {"q1": ["A", "X", "B"], "q2": ["C", "Y"]}
    r1 = evaluate_ir_run(qrels, run, depths=(1, 3, 5))
    r2 = evaluate_ir_run(qrels, run, depths=(1, 3, 5))
    assert r1 == r2


def test_evaluate_ir_run_rejects_empty_inputs() -> None:
    with pytest.raises(ValueError):
        evaluate_ir_run({}, {"q1": ["A"]})
    with pytest.raises(ValueError):
        evaluate_ir_run({"q1": ["A"]}, {})
