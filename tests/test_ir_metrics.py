"""Formula-level deterministic tests for the standard IR metrics (corresponding to Phase 11A
Step 11).

Every case uses a hand-written qrels/run fixture and does not depend on external retrieval,
embedding or Chroma.
Covered: the endpoint semantics of Precision / Recall / MRR / MAP / nDCG, plus boundaries such as
truncation K, duplicate chunk_id, empty qrels, no eligible qrels, and deterministic output.
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
    # qrels={A,B}, ranking=[A,X,B] => the first 3 hit A,B => 2/3
    rel = {"A", "B"}
    assert precision_at_k(["A", "X", "B"], rel, 3) == pytest.approx(2 / 3)
    assert precision_at_k(["A", "X", "B"], rel, 1) == pytest.approx(1.0)
    assert precision_at_k(["A", "X", "B"], rel, 2) == pytest.approx(1 / 2)


def test_recall_at_3_full() -> None:
    # Recall@3 = retrieved relevant / total relevant = 2/2 = 1
    rel = {"A", "B"}
    assert recall_at_k(["A", "X", "B"], rel, 3) == pytest.approx(1.0)
    # Only one hit => 1/2
    assert recall_at_k(["A", "X", "Y"], rel, 3) == pytest.approx(1 / 2)
    # A smaller cutoff misses the second relevant item
    assert recall_at_k(["A", "X", "B"], rel, 1) == pytest.approx(1 / 2)


def test_recall_denominator_is_relevant_count_not_k() -> None:
    # The standard Recall denominator is the total number of relevant items, not K; only the early
    # HitRate implementation used K as the denominator.
    rel = {"A", "B", "C"}
    assert recall_at_k(["A", "B", "C", "X", "Y"], rel, 5) == pytest.approx(1.0)
    assert recall_at_k(["A", "B", "X", "Y", "Z"], rel, 5) == pytest.approx(2 / 3)


def test_mrr_at_2_first_relevant_rank() -> None:
    rel = {"A", "B"}
    # ranking=[X,B,A] => the first relevant item is at rank 2 => RR=1/2
    assert reciprocal_rank_at_k(["X", "B", "A"], rel, 5) == pytest.approx(1 / 2)
    # The first relevant item is at rank 1
    assert reciprocal_rank_at_k(["A", "X", "B"], rel, 5) == pytest.approx(1.0)
    # No relevant item within Top-K
    assert reciprocal_rank_at_k(["X", "Y", "Z"], rel, 3) == 0.0


def test_map_multiple_relevant_ranks() -> None:
    rel = {"A", "B"}
    # rank1=A (P=1/1=1), rank2=B (P=2/2=1) => AP=(1+1)/2=1
    assert average_precision_at_k(["A", "B", "X"], rel, 3) == pytest.approx(1.0)
    # rank1=A (1), rank3=B (2/3) => AP=(1+2/3)/2 = 5/6
    assert average_precision_at_k(["A", "X", "B"], rel, 3) == pytest.approx(5 / 6)
    # Only one hit and it is rank1 => AP=1/2
    assert average_precision_at_k(["A", "X", "Y"], rel, 3) == pytest.approx(1 / 2)


def test_ndcg_perfect_shifted_and_none() -> None:
    rel = {"A", "B"}
    # Perfect ordering: rank1 and rank2 are relevant => DCG=1/log2(2)+1/log2(3); IDCG is the same
    # => 1
    assert ndcg_at_k(["A", "B", "X"], rel, 3) == pytest.approx(1.0)
    # Relevant items shifted later: rank2, rank3 => DCG < IDCG => < 1
    shifted = ndcg_at_k(["X", "A", "B"], rel, 3)
    assert 0.0 < shifted < 1.0
    # No relevant item
    assert ndcg_at_k(["X", "Y", "Z"], rel, 3) == 0.0


def test_cutoff_k_limits_computation() -> None:
    rel = {"A", "B", "C"}
    # @1 sees only A
    assert precision_at_k(["A", "B", "C"], rel, 1) == pytest.approx(1.0)
    assert recall_at_k(["A", "B", "C"], rel, 1) == pytest.approx(1 / 3)
    # @2 sees A,B
    assert recall_at_k(["A", "B", "C"], rel, 2) == pytest.approx(2 / 3)
    # Truncated to depth 2, the hit at rank3 does not count
    assert recall_at_k(["X", "Y", "A"], rel, 2) == pytest.approx(0.0)


def test_duplicate_chunk_id_ignored() -> None:
    rel = {"A", "B"}
    # A duplicate A does not create a new valid rank position; rank1=A is counted once
    assert precision_at_k(["A", "A", "B"], rel, 3) == pytest.approx(2 / 3)
    assert recall_at_k(["A", "A", "B"], rel, 3) == pytest.approx(1.0)
    # A duplicated non-relevant item does not affect the rank of the first relevant item
    assert reciprocal_rank_at_k(["X", "X", "A"], rel, 5) == pytest.approx(1 / 3)


def test_empty_qrels_recall_and_ndcg_zero() -> None:
    assert recall_at_k(["A", "B"], set(), 5) == 0.0
    assert ndcg_at_k(["A", "B"], set(), 5) == 0.0
    assert average_precision_at_k(["A", "B"], set(), 5) == 0.0


def test_query_without_eligible_qrels_excluded_from_aggregate() -> None:
    qrels = {"q1": ["A", "B"]}
    run = {"q1": ["A", "X", "B"], "q2": ["Y", "Z"]}  # q2 has no eligible qrels
    result = evaluate_ir_run(qrels, run, depths=(1, 3, 5))
    # Aggregation covers only the eligible q1
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
