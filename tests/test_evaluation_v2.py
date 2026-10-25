"""Verify evidence-group composition, fragment coverage, legacy metric migration and paired comparison."""

import json
from pathlib import Path

import pytest

from evidenceflow.domain.evaluation_v2 import (
    EvaluationDatasetV2, EvidenceLocator, GroupMapping, MappingArtifact, validate_dataset_splits,
)
from evidenceflow.eval.dataset import adapt_research_v1
from evidenceflow.eval.metrics import custom_evidence_group_ndcg, legacy_research_metrics
from evidenceflow.eval.comparison import paired_comparison, holm_adjust
from evidenceflow.eval.v2 import ranking_metrics, evaluate_rankings, save_artifact


def dataset():
    """Build a minimal Canonical Gold with one question and two groups."""
    return EvaluationDatasetV2.model_validate({
        "dataset_id": "d", "split": "development", "corpus_version": "c",
        "questions": [{"question_id": "q", "family_id": "f", "query": "compare",
                       "query_language": "en", "corpus_language": "en", "question_type": "compare",
                       "evidence_type": "body", "answerability": "answerable",
                       "required_evidence_groups": [{"group_id": gid, "alternatives": [{
                           "document_id": gid, "content_version": "h", "locator": {"page_start": 1, "page_end": 1},
                           "evidence_anchor": "answer"}]} for gid in ("g1", "g2")]}],
    })


def test_partial_complete_and_fragmented_groups():
    """The first relevant fragment and full-group coverage must not be conflated into one rank."""
    result = ranking_metrics(["a", "noise", "b", "c"], [[["a", "b"]], [["c"]]], (1, 3, 4))
    assert result["first_relevant_rr_at_1"] == 1
    assert result["evidence_group_recall_at_1"] == 0
    assert result["evidence_group_recall_at_3"] == .5
    assert result["all_required_evidence_at_4"] == 1
    assert result["custom_complete_evidence_rr_at_4"] == .25
    with pytest.raises(ValueError):
        ranking_metrics(["a", "a"], [[["a"]]])


def test_mapping_failure_is_excluded_not_retrieval_failure():
    """Incomplete mappings are reported separately, so they do not lower the Retriever score."""
    data = dataset()
    mapping = MappingArtifact(dataset_id="d", corpus_version="c", index_version="i", groups=[
        GroupMapping(question_id="q", group_id="g1", status="FULLY_MAPPED", chunk_sets=[["a"]], reason="exact"),
        GroupMapping(question_id="q", group_id="g2", status="UNMAPPED", reason="missing anchor"),
    ])
    report = evaluate_rankings(data, mapping, {"q": ["a"]})
    assert report["aggregate"] == {"question_count": 1, "eligible_count": 0}
    assert report["mapping_failures"] == ["q"]
    assert report["mapping_health"]["gold_mappability_rate"] == .5
    mapping.corpus_version = "wrong"
    with pytest.raises(ValueError):
        evaluate_rankings(data, mapping, {"q": ["a"]})


def test_schema_and_family_leakage():
    """Locator ranges, unanswerability and cross-split question families all have to be explicit."""
    with pytest.raises(ValueError):
        EvidenceLocator(page_start=2, page_end=1)
    data = dataset()
    holdout = data.model_copy(update={"split": "locked_holdout"})
    with pytest.raises(ValueError):
        validate_dataset_splits([data, holdout])
    raw = data.model_dump()
    raw["questions"][0]["answerability"] = "unanswerable"
    with pytest.raises(ValueError):
        EvaluationDatasetV2.model_validate(raw)
    raw["questions"][0]["required_evidence_groups"] = []
    unanswered = EvaluationDatasetV2.model_validate(raw)
    report = evaluate_rankings(unanswered, MappingArtifact(dataset_id="d", corpus_version="c", index_version="i", groups=[]), {"q": []})
    assert report["per_question"][0]["exclusion"] == "unanswerable"


def test_legacy_metrics_match_frozen_baseline():
    """Use the saved per-question results to verify that sharing the implementation did not change the historical averages."""
    root = Path(__file__).resolve().parents[1]
    payload = json.loads((root / "evals/research_retrieval/research_dense_baseline_v1.json").read_text(encoding="utf-8"))
    assert legacy_research_metrics(payload["question_results"]) == payload["breakdowns"]["overall"]
    assert custom_evidence_group_ndcg(["a", "b"], [["a", "b"]], 2) == 1
    assert custom_evidence_group_ndcg(["noise", "a"], [["a"]], 2) == pytest.approx(1 / 1.584962500721156)


def test_paired_comparison_deterministic_clusters_and_zero_baseline():
    """Parallel questions share a question family; statistics are reproducible and never fake a relative change over a zero denominator."""
    args = dict(baseline={"en": 0., "zh": 0.}, variant={"en": 1., "zh": 1.},
                families={"en": "f", "zh": "f"}, samples=100)
    result = paired_comparison(**args)
    assert result == paired_comparison(**args)
    assert result["family_count"] == 1
    assert result["relative_delta"] is None
    assert result["ci95"] == [1., 1.]
    assert holm_adjust({"a": .01, "b": .04}) == {"a": .02, "b": .04}
    with pytest.raises(ValueError):
        paired_comparison({"a": 1.}, {"b": 1.})


def test_artifact_never_overwrites(tmp_path):
    """A new experiment must use a new filename; old results can never be written back."""
    path = tmp_path / "report.json"
    save_artifact({"version": 2}, path)
    with pytest.raises(FileExistsError):
        save_artifact({"version": 3}, path)
    assert json.loads(path.read_text()) == {"version": 2}


def test_quality_requires_explicit_judgments():
    """A valid label does not mean the citation supports the claim, and a service error is not counted as abstention."""
    from evidenceflow.eval.quality import ClaimJudgment, answer_metrics, abstention_metrics, context_metrics
    claim = ClaimJudgment(correct=False, grounded=False, citations=[{"label_valid": True, "supports_claim": False}])
    result = answer_metrics([claim], [False], annotation_source="human-fixture")
    assert result["citation_label_validity"] == 1
    assert result["citation_support_precision"] == 0
    assert result["unsupported_cited_claim_rate"] == 1
    assert answer_metrics([], [], annotation_source="human-fixture")["claim_precision"] is None
    result = abstention_metrics([("unanswerable", "ANSWERED"), ("answerable", "ERROR")])
    assert result["unanswerable_false_answer_rate"] == 1
    assert result["answerable_false_abstention_rate"] is None
    context = context_metrics(["a", "b"], [[["a"], ["b"]]], generation_token_count=20, token_estimator="fake-generation-tokenizer")
    assert context["same_group_redundancy"] == .5
