"""Verify the already-generated Phase 7 frozen candidates and Gate artifacts."""

import json
from pathlib import Path


PHASE7 = Path(__file__).parents[1] / "evals" / "evaluation_v2" / "phase7_reranker_validation_v1"


def _load(name: str):
    """Read a deterministic experiment artifact, keeping the test independent of external models or the network."""

    return json.loads((PHASE7 / name).read_text(encoding="utf-8"))


def test_phase7_required_artifacts_and_gate_are_present() -> None:
    """The 11 JSON/Markdown artifacts required by the plan must all be present, and the Gate must be PASS."""

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
    """All three architectures preserve 100 unique candidates per question together with their source branches."""

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
    """Budget and architecture changes may affect ranking only, never candidate coverage."""

    results = _load("reranker_results.json")
    for architecture in ("c0_original_dense", "c1_translation_dense", "c2_hybrid"):
        for budget in ("20", "30", "50", "100"):
            assert results[architecture][budget]["coverage_unchanged"] is True
