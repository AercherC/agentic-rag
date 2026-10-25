"""Verify §13.1(e) failure-layer attribution and diagnosis per evidence group.

No model is loaded and no network access is needed: every input is constructed locally, so the
attribution layers stay deterministic and cheap to test.
"""

from __future__ import annotations

import json

import pytest

from evidenceflow.eval.e2e.dataset import E2EDataset, compute_fingerprint
from evidenceflow.eval.e2e.diagnosis import (
    E2EObservation,
    build_report,
    diagnose_case,
    diagnose_dataset,
    is_back_matter,
    render_markdown,
)
from evidenceflow.eval.e2e.mapping import E2EClaimMapping, E2EGroupMapping, E2EMapping


def _alt(document_id: str, anchor: str = "anchor") -> dict:
    return {"document_id": document_id, "content_version": "v", "evidence_anchor": anchor,
            "locator": {"page_start": 1, "page_end": 1}}


def _case(case_id: str, groups: dict[str, str], *, tags=(), expected="answered", acceptable=False,
          keys=(("PoinTr",),), unanswerable=False) -> dict:
    claims = [] if unanswerable else [
        {"claim_id": "C1", "text": "a grounded claim text", "required": True,
         "evidence_anchors": [_alt(next(iter(groups.values())))], "answer_keys": [list(k) for k in keys]}]
    return {
        "case_id": case_id, "family_id": case_id, "query": f"question {case_id}?", "query_language": "en",
        "question_type": "t", "tags": list(tags), "critical": False,
        "expected_status": "insufficient_context" if unanswerable else expected,
        "acceptable_abstention": acceptable,
        "required_evidence_groups": [] if unanswerable else [
            {"group_id": gid, "alternatives": [_alt(doc)]} for gid, doc in groups.items()],
        "required_claims": claims, "forbidden_claims": [],
        "unanswerable_kind": "out_of_corpus" if unanswerable else None,
        "provenance": {"drafted_by": "agent", "approved_by": None, "source": "test"},
    }


def _dataset(*cases: dict) -> E2EDataset:
    return E2EDataset.model_validate({"dataset_id": "d", "corpus_version": "c", "cases": list(cases)})


def _mapping(dataset: E2EDataset, groups: dict[tuple[str, str], tuple[str, list[list[str]]]],
             claims: dict[str, list[list[str]]] | None = None, chunk_count: int = 50) -> E2EMapping:
    return E2EMapping(
        dataset_fingerprint=compute_fingerprint(dataset), collection_name="col", index_signature="sig",
        chunk_count=chunk_count,
        groups=[E2EGroupMapping(case_id=c, group_id=g, status=status, chunk_sets=sets)
                for (c, g), (status, sets) in groups.items()],
        claim_anchors=[E2EClaimMapping(case_id=c, claim_id="C1", kind="required",
                                       status="FULLY_MAPPED" if sets else "UNMAPPED", chunk_sets=sets)
                       for c, sets in (claims or {}).items()],
    )


# Default corpus: docA provides a1..a30 and docB provides b1..b30, interleaved by rank; the
# document behind each chunk id is chosen by the id prefix (a* -> docA, b* -> docB).
def _dense(order: list[str], heading: dict[str, str] | None = None) -> list[dict]:
    heading = heading or {}
    return [{"chunk_id": cid, "document_id": "docA" if cid.startswith("a") else "docB" if cid.startswith("b")
             else "docC", "heading_path": heading.get(cid)} for cid in order]


def _obs(case_id: str, order: list[str], *, pool: int = 4, reranked: list[str] | None = None, final: int = 2,
         context: list[str] | None = None, generation: bool = True, status: str | None = "answered",
         answer: str | None = "PoinTr uses queries [S1]", sources: list[str] | None = None,
         error: str | None = None, heading: dict[str, str] | None = None, collection: str = "col",
         chunk_count: int = 50) -> E2EObservation:
    reranked = reranked if reranked is not None else order[:pool]
    context = context if context is not None else reranked[:final]
    payload = {"case_id": case_id, "collection_name": collection, "chunk_count": chunk_count,
               "dense_ranking": _dense(order, heading), "candidate_pool_k": pool, "reranked_pool": reranked,
               "final_top_k": final, "context_chunk_ids": context, "generation_ran": generation}
    if generation:
        payload.update(status=None if error else status, answer=None if error else answer,
                       source_chunk_ids=[] if error else (sources if sources is not None else context[:1]),
                       error_type=error)
    return E2EObservation.model_validate(payload)


ORDER = ["a1", "a2", "c1", "c2", "a3", "b9", "c3", "b1"]
MULTI = _case("M1", {"G1": "docA", "G2": "docB"}, tags=["multi_doc"])


def _diagnose(case: dict, groups, observation: E2EObservation, claims=None):
    dataset = _dataset(case)
    mapping = _mapping(dataset, groups, claims)
    return diagnose_case(dataset.cases[0], mapping, observation)


def test_candidate_miss_separates_document_absent_from_wrong_passage() -> None:
    groups = {("M1", "G1"): ("FULLY_MAPPED", [["a1"]]), ("M1", "G2"): ("FULLY_MAPPED", [["b1"]])}
    diagnosis = _diagnose(MULTI, groups, _obs("M1", ORDER))

    g1, g2 = diagnosis.groups
    assert diagnosis.layer == "CANDIDATE_MISS"
    assert g1.stage in {"CITED", "IN_CONTEXT_NOT_CITED"}
    # b1 ranks 8th while the candidate pool holds only 4 chunks, so docB never enters the pool:
    # a document-level recall failure.
    assert (g2.stage, g2.best_dense_rank, g2.document_in_candidates) == ("CANDIDATE_MISS", 8, False)

    order = ["a1", "b9", "c1", "c2", "b1"]  # docB is in the pool, but evidence passage b1 is not.
    diagnosis = _diagnose(MULTI, groups, _obs("M1", order))
    assert diagnosis.groups[1].document_in_candidates is True


def test_fragmented_evidence_partly_retrieved_is_flagged() -> None:
    groups = {("M1", "G1"): ("FULLY_MAPPED", [["a1"]]),
              ("M1", "G2"): ("FRAGMENTED_ACROSS_CHUNKS", [["b1", "b2"]])}
    order = ["a1", "b1", "c1", "c2", "c3", "b2"]
    group = _diagnose(MULTI, groups, _obs("M1", order)).groups[1]

    assert (group.stage, group.fragmented, group.partial_in_candidates) == ("CANDIDATE_MISS", True, True)
    assert group.best_dense_rank == 6  # the group rank is taken from the last chunk in the set


def test_evidence_beyond_observed_depth_has_no_rank() -> None:
    groups = {("M1", "G1"): ("FULLY_MAPPED", [["a1"]]), ("M1", "G2"): ("FULLY_MAPPED", [["b77"]])}
    group = _diagnose(MULTI, groups, _obs("M1", ORDER)).groups[1]

    assert group.stage == "CANDIDATE_MISS" and group.best_dense_rank is None


def test_rerank_miss_records_reranker_position() -> None:
    groups = {("M1", "G1"): ("FULLY_MAPPED", [["a1"]]), ("M1", "G2"): ("FULLY_MAPPED", [["b9"]])}
    order = ["a1", "b9", "c1", "c2"]
    observation = _obs("M1", order, reranked=["a1", "c1", "c2", "b9"])
    diagnosis = _diagnose(MULTI, groups, observation)

    assert diagnosis.layer == "RERANK_MISS"
    assert (diagnosis.groups[1].stage, diagnosis.groups[1].best_rerank_position) == ("RERANK_MISS", 4)
    assert diagnosis.groups[1].document_in_final is False


def test_context_drop_when_budget_excludes_evidence() -> None:
    groups = {("M1", "G1"): ("FULLY_MAPPED", [["a1"]]), ("M1", "G2"): ("FULLY_MAPPED", [["b9"]])}
    observation = _obs("M1", ["a1", "b9", "c1", "c2"], context=["a1"])
    diagnosis = _diagnose(MULTI, groups, observation)

    assert diagnosis.layer == "CONTEXT" and diagnosis.groups[1].stage == "CONTEXT_DROP"


def test_unmappable_group_makes_case_unattributable() -> None:
    groups = {("M1", "G1"): ("FULLY_MAPPED", [["a1"]]), ("M1", "G2"): ("PARTIALLY_MAPPED", [])}
    diagnosis = _diagnose(MULTI, groups, _obs("M1", ORDER))

    assert diagnosis.layer == "UNATTRIBUTABLE"
    assert (diagnosis.groups[1].stage, diagnosis.groups[1].mapping_status) == ("UNMAPPABLE", "PARTIALLY_MAPPED")


SINGLE = _case("S1", {"G1": "docA"})
SINGLE_GROUPS = {("S1", "G1"): ("FULLY_MAPPED", [["a1"]])}


@pytest.mark.parametrize(
    ("kwargs", "layer"),
    [
        ({}, "PASS"),
        ({"answer": "something else [S1]"}, "GENERATION"),
        ({"status": "insufficient_context", "answer": "not enough", "sources": []}, "STATUS"),
        ({"error": "GenerationContractError"}, "GENERATION"),
        ({"error": "CitationValidationError"}, "CITATION"),
        ({"error": "LLMRequestError"}, "ERROR"),
        ({"error": "RuntimeError"}, "ERROR"),
        ({"generation": False}, "RETRIEVAL_PASS"),
    ],
)
def test_generation_layers_follow_spec_order(kwargs, layer) -> None:
    observation = _obs("S1", ["a1", "c1", "c2", "c3"], **kwargs)
    diagnosis = _diagnose(SINGLE, SINGLE_GROUPS, observation, claims={"S1": [["a1"]]})

    assert diagnosis.layer == layer


def test_acceptable_abstention_is_pass() -> None:
    case = _case("S1", {"G1": "docA"}, acceptable=True)
    observation = _obs("S1", ["a1", "c1", "c2", "c3"], status="insufficient_context", answer="no", sources=[])

    assert _diagnose(case, SINGLE_GROUPS, observation).layer == "PASS"


def test_citation_layer_when_correct_answer_cites_no_evidence() -> None:
    observation = _obs("S1", ["a1", "c1", "c2", "c3"], sources=["c1"], context=["a1", "c1"])
    diagnosis = _diagnose(SINGLE, SINGLE_GROUPS, observation, claims={"S1": [["a1"]]})

    assert diagnosis.layer == "CITATION" and diagnosis.groups[0].stage == "IN_CONTEXT_NOT_CITED"


def test_unverified_without_key_matchable_claims() -> None:
    case = _case("S1", {"G1": "docA"}, keys=())
    assert _diagnose(case, SINGLE_GROUPS, _obs("S1", ["a1", "c1", "c2", "c3"])).layer == "UNVERIFIED"


def test_unanswerable_cases() -> None:
    case = _case("U1", {}, unanswerable=True)
    answered = _obs("U1", ["c1", "c2", "c3", "c4"], sources=[])
    abstained = _obs("U1", ["c1", "c2", "c3", "c4"], status="insufficient_context", answer="no", sources=[])

    assert _diagnose(case, {}, answered).layer == "STATUS"
    assert _diagnose(case, {}, abstained).layer == "PASS"
    assert _diagnose(case, {}, _obs("U1", ["c1", "c2", "c3", "c4"], generation=False)).layer == "NOT_RUN"


def test_final_top_k_composition_and_back_matter() -> None:
    groups = {("M1", "G1"): ("FULLY_MAPPED", [["a1"]]), ("M1", "G2"): ("FULLY_MAPPED", [["b1"]])}
    observation = _obs("M1", ["a1", "a2", "a3", "b1"], final=3,
                       heading={"a2": "References", "a3": "6 Conclusion"})
    diagnosis = _diagnose(MULTI, groups, observation)

    assert diagnosis.final_distinct_documents == 1
    assert diagnosis.final_dominant_share == 1.0
    assert diagnosis.final_back_matter_chunks == 1
    assert is_back_matter("Appendix > A.1") and is_back_matter("Supplementary Material")
    assert not is_back_matter("3 Method") and not is_back_matter(None)


def test_identity_mismatches_are_refused() -> None:
    dataset = _dataset(SINGLE)
    mapping = _mapping(dataset, SINGLE_GROUPS)
    fingerprint = compute_fingerprint(dataset)

    with pytest.raises(ValueError, match="collection"):
        diagnose_dataset(dataset, mapping, [_obs("S1", ORDER, collection="other")], fingerprint)
    with pytest.raises(ValueError, match="chunks"):
        diagnose_dataset(dataset, mapping, [_obs("S1", ORDER, chunk_count=7)], fingerprint)
    with pytest.raises(ValueError, match="fingerprint"):
        diagnose_dataset(dataset, mapping, [_obs("S1", ORDER)], "different")
    with pytest.raises(ValueError, match="unknown cases"):
        diagnose_dataset(dataset, mapping, [_obs("X9", ORDER)], fingerprint)


def test_observation_rejects_inconsistent_stages() -> None:
    with pytest.raises(ValueError, match="final top-k"):
        _obs("S1", ["a1", "c1", "c2", "c3"], context=["c3"])
    with pytest.raises(ValueError, match="candidate pool"):
        _obs("S1", ["a1", "c1", "c2", "c3", "c4"], reranked=["a1", "c4"])
    with pytest.raises(ValueError, match="retrieval-only"):
        E2EObservation.model_validate({**_obs("S1", ORDER, generation=False).model_dump(), "status": "answered"})


def test_report_slices_and_markdown() -> None:
    dataset = _dataset(MULTI, SINGLE, _case("U1", {}, unanswerable=True))
    mapping = _mapping(dataset, {("M1", "G1"): ("FULLY_MAPPED", [["a1"]]),
                                 ("M1", "G2"): ("FULLY_MAPPED", [["b1"]]), **SINGLE_GROUPS})
    observations = [_obs("M1", ORDER), _obs("S1", ORDER), _obs("U1", ORDER, sources=[])]
    diagnoses = diagnose_dataset(dataset, mapping, observations, compute_fingerprint(dataset))

    report = build_report(diagnoses, candidate_pool_k=4, final_top_k=2, generation_ran=True)
    multi = report["slices"]["multi_doc"]
    single = report["slices"]["single_doc_answerable"]
    assert (multi["cases"], single["cases"], report["slices"]["all"]["cases"]) == (1, 1, 3)
    assert multi["question_layers"] == {"CANDIDATE_MISS": 1}
    assert multi["candidate_miss"]["document_absent_from_candidates"] == 1
    assert multi["candidate_miss"]["dense_rank_buckets"] == {"rank_5_to_50": 1}
    assert single["question_layers"] == {"PASS": 1}
    markdown = render_markdown(report)
    assert "| CANDIDATE_MISS | 1 (100%) | 0 (0%) |" in markdown
    assert "required document absent from top-20 | 1 | 0" in markdown
    json.dumps(report)  # the report must be directly serializable to JSON
