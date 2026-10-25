"""Verify the rag_e2e_v1 schema, fingerprint and deterministic validation (SPEC §13.1(a)(c), including D8)."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from evidenceflow.eval.e2e.dataset import (
    E2EDataset,
    Provenance,
    compute_fingerprint,
    load_dataset,
    validate_dataset,
    with_fingerprint,
)

PAGES = {("doc-a", 1, 1): "SnowflakeNet achieves a Chamfer distance of 7.21 on PCN.",
         ("doc-b", 2, 2): "SeedFormer uses patch seeds with 512 points."}


def provider(document_id, page_start, page_end):
    """Small fixture: return the source text for one (document, page)."""
    return PAGES[(document_id, page_start, page_end)]


def anchor(doc="doc-a", page=1, text="Chamfer distance of 7.21"):
    return {"document_id": doc, "content_version": f"hash-{doc}", "evidence_anchor": text,
            "locator": {"page_start": page, "page_end": page}}


def answerable(case_id="E2E-M001", **overrides):
    case = {
        "case_id": case_id, "family_id": case_id, "query": "What CD does SnowflakeNet report?",
        "query_language": "en", "question_type": "numeric", "tags": [], "critical": False,
        "expected_status": "answered", "acceptable_abstention": False,
        "required_evidence_groups": [{"group_id": "G1", "alternatives": [anchor()]}],
        "required_claims": [{"claim_id": "C1", "text": "CD is 7.21", "required": True,
                             "evidence_anchors": [anchor()], "answer_keys": [["7.21"]]}],
        "forbidden_claims": [], "unanswerable_kind": None,
        "provenance": {"drafted_by": "agent", "approved_by": "user", "source": "new"},
    }
    case.update(overrides)
    return case


def unanswerable(case_id="E2E-U001", kind="near_miss", **overrides):
    case = answerable(case_id, expected_status="insufficient_context", required_evidence_groups=[],
                      required_claims=[], unanswerable_kind=kind, query="How many GPUs were used?")
    case.update(overrides)
    return case


def false_premise(case_id="E2E-M900", **overrides):
    case = answerable(case_id, tags=["false_premise"], acceptable_abstention=True,
                      query="Why does SnowflakeNet report a CD of 9.9 on PCN?",
                      forbidden_claims=[{"claim_id": "F1", "text": "CD is 9.9", "evidence_anchors": [anchor()]}])
    case.update(overrides)
    return case


def dataset(*cases):
    return E2EDataset.model_validate({"dataset_id": "t", "corpus_version": "c", "cases": list(cases)})


def rules(report, case_id=None):
    return sorted({i.rule for i in report.issues if case_id is None or i.case_id == case_id})


def check(*cases, require_approved=False):
    return validate_dataset(dataset(*cases), provider, require_approved=require_approved)


# ---------- schema and fingerprint ----------

def test_schema_forbids_extra_fields_and_old_false_premise_kind():
    with pytest.raises(ValidationError):
        dataset(answerable(extra_field=1))
    with pytest.raises(ValidationError):
        dataset(unanswerable(kind="false_premise"))


def test_fingerprint_is_stable_and_content_bound():
    ds = dataset(answerable(), unanswerable())
    assert compute_fingerprint(ds) == compute_fingerprint(dataset(answerable(), unanswerable()))
    filled = with_fingerprint(ds)
    assert ds.fingerprint is None and filled.fingerprint == compute_fingerprint(ds)
    # The fingerprint field itself does not take part in the computation.
    assert compute_fingerprint(filled.model_copy(update={"fingerprint": "x"})) == filled.fingerprint
    changed = dataset(answerable(query="Another question?"), unanswerable())
    assert compute_fingerprint(changed) != filled.fingerprint


def test_load_dataset_verifies_fingerprint(tmp_path):
    good = with_fingerprint(dataset(answerable()))
    path = tmp_path / "d.json"
    path.write_text(good.model_dump_json(), encoding="utf-8")
    assert load_dataset(path).fingerprint == good.fingerprint
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["cases"][0]["query"] = "tampered?"
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="mismatch"):
        load_dataset(path)
    assert load_dataset(path, verify_fingerprint=False).cases[0].query == "tampered?"
    path.write_text(dataset(answerable()).model_dump_json(), encoding="utf-8")
    with pytest.raises(ValueError, match="no fingerprint"):
        load_dataset(path)


def test_validate_reports_fingerprint_mismatch():
    ds = dataset(answerable()).model_copy(update={"fingerprint": "0" * 64})
    report = validate_dataset(ds, provider, require_approved=False)
    assert "fingerprint_mismatch" in rules(report)


# ---------- each rule: positive case + counterexample ----------

def test_valid_dataset_has_no_issues_and_ratios():
    report = check(answerable(), unanswerable(), unanswerable("E2E-U002", kind="out_of_corpus"), false_premise(),
                   require_approved=True)
    assert report.ok and report.issue_count == 0
    assert report.near_miss_ratio.numerator == 1 and report.near_miss_ratio.denominator == 2
    # False-premise cases count as answerable (D8), and their required claims enter the
    # denominator.
    assert report.key_matchable_required_claim_ratio.denominator == 2
    assert report.key_matchable_required_claim_ratio.value == 1.0


def test_duplicate_ids():
    report = check(answerable(), answerable())
    assert "duplicate_case_id" in rules(report)
    dup_claim = answerable(forbidden_claims=[], required_claims=[
        {"claim_id": "C1", "text": "a", "required": True, "evidence_anchors": [anchor()], "answer_keys": [["7.21"]]},
        {"claim_id": "C1", "text": "b", "required": False, "evidence_anchors": [anchor()], "answer_keys": []}])
    assert "duplicate_claim_id" in rules(check(dup_claim))
    dup_group = answerable(required_evidence_groups=[{"group_id": "G1", "alternatives": [anchor()]}] * 2)
    assert "duplicate_group_id" in rules(check(dup_group))


def test_answerable_needs_required_claim_with_anchors():
    assert rules(check(answerable())) == []
    optional_only = answerable(required_claims=[
        {"claim_id": "C1", "text": "a", "required": False, "evidence_anchors": [anchor()], "answer_keys": []}])
    assert "answerable_required_claim" in rules(check(optional_only))
    no_anchor = answerable(required_claims=[
        {"claim_id": "C1", "text": "a", "required": True, "evidence_anchors": [], "answer_keys": []}])
    assert "required_claim_anchors" in rules(check(no_anchor))


def test_multi_doc_rule():
    good = answerable(tags=["multi_doc"], required_evidence_groups=[
        {"group_id": "G1", "alternatives": [anchor()]},
        {"group_id": "G2", "alternatives": [anchor("doc-b", 2, "patch seeds")]}])
    assert rules(check(good)) == []
    same_doc = answerable(tags=["multi_doc"], required_evidence_groups=[
        {"group_id": "G1", "alternatives": [anchor()]}, {"group_id": "G2", "alternatives": [anchor()]}])
    assert "multi_doc" in rules(check(same_doc))
    assert "multi_doc" in rules(check(answerable(tags=["multi_doc"])))


def test_unanswerable_shape():
    assert rules(check(unanswerable())) == []
    bad = unanswerable(acceptable_abstention=True, unanswerable_kind=None,
                       required_evidence_groups=[{"group_id": "G1", "alternatives": [anchor()]}])
    report = check(bad)
    messages = [i.message for i in report.issues if i.rule == "unanswerable_shape"]
    assert len(messages) == 3
    with_claims = unanswerable(required_claims=answerable()["required_claims"])
    assert "unanswerable_shape" in rules(check(with_claims))


def test_answerable_kind_must_be_null():
    assert "answerable_unanswerable_kind" in rules(check(answerable(unanswerable_kind="near_miss")))


def test_false_premise_rules():
    assert rules(check(false_premise())) == []
    for overrides in ({"acceptable_abstention": False}, {"forbidden_claims": []},
                      {"forbidden_claims": [{"claim_id": "F1", "text": "x", "evidence_anchors": []}]},
                      {"required_claims": [{"claim_id": "C1", "text": "a", "required": True,
                                            "evidence_anchors": [anchor()], "answer_keys": []}]}):
        assert "false_premise" in rules(check(false_premise(**overrides))), overrides
    as_unanswerable = unanswerable(tags=["false_premise"])
    assert "false_premise" in rules(check(as_unanswerable))


def test_forbidden_claims_only_for_false_premise():
    bad = answerable(forbidden_claims=[{"claim_id": "F1", "text": "x", "evidence_anchors": [anchor()]}])
    assert "forbidden_claims_non_false_premise" in rules(check(bad))


def _with_keys(keys, query="What CD does SnowflakeNet report?", anchors=None):
    return answerable(query=query, required_claims=[
        {"claim_id": "C1", "text": "a", "required": True,
         "evidence_anchors": anchors if anchors is not None else [anchor()], "answer_keys": keys}])


def test_answerable_needs_evidence_group():
    """An answerable case needs at least 1 required evidence group; a case without one must be stopped before freezing."""
    assert "answerable_evidence_group" not in rules(check(answerable()))
    assert "answerable_evidence_group" in rules(check(answerable(required_evidence_groups=[])))
    # A false-premise case is also answerable per D8, so it likewise requires at least 1 evidence
    # group.
    assert "answerable_evidence_group" in rules(check(false_premise(required_evidence_groups=[])))


def test_old_pilot_dataset_still_validates():
    """The old pilot dataset (40 groups, no claims) still loads and validates, and a new rule does not make it raise."""
    pilot = Path(__file__).resolve().parents[1] / "artifacts/stage_b/B3a1_pilot_dataset.json"
    ds = E2EDataset.model_validate_json(pilot.read_text(encoding="utf-8"))
    report = validate_dataset(ds, provider, require_approved=False)
    # Only the existing "answerable case without a required claim" issue; the new rules must not
    # cause false hits.
    assert {issue.rule for issue in report.issues} == {"answerable_required_claim"}
    assert report.issue_count == 40


def test_answer_keys_rule_source_text():
    assert rules(check(_with_keys([["7.21"], ["倒角距离", "chamfer distance"]]))) == []
    assert "answer_keys_source" in rules(check(_with_keys([["8.88"]])))
    # Record an issue when the anchor page yields no text instead of raising.
    missing = _with_keys([["7.21"]], anchors=[anchor("doc-x")])
    assert "answer_keys_source" in rules(check(missing))


def test_answer_keys_rule_source_text_is_literal_substring():
    """Rule ① only requires a literal substring after normalization: `1.5mm` in the source allows the alias `1.5`, with no ASCII boundary applied."""

    def provider_mm(document_id, page_start, page_end):
        return "The reported error is 1.5mm on average."

    case = answerable(query="What is the reported error?", required_claims=[
        {"claim_id": "C1", "text": "error is 1.5", "required": True,
         "evidence_anchors": [anchor()], "answer_keys": [["1.5"]]}])
    report = validate_dataset(dataset(case), provider_mm, require_approved=False)
    assert "answer_keys_source" not in rules(report)


def test_answer_keys_rule_source_text_uses_same_hyphen_folding():
    """Rule ① shares normalize_text with (d): page text `view-\npoints` proves the alias `viewpoints`."""

    def provider_hyphen(document_id, page_start, page_end):
        return "the view-\npoints of the paper"

    case = answerable(query="What does the paper discuss?", required_claims=[
        {"claim_id": "C1", "text": "viewpoints", "required": True,
         "evidence_anchors": [anchor()], "answer_keys": [["viewpoints"]]}])
    report = validate_dataset(dataset(case), provider_hyphen, require_approved=False)
    assert "answer_keys_source" not in rules(report)


def test_answer_keys_rule_query_still_uses_ascii_boundary():
    """Rule ② still matches per (d) (with ASCII boundaries): 1.5 does not hit 11.5, but does hit a standalone 1.5."""
    assert "answer_keys_query" not in rules(check(_with_keys([["1.5"]], query="Is the error 11.5?")))
    assert "answer_keys_query" in rules(check(_with_keys([["1.5"]], query="Is the error 1.5?")))


def test_answer_keys_rule_query():
    assert "answer_keys_query" in rules(check(_with_keys([["SnowflakeNet"]])))
    assert "answer_keys_query" not in rules(check(_with_keys([["7.21"]])))


def test_answer_keys_rule_format():
    assert "answer_keys_format" in rules(check(_with_keys([["7"]])))
    assert "answer_keys_format" in rules(check(_with_keys([["7.21", "７.２１"]])))
    assert "answer_keys_format" in rules(check(_with_keys([[]])))
    assert "answer_keys_format" in rules(check(_with_keys([["  ", "7.21"]])))


def test_require_approved():
    pending = answerable(provenance={"drafted_by": "agent", "approved_by": None, "source": "new"})
    assert "not_approved" not in rules(check(pending))
    assert "not_approved" in rules(check(pending, require_approved=True))
    assert rules(check(answerable(), require_approved=True)) == []


def test_issues_grouped_by_case():
    report = check(answerable(), answerable("E2E-M002", unanswerable_kind="near_miss"))
    assert list(report.issues_by_case) == ["E2E-M002"]
    assert not report.ok


# ---------- §13.1(a): machine confirmation and approval_basis ----------

def test_provenance_approval_basis_contract():
    """machine must carry <path>#<sha256>; user / unconfirmed must not carry a basis."""
    with pytest.raises(ValidationError):
        Provenance(drafted_by="agent", approved_by="machine", source="new")
    with pytest.raises(ValidationError):
        Provenance(drafted_by="agent", approved_by="machine", approval_basis="no-hash", source="new")
    with pytest.raises(ValidationError):
        Provenance(drafted_by="agent", approved_by="user", approval_basis="a.json#abc", source="new")
    with pytest.raises(ValidationError):
        Provenance(drafted_by="agent", approval_basis="a.json#abc", source="new")
    # Legal combinations.
    assert Provenance(drafted_by="agent", approved_by="user", source="new").approval_basis is None
    machine = Provenance(
        drafted_by="agent", approved_by="machine",
        approval_basis="artifacts/stage_b/B3_gate_report.json#abc123", source="derived_t1",
    )
    assert machine.approval_basis.endswith("#abc123")

    # Backward compatibility: with approval_basis None, the fingerprint equals that of a dataset
    # from the old
    # model that had no such field.
    base_case = answerable()
    legacy = dataset(base_case)
    with_null = dataset({**base_case,
                         "provenance": {"drafted_by": "agent", "approved_by": "user",
                                        "approval_basis": None, "source": "new"}})
    assert compute_fingerprint(legacy) == compute_fingerprint(with_null)

    # With a machine acceptance basis, the basis content takes part in the fingerprint (a different
    # report or
    # a different sha256 means a different identity).
    def machine_case(basis):
        return answerable(provenance={"drafted_by": "agent", "approved_by": "machine",
                                      "approval_basis": basis, "source": "derived_t1"})

    assert compute_fingerprint(dataset(machine_case("a.json#1"))) != compute_fingerprint(
        dataset(machine_case("a.json#2"))
    )


def test_validate_dataset_allow_machine():
    machine = {"drafted_by": "agent", "approved_by": "machine",
               "approval_basis": "artifacts/stage_b/B3_gate_report.json#abc", "source": "derived_t1"}
    case = answerable(provenance=machine)
    # With allow_machine=True a machine acceptance passes; the default convention still accepts
    # only user.
    assert "not_approved" not in rules(validate_dataset(dataset(case), provider,
                                                        require_approved=True, allow_machine=True))
    assert "not_approved" in rules(check(case, require_approved=True))

    # machine but an empty basis: bypass constructor validation via model_copy to simulate tampered
    # persisted data.
    good = dataset(answerable()).cases[0]
    broken = good.model_copy(update={
        "provenance": good.provenance.model_copy(update={"approved_by": "machine", "approval_basis": None})
    })
    tampered = E2EDataset(dataset_id="t", corpus_version="c", cases=[broken])
    report = validate_dataset(tampered, provider, require_approved=True, allow_machine=True)
    assert "not_approved" in rules(report)


# ---------- §13.1(c′): grounding rules ----------

GROUNDING_TEXT = (
    "SnowflakeNet reports a chamfer distance of 7.21 on the PCN benchmark under the standard protocol."
)


def grounding_provider(document_id, page_start, page_end):
    return GROUNDING_TEXT


def _grounding_case(claim_text=GROUNDING_TEXT, keys=None,
                    query="What chamfer distance does SnowflakeNet report on PCN?"):
    return answerable(query=query, required_claims=[
        {"claim_id": "C1", "text": claim_text, "required": True,
         "evidence_anchors": [anchor()], "answer_keys": keys or []}])


def _grounding_rules(case, provider=grounding_provider, **kwargs):
    return rules(validate_dataset(dataset(case), provider, require_approved=False,
                                  enforce_grounding=True, **kwargs))


def test_grounding_claim_must_be_in_source_text():
    assert _grounding_rules(_grounding_case()) == []
    not_grounded = _grounding_case(
        claim_text="the paper describes an unrelated baseline method for comparison")
    assert "claim_grounded" in _grounding_rules(not_grounded)


def test_grounding_key_must_be_substring_of_claim():
    claim = "the measured chamfer distance is 7.21 on pcn"
    assert "claim_key_in_text" in _grounding_rules(_grounding_case(claim_text=claim, keys=[["8.88"]]))
    # Passes when the alias is a substring of the claim (7.21 both grounds and matches).
    assert "claim_key_in_text" not in _grounding_rules(_grounding_case(claim_text=claim, keys=[["7.21"]]))


def test_grounding_claim_too_short():
    def short_provider(document_id, page_start, page_end):
        return "the CD 7.21 value is reported"

    assert "claim_too_short" in _grounding_rules(
        _grounding_case(claim_text="CD 7.21"), provider=short_provider
    )


def test_grounding_query_copy_rule():
    claim = "snflk reports the chamfer distance value of 7.21 on the pcn benchmark under the protocol"
    # 9 consecutive shared words (the chamfer distance value of 7.21 on the pcn) > 8 => triggered.
    copied = _grounding_case(
        claim_text=claim,
        query="what does snflk report about the chamfer distance value of 7.21 on the pcn",
    )
    assert "query_claim_copy" in _grounding_rules(copied)

    # Only 5 consecutive shared words (the chamfer distance value of) is below the threshold and
    # does not
    # trigger, so numeric questions are not killed by mistake.
    clean = _grounding_case(
        claim_text=claim,
        query="what does snflk report about the chamfer distance value of",
    )
    assert "query_claim_copy" not in _grounding_rules(clean)


def test_grounding_forbidden_claims_only_need_length_and_copy():
    forbidden = {"claim_id": "F1", "text": "the chamfer distance is 9.9 on pcn",
                 "evidence_anchors": [anchor()]}
    required = [{"claim_id": "C1", "text": GROUNDING_TEXT, "required": True,
                 "evidence_anchors": [anchor()], "answer_keys": [["7.21"]]}]
    case = false_premise(required_claims=required, forbidden_claims=[forbidden])
    report_rules = _grounding_rules(case)
    # F1 is the false premise and is by design not in the source: ① is not required and all other
    # rules pass.
    assert "claim_grounded" not in report_rules and "claim_too_short" not in report_rules
    assert report_rules == []

    short_forbidden = dict(forbidden, text="9.9 pcn")
    short_case = false_premise(required_claims=required, forbidden_claims=[short_forbidden])
    assert "claim_too_short" in _grounding_rules(short_case)


def test_grounding_is_opt_in():
    """With enforce_grounding=False the new rules never appear, keeping the existing semantics."""
    case = _grounding_case(claim_text="the paper describes an unrelated baseline method")
    assert _grounding_rules(case) == ["claim_grounded"]
    report = validate_dataset(dataset(case), grounding_provider, require_approved=False)
    assert not {"claim_grounded", "claim_key_in_text", "claim_too_short", "query_claim_copy"} & set(
        rules(report)
    )
