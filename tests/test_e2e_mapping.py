"""Verify that rag_e2e_v1 evidence mapping reuses stable_mapping and correctly combines group and claim anchor statuses."""

import pytest

from evidenceflow.domain.chunk import DocumentChunk
from evidenceflow.domain.document import Document, DocumentSegment
from evidenceflow.domain.evaluation_v2 import EvaluationDatasetV2
from evidenceflow.eval.e2e.dataset import E2EDataset, compute_fingerprint
from evidenceflow.eval.e2e.mapping import E2EMapping, compute_chunk_id_hash, is_usable, map_dataset
from evidenceflow.eval.stable_mapping import map_stable_dataset

TEXT = "Intro text. The answer spans two chunks here. Unrelated tail sentence."


def document():
    return Document(document_id="doc", content_hash="hash", source_path="p.pdf", file_name="p.pdf",
                    file_type="pdf", segments=[DocumentSegment(content=TEXT, page_number=1)])


def chunk(cid, text, index):
    return DocumentChunk(chunk_id=cid, document_id="doc", content_hash="hash", file_name="p.pdf",
                         content=text, token_count=3, chunk_index=index, page_start=1, page_end=1)


CHUNKS = [chunk("c1", "Intro text. The answer spans", 0),
          chunk("c2", "two chunks here. Unrelated tail sentence.", 1)]


def alt(text, page=1, version="hash"):
    return {"document_id": "doc", "content_version": version, "evidence_anchor": text,
            "locator": {"page_start": page, "page_end": page}}


def case(groups, claims=(), forbidden=()):
    return {"case_id": "E2E-M001", "family_id": "f", "query": "q?", "query_language": "en",
            "question_type": "t", "tags": ["false_premise"] if forbidden else [], "critical": False,
            "expected_status": "answered", "acceptable_abstention": bool(forbidden),
            "required_evidence_groups": groups, "required_claims": list(claims),
            "forbidden_claims": list(forbidden), "unanswerable_kind": None,
            "provenance": {"drafted_by": "agent", "approved_by": None, "source": "test"}}


def run(cases):
    ds = E2EDataset.model_validate({"dataset_id": "d", "corpus_version": "c", "cases": cases})
    return ds, map_dataset(ds, [document()], CHUNKS, collection_name="col", index_signature="sig")


def test_full_fragmented_unmapped_and_source_failure():
    groups = [{"group_id": "G1", "alternatives": [alt("Intro text.")]},
              {"group_id": "G2", "alternatives": [alt("answer spans two chunks")]},
              {"group_id": "G3", "alternatives": [alt("not in source at all")]},
              {"group_id": "G4", "alternatives": [alt("Intro text.", version="other")]}]
    ds, mapping = run([case(groups)])
    status = {g.group_id: (g.status, g.chunk_sets) for g in mapping.groups}
    assert status["G1"] == ("FULLY_MAPPED", [["c1"]])
    assert status["G2"] == ("FRAGMENTED_ACROSS_CHUNKS", [["c1", "c2"]])
    assert status["G3"][0] == "SOURCE_EXTRACTION_FAILURE"
    assert status["G4"][0] == "SOURCE_EXTRACTION_FAILURE"
    assert mapping.dataset_fingerprint == compute_fingerprint(ds)
    assert (mapping.collection_name, mapping.index_signature) == ("col", "sig")


def test_unmapped_when_chunks_do_not_cover_source():
    """Source location succeeds but no chunk covers it (for example the page's chunk does not exist) => UNMAPPED."""
    ds = E2EDataset.model_validate({"dataset_id": "d", "corpus_version": "c",
                                    "cases": [case([{"group_id": "G1", "alternatives": [alt("Intro text.")]}])]})
    mapping = map_dataset(ds, [document()], [CHUNKS[1]], collection_name="col", index_signature="sig")
    assert mapping.groups[0].status == "UNMAPPED" and mapping.groups[0].chunk_sets == []


def test_claim_anchors_any_alternative_usable():
    claims = [{"claim_id": "C1", "text": "x", "required": True, "answer_keys": [],
               "evidence_anchors": [alt("not in source at all"), alt("Unrelated tail")]},
              {"claim_id": "C2", "text": "y", "required": True, "answer_keys": [], "evidence_anchors": []}]
    forbidden = [{"claim_id": "F1", "text": "z", "evidence_anchors": [alt("answer spans two chunks")]}]
    _, mapping = run([case([{"group_id": "G1", "alternatives": [alt("Intro text.")]}], claims, forbidden)])
    rows = {c.claim_id: c for c in mapping.claim_anchors}
    assert rows["C1"].status == "FULLY_MAPPED" and rows["C1"].chunk_sets == [["c2"]]
    assert rows["C2"].status == "UNMAPPED"
    assert rows["F1"].kind == "forbidden" and rows["F1"].status == "FRAGMENTED_ACROSS_CHUNKS"


def test_groups_match_map_stable_dataset_when_sources_valid():
    """When every anchor's source text is valid, the group-level results match the canonical map_stable_dataset exactly."""
    groups = [{"group_id": "G1", "alternatives": [alt("Intro text."), alt("tail sentence")]},
              {"group_id": "G2", "alternatives": [alt("answer spans two chunks")]}]
    _, mapping = run([case(groups)])
    legacy = EvaluationDatasetV2.model_validate({
        "dataset_id": "d", "split": "development", "corpus_version": "c",
        "questions": [{"question_id": "E2E-M001", "family_id": "f", "query": "q?", "query_language": "en",
                       "corpus_language": "en", "question_type": "t", "evidence_type": "text",
                       "answerability": "answerable", "required_evidence_groups": groups}]})
    artifact, _, _ = map_stable_dataset(legacy, [document()], CHUNKS, index_version="i")
    assert [(g.group_id, g.status, g.chunk_sets) for g in mapping.groups] == \
        [(g.group_id, g.status, g.chunk_sets) for g in artifact.groups]


def test_is_usable_and_record_consistency():
    assert is_usable("FULLY_MAPPED") and is_usable("FRAGMENTED_ACROSS_CHUNKS")
    for status in ("PARTIALLY_MAPPED", "UNMAPPED", "SOURCE_EXTRACTION_FAILURE",
                   "ANCHOR_NORMALIZATION_FAILURE", "AMBIGUOUS_MAPPING"):
        assert not is_usable(status)
    with pytest.raises(ValueError):
        E2EMapping.model_validate({"dataset_fingerprint": "f", "collection_name": "c", "index_signature": "s",
                                   "groups": [{"case_id": "a", "group_id": "G1", "status": "UNMAPPED",
                                               "chunk_sets": [["c1"]]}], "claim_anchors": []})


def test_chunk_id_hash_matches_frozen_a1_formula():
    """The chunk_id_hash standard must match the A1 baseline: sha256 of the canonical JSON of the sorted id list."""
    assert compute_chunk_id_hash(["b", "a", "c"]) == \
        "fa1844c2988ad15ab7b49e0ece09684500fad94df916859fb9a43ff85f5bb477"
    assert compute_chunk_id_hash([]) == \
        "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945"


def test_mapping_records_measured_chunk_identity():
    """The measured chunk_count / chunk_id_hash are written into the artifact; no error is raised when they match the expectations."""
    ds = E2EDataset.model_validate({"dataset_id": "d", "corpus_version": "c",
                                    "cases": [case([{"group_id": "G1", "alternatives": [alt("Intro text.")]}])]})
    mapping = map_dataset(ds, [document()], CHUNKS, collection_name="col", index_signature="sig",
                          expected_chunk_count=2, expected_chunk_id_hash=compute_chunk_id_hash(["c1", "c2"]))
    assert mapping.chunk_count == 2
    assert mapping.chunk_id_hash == compute_chunk_id_hash(["c2", "c1"])


def test_expected_chunk_identity_mismatch_raises():
    ds = E2EDataset.model_validate({"dataset_id": "d", "corpus_version": "c",
                                    "cases": [case([{"group_id": "G1", "alternatives": [alt("Intro text.")]}])]})
    with pytest.raises(ValueError, match="chunk_count mismatch"):
        map_dataset(ds, [document()], CHUNKS, collection_name="col", index_signature="sig",
                    expected_chunk_count=3)
    with pytest.raises(ValueError, match="chunk_id_hash mismatch"):
        map_dataset(ds, [document()], CHUNKS, collection_name="col", index_signature="sig",
                    expected_chunk_id_hash="0" * 64)


def test_old_mapping_without_identity_fields_still_validates():
    """Older pilot artifacts have no chunk_count / chunk_id_hash and must still pass model_validate."""
    old = {"dataset_fingerprint": "f", "collection_name": "c", "index_signature": "s",
           "groups": [], "claim_anchors": []}
    mapping = E2EMapping.model_validate(old)
    assert mapping.chunk_count is None and mapping.chunk_id_hash is None
