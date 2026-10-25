"""Verify mapping classification with fixed source text, without depending on a real model or the network."""

import pytest

from evidenceflow.domain.document import Document, DocumentSegment
from evidenceflow.domain.chunk import DocumentChunk
from evidenceflow.eval.mapping import map_dataset
from test_evaluation_v2 import dataset


def source(text="the answer exists"):
    """Build a single-page source with the same version as the canonical fixture."""
    return Document(document_id="g1", source_path="s.pdf", file_name="s.pdf", file_type="pdf",
                    content_hash="h", segments=[DocumentSegment(content=text, page_number=1)])


def chunk(cid, text):
    """Build fixed index text with the same document and the same version."""
    return DocumentChunk(chunk_id=cid, document_id="g1", content_hash="h", file_name="s.pdf",
                         content=text, token_count=3, chunk_index=0, page_start=1, page_end=1)


def mapped(text, chunks, anchor="answer"):
    """Keep only one group, so the classification can be verified independently."""
    data = dataset()
    data.questions[0].required_evidence_groups = data.questions[0].required_evidence_groups[:1]
    data.questions[0].required_evidence_groups[0].alternatives[0].evidence_anchor = anchor
    return map_dataset(data, [source(text)], chunks, index_version="i").groups[0]


def test_complete_and_fragmented_source_alignment():
    """Crossing chunks must still map back to a contiguous source position; two keyword hits do not equal complete evidence."""
    assert mapped("the answer exists", [chunk("a", "the answer exists")]).status == "FULLY_MAPPED"
    result = mapped("alpha beta gamma delta", [chunk("a", "alpha beta"), chunk("b", "gamma delta")], "beta gamma")
    assert result.status == "FRAGMENTED_ACROSS_CHUNKS"
    assert result.chunk_sets == [["a", "b"]]
    assert mapped("alpha beta gamma delta", [chunk("a", "alpha beta")], "beta gamma").status == "PARTIALLY_MAPPED"


def test_normalization_ambiguity_and_missing_source():
    """Loose hyphen breaking only diagnoses, it never becomes Gold automatically."""
    result = mapped("additional detail", [chunk("a", "addi- tional detail")], "additional detail")
    assert result.status == "ANCHOR_NORMALIZATION_FAILURE"
    assert result.chunk_sets == []
    assert mapped("answer answer", [chunk("a", "answer answer")]).status == "AMBIGUOUS_MAPPING"
    assert mapped("other text", [chunk("a", "other text")]).status == "UNMAPPED"
    assert mapped("", []).status == "SOURCE_EXTRACTION_FAILURE"


def test_index_version_mismatch_rejected():
    """A content-version change requires a new audit instead of continuing to use the old Gold mapping."""
    item = chunk("a", "answer")
    item.content_hash = "wrong-version"
    with pytest.raises(ValueError, match="index content version"):
        mapped("answer", [item])


def test_alternatives_are_or_and_source_missing_is_explicit():
    """One usable alternative source is enough to cover the group; when all sources are missing the class is explicit."""
    data = dataset()
    data.questions[0].required_evidence_groups = data.questions[0].required_evidence_groups[:1]
    alternative = data.questions[0].required_evidence_groups[0].alternatives[0]
    data.questions[0].required_evidence_groups[0].alternatives.insert(0, alternative.model_copy(update={"document_id": "absent"}))
    result = map_dataset(data, [source()], [chunk("a", "the answer exists")], index_version="i")
    assert result.groups[0].status == "FULLY_MAPPED"
    assert map_dataset(data, [], [], index_version="i").groups[0].status == "SOURCE_EXTRACTION_FAILURE"


def test_frozen_full_audit_covers_every_group():
    """Check that the formal audit artifact misses no question or group, and failures do not leak into the retrieval denominator."""
    import json
    from pathlib import Path
    from evidenceflow.domain.evaluation_v2 import EvaluationDatasetV2, MappingArtifact
    from evidenceflow.eval.v2 import evaluate_rankings
    root = Path(__file__).resolve().parents[1] / "evals/evaluation_v2/phase3_gold_audit_v1"
    data = EvaluationDatasetV2.model_validate_json((root / "research_dataset_v2.json").read_text(encoding="utf-8"))
    mapping = MappingArtifact.model_validate_json((root / "gold_mapping_v2.json").read_text(encoding="utf-8"))
    replay = json.loads((root / "historical_rankings_replay_v2.json").read_text(encoding="utf-8"))
    assert len(mapping.groups) == sum(len(q.required_evidence_groups) for q in data.questions)
    rankings = {row["question_id"]: row["ranked_chunk_ids"] for row in replay["per_question"]}
    result = evaluate_rankings(data, mapping, rankings, depths=replay["depths"], configuration=replay["configuration"])
    assert result == replay
    assert result["aggregate"]["eligible_count"] + len(result["mapping_failures"]) == len(data.questions)
