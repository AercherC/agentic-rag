"""用固定原文验证映射分类，不依赖真实模型或网络。"""

import pytest

from evidenceflow.domain.document import Document, DocumentSegment
from evidenceflow.domain.chunk import DocumentChunk
from evidenceflow.eval.mapping import map_dataset
from test_evaluation_v2 import dataset


def source(text="the answer exists"):
    """构造与 canonical fixture 相同版本的单页来源。"""
    return Document(document_id="g1", source_path="s.pdf", file_name="s.pdf", file_type="pdf",
                    content_hash="h", segments=[DocumentSegment(content=text, page_number=1)])


def chunk(cid, text):
    """构造同文档同版本的固定索引文本。"""
    return DocumentChunk(chunk_id=cid, document_id="g1", content_hash="h", file_name="s.pdf",
                         content=text, token_count=3, chunk_index=0, page_start=1, page_end=1)


def mapped(text, chunks, anchor="answer"):
    """仅保留一组，方便独立验证分类。"""
    data = dataset()
    data.questions[0].required_evidence_groups = data.questions[0].required_evidence_groups[:1]
    data.questions[0].required_evidence_groups[0].alternatives[0].evidence_anchor = anchor
    return map_dataset(data, [source(text)], chunks, index_version="i").groups[0]


def test_complete_and_fragmented_source_alignment():
    """跨块必须能回到连续原文位置，两个关键词命中不等于完整证据。"""
    assert mapped("the answer exists", [chunk("a", "the answer exists")]).status == "FULLY_MAPPED"
    result = mapped("alpha beta gamma delta", [chunk("a", "alpha beta"), chunk("b", "gamma delta")], "beta gamma")
    assert result.status == "FRAGMENTED_ACROSS_CHUNKS"
    assert result.chunk_sets == [["a", "b"]]
    assert mapped("alpha beta gamma delta", [chunk("a", "alpha beta")], "beta gamma").status == "PARTIALLY_MAPPED"


def test_normalization_ambiguity_and_missing_source():
    """宽松断词只诊断，不自动转换为 Gold。"""
    result = mapped("additional detail", [chunk("a", "addi- tional detail")], "additional detail")
    assert result.status == "ANCHOR_NORMALIZATION_FAILURE"
    assert result.chunk_sets == []
    assert mapped("answer answer", [chunk("a", "answer answer")]).status == "AMBIGUOUS_MAPPING"
    assert mapped("other text", [chunk("a", "other text")]).status == "UNMAPPED"
    assert mapped("", []).status == "SOURCE_EXTRACTION_FAILURE"


def test_index_version_mismatch_rejected():
    """内容版本变化必须重新审计，而非继续使用旧 Gold 映射。"""
    item = chunk("a", "answer")
    item.content_hash = "wrong-version"
    with pytest.raises(ValueError, match="index content version"):
        mapped("answer", [item])


def test_alternatives_are_or_and_source_missing_is_explicit():
    """一个可用替代来源足以覆盖组；全部缺来源则明确归类。"""
    data = dataset()
    data.questions[0].required_evidence_groups = data.questions[0].required_evidence_groups[:1]
    alternative = data.questions[0].required_evidence_groups[0].alternatives[0]
    data.questions[0].required_evidence_groups[0].alternatives.insert(0, alternative.model_copy(update={"document_id": "absent"}))
    result = map_dataset(data, [source()], [chunk("a", "the answer exists")], index_version="i")
    assert result.groups[0].status == "FULLY_MAPPED"
    assert map_dataset(data, [], [], index_version="i").groups[0].status == "SOURCE_EXTRACTION_FAILURE"


def test_frozen_full_audit_covers_every_group():
    """检查正式审计产物无漏题漏组，失败不混入检索分母。"""
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
