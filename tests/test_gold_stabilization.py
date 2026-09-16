"""验证 Phase 3.5 来源定位、受约束规范化、Corpus 映射和全量分母。"""

import json
from pathlib import Path

import pytest

from evidenceflow.domain.evaluation_v2 import EvaluationDatasetV2, EvidenceAlternative, GroupMapping, MappingArtifact
from evidenceflow.domain.document import Document, DocumentSegment
from evidenceflow.domain.chunk import DocumentChunk
from evidenceflow.eval.stable_mapping import hyphen_rules, normalize_mapping, validate_source, map_source_to_chunks
from evidenceflow.eval.v2 import evaluate_evidence_coverage


def source(text, page=1):
    """固定版本单页原文。"""
    return Document(document_id="doc", content_hash="hash", source_path="paper.pdf", file_name="paper.pdf",
                    file_type="pdf", segments=[DocumentSegment(content=text, page_number=page)])


def alternative(anchor="answer", **locator):
    """构造独立于 Chunk 的来源定位。"""
    return EvidenceAlternative(document_id="doc", content_version="hash", evidence_anchor=anchor,
                               locator={"page_start": 1, "page_end": 1, **locator})


def chunk(cid, text, page=1):
    """构造只读 Corpus 的测试记录。"""
    return DocumentChunk(chunk_id=cid, document_id="doc", content_hash="hash", file_name="paper.pdf",
                         content=text, token_count=2, chunk_index=0, page_start=page, page_end=page)


def test_unique_anchor_and_repeated_anchor():
    """重复出现必须消歧，不能默认选择首个。"""
    assert validate_source(alternative(), source("the answer exists"))["source_valid"]
    result = validate_source(alternative(), source("wrong answer. right answer."))
    assert not result["source_valid"]
    assert result["source_match_count"] == 2


def test_context_and_original_occurrence_disambiguation():
    """上下文先过滤，occurrence 不按过滤结果重新编号。"""
    doc = source("wrong answer before. right answer after.")
    a = alternative(left_context="right", right_context="after.")
    result = validate_source(a, doc)
    assert result["source_valid"] and result["selected_occurrence"] == 2
    assert validate_source(alternative(occurrence=2), doc)["source_valid"]
    assert not validate_source(alternative(left_context="right", occurrence=1), doc)["source_valid"]
    assert not validate_source(alternative(occurrence=3), doc)["source_valid"]


def test_hash_page_and_auxiliary_offsets():
    """页码/hash 错误失败；辅助 offset 漂移不能覆盖上下文真值。"""
    a = alternative()
    a.content_version = "changed"
    assert not validate_source(a, source("answer"))["source_valid"]
    assert not validate_source(alternative(), source("answer", page=2))["source_valid"]
    result = validate_source(alternative(source_span_start=10, source_span_end=16), source("answer"))
    assert result["source_valid"] and result["notes"]


@pytest.mark.parametrize("legal", ["state-of-the-art", "geometry-aware", "cross-modal"])
def test_legal_hyphens_preserved(legal):
    """合法连接符不能被全局删除，其他词的断词规则也不能影响它。"""
    rules = hyphen_rules("addi-\ntional evidence", "additional evidence")
    assert normalize_mapping(legal, rules) == legal
    assert normalize_mapping(legal, hyphen_rules(legal, legal)) == legal


def test_hyphen_repair_requires_source_proof():
    """只有原页换行与完整词共同证明，才能修复 Chunk 的 addi- tional。"""
    rules = hyphen_rules("addi-\ntional", "additional")
    assert normalize_mapping("addi- tional", rules) == "additional"
    assert normalize_mapping("addi- tional", hyphen_rules("additional", "additional")) == "addi- tional"
    rules = hyphen_rules("geometry-\naware", "geometry-aware")
    assert normalize_mapping("geometry- aware", rules) == "geometry-aware"


@pytest.mark.parametrize("texts,expected", [
    (["alpha beta gamma delta"], "FULLY_MAPPED"),
    (["alpha beta", "gamma delta"], "FRAGMENTED_ACROSS_CHUNKS"),
    (["alpha beta"], "PARTIALLY_MAPPED"),
    (["irrelevant text"], "UNMAPPED"),
])
def test_source_valid_corpus_states(texts, expected):
    """Source 有效与 Corpus 完整覆盖是两个独立判断。"""
    doc = source("alpha beta gamma delta")
    a = alternative("beta gamma")
    validation = validate_source(a, doc)
    status, _, _ = map_source_to_chunks(a, doc, [chunk(str(i), t) for i, t in enumerate(texts)], validation)
    assert validation["source_valid"] and status == expected


def test_wrong_occurrence_chunk_cannot_satisfy_gold():
    """即使同文档同页含相同短词，也不能用错误 occurrence 替代目标。"""
    doc = source("wrong answer before. right answer after.")
    a = alternative(left_context="right", right_context="after.", occurrence=2)
    status, bundles, _ = map_source_to_chunks(a, doc, [chunk("wrong", "wrong answer before.")], validate_source(a, doc))
    assert status == "UNMAPPED" and not bundles


def test_overall_denominator_retains_unmapped_groups():
    """防止只有易映射子集的方案因分母缩水而被错误选优。"""
    data = EvaluationDatasetV2.model_validate({
        "schema_version": "2.1", "dataset_id": "d", "split": "development", "corpus_version": "c",
        "questions": [{"question_id": "q", "family_id": "f", "query": "query", "query_language": "en",
                       "corpus_language": "en", "question_type": "multi_document", "evidence_type": "body",
                       "answerability": "answerable", "required_evidence_groups": [
                           {"group_id": g, "alternatives": [alternative().model_dump()]} for g in ("a", "b")]}]})
    mapping = MappingArtifact(dataset_id="d", corpus_version="c", index_version="i", groups=[
        GroupMapping(question_id="q", group_id="a", status="FULLY_MAPPED", chunk_sets=[["hit"]], reason="fixture"),
        GroupMapping(question_id="q", group_id="b", status="UNMAPPED", reason="valid source absent from index")])
    metrics = evaluate_evidence_coverage(data, mapping, {"q": ["hit"]}, depths=(1,))["aggregate"]
    assert metrics["eligible_evidence_group_recall_at_1"] == 1
    assert metrics["overall_evidence_coverage_at_1"] == .5
    mapping.groups[0] = GroupMapping(question_id="q", group_id="a", status="UNMAPPED", reason="empty corpus")
    metrics = evaluate_evidence_coverage(data, mapping, {"q": []}, depths=(1,))["aggregate"]
    assert metrics["eligible_evidence_group_recall_at_1"] is None
    assert metrics["overall_evidence_coverage_at_1"] == 0


def test_stable_artifacts_reproduce_from_source_without_rankings():
    """全量重建保持内容一致；函数输入不含历史排名。"""
    from experiments.stabilize_gold_v3 import build
    root = Path(__file__).resolve().parents[1] / "evals/evaluation_v2/phase35_gold_stabilization_v1"
    artifacts = build()
    for name, payload in artifacts.items():
        assert payload == json.loads((root / name).read_text(encoding="utf-8"))
    summary = artifacts["phase35_summary.json"]
    assert summary["source_valid_groups"] == 44
    assert len(summary["multi_document"]) == 4
    assert all(row["source_valid"] for row in summary["multi_document"])
