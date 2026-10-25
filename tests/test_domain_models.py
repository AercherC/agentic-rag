"""Test the Step 1 Domain Contracts and the deterministic ID helpers."""

import pytest
from pydantic import ValidationError

from evidenceflow.domain import (
    AnswerStatus,
    Document,
    DocumentChunk,
    DocumentSegment,
    EvaluationQuestion,
    EvaluationSource,
    QueryRequest,
    RAGAnswer,
    RerankedRetrievedChunk,
    RetrievedChunk,
    SourceReference,
    build_chunk_id,
    build_content_hash,
    build_document_id,
)


def make_chunk(**overrides: object) -> DocumentChunk:
    """Create a valid baseline Chunk; tests may override individual fields as needed."""

    # One shared valid baseline, so each validation test only changes the input condition it is
    # about.
    values = {
        "chunk_id": "chunk-1",
        "document_id": "document-1",
        "content": "EvidenceFlow content",
        "token_count": 3,
        "chunk_index": 0,
        "page_start": 12,
        "page_end": 13,
        "file_name": "guide.pdf",
        "content_hash": "content-hash",
    }
    values.update(overrides)
    return DocumentChunk(**values)


def test_domain_models_construct_with_valid_data() -> None:
    segment = DocumentSegment(content="Page content", page_number=1)
    document = Document(
        document_id="document-1",
        source_path="docs/guide.pdf",
        file_name="guide.pdf",
        file_type="pdf",
        content_hash="content-hash",
        segments=[segment],
    )
    chunk = make_chunk()
    retrieved = RetrievedChunk(chunk=chunk, rank=1, similarity_score=0.92)
    source = SourceReference(
        source_label="[S1]",
        document_id=document.document_id,
        file_name=document.file_name,
        page_start=12,
        page_end=13,
        chunk_id=chunk.chunk_id,
    )
    answer = RAGAnswer(
        query="What is EvidenceFlow?",
        status=AnswerStatus.ANSWERED,
        answer="A local RAG assistant. [S1]",
        sources=[source],
        retrieved_chunks=[retrieved],
    )
    question = EvaluationQuestion(
        question="What is EvidenceFlow?",
        relevant_sources=[
            EvaluationSource(document_id=document.document_id, page_start=12, page_end=13)
        ],
    )

    assert answer.sources[0].source_label == "[S1]"
    assert question.relevant_sources[0].page_start == 12
    assert QueryRequest(query="What is RAG?").query == "What is RAG?"


def test_missing_required_field_fails_validation() -> None:
    with pytest.raises(ValidationError):
        DocumentSegment()


@pytest.mark.parametrize("model", [make_chunk, lambda **kwargs: EvaluationSource(document_id="d", **kwargs)])
def test_page_start_after_page_end_fails_validation(model) -> None:
    with pytest.raises(ValidationError):
        model(page_start=4, page_end=3)


def test_rank_below_one_fails_validation() -> None:
    with pytest.raises(ValidationError):
        RetrievedChunk(chunk=make_chunk(), rank=0, similarity_score=0.5)


def test_reranked_result_keeps_dense_and_learned_scores_separate() -> None:
    """A Phase 9 result keeps the Dense candidate rank, the cosine score, and an independent Reranker score side by side."""

    result = RerankedRetrievedChunk(
        chunk=make_chunk(),
        rank=1,
        candidate_rank=7,
        similarity_score=0.42,
        reranker_score=3.5,
    )

    assert result.candidate_rank == 7
    assert result.similarity_score == 0.42
    assert result.reranker_score == 3.5


def test_negative_token_count_fails_validation() -> None:
    with pytest.raises(ValidationError):
        make_chunk(token_count=-1)


def test_negative_chunk_index_fails_validation() -> None:
    with pytest.raises(ValidationError):
        make_chunk(chunk_index=-1)


def test_unpaged_text_models_allow_none_page_ranges() -> None:
    segment = DocumentSegment(content="Markdown content", page_number=None)
    chunk = make_chunk(page_start=None, page_end=None, file_name="notes.md")
    source = EvaluationSource(document_id="document-1", page_start=None, page_end=None)

    assert segment.page_number is None
    assert chunk.page_start is None and chunk.page_end is None
    assert source.page_start is None and source.page_end is None


def test_document_id_is_deterministic_and_path_sensitive() -> None:
    assert build_document_id("docs/guide.pdf") == build_document_id("docs\\guide.pdf")
    assert build_document_id("docs/guide.pdf") != build_document_id("docs/other.pdf")


def test_document_id_normalizes_unicode_to_nfc() -> None:
    assert build_document_id("docs/caf\u00e9.pdf") == build_document_id("docs/cafe\u0301.pdf")


def test_content_hash_is_deterministic_and_content_sensitive() -> None:
    assert build_content_hash(b"same") == build_content_hash(b"same")
    assert build_content_hash(b"same") != build_content_hash(b"different")


def test_chunk_id_is_deterministic_and_each_identity_input_matters() -> None:
    base = {
        "document_id": "document-1",
        "chunk_size": 512,
        "chunk_overlap": 64,
        "chunk_index": 0,
        "content": "chunk content",
    }
    baseline = build_chunk_id(**base)

    assert baseline == build_chunk_id(**base)
    for field, value in {
        "chunk_size": 256,
        "chunk_overlap": 32,
        "chunk_index": 1,
        "content": "changed content",
    }.items():
        changed = {**base, field: value}
        assert build_chunk_id(**changed) != baseline


def test_rag_answer_requires_status() -> None:
    """RAGAnswer.status is a required field; both a missing and an unknown value are rejected."""

    with pytest.raises(ValidationError):
        RAGAnswer(query="q", answer="a", sources=[], retrieved_chunks=[])
    with pytest.raises(ValidationError):
        RAGAnswer(query="q", status="maybe", answer="a", sources=[], retrieved_chunks=[])
