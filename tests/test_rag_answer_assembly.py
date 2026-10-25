"""Test the deterministic source assembly from Citation validation into the existing RAGAnswer."""

from copy import deepcopy

import pytest

from evidenceflow.citation import (
    CitationValidationError,
    assemble_rag_answer,
    check_citation_rules,
    classify_citation_labels,
)
from evidenceflow.domain import (
    AnswerStatus,
    BuiltContext,
    ContextEntry,
    DocumentChunk,
    GeneratedAnswer,
    RetrievedChunk,
    SourceReference,
)


def make_source_entry(
    index: int,
    *,
    file_name: str,
    page_start: int | None,
    page_end: int | None,
) -> ContextEntry:
    """Create a Context Entry that carries PDF or Markdown page-number semantics."""

    label = f"[S{index}]"
    source = SourceReference(
        source_label=label,
        document_id=f"document-{index}",
        file_name=file_name,
        page_start=page_start,
        page_end=page_end,
        chunk_id=f"chunk-{index}",
    )
    return ContextEntry(
        source_label=label,
        retrieval_rank=index,
        chunk_id=f"chunk-{index}",
        content=f"evidence {index}",
        source_reference=source,
        token_count=10,
    )


def make_context() -> BuiltContext:
    """Create a formal Context containing PDF, Markdown, and an unreferenced source at once."""

    entries = [
        make_source_entry(1, file_name="guide.pdf", page_start=7, page_end=8),
        make_source_entry(2, file_name="classic_patterns.md", page_start=None, page_end=None),
        make_source_entry(3, file_name="unused.txt", page_start=None, page_end=None),
    ]
    return BuiltContext(
        context_text="context with [S1], [S2], and [S3]",
        entries=entries,
        total_tokens=30,
        candidate_count=3,
        included_count=3,
        skipped_count=0,
    )


def make_retrieved() -> list[RetrievedChunk]:
    """Create the retrieval trace record required by the existing RAGAnswer contract."""

    chunk = DocumentChunk(
        chunk_id="retrieved-chunk",
        document_id="retrieved-document",
        content="retrieved evidence",
        token_count=2,
        chunk_index=0,
        page_start=1,
        page_end=1,
        file_name="retrieved.pdf",
        content_hash="hash",
    )
    return [RetrievedChunk(chunk=chunk, rank=1, similarity_score=0.9)]


def empty_context() -> BuiltContext:
    """Create an empty Context that permits a citation-free answer."""

    return BuiltContext(
        context_text="",
        entries=[],
        total_tokens=0,
        candidate_count=0,
        included_count=0,
        skipped_count=0,
    )


def test_assembly_maps_only_actual_sources_in_answer_order() -> None:
    context = make_context()
    generated = GeneratedAnswer(
        status=AnswerStatus.ANSWERED,
        answer_text="Markdown fact [S2]. PDF fact [S1]. Again [S2].",
    )

    result = assemble_rag_answer(
        query="What are the facts?",
        generated_answer=generated,
        built_context=context,
        retrieved_chunks=make_retrieved(),
    )

    assert result.answer == generated.answer_text
    assert [source.source_label for source in result.sources] == ["[S2]", "[S1]"]
    assert all(source.source_label != "[S3]" for source in result.sources)
    # Markdown does not fake page numbers; the PDF's original 7-8 range is preserved intact.
    assert result.sources[0].file_name == "classic_patterns.md"
    assert result.sources[0].page_start is None and result.sources[0].page_end is None
    assert result.sources[1].file_name == "guide.pdf"
    assert (result.sources[1].page_start, result.sources[1].page_end) == (7, 8)
    assert result.retrieved_chunks == make_retrieved()


@pytest.mark.parametrize(
    "answer_text",
    ["Unknown [S9].", "Valid [S1], but unknown [S9].", "Invalid zero [S0]."],
)
def test_unknown_or_mixed_citation_rejects_entire_answer(answer_text: str) -> None:
    with pytest.raises(CitationValidationError, match="unknown citation"):
        assemble_rag_answer(
            query="Question",
            generated_answer=GeneratedAnswer(status=AnswerStatus.ANSWERED, answer_text=answer_text),
            built_context=make_context(),
            retrieved_chunks=[],
        )


def test_nonempty_context_without_formal_citation_is_rejected() -> None:
    with pytest.raises(CitationValidationError, match="at least one"):
        assemble_rag_answer(
            query="Question",
            generated_answer=GeneratedAnswer(
                status=AnswerStatus.ANSWERED,
                answer_text="Answer without citation. [Sx]",
            ),
            built_context=make_context(),
            retrieved_chunks=[],
        )


def test_empty_context_without_citation_is_valid_with_no_sources() -> None:
    generated = GeneratedAnswer(
        status=AnswerStatus.INSUFFICIENT_CONTEXT,
        answer_text="知识库中没有足够的相关信息来回答这个问题。",
    )

    result = assemble_rag_answer(
        query="Missing question",
        generated_answer=generated,
        built_context=empty_context(),
        retrieved_chunks=[],
    )

    assert result.answer == generated.answer_text
    assert result.sources == []


def test_nonempty_context_insufficient_status_allows_no_citation() -> None:
    """A structured refusal must not be misjudged as missing citations even when candidate context is retained."""

    generated = GeneratedAnswer(
        status=AnswerStatus.INSUFFICIENT_CONTEXT,
        answer_text="The supplied evidence is insufficient to answer.",
    )

    result = assemble_rag_answer(
        query="Question",
        generated_answer=generated,
        built_context=make_context(),
        retrieved_chunks=[],
    )

    assert result.answer == generated.answer_text
    assert result.sources == []


def test_empty_context_with_citation_is_rejected() -> None:
    with pytest.raises(CitationValidationError, match="unknown citation"):
        assemble_rag_answer(
            query="Question",
            generated_answer=GeneratedAnswer(
                status=AnswerStatus.INSUFFICIENT_CONTEXT,
                answer_text="Unexpected [S1].",
            ),
            built_context=empty_context(),
            retrieved_chunks=[],
        )


def test_assembly_preserves_answer_context_and_generated_input() -> None:
    answer_text = "  Fact [S1].  "
    generated = GeneratedAnswer(status=AnswerStatus.ANSWERED, answer_text=answer_text)
    context = make_context()
    retrieved = make_retrieved()
    original_generated = deepcopy(generated)
    original_context = deepcopy(context)
    original_retrieved = deepcopy(retrieved)

    result = assemble_rag_answer(
        query="Question",
        generated_answer=generated,
        built_context=context,
        retrieved_chunks=retrieved,
    )

    assert result.answer == answer_text
    assert generated == original_generated
    assert context == original_context
    assert retrieved == original_retrieved


def test_assembly_rejects_blank_query_before_building_rag_answer() -> None:
    with pytest.raises(ValueError, match="query"):
        assemble_rag_answer(
            query="   ",
            generated_answer=GeneratedAnswer(status=AnswerStatus.ANSWERED, answer_text="Fact [S1]."),
            built_context=make_context(),
            retrieved_chunks=[],
        )


def test_status_is_passed_through_to_rag_answer() -> None:
    """answered and insufficient_context are both passed through from GeneratedAnswer to RAGAnswer.status."""

    answered = assemble_rag_answer(
        query="q",
        generated_answer=GeneratedAnswer(status=AnswerStatus.ANSWERED, answer_text="Fact [S1]."),
        built_context=make_context(),
        retrieved_chunks=make_retrieved(),
    )
    refused = assemble_rag_answer(
        query="q",
        generated_answer=GeneratedAnswer(
            status=AnswerStatus.INSUFFICIENT_CONTEXT, answer_text="知识库信息不足。"
        ),
        built_context=make_context(),
        retrieved_chunks=make_retrieved(),
    )

    assert answered.status is AnswerStatus.ANSWERED
    assert refused.status is AnswerStatus.INSUFFICIENT_CONTEXT
    assert refused.sources == []


def test_refusal_with_unknown_label_is_still_rejected() -> None:
    """No status permits an unknown label (independent of the rule that answered answers must cite)."""

    with pytest.raises(CitationValidationError, match="unknown citation"):
        assemble_rag_answer(
            query="q",
            generated_answer=GeneratedAnswer(
                status=AnswerStatus.INSUFFICIENT_CONTEXT, answer_text="Not enough [S9]."
            ),
            built_context=make_context(),
            retrieved_chunks=make_retrieved(),
        )


def test_label_classification_and_rules_work_without_built_context() -> None:
    """The allowlist version used by offline re-parsing follows the same rules as the BuiltContext version."""

    parsed = classify_citation_labels("A [S2]. B [S1]. C [S9]. D [S2].", {"[S1]", "[S2]"})

    assert parsed.valid_labels == ("[S2]", "[S1]")
    assert parsed.invalid_labels == ("[S9]",)
    with pytest.raises(CitationValidationError, match="unknown"):
        check_citation_rules(AnswerStatus.ANSWERED, parsed)
    ok = classify_citation_labels("no citations", {"[S1]"})
    with pytest.raises(CitationValidationError, match="at least one"):
        check_citation_rules(AnswerStatus.ANSWERED, ok)
    check_citation_rules(AnswerStatus.INSUFFICIENT_CONTEXT, ok)
