"""测试 Citation 标签的严格解析、去重、顺序和 allowlist 分类。"""

from copy import deepcopy

from evidenceflow.citation import parse_citations
from evidenceflow.domain import BuiltContext, ContextEntry, SourceReference


def make_context(count: int = 3) -> BuiltContext:
    """创建带连续正式标签的 Context，支持验证多位数标签。"""

    entries: list[ContextEntry] = []
    rendered: list[str] = []
    for index in range(1, count + 1):
        label = f"[S{index}]"
        source = SourceReference(
            source_label=label,
            document_id=f"document-{index}",
            file_name=f"source-{index}.md",
            page_start=None,
            page_end=None,
            chunk_id=f"chunk-{index}",
        )
        entries.append(
            ContextEntry(
                source_label=label,
                retrieval_rank=index,
                chunk_id=f"chunk-{index}",
                content=f"evidence {index}",
                source_reference=source,
                token_count=10,
            )
        )
        rendered.append(f"{label}\nFile: source-{index}.md\nContent:\nevidence {index}")
    text = "\n\n".join(rendered)
    return BuiltContext(
        context_text=text,
        entries=entries,
        total_tokens=max(1, len(text)),
        candidate_count=count,
        included_count=count,
        skipped_count=0,
    )


def empty_context() -> BuiltContext:
    """创建没有任何允许标签的正式空 Context。"""

    return BuiltContext(
        context_text="",
        entries=[],
        total_tokens=0,
        candidate_count=0,
        included_count=0,
        skipped_count=0,
    )


def test_single_and_multiple_citations_preserve_first_occurrence_order() -> None:
    context = make_context()

    single = parse_citations("Fact [S1].", context)
    multiple = parse_citations("A [S2]. B [S1]. C [S3].", context)

    assert single.valid_labels == ("[S1]",)
    assert single.invalid_labels == ()
    assert multiple.valid_labels == ("[S2]", "[S1]", "[S3]")


def test_duplicate_citations_are_deduplicated_at_first_occurrence() -> None:
    parsed = parse_citations("A [S2]. B [S1]. C [S2]. D [S1].", make_context())

    assert parsed.valid_labels == ("[S2]", "[S1]")


def test_unknown_and_mixed_labels_are_reported_without_silent_drop() -> None:
    context = make_context(2)

    unknown = parse_citations("A [S7].", context)
    mixed = parse_citations("A [S1]. B [S9]. C [S1]. D [S8].", context)

    assert unknown.valid_labels == ()
    assert unknown.invalid_labels == ("[S7]",)
    assert mixed.valid_labels == ("[S1]",)
    assert mixed.invalid_labels == ("[S9]", "[S8]")


def test_s10_and_s0_are_parsed_as_candidates_then_validated_against_context() -> None:
    parsed = parse_citations("Valid [S10], invalid [S0].", make_context(10))

    assert parsed.valid_labels == ("[S10]",)
    assert parsed.invalid_labels == ("[S0]",)


def test_citation_like_noise_is_not_recognized() -> None:
    parsed = parse_citations(
        "Use [Section 1], [Sx], [S-1], [S 1], [S01], and S1.",
        make_context(),
    )

    assert parsed.valid_labels == ()
    assert parsed.invalid_labels == ()


def test_empty_context_classifies_formal_label_as_invalid() -> None:
    parsed = parse_citations("Unexpected [S1].", empty_context())

    assert parsed.valid_labels == ()
    assert parsed.invalid_labels == ("[S1]",)


def test_parser_does_not_modify_answer_or_built_context() -> None:
    answer = "  A [S2]. B [S1].  "
    context = make_context()
    original_context = deepcopy(context)

    parse_citations(answer, context)

    assert answer == "  A [S2]. B [S1].  "
    assert context == original_context
