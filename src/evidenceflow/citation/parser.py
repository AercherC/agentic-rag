"""Parse Source Labels out of a Generated Answer and assemble the final RAGAnswer."""

from __future__ import annotations

import re
from collections.abc import Sequence

from evidenceflow.domain.answer import RAGAnswer
from evidenceflow.domain.chunk import (
    RerankedRetrievedChunk,
    RetrievedChunk,
)
from evidenceflow.domain.citation import CitationParseResult
from evidenceflow.domain.context import BuiltContext
from evidenceflow.domain.generation import GeneratedAnswer
from evidenceflow.domain.generation import AnswerStatus

# S0 is captured as a well-formed candidate too, then explicitly judged unknown against the Context
# allowlist.
_CITATION_PATTERN = re.compile(r"\[S(?:0|[1-9]\d*)\]")


class CitationValidationError(ValueError):
    """Signals that the answer violates the labels allowed by the current Context or the must-cite-a-source contract."""


# Keep the old import name as a compatibility alias; callers never handled the two exception layers
# differently, so an empty intermediate class is not worth keeping.
CitationError = CitationValidationError


def parse_citations(answer_text: str, built_context: BuiltContext) -> CitationParseResult:
    """Extract labels in order of first appearance, deduplicate, and classify them against the BuiltContext allowlist."""

    allowed_labels = {entry.source_label for entry in built_context.entries}
    return classify_citation_labels(answer_text, allowed_labels)


def classify_citation_labels(
    answer_text: str, allowed_labels: set[str] | frozenset[str]
) -> CitationParseResult:
    """Classify labels using the allowlist alone, so the offline re-parse script can reuse this without rebuilding a full BuiltContext."""

    if not isinstance(answer_text, str) or not answer_text.strip():
        raise CitationValidationError("answer_text must not be blank")

    seen: set[str] = set()
    valid_labels: list[str] = []
    invalid_labels: list[str] = []
    for match in _CITATION_PATTERN.finditer(answer_text):
        label = match.group(0)
        # A label only enters the result on its first appearance, keeping the final Source List
        # aligned with reading order.
        if label in seen:
            continue
        seen.add(label)
        target = valid_labels if label in allowed_labels else invalid_labels
        target.append(label)

    return CitationParseResult(
        valid_labels=tuple(valid_labels),
        invalid_labels=tuple(invalid_labels),
    )


def check_citation_rules(status: AnswerStatus, parsed: CitationParseResult) -> None:
    """Citation contract: no status may carry unknown labels, and answered must cite at least one source."""

    if parsed.invalid_labels:
        labels = ", ".join(parsed.invalid_labels)
        raise CitationValidationError(f"answer contains unknown citation labels: {labels}")
    if status is AnswerStatus.ANSWERED and not parsed.valid_labels:
        raise CitationValidationError("answer must cite at least one supplied source")


def assemble_rag_answer(
    *,
    query: str,
    generated_answer: GeneratedAnswer,
    built_context: BuiltContext,
    retrieved_chunks: Sequence[RetrievedChunk | RerankedRetrievedChunk],
) -> RAGAnswer:
    """Validate the model's labels and assemble the final structured answer from the existing SourceReference objects."""

    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must not be blank")

    parsed = parse_citations(generated_answer.answer_text, built_context)
    check_citation_rules(generated_answer.status, parsed)

    source_by_label = {
        entry.source_label: entry.source_reference for entry in built_context.entries
    }
    # Reuse the already-validated source objects from Context; never guess file names, pages, or
    # Chunk identity from the Answer.
    sources = [source_by_label[label] for label in parsed.valid_labels]
    return RAGAnswer(
        query=query,
        # status comes straight from the Generator, so downstream code can tell "answered" from
        # "abstained" without guessing from natural language.
        status=generated_answer.status,
        answer=generated_answer.answer_text,
        sources=sources,
        retrieved_chunks=list(retrieved_chunks),
    )
