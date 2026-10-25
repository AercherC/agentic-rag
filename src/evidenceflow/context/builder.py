"""Assemble the formal retrieval results into a budget-constrained, complete LLM Context."""

from __future__ import annotations

from collections.abc import Sequence
from evidenceflow.domain.answer import SourceReference
from evidenceflow.domain.chunk import (
    RerankedRetrievedChunk,
    RetrievedChunk,
)
from evidenceflow.domain.context import BuiltContext, ContextEntry
from evidenceflow.ingestion.tokenizer import TokenizerError, TokenizerProtocol

class ContextBuilder:
    """Include complete Chunks strictly in Retrieval Rank order, with no truncation and no re-ranking."""

    _ENTRY_SEPARATOR = "\n\n"

    def __init__(self, tokenizer: TokenizerProtocol, max_context_tokens: int) -> None:
        """Inject the Tokenizer that matches Embedding plus a positive-integer default budget."""

        self._tokenizer = tokenizer
        self._max_context_tokens = self._validate_budget(max_context_tokens)

    @staticmethod
    def _validate_budget(max_tokens: int) -> int:
        """Reject booleans, zero, and negatives to avoid ambiguous budget semantics."""

        if type(max_tokens) is not int or max_tokens <= 0:
            raise ValueError("max_tokens must be a positive integer")
        return max_tokens

    def _count_tokens(self, text: str) -> int:
        """Count with the formal Tokenizer and verify that test doubles also honour the interface contract."""

        token_ids = self._tokenizer.encode(text)
        if not isinstance(token_ids, list) or not all(type(item) is int for item in token_ids):
            raise TokenizerError("Tokenizer encode returned invalid token IDs")
        return len(token_ids)

    @staticmethod
    def _render_entry(
        source_label: str,
        item: RetrievedChunk | RerankedRetrievedChunk,
    ) -> str:
        """Render the label, source header, and unmodified full body using a stable template."""

        chunk = item.chunk
        lines = [source_label, f"File: {chunk.file_name}"]
        if chunk.page_start is not None:
            # The DocumentChunk model already guarantees a complete page range; a single page
            # avoids the redundant 12-12.
            page_range = (
                str(chunk.page_start)
                if chunk.page_start == chunk.page_end
                else f"{chunk.page_start}-{chunk.page_end}"
            )
            lines.append(f"Pages: {page_range}")
        lines.extend(["Content:", chunk.content])
        return "\n".join(lines)

    @staticmethod
    def _source_reference(
        source_label: str,
        item: RetrievedChunk | RerankedRetrievedChunk,
    ) -> SourceReference:
        """Map the existing SourceReference from the complete Chunk without carrying retrieval-internal scores."""

        chunk = item.chunk
        return SourceReference(
            source_label=source_label,
            document_id=chunk.document_id,
            file_name=chunk.file_name,
            page_start=chunk.page_start,
            page_end=chunk.page_end,
            chunk_id=chunk.chunk_id,
        )

    @staticmethod
    def _ranked_candidates(
        retrieved_chunks: Sequence[RetrievedChunk | RerankedRetrievedChunk],
    ) -> list[RetrievedChunk | RerankedRetrievedChunk]:
        """Restore the formal order from the explicit Rank, rejecting ambiguous duplicate identities or ranks."""

        candidates = list(retrieved_chunks)
        ranks = [item.rank for item in candidates]
        chunk_ids = [item.chunk.chunk_id for item in candidates]
        if len(set(ranks)) != len(ranks):
            raise ValueError("retrieved_chunks must not contain duplicate ranks")
        if len(set(chunk_ids)) != len(chunk_ids):
            raise ValueError("retrieved_chunks must not contain duplicate chunk_id values")
        # Order by the Retriever's formal Rank only; introduce no second relevance judgment.
        return sorted(candidates, key=lambda item: item.rank)

    def build(
        self,
        retrieved_chunks: Sequence[RetrievedChunk | RerankedRetrievedChunk],
        max_tokens: int | None = None,
    ) -> BuiltContext:
        """Fit as many complete candidates as the budget allows, and keep checking later candidates after an overrun."""

        budget = self._max_context_tokens if max_tokens is None else self._validate_budget(max_tokens)
        candidates = self._ranked_candidates(retrieved_chunks)
        entries: list[ContextEntry] = []
        rendered_entries: list[str] = []
        total_tokens = 0

        for item in candidates:
            # The label follows the number of actually included entries, so skipped candidates
            # leave no gaps.
            source_label = f"[S{len(entries) + 1}]"
            rendered_entry = self._render_entry(source_label, item)
            prospective_text = self._ENTRY_SEPARATOR.join([*rendered_entries, rendered_entry])
            # Re-encode the whole prospective Context so separators between entries and token
            # boundaries also count against the budget.
            prospective_tokens = self._count_tokens(prospective_text)
            if prospective_tokens > budget:
                continue

            entries.append(
                ContextEntry(
                    source_label=source_label,
                    retrieval_rank=item.rank,
                    chunk_id=item.chunk.chunk_id,
                    content=item.chunk.content,
                    source_reference=self._source_reference(source_label, item),
                    token_count=self._count_tokens(rendered_entry),
                )
            )
            rendered_entries.append(rendered_entry)
            total_tokens = prospective_tokens

        context_text = self._ENTRY_SEPARATOR.join(rendered_entries)
        return BuiltContext(
            context_text=context_text,
            entries=entries,
            total_tokens=total_tokens,
            candidate_count=len(candidates),
            included_count=len(entries),
            skipped_count=len(candidates) - len(entries),
        )
