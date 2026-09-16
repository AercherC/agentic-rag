"""将正式检索结果组装为受预算约束的完整 LLM Context。"""

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
    """严格按 Retrieval Rank 纳入完整 Chunk，不执行截断或重新排序。"""

    _ENTRY_SEPARATOR = "\n\n"

    def __init__(self, tokenizer: TokenizerProtocol, max_context_tokens: int) -> None:
        """注入与 Embedding 一致的 Tokenizer 和正整数默认预算。"""

        self._tokenizer = tokenizer
        self._max_context_tokens = self._validate_budget(max_context_tokens)

    @staticmethod
    def _validate_budget(max_tokens: int) -> int:
        """拒绝布尔值、零和负数，避免产生含糊的预算语义。"""

        if type(max_tokens) is not int or max_tokens <= 0:
            raise ValueError("max_tokens must be a positive integer")
        return max_tokens

    def _count_tokens(self, text: str) -> int:
        """使用正式 Tokenizer 计数，并校验测试替身也遵守接口合同。"""

        token_ids = self._tokenizer.encode(text)
        if not isinstance(token_ids, list) or not all(type(item) is int for item in token_ids):
            raise TokenizerError("Tokenizer encode returned invalid token IDs")
        return len(token_ids)

    @staticmethod
    def _render_entry(
        source_label: str,
        item: RetrievedChunk | RerankedRetrievedChunk,
    ) -> str:
        """按稳定模板渲染标签、来源头和未经修改的完整正文。"""

        chunk = item.chunk
        lines = [source_label, f"File: {chunk.file_name}"]
        if chunk.page_start is not None:
            # 页码范围由 DocumentChunk 模型保证完整；单页避免输出冗余的 12-12。
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
        """从完整 Chunk 映射已有 SourceReference，不携带检索内部评分。"""

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
        """按显式 Rank 恢复正式顺序，并拒绝不明确的重复身份或名次。"""

        candidates = list(retrieved_chunks)
        ranks = [item.rank for item in candidates]
        chunk_ids = [item.chunk.chunk_id for item in candidates]
        if len(set(ranks)) != len(ranks):
            raise ValueError("retrieved_chunks must not contain duplicate ranks")
        if len(set(chunk_ids)) != len(chunk_ids):
            raise ValueError("retrieved_chunks must not contain duplicate chunk_id values")
        # 仅依据 Retriever 的正式 Rank 排序，不引入任何二次相关性判断。
        return sorted(candidates, key=lambda item: item.rank)

    def build(
        self,
        retrieved_chunks: Sequence[RetrievedChunk | RerankedRetrievedChunk],
        max_tokens: int | None = None,
    ) -> BuiltContext:
        """在预算内尽量纳入完整候选，超限后继续检查后续候选。"""

        budget = self._max_context_tokens if max_tokens is None else self._validate_budget(max_tokens)
        candidates = self._ranked_candidates(retrieved_chunks)
        entries: list[ContextEntry] = []
        rendered_entries: list[str] = []
        total_tokens = 0

        for item in candidates:
            # 标签由实际纳入数决定；被跳过的候选不会造成标签空洞。
            source_label = f"[S{len(entries) + 1}]"
            rendered_entry = self._render_entry(source_label, item)
            prospective_text = self._ENTRY_SEPARATOR.join([*rendered_entries, rendered_entry])
            # 对完整候选 Context 重新编码，确保条目间分隔符及 Token 边界也计入预算。
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
