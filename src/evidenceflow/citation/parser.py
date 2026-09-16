"""解析 Generated Answer 中的 Source Label 并组装最终 RAGAnswer。"""

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

# S0 也先作为正式形状的 Candidate 捕获，再由 Context allowlist 明确判定为未知标签。
_CITATION_PATTERN = re.compile(r"\[S(?:0|[1-9]\d*)\]")


class CitationValidationError(ValueError):
    """表示答案违反当前 Context 允许标签或必须引用来源的合同。"""


# 保留旧导入名作为兼容别名；上层从未对两层异常采取不同处理，无需维持空中间类。
CitationError = CitationValidationError


def parse_citations(answer_text: str, built_context: BuiltContext) -> CitationParseResult:
    """按首次出现顺序提取、去重并按 BuiltContext allowlist 分类标签。"""

    if not isinstance(answer_text, str) or not answer_text.strip():
        raise CitationValidationError("answer_text must not be blank")

    allowed_labels = {entry.source_label for entry in built_context.entries}
    seen: set[str] = set()
    valid_labels: list[str] = []
    invalid_labels: list[str] = []
    for match in _CITATION_PATTERN.finditer(answer_text):
        label = match.group(0)
        # 同一标签只在第一次出现时进入结果，保持最终 Source List 与阅读顺序一致。
        if label in seen:
            continue
        seen.add(label)
        target = valid_labels if label in allowed_labels else invalid_labels
        target.append(label)

    return CitationParseResult(
        valid_labels=tuple(valid_labels),
        invalid_labels=tuple(invalid_labels),
    )


def assemble_rag_answer(
    *,
    query: str,
    generated_answer: GeneratedAnswer,
    built_context: BuiltContext,
    retrieved_chunks: Sequence[RetrievedChunk | RerankedRetrievedChunk],
) -> RAGAnswer:
    """校验模型标签并以既有 SourceReference 组装最终结构化回答。"""

    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must not be blank")

    parsed = parse_citations(generated_answer.answer_text, built_context)
    if parsed.invalid_labels:
        labels = ", ".join(parsed.invalid_labels)
        raise CitationValidationError(f"answer contains unknown citation labels: {labels}")
    if generated_answer.status is AnswerStatus.ANSWERED and not parsed.valid_labels:
        raise CitationValidationError("answer must cite at least one supplied source")

    source_by_label = {
        entry.source_label: entry.source_reference for entry in built_context.entries
    }
    # 直接复用 Context 已验证的来源对象；不从 Answer 猜测文件名、页码或 Chunk 身份。
    sources = [source_by_label[label] for label in parsed.valid_labels]
    return RAGAnswer(
        query=query,
        answer=generated_answer.answer_text,
        sources=sources,
        retrieved_chunks=list(retrieved_chunks),
    )
