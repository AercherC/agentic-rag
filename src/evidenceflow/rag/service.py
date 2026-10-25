"""组合正式 Retrieval、Context、Generation 与 Citation 层完成单轮 RAG。"""

from __future__ import annotations

import hashlib
from typing import Protocol

from evidenceflow.citation import assemble_rag_answer, parse_citations
from evidenceflow.domain.answer import RAGAnswer
from evidenceflow.domain.chunk import (
    RerankedRetrievedChunk,
    RetrievedChunk,
)
from evidenceflow.domain.context import BuiltContext
from evidenceflow.domain.generation import GeneratedAnswer
from evidenceflow.tracing import TraceRecorder, resolve_recorder, safe_record


class RetrieverProtocol(Protocol):
    """RAGService 所需的最小正式检索接口。"""

    def retrieve(
        self, query: str
    ) -> list[RetrievedChunk | RerankedRetrievedChunk]: ...


class ContextBuilderProtocol(Protocol):
    """RAGService 所需的最小 Context 构建接口。"""

    def build(
        self,
        retrieved_chunks: list[RetrievedChunk | RerankedRetrievedChunk],
    ) -> BuiltContext: ...


class GeneratorProtocol(Protocol):
    """RAGService 所需的最小 Grounded Generation 接口。"""

    def generate(self, query: str, context: BuiltContext) -> GeneratedAnswer: ...


class RAGService:
    """仅按冻结顺序编排已有组件，并返回经过 Citation 校验的 RAGAnswer。"""

    def __init__(
        self,
        retriever: RetrieverProtocol,
        context_builder: ContextBuilderProtocol,
        generator: GeneratorProtocol,
        *,
        trace_recorder: TraceRecorder | None = None,
    ) -> None:
        """注入已创建的组件，避免 Service 隐式加载昂贵 Runtime 依赖。

        ``trace_recorder`` 可选（默认 None）：只记录 run.*、context.*、citation_validation.completed，
        不改变回答与异常（PROJECT_SPEC §12.5）。检索与生成内部事件由各组件自己的 recorder 记录。
        """

        self._retriever = retriever
        self._context_builder = context_builder
        self._generator = generator
        self.trace_recorder = trace_recorder

    def answer(self, query: str) -> RAGAnswer:
        """执行 Retrieve→Context→Generate→Citation，错误保持原样向上传播。"""

        # 空白 query 校验保持在最前：此时 run 尚未开始，不记录任何事件。
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must not be blank")
        recorder = resolve_recorder(self.trace_recorder)
        safe_record(recorder, "run.started")
        # 记录当前阶段名，失败时据此写 run.failed.stage。
        stage = "retrieval"
        try:
            # 不向 Retriever 传覆盖参数，确保使用 Composition Root 冻结的正式 Final Top-K。
            retrieved_chunks = self._retriever.retrieve(query)
            stage = "context"
            safe_record(recorder, "context.started")
            built_context = self._context_builder.build(retrieved_chunks)
            if recorder is not None:
                # 只记标签、chunk_id、token 数与 context_text 的 UTF-8 sha256，不记正文。
                safe_record(
                    recorder,
                    "context.completed",
                    included_count=built_context.included_count,
                    skipped_count=built_context.skipped_count,
                    total_tokens=built_context.total_tokens,
                    entries=[
                        {
                            "source_label": entry.source_label,
                            "chunk_id": entry.chunk_id,
                            "token_count": entry.token_count,
                        }
                        for entry in built_context.entries
                    ],
                    context_text_sha256=hashlib.sha256(
                        built_context.context_text.encode("utf-8")
                    ).hexdigest(),
                )
            stage = "generation"
            generated_answer = self._generator.generate(query, built_context)
            stage = "citation_validation"
            answer = assemble_rag_answer(
                query=query,
                generated_answer=generated_answer,
                built_context=built_context,
                retrieved_chunks=retrieved_chunks,
            )
        except Exception as exc:
            # 先记 run.failed（只记异常类名与阶段，不记消息），再原样 raise：
            # 不包装、不改异常链，调用方（CLI 错误前缀映射等）看到的异常与未接 trace 时完全相同。
            safe_record(recorder, "run.failed", error_type=type(exc).__name__, stage=stage)
            raise
        if recorder is not None:
            # 标签在 assemble_rag_answer 成功后用同一 parser 重算；只在 trace 开启时付出这份开销。
            parsed = parse_citations(answer.answer, built_context)
            safe_record(
                recorder,
                "citation_validation.completed",
                valid_labels=list(parsed.valid_labels),
                invalid_labels=list(parsed.invalid_labels),
                source_count=len(answer.sources),
            )
        safe_record(recorder, "run.completed")
        return answer
