"""组合正式 Retrieval、Context、Generation 与 Citation 层完成单轮 RAG。"""

from __future__ import annotations

from typing import Protocol

from evidenceflow.citation import assemble_rag_answer
from evidenceflow.domain.answer import RAGAnswer
from evidenceflow.domain.chunk import (
    RerankedRetrievedChunk,
    RetrievedChunk,
)
from evidenceflow.domain.context import BuiltContext
from evidenceflow.domain.generation import GeneratedAnswer


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
    ) -> None:
        """注入已创建的组件，避免 Service 隐式加载昂贵 Runtime 依赖。"""

        self._retriever = retriever
        self._context_builder = context_builder
        self._generator = generator

    def answer(self, query: str) -> RAGAnswer:
        """执行 Retrieve→Context→Generate→Citation，错误保持原样向上传播。"""

        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must not be blank")
        # 不向 Retriever 传覆盖参数，确保使用 Composition Root 冻结的正式 Final Top-K。
        retrieved_chunks = self._retriever.retrieve(query)
        built_context = self._context_builder.build(retrieved_chunks)
        generated_answer = self._generator.generate(query, built_context)
        return assemble_rag_answer(
            query=query,
            generated_answer=generated_answer,
            built_context=built_context,
            retrieved_chunks=retrieved_chunks,
        )
