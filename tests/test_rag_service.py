"""测试 RAGService 的固定编排顺序、数据传递与错误传播。"""

from copy import deepcopy

import pytest

from evidenceflow.citation import CitationValidationError
from evidenceflow.domain import (
    AnswerStatus,
    BuiltContext,
    ContextEntry,
    DocumentChunk,
    GeneratedAnswer,
    RerankedRetrievedChunk,
    SourceReference,
)
from evidenceflow.generation import Generator
from evidenceflow.llm import LLMRequestError
from evidenceflow.rag import RAGService


def make_retrieved() -> list[RerankedRetrievedChunk]:
    """创建正式 Dense + Learned Reranker 检索追踪记录。"""

    chunk = DocumentChunk(
        chunk_id="chunk-1",
        document_id="document-1",
        content="Agent tools provide external capabilities.",
        token_count=6,
        chunk_index=0,
        page_start=7,
        page_end=8,
        file_name="agents.pdf",
        content_hash="hash-1",
    )
    return [
        RerankedRetrievedChunk(
            chunk=chunk,
            rank=1,
            candidate_rank=1,
            similarity_score=0.9,
            reranker_score=3.0,
        )
    ]


def make_context() -> BuiltContext:
    """创建与正式候选一致的单来源 Context。"""

    source = SourceReference(
        source_label="[S1]",
        document_id="document-1",
        file_name="agents.pdf",
        page_start=7,
        page_end=8,
        chunk_id="chunk-1",
    )
    entry = ContextEntry(
        source_label="[S1]",
        retrieval_rank=1,
        chunk_id="chunk-1",
        content="Agent tools provide external capabilities.",
        source_reference=source,
        token_count=20,
    )
    return BuiltContext(
        context_text="[S1]\nFile: agents.pdf\nPages: 7-8\nContent:\nAgent tools provide external capabilities.",
        entries=[entry],
        total_tokens=20,
        candidate_count=1,
        included_count=1,
        skipped_count=0,
    )


class FakeRetriever:
    def __init__(self, results=None, error: Exception | None = None, events=None) -> None:
        self.results = make_retrieved() if results is None else results
        self.error = error
        self.calls = []
        self.events = events if events is not None else []

    def retrieve(self, query):
        self.calls.append(query)
        self.events.append("retrieve")
        if self.error:
            raise self.error
        return self.results


class FakeContextBuilder:
    def __init__(self, context=None, error: Exception | None = None, events=None) -> None:
        self.context = make_context() if context is None else context
        self.error = error
        self.calls = []
        self.events = events if events is not None else []

    def build(self, results):
        self.calls.append(results)
        self.events.append("build")
        if self.error:
            raise self.error
        return self.context


class FakeGenerator:
    def __init__(
        self,
        answer="Agent tools provide external capabilities [S1].",
        error=None,
        events=None,
        status=AnswerStatus.ANSWERED,
    ) -> None:
        self.answer = answer
        self.error = error
        self.calls = []
        self.events = events if events is not None else []
        self.status = status

    def generate(self, query, context):
        self.calls.append((query, context))
        self.events.append("generate")
        if self.error:
            raise self.error
        return GeneratedAnswer(status=self.status, answer_text=self.answer)


def test_happy_path_preserves_query_results_sources_and_call_order() -> None:
    events = []
    retrieved = make_retrieved()
    context = make_context()
    retriever = FakeRetriever(retrieved, events=events)
    builder = FakeContextBuilder(context, events=events)
    generator = FakeGenerator(events=events)
    query = "为什么 Agent 需要工具？"

    answer = RAGService(retriever, builder, generator).answer(query)

    assert events == ["retrieve", "build", "generate"]
    assert retriever.calls == [query]
    assert builder.calls == [retrieved]
    assert generator.calls == [(query, context)]
    assert answer.query == query
    assert answer.answer == "Agent tools provide external capabilities [S1]."
    assert answer.retrieved_chunks == retrieved
    assert [source.source_label for source in answer.sources] == ["[S1]"]


def test_query_is_only_checked_for_blank_and_otherwise_preserved() -> None:
    query = "  What are tools?  "
    retriever = FakeRetriever()
    generator = FakeGenerator()
    answer = RAGService(retriever, FakeContextBuilder(), generator).answer(query)

    assert retriever.calls == [query]
    assert generator.calls[0][0] == query
    assert answer.query == query


def test_empty_retrieval_flows_through_existing_empty_context_and_generator_contract() -> None:
    empty = BuiltContext(context_text="", entries=[], total_tokens=0, candidate_count=0, included_count=0, skipped_count=0)

    class NeverCalledLLM:
        def generate(self, messages):
            raise AssertionError("LLM must not be called for empty context")

    answer = RAGService(FakeRetriever(results=[]), FakeContextBuilder(context=empty), Generator(NeverCalledLLM())).answer("unknown")

    assert answer.sources == []
    assert answer.retrieved_chunks == []
    assert "没有足够" in answer.answer


def test_retriever_or_context_failure_stops_downstream_components() -> None:
    builder = FakeContextBuilder()
    generator = FakeGenerator()
    with pytest.raises(RuntimeError, match="retrieval failed"):
        RAGService(FakeRetriever(error=RuntimeError("retrieval failed")), builder, generator).answer("query")
    assert builder.calls == [] and generator.calls == []

    generator = FakeGenerator()
    with pytest.raises(RuntimeError, match="context failed"):
        RAGService(FakeRetriever(), FakeContextBuilder(error=RuntimeError("context failed")), generator).answer("query")
    assert generator.calls == []


@pytest.mark.parametrize(
    "error",
    [LLMRequestError("LLM failed"), CitationValidationError("bad citation")],
)
def test_generation_and_citation_errors_propagate_without_fallback(error: Exception) -> None:
    generator = FakeGenerator(error=error) if isinstance(error, LLMRequestError) else FakeGenerator(answer="Bad [S9].")
    with pytest.raises(type(error)):
        RAGService(FakeRetriever(), FakeContextBuilder(), generator).answer("query")


def test_service_does_not_mutate_component_outputs() -> None:
    retrieved = make_retrieved()
    context = make_context()
    generated = GeneratedAnswer(status=AnswerStatus.ANSWERED, answer_text="Fact [S1].")
    originals = deepcopy((retrieved, context, generated))

    class ExistingAnswerGenerator:
        def generate(self, query, built_context):
            return generated

    RAGService(FakeRetriever(retrieved), FakeContextBuilder(context), ExistingAnswerGenerator()).answer("query")

    assert (retrieved, context, generated) == originals


def test_blank_query_is_rejected_before_retrieval() -> None:
    retriever = FakeRetriever()
    with pytest.raises(ValueError, match="query"):
        RAGService(retriever, FakeContextBuilder(), FakeGenerator()).answer("   ")
    assert retriever.calls == []


def test_status_flows_from_generator_to_rag_answer() -> None:
    """拒答状态经 Generator → Citation → RAGAnswer 保持不变。"""

    generator = FakeGenerator(answer="知识库信息不足。", status=AnswerStatus.INSUFFICIENT_CONTEXT)

    result = RAGService(FakeRetriever(), FakeContextBuilder(), generator).answer("query")

    assert result.status is AnswerStatus.INSUFFICIENT_CONTEXT
    assert result.sources == []
