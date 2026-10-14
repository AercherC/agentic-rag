"""测试 ask CLI 的委托、用户输出、来源格式和失败退出码。"""

import pytest

from evidenceflow.__main__ import main, render_rag_answer
from evidenceflow.citation import CitationValidationError
from evidenceflow.config import Settings
from evidenceflow.generation import GenerationContractError
from evidenceflow.domain import AnswerStatus, RAGAnswer, SourceReference
from evidenceflow.llm import LLMConfigurationError, LLMRequestError


class FakeRAGService:
    """记录 CLI Query 并返回固定回答或抛出指定错误。"""

    def __init__(self, answer: RAGAnswer | None = None, error: Exception | None = None) -> None:
        self.result = answer
        self.error = error
        self.calls: list[str] = []

    def answer(self, query: str) -> RAGAnswer:
        self.calls.append(query)
        if self.error is not None:
            raise self.error
        assert self.result is not None
        return self.result


def make_answer(*, sources: list[SourceReference] | None = None) -> RAGAnswer:
    """创建无需真实 Retriever/LLM 的 CLI 输出结果。"""

    return RAGAnswer(
        query="为什么 Agent 需要工具？",
        status=AnswerStatus.ANSWERED if sources else AnswerStatus.INSUFFICIENT_CONTEXT,
        answer="Agent 通过工具获得外部能力 [S1]。" if sources else "知识库信息不足。",
        sources=[] if sources is None else sources,
        retrieved_chunks=[],
    )


def test_ask_cli_preserves_quoted_multiword_query_and_renders_answer(capsys) -> None:
    source = SourceReference(
        source_label="[S1]",
        document_id="doc",
        file_name="agents.pdf",
        page_start=7,
        page_end=8,
        chunk_id="chunk",
    )
    service = FakeRAGService(make_answer(sources=[source]))

    exit_code = main(
        ["ask", "为什么 Agent 需要工具？"],
        settings=Settings(_env_file=None),
        rag_service=service,
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert service.calls == ["为什么 Agent 需要工具？"]
    assert "Question:\n为什么 Agent 需要工具？" in captured.out
    assert "Status: answered\nAnswer:\nAgent 通过工具获得外部能力 [S1]。" in captured.out
    assert "[S1]\nFile: agents.pdf\nPages: 7-8" in captured.out
    assert "document_id" not in captured.out and "chunk" not in captured.out
    assert captured.err == ""


def test_renderer_formats_single_pdf_page_markdown_and_empty_sources() -> None:
    sources = [
        SourceReference(source_label="[S1]", document_id="d1", file_name="one.pdf", page_start=7, page_end=7, chunk_id="c1"),
        SourceReference(source_label="[S2]", document_id="d2", file_name="notes.md", page_start=None, page_end=None, chunk_id="c2"),
    ]
    rendered = render_rag_answer(make_answer(sources=sources))

    assert "File: one.pdf\nPages: 7" in rendered
    assert "Pages: 7-7" not in rendered
    assert "[S2]\nFile: notes.md" in rendered
    assert "Pages: None" not in rendered
    assert "No cited sources." in render_rag_answer(make_answer())


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (LLMConfigurationError("missing key"), "LLM configuration error: missing key"),
        (LLMRequestError("timeout"), "LLM request failed: timeout"),
        (GenerationContractError("bad json"), "Generation format error: bad json"),
        (CitationValidationError("unknown [S9]"), "Citation validation failed: unknown [S9]"),
        (RuntimeError("store unavailable"), "Retrieval or RAG processing failed: store unavailable"),
    ],
)
def test_ask_cli_renders_controlled_failure_without_unvalidated_answer(
    error: Exception, expected: str, capsys
) -> None:
    service = FakeRAGService(error=error)

    exit_code = main(
        ["ask", "question"],
        settings=Settings(_env_file=None),
        rag_service=service,
    )
    captured = capsys.readouterr()

    assert exit_code == 1
    assert captured.out == ""
    assert expected in captured.err


def test_ask_cli_does_not_swallow_keyboard_interrupt() -> None:
    service = FakeRAGService(error=KeyboardInterrupt())
    with pytest.raises(KeyboardInterrupt):
        main(["ask", "question"], settings=Settings(_env_file=None), rag_service=service)


def test_renderer_shows_insufficient_context_status() -> None:
    """拒答状态也要在 CLI 输出中可见，用户不必从自然语言里猜。"""

    rendered = render_rag_answer(make_answer())

    assert "Status: insufficient_context\nAnswer:\n知识库信息不足。" in rendered
