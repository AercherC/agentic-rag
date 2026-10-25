"""Domain Models for generated answers and the sources they actually cite."""

from pydantic import BaseModel, Field, model_validator

from evidenceflow.domain.chunk import (
    RerankedRetrievedChunk,
    RetrievedChunk,
)
from evidenceflow.domain.generation import AnswerStatus


class SourceReference(BaseModel):
    """The compact source object actually cited by the final answer."""

    source_label: str = Field(
        min_length=1,
        description="LLM 在 Context 中引用的临时标签，例如 [S1]。",
    )
    document_id: str = Field(min_length=1, description="被引用逻辑文档的稳定身份。")
    file_name: str = Field(min_length=1, description="向用户展示的被引用文件名。")
    page_start: int | None = Field(
        default=None,
        ge=1,
        description="引用范围的首个 1-based PDF 页码；无页码来源使用 None。",
    )
    page_end: int | None = Field(
        default=None,
        ge=1,
        description="引用范围的末个 1-based PDF 页码；无页码来源使用 None。",
    )
    chunk_id: str = Field(min_length=1, description="被引用完整 Chunk 的稳定身份。")

    @model_validator(mode="after")
    def validate_page_range(self) -> "SourceReference":
        """Guarantees the optional page range of a final citation is complete and ordered."""

        # A source without page numbers must leave both ends empty; a PDF citation must provide a
        # complete start and end page.
        if (self.page_start is None) != (self.page_end is None):
            raise ValueError("page_start and page_end must both be set or both be None")
        # A start page later than the end page cannot express a valid source range.
        if self.page_start is not None and self.page_start > self.page_end:
            raise ValueError("page_start must be less than or equal to page_end")
        return self


class RAGAnswer(BaseModel):
    """Structured RAG answer carrying the actual citations plus retrieval trace information."""

    query: str = Field(min_length=1, description="原始用户查询。")
    status: AnswerStatus = Field(
        description="回答状态：answered 表示有证据回答，insufficient_context 表示知识库证据不足（拒答）。"
    )
    answer: str = Field(description="LLM 基于 Context 生成的回答文本。")
    sources: list[SourceReference] = Field(
        description="仅包含最终回答中实际出现标签所对应的来源。"
    )
    retrieved_chunks: list[RetrievedChunk | RerankedRetrievedChunk] = Field(
        description="为评测和调试保留的完整排序检索记录。"
    )
