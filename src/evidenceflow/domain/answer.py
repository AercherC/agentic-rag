"""生成回答及其实际引用来源的 Domain Models。"""

from pydantic import BaseModel, Field, model_validator

from evidenceflow.domain.chunk import (
    RerankedRetrievedChunk,
    RetrievedChunk,
)


class SourceReference(BaseModel):
    """最终回答实际引用的精简来源对象。"""

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
        """保证最终引用的可选页码范围完整且有序。"""

        # 无页码来源必须两端同时为空；PDF 引用必须提供完整起止页码。
        if (self.page_start is None) != (self.page_end is None):
            raise ValueError("page_start and page_end must both be set or both be None")
        # 起始页晚于结束页时无法表示有效的来源范围。
        if self.page_start is not None and self.page_start > self.page_end:
            raise ValueError("page_start must be less than or equal to page_end")
        return self


class RAGAnswer(BaseModel):
    """包含实际引用与检索追踪信息的结构化 RAG 回答。"""

    query: str = Field(min_length=1, description="原始用户查询。")
    answer: str = Field(description="LLM 基于 Context 生成的回答文本。")
    sources: list[SourceReference] = Field(
        description="仅包含最终回答中实际出现标签所对应的来源。"
    )
    retrieved_chunks: list[RetrievedChunk | RerankedRetrievedChunk] = Field(
        description="为评测和调试保留的完整排序检索记录。"
    )
