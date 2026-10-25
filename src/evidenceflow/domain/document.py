"""Domain Models describing loaded documents and their source segments."""

from typing import Any, Literal

from pydantic import BaseModel, Field


class DocumentSegment(BaseModel):
    """A text unit that still carries the original page boundaries before entering the Chunker."""

    content: str = Field(description="从当前来源片段提取的文本内容。")
    page_number: int | None = Field(
        default=None,
        ge=1,
        description="PDF 的 1-based 页码；Markdown 和 TXT 使用 None。",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Loader 随片段保留的可选来源元数据。",
    )


class Document(BaseModel):
    """A logical source document with a stable path-based identity."""

    document_id: str = Field(
        min_length=1,
        description="由规范化相对路径生成的稳定 SHA-256 文档身份。",
    )
    source_path: str = Field(
        min_length=1,
        description="相对于 ingestion root 的文档路径。",
    )
    file_name: str = Field(min_length=1, description="包含扩展名的来源文件名。")
    file_type: Literal["pdf", "md", "txt"] = Field(
        description="不带前导点号的受支持文件类型。"
    )
    content_hash: str = Field(
        min_length=1,
        description="原始文件 bytes 的 SHA-256，用于检测内容变化。",
    )
    segments: list[DocumentSegment] = Field(
        description="按原文顺序排列的来源片段；PDF 片段保留页面边界。"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="可选的文档级来源元数据。",
    )
