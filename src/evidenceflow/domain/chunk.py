"""文档 Chunk 与检索结果的 Domain Models。"""

from typing import Any

from pydantic import BaseModel, Field, model_validator


class DocumentChunk(BaseModel):
    """受 Token 数限制且可追踪来源页码的文档片段。"""

    chunk_id: str = Field(
        min_length=1,
        description="由文档、Chunk 配置、位置和内容共同确定的稳定身份。",
    )
    document_id: str = Field(min_length=1, description="所属逻辑文档的稳定身份。")
    content: str = Field(description="当前 Chunk 表示的完整文本。")
    token_count: int = Field(
        ge=0,
        description="Chunk 内容按 Embedding Tokenizer 计算的 Token 数。",
    )
    chunk_index: int = Field(
        ge=0,
        description="Chunk 在所属文档中的 0-based 顺序位置。",
    )
    page_start: int | None = Field(
        default=None,
        ge=1,
        description="覆盖的首个 1-based PDF 页码；Markdown 和 TXT 使用 None。",
    )
    page_end: int | None = Field(
        default=None,
        ge=1,
        description="覆盖的末个 1-based PDF 页码；Markdown 和 TXT 使用 None。",
    )
    file_name: str = Field(min_length=1, description="所属来源文档的文件名。")
    content_hash: str = Field(
        min_length=1,
        description="生成当前 Chunk 的父文件版本 Hash。",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="可选的来源信息及 Chunk 切分元数据。",
    )

    @model_validator(mode="after")
    def validate_page_range(self) -> "DocumentChunk":
        """保证页码要么是完整有序范围，要么整体为空。"""

        # Markdown/TXT 两端都为 None；PDF 两端必须同时存在，避免半截页码范围。
        if (self.page_start is None) != (self.page_end is None):
            raise ValueError("page_start and page_end must both be set or both be None")
        # PDF 使用用户可读的正向页码范围，跨页 Chunk 也必须从小到大记录。
        if self.page_start is not None and self.page_start > self.page_end:
            raise ValueError("page_start must be less than or equal to page_end")
        return self


class RetrievedChunk(BaseModel):
    """某个 DocumentChunk 针对一次查询的排序结果。"""

    chunk: DocumentChunk = Field(description="检索返回的项目内部 DocumentChunk。")
    rank: int = Field(ge=1, description="1-based 检索名次，1 表示最佳匹配。")
    similarity_score: float = Field(
        allow_inf_nan=False,
        description="Cosine 相似度，数值越大表示语义越相似。",
    )


class RerankedRetrievedChunk(BaseModel):
    """保存 Dense 候选证据与 Learned Reranker 最终排序。"""

    chunk: DocumentChunk = Field(description="重排候选对应的内部 DocumentChunk。")
    rank: int = Field(ge=1, description="Reranker 输出的 1-based 最终名次。")
    candidate_rank: int = Field(
        ge=1,
        description="候选进入 Reranker 前的 1-based Dense 候选名次。",
    )
    similarity_score: float = Field(
        allow_inf_nan=False,
        description="Original Dense 返回的 cosine 相似度；不得被 Reranker 分数覆盖。",
    )
    reranker_score: float = Field(
        allow_inf_nan=False,
        description="Cross-Encoder relevance score；只用于最终重排。",
    )

