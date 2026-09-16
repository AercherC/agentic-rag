"""Retrieval Evaluation Dataset 使用的 Gold Source Models。"""

from pydantic import BaseModel, Field, field_validator, model_validator


class EvaluationSource(BaseModel):
    """Gold 文档及其页码、原文锚点或历史文档级定位信息。"""

    document_id: str = Field(min_length=1, description="Gold 相关文档的稳定身份。")
    page_start: int | None = Field(
        default=None,
        ge=1,
        description="Gold 相关范围首个 1-based 页码；None 表示按文档匹配。",
    )
    page_end: int | None = Field(
        default=None,
        ge=1,
        description="Gold 相关范围末个 1-based 页码；None 表示按文档匹配。",
    )
    reference_text: str | None = Field(
        default=None,
        description="Markdown/TXT 的连续 Source 原文锚点；None 保留历史文档级匹配。",
    )

    @field_validator("reference_text")
    @classmethod
    def validate_reference_text(cls, value: str | None) -> str | None:
        """拒绝空锚点，同时保留原文空白供确定性规范化处理。"""

        if value is not None and not value.strip():
            raise ValueError("reference_text must not be empty or whitespace-only")
        return value

    @model_validator(mode="after")
    def validate_page_range(self) -> "EvaluationSource":
        """在页码级评测时保证 Gold 范围完整且有序。"""

        # 两端均为空时使用 reference_text 或历史 document_id 匹配；否则必须完整。
        if (self.page_start is None) != (self.page_end is None):
            raise ValueError("page_start and page_end must both be set or both be None")
        # 有效 Gold 范围必须从较小页码指向较大页码。
        if self.page_start is not None and self.page_start > self.page_end:
            raise ValueError("page_start must be less than or equal to page_end")
        # 页码和文本锚点是两套不同定位语义，禁止组合后产生不明确的 AND/OR 行为。
        if self.page_start is not None and self.reference_text is not None:
            raise ValueError("reference_text cannot be combined with a page range")
        return self


class EvaluationQuestion(BaseModel):
    """一个检索问题及其一个或多个 Gold Sources。"""

    question: str = Field(min_length=1, description="送入 Retriever 的评测问题。")
    relevant_sources: list[EvaluationSource] = Field(
        min_length=1,
        description="当前问题可判定为 relevant 的 Gold Sources。",
    )
