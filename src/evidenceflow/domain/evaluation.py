"""Gold Source Models used by the Retrieval Evaluation Dataset.

COMPATIBILITY_ONLY: legacy HitRate semantics; new code uses eval/ir_metrics.py.
"""

from pydantic import BaseModel, Field, field_validator, model_validator


class EvaluationSource(BaseModel):
    """A Gold document plus its pages, text anchor, or historical document-level locator."""

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
        """Reject an empty anchor while preserving original whitespace for deterministic normalization."""

        if value is not None and not value.strip():
            raise ValueError("reference_text must not be empty or whitespace-only")
        return value

    @model_validator(mode="after")
    def validate_page_range(self) -> "EvaluationSource":
        """Guarantees the Gold range is complete and ordered when evaluating at page level."""

        # When both ends are empty, match by reference_text or the historical document_id;
        # otherwise the range must be complete.
        if (self.page_start is None) != (self.page_end is None):
            raise ValueError("page_start and page_end must both be set or both be None")
        # A valid Gold range must point from the smaller page to the larger page.
        if self.page_start is not None and self.page_start > self.page_end:
            raise ValueError("page_start must be less than or equal to page_end")
        # Page numbers and text anchors are two different locating semantics; combining them would
        # create ambiguous AND/OR behaviour.
        if self.page_start is not None and self.reference_text is not None:
            raise ValueError("reference_text cannot be combined with a page range")
        return self


class EvaluationQuestion(BaseModel):
    """One retrieval question and its one or more Gold Sources."""

    question: str = Field(min_length=1, description="送入 Retriever 的评测问题。")
    relevant_sources: list[EvaluationSource] = Field(
        min_length=1,
        description="当前问题可判定为 relevant 的 Gold Sources。",
    )
