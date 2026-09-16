"""Citation Parser 的确定性分类结果模型。"""

import re

from pydantic import BaseModel, Field, field_validator, model_validator

_CITATION_LABEL_PATTERN = r"^\[S(?:0|[1-9]\d*)\]$"


class CitationParseResult(BaseModel):
    """记录按答案首次出现顺序去重后的合法与未知 Citation 标签。"""

    valid_labels: tuple[str, ...] = Field(
        default=(),
        description="存在于 BuiltContext 的正式 Citation 标签。",
    )
    invalid_labels: tuple[str, ...] = Field(
        default=(),
        description="格式正确但不存在于 BuiltContext 的 Citation 标签。",
    )

    @field_validator("valid_labels", "invalid_labels")
    @classmethod
    def validate_labels(cls, labels: tuple[str, ...]) -> tuple[str, ...]:
        """保证结果只包含唯一、格式规范的方括号标签。"""

        if len(labels) != len(set(labels)):
            raise ValueError("citation labels must be unique")
        if any(re.fullmatch(_CITATION_LABEL_PATTERN, label) is None for label in labels):
            raise ValueError("citation labels must use the [S<number>] syntax")
        return labels

    @model_validator(mode="after")
    def validate_disjoint_groups(self) -> "CitationParseResult":
        """同一标签不能同时被分类为合法和无效。"""

        if set(self.valid_labels) & set(self.invalid_labels):
            raise ValueError("valid and invalid citation labels must be disjoint")
        return self
