"""定义与 Chunk 策略解耦的 Evaluation v2 真值和映射合同。"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ContractModel(BaseModel):
    """拒绝未声明字段，防止版本迁移时悄悄丢弃标注。"""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class EvidenceLocator(ContractModel):
    """定位页码或有序 Segment，字符范围为块内半开区间。"""

    page_start: int | None = Field(default=None, ge=1, description="1-based 起始页。")
    page_end: int | None = Field(default=None, ge=1, description="1-based 结束页。")
    block_index: int | None = Field(default=None, ge=0, description="0-based Segment 序号。")
    span_start: int | None = Field(default=None, ge=0, description="块内起始字符偏移。")
    span_end: int | None = Field(default=None, ge=1, description="块内结束字符偏移，不包含该位置。")
    left_context: str | None = Field(default=None, min_length=1, description="紧邻 anchor 左侧的来源上下文。")
    right_context: str | None = Field(default=None, min_length=1, description="紧邻 anchor 右侧的来源上下文。")
    occurrence: int | None = Field(default=None, ge=1, description="原 locator 范围内 anchor 的 1-based 序号。")
    source_span_start: int | None = Field(default=None, ge=0, description="规范化 locator 文本中的辅助起始偏移。")
    source_span_end: int | None = Field(default=None, ge=1, description="规范化 locator 文本中的辅助结束偏移。")

    @model_validator(mode="after")
    def validate_location(self):
        """拒绝不完整或相互冲突的定位范围。"""
        if (self.page_start is None) != (self.page_end is None):
            raise ValueError("page endpoints must be paired")
        if self.page_start is not None and self.page_start > self.page_end:
            raise ValueError("page range is reversed")
        if (self.page_start is None) == (self.block_index is None):
            raise ValueError("choose page range or block_index")
        if (self.span_start is None) != (self.span_end is None):
            raise ValueError("span endpoints must be paired")
        if self.span_start is not None:
            if self.block_index is None or self.span_start >= self.span_end:
                raise ValueError("span requires a block and positive length")
        if (self.source_span_start is None) != (self.source_span_end is None):
            raise ValueError("source span endpoints must be paired")
        if self.source_span_start is not None and self.source_span_start >= self.source_span_end:
            raise ValueError("source span is reversed")
        return self


class EvidenceAlternative(ContractModel):
    """同组内可独立支持所需事实的一种来源。"""

    document_id: str = Field(min_length=1, description="稳定文档身份。")
    content_version: str = Field(min_length=1, description="原始文件 content_hash。")
    locator: EvidenceLocator = Field(description="原始文档定位。")
    evidence_anchor: str = Field(min_length=1, description="连续原文锚点，不绑定 Chunk。")


class EvidenceGroup(ContractModel):
    """必需证据组；alternatives 为 OR，不同组为 AND。"""

    group_id: str = Field(min_length=1, description="题内唯一证据组 ID。")
    alternatives: list[EvidenceAlternative] = Field(min_length=1, description="可互换的来源。")


class EvaluationQuestionV2(ContractModel):
    """保存问题、切片和独立标注真值。"""

    question_id: str = Field(min_length=1, description="数据集内唯一 ID。")
    family_id: str = Field(min_length=1, description="平行改写族；不能跨 split。")
    query: str = Field(min_length=1, description="原始检索问题。")
    query_language: str = Field(min_length=1, description="查询语言。")
    corpus_language: str = Field(min_length=1, description="目标语料语言。")
    question_type: str = Field(min_length=1, description="题型切片。")
    evidence_type: str = Field(min_length=1, description="正文、表格等证据切片。")
    answerability: Literal["answerable", "unanswerable"] = Field(description="人工可回答性标注。")
    required_evidence_groups: list[EvidenceGroup] = Field(default_factory=list, description="必需证据组。")
    reference_claims: list[str] = Field(default_factory=list, description="已标注原子事实；空表示尚未标注。")
    reference_answer: str | None = Field(default=None, description="可选参考答案。")

    @model_validator(mode="after")
    def validate_gold(self):
        """可回答题必须有证据；不可回答题不伪造正例。"""
        if bool(self.required_evidence_groups) != (self.answerability == "answerable"):
            raise ValueError("answerability and required groups disagree")
        ids = [g.group_id for g in self.required_evidence_groups]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate group_id")
        if any(not claim.strip() for claim in self.reference_claims):
            raise ValueError("reference claims must not be blank")
        return self


class EvaluationDatasetV2(ContractModel):
    """冻结同一 split 与语料版本的问题集合。"""

    schema_version: Literal["2.0", "2.1"] = "2.0"
    dataset_id: str = Field(min_length=1, description="数据集版本身份。")
    split: Literal["legacy_dev_diagnostic", "development", "locked_holdout"]
    corpus_version: str = Field(min_length=1, description="原始语料版本或 manifest hash。")
    questions: list[EvaluationQuestionV2] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_ids(self):
        """拒绝重复问题，确保配对比较不会错位。"""
        ids = [q.question_id for q in self.questions]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate question_id")
        return self


MappingStatus = Literal[
    "FULLY_MAPPED", "FRAGMENTED_ACROSS_CHUNKS", "PARTIALLY_MAPPED", "UNMAPPED",
    "SOURCE_EXTRACTION_FAILURE", "ANCHOR_NORMALIZATION_FAILURE", "AMBIGUOUS_MAPPING",
]


class GroupMapping(ContractModel):
    """保存组的候选证据集合；外层 OR、内层 AND。"""

    question_id: str
    group_id: str
    status: MappingStatus
    chunk_sets: list[list[str]] = Field(default_factory=list, description="每个集合都足以完整覆盖本组。")
    reason: str = Field(description="基于原文和索引的诊断，不猜测因果。")

    @model_validator(mode="after")
    def validate_sets(self):
        """未解决映射不得携带可评分的伪 Gold。"""
        mapped = self.status in {"FULLY_MAPPED", "FRAGMENTED_ACROSS_CHUNKS"}
        if mapped != bool(self.chunk_sets):
            raise ValueError("mapping status and chunk_sets disagree")
        if any(not bundle or len(bundle) != len(set(bundle)) for bundle in self.chunk_sets):
            raise ValueError("chunk bundles must be non-empty and unique")
        return self


class MappingArtifact(ContractModel):
    """把映射绑定到数据、语料与索引版本。"""

    mapping_version: str = "gold-mapper-v2"
    dataset_id: str
    corpus_version: str
    index_version: str
    normalization_version: str = "research-anchor-v1"
    groups: list[GroupMapping]

    @model_validator(mode="after")
    def validate_group_identity(self):
        """重复映射不能覆盖另一条记录。"""
        keys = [(g.question_id, g.group_id) for g in self.groups]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate mapping group")
        return self


def validate_dataset_splits(datasets: list[EvaluationDatasetV2]) -> None:
    """检查题族泄漏；不推断文本相似性或自动拆分人工题族。"""
    splits: dict[str, str] = {}
    for dataset in datasets:
        for question in dataset.questions:
            previous = splits.setdefault(question.family_id, dataset.split)
            if previous != dataset.split:
                raise ValueError(f"family crosses splits: {question.family_id}")
