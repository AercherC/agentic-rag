"""LLM Context domain models constrained by the Token budget."""

from pydantic import BaseModel, Field, model_validator

from evidenceflow.domain.answer import SourceReference


class ContextEntry(BaseModel):
    """Records one Chunk that was fully included in the Context together with its source mapping."""

    source_label: str = Field(
        pattern=r"^\[S[1-9]\d*\]$",
        description="按实际纳入顺序分配的连续临时来源标签。",
    )
    retrieval_rank: int = Field(ge=1, description="该 Chunk 的原始融合检索名次。")
    chunk_id: str = Field(min_length=1, description="被完整纳入的 Chunk 稳定身份。")
    content: str = Field(description="未经截断或改写的完整 Chunk 正文。")
    source_reference: SourceReference = Field(
        description="与当前标签及 Chunk 一一对应的精简来源映射。"
    )
    token_count: int = Field(
        ge=1,
        description="标签、来源头和正文组成的单个完整渲染条目 Token 数。",
    )

    @model_validator(mode="after")
    def validate_source_mapping(self) -> "ContextEntry":
        """Guarantees entry identity, label, and public source mapping do not drift."""

        if self.source_reference.source_label != self.source_label:
            raise ValueError("source_reference label must match context entry label")
        if self.source_reference.chunk_id != self.chunk_id:
            raise ValueError("source_reference chunk_id must match context entry chunk_id")
        return self


class BuiltContext(BaseModel):
    """Context that can be handed straight to the Generator, plus deterministic statistics."""

    context_text: str = Field(description="由已纳入完整条目按顺序拼接的 LLM Context。")
    entries: list[ContextEntry] = Field(description="按实际纳入顺序排列的来源条目。")
    total_tokens: int = Field(
        ge=0,
        description="完整 context_text 使用注入 Tokenizer 计算的实际 Token 数。",
    )
    candidate_count: int = Field(ge=0, description="本次检查的检索候选总数。")
    included_count: int = Field(ge=0, description="完整纳入 Context 的候选数。")
    skipped_count: int = Field(ge=0, description="因超预算而跳过的候选数。")

    @model_validator(mode="after")
    def validate_counts(self) -> "BuiltContext":
        """Guarantees the aggregate counters agree with the actual number of entries."""

        if self.included_count != len(self.entries):
            raise ValueError("included_count must match entries length")
        if self.candidate_count != self.included_count + self.skipped_count:
            raise ValueError("candidate_count must equal included_count plus skipped_count")
        if not self.entries and (self.context_text or self.total_tokens != 0):
            raise ValueError("empty entries require empty context_text and zero total_tokens")
        return self
