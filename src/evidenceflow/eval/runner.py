"""编排 Retriever、指标计算与可重复保存的 Evaluation Report。"""

import json
from collections.abc import Collection, Mapping, Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from evidenceflow.domain.chunk import RetrievedChunk
from evidenceflow.domain.evaluation import EvaluationQuestion, EvaluationSource
from evidenceflow.eval.metrics import (
    AggregateMetrics,
    QuestionMetrics,
    aggregate_question_metrics,
    calculate_question_metrics,
    is_relevant_to_any,
)
from evidenceflow.retrieval.retriever import DenseRetriever


class RetrievedResultSummary(BaseModel):
    """用于失败分析的精简检索结果，不包含 Embedding Vector。"""

    rank: int = Field(ge=1, description="1-based 检索名次。")
    similarity_score: float = Field(description="项目统一的 cosine 相似度。")
    chunk_id: str = Field(min_length=1, description="当前 Chunk 的稳定身份。")
    document_id: str = Field(min_length=1, description="当前 Chunk 所属文档身份。")
    file_name: str = Field(min_length=1, description="来源文件名。")
    page_start: int | None = Field(default=None, description="PDF 起始页码。")
    page_end: int | None = Field(default=None, description="PDF 结束页码。")
    content_preview: str = Field(description="便于人工排查的规范化内容预览。")
    relevant: bool = Field(description="是否匹配当前问题的任一 Gold Source。")


class PerQuestionResult(BaseModel):
    """保存单题 Gold、客观指标及本次 Top-5 检索明细。"""

    question: str
    relevant_sources: list[EvaluationSource]
    first_relevant_rank: int | None
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    reciprocal_rank: float
    retrieved_results: list[RetrievedResultSummary]


class MetricSummary(BaseModel):
    """适合 JSON 序列化的一组聚合 Retrieval Metrics。"""

    question_count: int = Field(gt=0)
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    mrr: float

    @classmethod
    def from_metrics(cls, metrics: AggregateMetrics) -> "MetricSummary":
        """将纯指标层结果转换为报告层模型。"""

        return cls(**metrics.__dict__)


class EvaluationReport(BaseModel):
    """保存配置、总体指标、可靠分组、逐题结果与失败案例。"""

    configuration: dict[str, Any] = Field(default_factory=dict)
    aggregate_metrics: MetricSummary
    breakdowns: dict[str, MetricSummary]
    per_question_results: list[PerQuestionResult]
    failure_cases: list[PerQuestionResult]
    near_miss_cases: list[PerQuestionResult]


def _summarize_retrieved(
    result: RetrievedChunk,
    gold_sources: Sequence[EvaluationSource],
    *,
    preview_length: int,
) -> RetrievedResultSummary:
    """把 Domain 检索结果压缩为可读、可持久化的失败分析记录。"""

    # 合并空白可让 PDF 换行文本在 JSON 中保持紧凑，同时不改变实际检索内容。
    preview = " ".join(result.chunk.content.split())[:preview_length]
    return RetrievedResultSummary(
        rank=result.rank,
        similarity_score=result.similarity_score,
        chunk_id=result.chunk.chunk_id,
        document_id=result.chunk.document_id,
        file_name=result.chunk.file_name,
        page_start=result.chunk.page_start,
        page_end=result.chunk.page_end,
        content_preview=preview,
        relevant=is_relevant_to_any(result, gold_sources),
    )


def _gold_format_group(question: EvaluationQuestion) -> str:
    """依据 Gold 页码合同可靠区分 PDF、Markdown 与混合来源问题。"""

    has_paged = any(source.page_start is not None for source in question.relevant_sources)
    has_unpaged = any(source.page_start is None for source in question.relevant_sources)
    if has_paged and has_unpaged:
        return "mixed_gold"
    return "pdf_gold" if has_paged else "markdown_gold"


class EvaluationRunner:
    """每题只执行一次检索，并汇总固定 Recall@1/3/5 与 MRR。"""

    def __init__(
        self,
        retriever: DenseRetriever,
        top_k: int = 5,
        preview_length: int = 240,
    ) -> None:
        """注入现有 Retriever，并固定本阶段唯一允许的 Top-5 检索深度。"""

        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k != 5:
            raise ValueError("top_k must be 5 for the current evaluation contract")
        if (
            isinstance(preview_length, bool)
            or not isinstance(preview_length, int)
            or preview_length <= 0
        ):
            raise ValueError("preview_length must be a positive integer")
        self.retriever = retriever
        self.top_k = top_k
        self.preview_length = preview_length

    def run(
        self,
        questions: Sequence[EvaluationQuestion],
        *,
        configuration: Mapping[str, Any] | None = None,
        question_groups: Mapping[str, Collection[int]] | None = None,
    ) -> EvaluationReport:
        """遍历非空问题集，计算总体、来源格式及显式辅助分组指标。"""

        question_list = list(questions)
        if not question_list:
            raise ValueError("questions must not be empty")

        per_question: list[PerQuestionResult] = []
        raw_metrics: list[QuestionMetrics] = []
        for question in question_list:
            # 每题只执行一次 Top-K，所有 @K 指标均复用这组结果。
            retrieved = self.retriever.retrieve(question.question, top_k=self.top_k)
            metrics = calculate_question_metrics(retrieved, question.relevant_sources)
            raw_metrics.append(metrics)
            per_question.append(
                PerQuestionResult(
                    question=question.question,
                    relevant_sources=question.relevant_sources,
                    first_relevant_rank=metrics.first_relevant_rank,
                    recall_at_1=metrics.recall_at_1,
                    recall_at_3=metrics.recall_at_3,
                    recall_at_5=metrics.recall_at_5,
                    reciprocal_rank=metrics.reciprocal_rank,
                    retrieved_results=[
                        _summarize_retrieved(
                            result,
                            question.relevant_sources,
                            preview_length=self.preview_length,
                        )
                        for result in retrieved
                        if result.rank <= self.top_k
                    ],
                )
            )

        breakdown_indices: dict[str, set[int]] = {}
        for index, question in enumerate(question_list):
            breakdown_indices.setdefault(_gold_format_group(question), set()).add(index)

        # 语言与跨语言分组只能由人工显式映射提供，Runner 不通过字符特征猜测。
        if question_groups:
            for name, indices in question_groups.items():
                if not isinstance(name, str) or not name.strip():
                    raise ValueError("question group names must not be empty")
                normalized = set(indices)
                if not normalized:
                    raise ValueError(f"question group {name!r} must not be empty")
                if any(
                    isinstance(index, bool)
                    or not isinstance(index, int)
                    or index < 0
                    or index >= len(question_list)
                    for index in normalized
                ):
                    raise ValueError(f"question group {name!r} contains an invalid index")
                breakdown_indices[name] = normalized

        breakdowns = {
            name: MetricSummary.from_metrics(
                aggregate_question_metrics([raw_metrics[index] for index in sorted(indices)])
            )
            for name, indices in breakdown_indices.items()
        }
        failures = [result for result in per_question if result.recall_at_5 == 0.0]
        near_misses = [
            result
            for result in per_question
            if result.first_relevant_rank is not None
            and result.first_relevant_rank >= 3
        ]
        return EvaluationReport(
            configuration=dict(configuration or {}),
            aggregate_metrics=MetricSummary.from_metrics(
                aggregate_question_metrics(raw_metrics)
            ),
            breakdowns=breakdowns,
            per_question_results=per_question,
            failure_cases=failures,
            near_miss_cases=near_misses,
        )


def save_evaluation_report(
    report: EvaluationReport,
    path: Path,
    *,
    overwrite: bool = False,
) -> None:
    """以 UTF-8 JSON 保存报告，默认拒绝静默覆盖既有实验。"""

    output_path = Path(path)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Evaluation report already exists: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # ensure_ascii=False 保留中英文问题可读性；缩进便于人工审查失败案例。
    payload = report.model_dump(mode="json")
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
