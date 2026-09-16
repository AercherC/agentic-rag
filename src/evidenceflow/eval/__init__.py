"""提供 Retrieval Evaluation Dataset、Metrics 与 Runner 公共接口。"""

from evidenceflow.eval.dataset import EvaluationDatasetError, load_evaluation_questions
from evidenceflow.eval.metrics import (
    AggregateMetrics,
    QuestionMetrics,
    aggregate_question_metrics,
    calculate_question_metrics,
    is_relevant,
    is_relevant_to_any,
)
from evidenceflow.eval.gold import (
    aggregate_hits_at_depths,
    gold_chunk_hit,
    gold_chunk_rank,
    hits_at_depths,
    is_hit_at,
)
from evidenceflow.eval.runner import (
    EvaluationReport,
    EvaluationRunner,
    MetricSummary,
    PerQuestionResult,
    RetrievedResultSummary,
    save_evaluation_report,
)

__all__ = [
    "AggregateMetrics",
    "EvaluationDatasetError",
    "EvaluationReport",
    "EvaluationRunner",
    "MetricSummary",
    "PerQuestionResult",
    "QuestionMetrics",
    "RetrievedResultSummary",
    "aggregate_hits_at_depths",
    "aggregate_question_metrics",
    "calculate_question_metrics",
    "gold_chunk_hit",
    "gold_chunk_rank",
    "hits_at_depths",
    "is_hit_at",
    "is_relevant",
    "is_relevant_to_any",
    "load_evaluation_questions",
    "save_evaluation_report",
]
