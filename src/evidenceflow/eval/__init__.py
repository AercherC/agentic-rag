"""提供 Retrieval Evaluation Dataset 与 Metrics 公共接口。

legacy ``eval/runner.py``（EvaluationRunner 等）已在 Stage A A4 删除，可从 tag ``stage-a-start`` 恢复。
"""

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

__all__ = [
    "AggregateMetrics",
    "EvaluationDatasetError",
    "QuestionMetrics",
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
]
