"""Public interfaces for the Retrieval Evaluation Dataset and Metrics.

The legacy ``eval/runner.py`` (EvaluationRunner and friends) was removed in Stage A A4
and can be restored from tag ``stage-a-start``.
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
