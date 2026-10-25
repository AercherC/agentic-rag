"""Retrieval Gold matching, per-question metrics, and aggregate metric computation.

COMPATIBILITY_ONLY: legacy HitRate semantics; new code uses eval/ir_metrics.py.
"""

from dataclasses import dataclass
from collections.abc import Sequence
import re
import unicodedata
import math

from evidenceflow.domain.chunk import RetrievedChunk
from evidenceflow.domain.evaluation import EvaluationSource


@dataclass(frozen=True)
class QuestionMetrics:
    """Objective retrieval metrics of one question over the fixed Top-5 results."""

    first_relevant_rank: int | None
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    reciprocal_rank: float


@dataclass(frozen=True)
class AggregateMetrics:
    """Question count, mean Recall@K, and MRR of a question set."""

    question_count: int
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    mrr: float


def normalize_gold_text(text: str) -> str:
    """Normalize Gold anchors and candidate text with the minimal deterministic rules."""

    # NFC plus newline unification remove cross-platform differences first, then all runs of
    # whitespace are collapsed.
    normalized = unicodedata.normalize("NFC", text)
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    return re.sub(r"\s+", " ", normalized).strip()


def is_relevant(retrieved: RetrievedChunk, gold: EvaluationSource) -> bool:
    """Decide relevance from document identity plus pages or the text anchor."""

    if retrieved.chunk.document_id != gold.document_id:
        return False

    if gold.reference_text is not None:
        # Fine-grained Gold must hit the actual answer text; other Chunks of the same document are
        # no longer misjudged.
        anchor = normalize_gold_text(gold.reference_text)
        content = normalize_gold_text(retrieved.chunk.content)
        return anchor in content

    # No pages and no anchor is the historical Markdown/TXT contract, so document identity still
    # matches.
    if gold.page_start is None:
        return True

    retrieved_start = retrieved.chunk.page_start
    retrieved_end = retrieved.chunk.page_end
    # PDF Gold must match real page numbers; missing metadata must not degrade into a
    # document-level hit.
    if retrieved_start is None or retrieved_end is None:
        return False

    # Two closed intervals intersect exactly when neither one lies entirely outside the other.
    return retrieved_start <= gold.page_end and retrieved_end >= gold.page_start


def is_relevant_to_any(
    retrieved: RetrievedChunk,
    gold_sources: Sequence[EvaluationSource],
) -> bool:
    """Decide whether a retrieval result matches any human-annotated Gold Source of the question."""

    return any(is_relevant(retrieved, gold) for gold in gold_sources)


def calculate_question_metrics(
    retrieved_results: Sequence[RetrievedChunk],
    gold_sources: Sequence[EvaluationSource],
) -> QuestionMetrics:
    """Compute First Rank, Recall@K, and RR from one Top-5 retrieval result."""

    if not gold_sources:
        raise ValueError("gold_sources must not be empty")

    # Use the explicit 1-based rank from the Domain Model, never rely on the input list happening
    # to be sorted.
    relevant_ranks = [
        result.rank
        for result in retrieved_results
        if result.rank <= 5 and is_relevant_to_any(result, gold_sources)
    ]
    first_rank = min(relevant_ranks, default=None)

    def hit_at(k: int) -> float:
        """Convert "Top-K hit at least one Gold" into an aggregatable 0/1 float."""

        return float(first_rank is not None and first_rank <= k)

    reciprocal_rank = 0.0 if first_rank is None else 1.0 / first_rank
    return QuestionMetrics(
        first_relevant_rank=first_rank,
        recall_at_1=hit_at(1),
        recall_at_3=hit_at(3),
        recall_at_5=hit_at(5),
        reciprocal_rank=reciprocal_rank,
    )


def aggregate_question_metrics(
    metrics: Sequence[QuestionMetrics],
) -> AggregateMetrics:
    """Take the arithmetic mean over a non-empty question set to obtain Recall@K and MRR."""

    if not metrics:
        raise ValueError("metrics must not be empty")

    count = len(metrics)
    return AggregateMetrics(
        question_count=count,
        recall_at_1=sum(item.recall_at_1 for item in metrics) / count,
        recall_at_3=sum(item.recall_at_3 for item in metrics) / count,
        recall_at_5=sum(item.recall_at_5 for item in metrics) / count,
        mrr=sum(item.reciprocal_rank for item in metrics) / count,
    )


def custom_evidence_group_ndcg(ranked_ids, groups, depth: int) -> float:
    """Keep the research v1 CUSTOM first-group gain; each rank contributes at most one new group."""
    return custom_bundle_ndcg(ranked_ids, [[[cid] for cid in group] for group in groups], depth)


def custom_bundle_ndcg(ranked_ids, groups, depth: int) -> float:
    """Shared CUSTOM gain core, supporting single-block alternatives and cross-block AND sets."""
    if type(depth) is not int or depth <= 0:
        raise ValueError("depth must be positive")
    seen = set()
    prefix = set()
    dcg = 0.0
    for rank, chunk_id in enumerate(ranked_ids[:depth], 1):
        prefix.add(chunk_id)
        for index, group in enumerate(groups):
            if index not in seen and any(chunk_id in bundle and set(bundle) <= prefix for bundle in group):
                dcg += 1.0 / math.log2(rank + 1)
                seen.add(index)
                break
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(len(groups), depth) + 1))
    return dcg / ideal if ideal else 0.0


def legacy_research_metrics(rows, depths=(1, 3, 5, 10, 20)) -> dict:
    """Historical report adapter: Hit/MRR still mean complete-group coverage and must not name v2 fields."""
    count = len(rows)
    if not count:
        return {"N": 0}
    result = {"N": count}
    for depth in depths:
        result[f"hit_at_{depth}"] = sum(
            row["gold_group_rank"] is not None and row["gold_group_rank"] <= depth for row in rows
        ) / count
        result[f"ndcg_at_{depth}"] = sum(row[f"ndcg_at_{depth}"] for row in rows) / count
    result["mrr"] = sum(0.0 if row["gold_group_rank"] is None else 1.0 / row["gold_group_rank"]
                        for row in rows) / count
    return result
