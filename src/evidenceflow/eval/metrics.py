"""实现 Retrieval Gold 匹配、单题指标与聚合指标计算。

COMPATIBILITY_ONLY：legacy HitRate 语义，新代码用 eval/ir_metrics.py。
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
    """保存一道问题在固定 Top-5 结果上的客观检索指标。"""

    first_relevant_rank: int | None
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    reciprocal_rank: float


@dataclass(frozen=True)
class AggregateMetrics:
    """保存一组问题的数量、平均 Recall@K 与 MRR。"""

    question_count: int
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    mrr: float


def normalize_gold_text(text: str) -> str:
    """以最小确定性规则规范化 Gold 锚点与待匹配正文。"""

    # NFC 与统一换行先消除跨平台表示差异，再折叠所有连续 whitespace。
    normalized = unicodedata.normalize("NFC", text)
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    return re.sub(r"\s+", " ", normalized).strip()


def is_relevant(retrieved: RetrievedChunk, gold: EvaluationSource) -> bool:
    """按文档身份及页码或原文锚点判断检索结果是否相关。"""

    if retrieved.chunk.document_id != gold.document_id:
        return False

    if gold.reference_text is not None:
        # Fine-grained Gold 必须命中真正答案文本；同文档的其他 Chunk 不再误判。
        anchor = normalize_gold_text(gold.reference_text)
        content = normalize_gold_text(retrieved.chunk.content)
        return anchor in content

    # 无页码、无锚点是历史 Markdown/TXT 合同，仍按文档身份命中。
    if gold.page_start is None:
        return True

    retrieved_start = retrieved.chunk.page_start
    retrieved_end = retrieved.chunk.page_end
    # PDF Gold 必须匹配真实页码；元数据缺失时不能退化为文档级命中。
    if retrieved_start is None or retrieved_end is None:
        return False

    # 两个闭区间有交集，当且仅当任一区间没有完全落在另一区间外侧。
    return retrieved_start <= gold.page_end and retrieved_end >= gold.page_start


def is_relevant_to_any(
    retrieved: RetrievedChunk,
    gold_sources: Sequence[EvaluationSource],
) -> bool:
    """判断检索结果是否匹配问题的任意一个人工 Gold Source。"""

    return any(is_relevant(retrieved, gold) for gold in gold_sources)


def calculate_question_metrics(
    retrieved_results: Sequence[RetrievedChunk],
    gold_sources: Sequence[EvaluationSource],
) -> QuestionMetrics:
    """从一次 Top-5 检索结果计算 First Rank、Recall@K 与 RR。"""

    if not gold_sources:
        raise ValueError("gold_sources must not be empty")

    # 使用 Domain Model 中的显式 1-based rank，避免依赖输入列表恰好已经排序。
    relevant_ranks = [
        result.rank
        for result in retrieved_results
        if result.rank <= 5 and is_relevant_to_any(result, gold_sources)
    ]
    first_rank = min(relevant_ranks, default=None)

    def hit_at(k: int) -> float:
        """将 Top-K 是否至少命中一个 Gold 转换为可聚合的 0/1 浮点值。"""

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
    """对非空问题集合取算术平均，得到 Recall@K 和 MRR。"""

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
    """保留 research v1 的 CUSTOM 首次组增益；每个排名最多贡献一个新组。"""
    return custom_bundle_ndcg(ranked_ids, [[[cid] for cid in group] for group in groups], depth)


def custom_bundle_ndcg(ranked_ids, groups, depth: int) -> float:
    """共享 CUSTOM 增益核心，支持单块替代与跨块 AND 集合。"""
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
    """历史报告适配器：Hit/MRR 仍表示完整组覆盖，禁止用于 v2 字段命名。"""
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
