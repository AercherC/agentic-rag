"""标准 IR 指标（TREC / BEIR 风格），作为 Evaluation v3 的共享实现核心。

设计约束：
- 检索排名单位为 DocumentChunk / chunk_id；查询单位为 question_id。
- 相关性采用二元标注（relevance=1），绝不自行创造 0/1/2/3 分级相关性。
- 所有函数均为纯函数，依赖确定性输入（固定 qrels 与 run），便于公式级单元测试。
- 本模块不调用检索器、Reranker、Embedding 或 Chroma；只做指标计算。

命名澄清（对应 Phase 11A Step 9）：
- 本项目早期 v1 `metrics.calculate_question_metrics` 的 `recall_at_k` 实为 HitRate@K
  （Top-K 任意相关即计 1，否则 0），并非标准 Recall@K。
- 本模块的 `recall_at_k` 采用标准定义：检索到的相关文档数 / 相关文档总数。
- 早期 phase9/10 的 `ndcg_at_5` 实为 `custom_evidence_group_ndcg`，并非标准 nDCG@K。
"""

from collections.abc import Mapping, Sequence

import math


# 标准 IR 指标建议截断集合（对应 Phase 11A Step 3-7）。
STANDARD_DEPTHS: tuple[int, ...] = (1, 3, 5, 10, 20)


def precision_at_k(ranked_ids: Sequence[str], relevant: set[str], k: int) -> float:
    """标准 Precision@K = 检索到的相关文档数 / K（位置截断）。

    K 为固定截断，与语料或 qrels 规模无关。Top-K 内无相关结果时为 0。
    重复 chunk_id 只在其首次出现位置计一次命中（按文档去重，不重复计数）。
    """

    if k <= 0:
        raise ValueError("k must be a positive integer")
    seen: set[str] = set()
    hits = 0
    for rank, cid in enumerate(ranked_ids, 1):
        if rank > k:
            break
        if cid in seen:
            continue
        seen.add(cid)
        if cid in relevant:
            hits += 1
    return hits / k


def recall_at_k(ranked_ids: Sequence[str], relevant: set[str], k: int) -> float:
    """标准 Recall@K = 检索到的相关文档数 / 相关文档总数。

    与早期 v1 的 HitRate@K（任意相关即 1）不同：这里分母是所有相关 chunk 数。
    若 qrels 中该 query 无相关文档，按 0.0 处理（避免 0/0）。
    重复 chunk_id 只计一次命中。
    """

    if k <= 0:
        raise ValueError("k must be a positive integer")
    if not relevant:
        return 0.0
    seen: set[str] = set()
    hits = 0
    for rank, cid in enumerate(ranked_ids, 1):
        if rank > k:
            break
        if cid in seen:
            continue
        seen.add(cid)
        if cid in relevant:
            hits += 1
    return hits / len(relevant)


def reciprocal_rank_at_k(ranked_ids: Sequence[str], relevant: set[str], k: int) -> float:
    """MRR 的单题贡献：Top-K 内第一个相关结果的 1/rank；无相关则为 0。

    rank 为 1-based 名次；重复 chunk_id 不会制造新的首个相关名次。
    """

    if k <= 0:
        raise ValueError("k must be a positive integer")
    seen: set[str] = set()
    for rank, cid in enumerate(ranked_ids, 1):
        if rank > k:
            break
        if cid in seen:
            continue
        seen.add(cid)
        if cid in relevant:
            return 1.0 / rank
    return 0.0


def average_precision_at_k(ranked_ids: Sequence[str], relevant: set[str], k: int) -> float:
    """标准 Average Precision@K：仅在相关位置累积 Precision，按相关总数归一化。

    公式：AP@K = (1 / |rel|) * Σ_{rank=1..K} (P@rank * rel_at_rank)，其中每个相关
    文档只在其首次出现位置贡献 hits/rank。无相关 qrels 时返回 0.0。
    """

    if k <= 0:
        raise ValueError("k must be a positive integer")
    if not relevant:
        return 0.0
    seen: set[str] = set()
    hits = 0
    score = 0.0
    for rank, cid in enumerate(ranked_ids, 1):
        if rank > k:
            break
        if cid in seen:
            continue
        seen.add(cid)
        if cid in relevant:
            hits += 1
            score += hits / rank
    return score / len(relevant)


def ndcg_at_k(ranked_ids: Sequence[str], relevant: set[str], k: int) -> float:
    """标准 nDCG@K（二元相关性）：DCG@K / IDCG@K。

    DCG 采用对数折扣：命中位置 rank 贡献 1 / log2(rank+1)，每个相关文档只计一次。
    IDCG 为理想排序（所有相关文档排在最前）的 DCG，截断到 min(|rel|, K)。
    无相关 qrels 或理想 DCG 为 0 时返回 0.0。
    """

    if k <= 0:
        raise ValueError("k must be a positive integer")
    if not relevant:
        return 0.0
    seen: set[str] = set()
    dcg = 0.0
    for rank, cid in enumerate(ranked_ids, 1):
        if rank > k:
            break
        if cid in seen:
            continue
        seen.add(cid)
        if cid in relevant:
            dcg += 1.0 / math.log2(rank + 1)
    m = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, m + 1))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_ir_run(
    qrels: Mapping[str, Sequence[str]],
    run: Mapping[str, Sequence[str]],
    *,
    depths: Sequence[int] = STANDARD_DEPTHS,
) -> dict:
    """对一组 query 的 qrels 与 run 计算标准 IR 指标聚合与逐题指标。

    qrels: question_id -> 相关 chunk_id 列表（二元相关性，rel=1）。
    run:   question_id -> 排名后的 chunk_id 列表（与 qrels 同查询空间）。

    仅对拥有非空前 qrels 的 eligible query 做聚合，避免 0/0 稀释；
    无 eligible qrels 的 query 单独记录到 excluded_queries。
    """

    if not qrels:
        raise ValueError("qrels must not be empty")
    if not run:
        raise ValueError("run must not be empty")

    sorted_depths = sorted(set(depths))
    per_query: dict[str, dict] = {}
    excluded: list[str] = []

    for qid in run:
        rel = set(qrels.get(qid, []))
        ranked = list(run[qid])
        if not rel:
            excluded.append(qid)
            per_query[qid] = {
                "question_id": qid,
                "eligible": False,
                "num_relevant": 0,
                "metrics": None,
            }
            continue
        metrics: dict[str, float] = {}
        for k in sorted_depths:
            metrics[f"precision_at_{k}"] = precision_at_k(ranked, rel, k)
            metrics[f"recall_at_{k}"] = recall_at_k(ranked, rel, k)
            metrics[f"mrr_at_{k}"] = reciprocal_rank_at_k(ranked, rel, k)
            metrics[f"map_at_{k}"] = average_precision_at_k(ranked, rel, k)
            metrics[f"ndcg_at_{k}"] = ndcg_at_k(ranked, rel, k)
        per_query[qid] = {
            "question_id": qid,
            "eligible": True,
            "num_relevant": len(rel),
            "metrics": metrics,
        }

    eligible_rows = [row for row in per_query.values() if row["eligible"]]
    if not eligible_rows:
        raise ValueError("no query has eligible qrels; cannot aggregate standard IR metrics")

    def mean_at(prefix: str) -> dict[int, float]:
        out: dict[int, float] = {}
        for k in sorted_depths:
            key = f"{prefix}_at_{k}"
            out[k] = sum(row["metrics"][key] for row in eligible_rows) / len(eligible_rows)
        return out

    aggregate = {
        "question_count": len(run),
        "eligible_count": len(eligible_rows),
        "excluded_count": len(excluded),
        "precision_at_k": mean_at("precision"),
        "recall_at_k": mean_at("recall"),
        "mrr_at_k": mean_at("mrr"),
        "map_at_k": mean_at("map"),
        "ndcg_at_k": mean_at("ndcg"),
    }

    return {
        "evaluation_version": "3.0",
        "depths": list(sorted_depths),
        "aggregate": aggregate,
        "per_query": per_query,
        "excluded_queries": excluded,
    }
