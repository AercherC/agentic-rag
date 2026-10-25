"""Standard IR metrics (TREC / BEIR style), the shared implementation core of Evaluation v3.

Design constraints:
- the retrieval ranking unit is DocumentChunk / chunk_id; the query unit is question_id.
- relevance is binary (relevance=1); graded 0/1/2/3 relevance is never invented.
- every function is pure and depends on deterministic inputs (fixed qrels and run), which keeps the
  formulas unit-testable.
- this module calls no retriever, Reranker, Embedding or Chroma; it only computes metrics.

Naming clarification (corresponding to Phase 11A Step 9):
- in this project's early v1 `metrics.calculate_question_metrics`, `recall_at_k` was really
  HitRate@K (1 if any relevant item is in the Top-K, otherwise 0), not standard Recall@K.
- `recall_at_k` in this module uses the standard definition: retrieved relevant documents / total
  relevant documents.
- the early phase9/10 `ndcg_at_5` was really `custom_evidence_group_ndcg`, not standard nDCG@K.
"""

from collections.abc import Mapping, Sequence

import math


# Suggested truncation depths for standard IR metrics (corresponding to Phase 11A Step 3-7).
STANDARD_DEPTHS: tuple[int, ...] = (1, 3, 5, 10, 20)


def precision_at_k(ranked_ids: Sequence[str], relevant: set[str], k: int) -> float:
    """Standard Precision@K = retrieved relevant documents / K (positionally truncated).

    K is a fixed cutoff, independent of the corpus or qrels size. A Top-K with no relevant result
    scores 0.
    A duplicate chunk_id is counted as a hit only at its first position (deduplicated per document,
    never double counted).
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
    """Standard Recall@K = retrieved relevant documents / total relevant documents.

    Unlike early v1's HitRate@K (1 when anything relevant appears): the denominator here is the number
    of all relevant chunks.
    If the query has no relevant document in qrels, it is treated as 0.0 (avoiding 0/0).
    A duplicate chunk_id counts as a hit only once.
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
    """A single question's contribution to MRR: 1/rank of the first relevant result within the Top-K,
    or 0 when there is none.

    rank is the 1-based position; duplicate chunk_ids cannot create a new first-relevant rank.
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
    """Standard Average Precision@K: accumulate Precision only at relevant positions, normalized by
    the total number of relevant documents.

    Formula: AP@K = (1 / |rel|) * sum_{rank=1..K} (P@rank * rel_at_rank), where each relevant document
    contributes hits/rank only at its first occurrence. Returns 0.0 when the qrels hold nothing
    relevant.
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
    """Standard nDCG@K (binary relevance): DCG@K / IDCG@K.

    DCG uses logarithmic discounting: a hit at position rank contributes 1 / log2(rank+1), and each
    relevant document counts only once.
    IDCG is the DCG of the ideal ranking (all relevant documents first), truncated to min(|rel|, K).
    Returns 0.0 when the qrels hold nothing relevant or the ideal DCG is 0.
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
    """Compute standard IR metric aggregates and per-question metrics for a set of qrels and runs.

    qrels: question_id -> list of relevant chunk_ids (binary relevance, rel=1).
    run:   question_id -> ranked list of chunk_ids (same query space as qrels).

    Aggregation covers only eligible queries with non-empty qrels, so 0/0 cannot dilute the mean;
    queries with no eligible qrels are recorded separately in excluded_queries.
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
