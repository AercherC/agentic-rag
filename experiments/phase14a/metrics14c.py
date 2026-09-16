"""Phase 14A-C 指标模块（方案 §7-§10 重建执行版）。

指标分三层（与压缩摘要中的方案一致）：
1. 标准 IR 指标（chunk 级）：复用 ``src/evidenceflow/eval/ir_metrics.py``（Phase 11A 命名，
   Recall ≠ HitRate），qrels 由 anchor 逐 arm 投影生成；
2. Evidence Anchor 指标（element 级）：EvidenceAnchorRecall@K（任一 anchor 命中的元素覆盖率）
   与 AllRequired@K（全部 required anchor 被覆盖的比例）；
3. 结构指标：chunk 数/token 分布/覆盖率/泄漏计数/膨胀率/延迟。

配对 bootstrap：n=1000，seed=20260915，query 级重采样，两两 arm 对比（causal pairwise，
不做加权总分）。
"""

from __future__ import annotations

import random
from typing import Any, Callable

from evidenceflow.eval.ir_metrics import evaluate_ir_run

# 冻结的评测深度集合。
DEPTHS = (1, 3, 5, 10, 20)
BOOTSTRAP_N = 1000
BOOTSTRAP_SEED = 20260915

# 选择顺序使用的指标（方案 §19 重建）。
SELECTION_METRICS = ("evidence_anchor_recall_at_20", "all_required_at_20", "ndcg_at_5", "mrr_at_5", "recall_at_10")


# --------------------------------------------------------------------------------------
# anchor 投影（Gold 不绑定 chunk_id，逐 arm 投影）
# --------------------------------------------------------------------------------------

def project_qrels(
    queries: list[dict[str, Any]], chunks_by_doc: dict[str, list[dict[str, Any]]]
) -> tuple[dict[str, list[str]], dict[str, dict[str, Any]]]:
    """anchor → chunk 空间投影。

    返回 (qrels, projection_report)：
    - qrels: qid → 相关 chunk_id 列表（chunk.element_ids ∩ 任一 anchor element ≠ ∅）；
    - projection_report: qid → {relevant_count, anchor_element_ids, required_coverable,
      required_element_ids, unprojectable_required}。
    """

    qrels: dict[str, list[str]] = {}
    report: dict[str, dict[str, Any]] = {}
    for q in queries:
        anchor_ids = {a["element_id"] for a in q["anchors"] if a["element_id"]}
        doc_chunks = chunks_by_doc.get(q["doc_id"], [])
        rel = [c["chunk_id"] for c in doc_chunks if anchor_ids & set(c["element_ids"])]
        # required anchor 是否至少投影到一个 chunk（门 C-G4 前提）。
        unprojectable = []
        for rid in q["required_element_ids"]:
            if not any(rid in c["element_ids"] for c in doc_chunks):
                unprojectable.append(rid)
        qrels[q["qid"]] = rel
        report[q["qid"]] = {
            "relevant_chunk_count": len(rel),
            "relevant_chunk_ids": rel,
            "anchor_element_ids": sorted(anchor_ids),
            "required_element_ids": list(q["required_element_ids"]),
            "unprojectable_required": unprojectable,
        }
    return qrels, report


# --------------------------------------------------------------------------------------
# 标准 IR 指标（复用 src/evidenceflow/eval/ir_metrics.py）+ HitRate（独立实现，禁止混淆）
# --------------------------------------------------------------------------------------

def hit_rate_at_k(ranked_ids: list[str], relevant: set[str], k: int) -> float:
    """HitRate@K：Top-K 内**任意**相关即 1，否则 0（分母是 query 数，不是相关文档数）。

    与标准 Recall@K 的区别由 tests/test_phase14a_c.py 的 mutation 守护
    （把 HitRate 改名为 Recall 必须被抓到）。
    """

    if k <= 0:
        raise ValueError("k must be a positive integer")
    if not relevant:
        return 0.0
    return 1.0 if any(cid in relevant for cid in ranked_ids[:k]) else 0.0


def compute_ir_metrics(
    qrels: dict[str, list[str]], run: dict[str, list[str]]
) -> dict[str, Any]:
    """标准 IR 指标 + HitRate@5/10/20（方案 §7 的完整指标集）。

    - 复用 evaluate_ir_run（P@5、Recall@5/10/20、MRR@5、MAP@5、nDCG@5）；
    - HitRate 单独计算（Phase 11A 命名纪律）；
    - 无相关 qrels 的 query 由 evaluate_ir_run 排除并在报告中记录。
    """

    result = evaluate_ir_run(qrels, run, depths=DEPTHS)
    agg = result["aggregate"]
    hit: dict[int, float] = {}
    eligible = [row["question_id"] for row in result["per_query"].values() if row["eligible"]]
    for k in (5, 10, 20):
        vals = [hit_rate_at_k(run[qid], set(qrels.get(qid, [])), k) for qid in eligible]
        hit[k] = sum(vals) / len(vals) if vals else 0.0
    metrics = {
        "precision_at_5": agg["precision_at_k"][5],
        "recall_at_5": agg["recall_at_k"][5],
        "recall_at_10": agg["recall_at_k"][10],
        "recall_at_20": agg["recall_at_k"][20],
        "mrr_at_5": agg["mrr_at_k"][5],
        "map_at_5": agg["map_at_k"][5],
        "ndcg_at_5": agg["ndcg_at_k"][5],
        "hit_rate_at_5": hit[5],
        "hit_rate_at_10": hit[10],
        "hit_rate_at_20": hit[20],
        "eligible_count": agg["eligible_count"],
        "excluded_count": agg["excluded_count"],
    }
    return {"aggregate": metrics, "per_query": result["per_query"], "depths": list(DEPTHS)}


def per_query_metric_values(
    ir_result: dict[str, Any], metric: str
) -> dict[str, float]:
    """抽取逐题指标值（bootstrap 重采样单位）。支持 recall_at_*/ndcg_at_*/mrr_at_* 等。"""

    out: dict[str, float] = {}
    for qid, row in ir_result["per_query"].items():
        if row["eligible"] and row["metrics"] and metric in row["metrics"]:
            out[qid] = row["metrics"][metric]
    return out


# --------------------------------------------------------------------------------------
# Evidence Anchor 指标（element 级）
# --------------------------------------------------------------------------------------

def compute_anchor_metrics(
    queries: list[dict[str, Any]], run: dict[str, list[str]],
    chunks_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """EvidenceAnchorRecall@K 与 AllRequired@K（element 级覆盖）。

    - anchor element 被覆盖 = Top-K 内存在包含该 element_id 的 chunk；
    - EvidenceAnchorRecall@K 分母 = 该 query 的**可投影** anchor 元素数（required 必可投影）；
    - AllRequired@K = 全部 required anchor 被覆盖 → 1，否则 0（query 级均值）。
    """

    recall: dict[int, list[float]] = {k: [] for k in (5, 10, 20)}
    allreq: dict[int, list[float]] = {k: [] for k in (5, 10, 20)}
    for q in queries:
        top = run.get(q["qid"], [])
        # 可投影 anchor 元素（在至少一个 chunk 中出现）。
        projectable: list[str] = []
        for a in q["anchors"]:
            eid = a["element_id"]
            if eid and any(eid in c["element_ids"] for c in chunks_by_id.values()
                           if c["document_id"] == q["doc_id"]):
                projectable.append(eid)
        covered = {eid for eid in set(projectable) for c in top
                   if chunks_by_id.get(c, {}).get("document_id") == q["doc_id"]
                   and eid in chunks_by_id[c]["element_ids"]}
        for k in (5, 10, 20):
            topk = top[:k]
            cov_k = {eid for eid in projectable
                     for c in topk if eid in chunks_by_id.get(c, {}).get("element_ids", [])}
            if projectable:
                recall[k].append(len(cov_k) / len(projectable))
            req = [e for e in q["required_element_ids"] if e in set(projectable)]
            allreq[k].append(1.0 if req and all(e in cov_k for e in req) else 0.0)
    agg = {}
    for k in (5, 10, 20):
        agg[f"evidence_anchor_recall_at_{k}"] = (
            sum(recall[k]) / len(recall[k]) if recall[k] else 0.0
        )
        agg[f"all_required_at_{k}"] = (
            sum(allreq[k]) / len(allreq[k]) if allreq[k] else 0.0
        )
    return {"aggregate": agg}


# --------------------------------------------------------------------------------------
# 结构指标
# --------------------------------------------------------------------------------------

def compute_structural_metrics(
    chunks: list[dict[str, Any]],
    docs: dict[str, dict[str, Any]],
    indexable_element_ids: dict[str, list[str]],
) -> dict[str, Any]:
    """结构指标：chunk 数/token 分布/覆盖/泄漏/膨胀。

    - 覆盖：indexable 元素（role ∈ body/structure/reference 且非隐藏）出现在 ≥1 chunk；
    - 泄漏：furniture/hidden role 元素或隐藏行/列 cell 出现在 chunk 中（必须为 0）；
    - **element_id 只在文档内唯一**（跨文档同名，如 X4 与 X7 都有 el-xlsx-cell-0-A3），
      covered 集合必须用 (document_id, element_id) 二元组，否则跨文档互相污染。
    """

    token_counts = [c["token_count"] for c in chunks]
    covered: set[tuple[str, str]] = {
        (c["document_id"], eid) for c in chunks for eid in c["element_ids"]
    }
    per_doc_coverage = {}
    for doc_id, ids in indexable_element_ids.items():
        got = [i for i in ids if (doc_id, i) in covered]
        per_doc_coverage[doc_id] = {"total": len(ids), "covered": len(got),
                                    "missing": sorted(set(ids) - set(got))}
    # 泄漏：被 chunk 引用但不可入索引（furniture/hidden role 或隐藏行/列）。
    leaks: list[str] = []
    for doc_id, doc in docs.items():
        for el in doc["elements"]:
            md = el.get("metadata") or {}
            hidden = el.get("retrieval_role") in ("furniture", "hidden") or \
                md.get("is_hidden_row") or md.get("is_hidden_col")
            if hidden and (doc_id, el["element_id"]) in covered:
                leaks.append(f"{doc_id}:{el['element_id']}")
    indexable_flat = [i for ids in indexable_element_ids.values() for i in ids]
    return {
        "chunk_count": len(chunks),
        "token_sum": sum(token_counts),
        "token_mean": sum(token_counts) / len(chunks) if chunks else 0.0,
        "token_max": max(token_counts) if token_counts else 0,
        "oversized_split_count": sum(1 for c in chunks if c.get("split")),
        "empty_chunk_count": sum(1 for c in chunks if not c["text"].strip()),
        "coverage": per_doc_coverage,
        "coverage_ratio": (
            sum(v["covered"] for v in per_doc_coverage.values())
            / max(1, sum(v["total"] for v in per_doc_coverage.values()))
        ),
        "leak_count": len(leaks),
        "leak_element_ids": sorted(set(leaks)),
        "indexable_element_count": len(indexable_flat),
        "chunk_count_over_docs": len({c["document_id"] for c in chunks}),
    }


def _hidden_sheets(doc: dict[str, Any]) -> set[str]:
    """文档中隐藏 sheet 名集合（metadata.hidden 或 state != visible）。"""

    out: set[str] = set()
    for el in doc["elements"]:
        if el["type"] == "sheet":
            md = el.get("metadata") or {}
            name = md.get("name") or (el.get("source_ref") or {}).get("sheet")
            if bool(md.get("hidden")) or md.get("state") != "visible":
                out.add(name)
    return out


def indexable_elements_by_doc(docs: dict[str, dict[str, Any]]) -> dict[str, list[str]]:
    """每文档可入索引元素 id 列表（覆盖率分母；与 chunk14c.INDEXABLE_ROLES 同口径）。

    按设计不计入分母（无检索内容）：
    - 隐藏 sheet 上的元素；隐藏行/列 cell；
    - 空文本元素（图片/分页符/文本框位置的空载体段落，B1 中 normalized_text 为空，
      实测 W4/W6/W7/W12 各 1 个——它们承载版面而非文本，无内容可检索）。
    """

    from .chunk14c import INDEXABLE_ROLES

    out: dict[str, list[str]] = {}
    for doc_id, doc in docs.items():
        hidden_sheets = _hidden_sheets(doc)
        ids = []
        for el in doc["elements"]:
            md = el.get("metadata") or {}
            if el.get("retrieval_role") not in INDEXABLE_ROLES:
                continue
            if md.get("is_hidden_row") or md.get("is_hidden_col"):
                continue
            if (el.get("source_ref") or {}).get("sheet") in hidden_sheets:
                continue
            if el["type"] != "table" and not (el.get("normalized_text") or "").strip():
                continue
            ids.append(el["element_id"])
        out[doc_id] = ids
    return out


# --------------------------------------------------------------------------------------
# 配对 bootstrap（causal pairwise）
# --------------------------------------------------------------------------------------

def paired_bootstrap(
    values_a: dict[str, float], values_b: dict[str, float],
    n: int = BOOTSTRAP_N, seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """A vs B 配对 bootstrap：query 级有放回重采样，报告 Δ=A-B 的 95% CI 与 P(Δ>0)。

    只对两 arm 共有的 eligible query 做配对；共有 query 少于 2 时无法重采样（返回 ineligible）。
    """

    common = sorted(set(values_a) & set(values_b))
    if len(common) < 2:
        return {"eligible": False, "common_queries": len(common), "n": n, "seed": seed}
    rng = random.Random(seed)
    da = [values_a[q] for q in common]
    db = [values_b[q] for q in common]
    deltas = [a - b for a, b in zip(da, db)]
    boots = []
    for _ in range(n):
        idx = [rng.randrange(len(deltas)) for _ in range(len(deltas))]
        boots.append(sum(deltas[i] for i in idx) / len(idx))
    boots.sort()
    mean_delta = sum(deltas) / len(deltas)
    return {
        "eligible": True,
        "common_queries": len(common),
        "n": n,
        "seed": seed,
        "mean_delta": mean_delta,
        "ci95_low": boots[int(0.025 * n)],
        "ci95_high": boots[int(0.975 * n) - 1],
        "p_delta_gt_0": sum(1 for b in boots if b > 0) / n,
        "significant": not (boots[int(0.025 * n)] <= 0 <= boots[int(0.975 * n) - 1]),
    }


def pairwise_bootstrap_all_arms(
    per_query_by_arm: dict[str, dict[str, float]],
    arm_order: list[str],
    metric_label: str,
) -> list[dict[str, Any]]:
    """对 arm 两两做 bootstrap（按 arm_order 展开成对）。"""

    out = []
    for i, a in enumerate(arm_order):
        for b in arm_order[i + 1:]:
            res = paired_bootstrap(per_query_by_arm.get(a, {}), per_query_by_arm.get(b, {}))
            out.append({"metric": metric_label, "arm_a": a, "arm_b": b, **res})
    return out
