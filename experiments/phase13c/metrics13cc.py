# -*- coding: utf-8 -*-
"""Phase 13C-C 指标与门检查模块（方案 PLAN.md §5-§8）。

三层：
1. anchor 严格制 qrels 投影（per-arm）：chunk 归一化文本包含 gold evidence_anchor
   才算 relevant；页码只作诊断不参与 relevant 判定；
2. 标准 IR 指标：复用 14A-C ``compute_ir_metrics``（其内部走
   ``src/evidenceflow/eval/ir_metrics.py`` 的 canonical 实现，禁止重写公式）；
3. 结构门检查：token 预算（C-G5）、provenance（C-G4）。

配对 bootstrap 复用 14A-C ``paired_bootstrap``（n/seed 可参数化；方案要求 n=10000、
seed=20260915、重复两次结果全等 = C-G7）。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# 复用归档实验的指标纯函数（import 不复制、不修改）。
from phase14a.metrics14c import (  # noqa: E402
    compute_ir_metrics,
    per_query_metric_values,
    paired_bootstrap,
    pairwise_bootstrap_all_arms,
)

__all__ = [
    "normalize_text",
    "project_qrels_anchor",
    "check_token_budget",
    "check_provenance",
    "compute_ir_metrics",
    "per_query_metric_values",
    "paired_bootstrap",
    "pairwise_bootstrap_all_arms",
]


def normalize_text(text: str) -> str:
    """anchor 匹配归一化：NFKC（连字 ﬁ→fi）+ casefold（大小写变体）+ 折叠空白。

    实证依据（冒烟测试）：gold anchor "Figure 2: PCN architecture" 与提取文本
    "Figure 2: PCN Architecture" 仅大小写不同；PDF 正文存在 ﬁ/ﬂ 连字。二者若不
    归一化会把等价文本误判为未覆盖。归一化只影响匹配，不放宽二元相关语义。
    """

    import unicodedata

    t = unicodedata.normalize("NFKC", text)
    return " ".join(t.split()).casefold()


def project_qrels_anchor(
    chunks_by_doc: dict[str, list[dict[str, Any]]],
    questions: list[dict[str, Any]],
) -> dict[str, Any]:
    """anchor 严格制 qrels 投影（PLAN.md §5）。

    - relevant(chunk) = 该 chunk 文本（归一化）包含 gold source 的 evidence_anchor
      （归一化后子串匹配），且 chunk 属于该 gold source 的 document_id；
    - 多 gold source 取并集；二元相关；
    - 页码不参与 relevant，仅记录 owner 页码与 gold 页范围的重叠作诊断；
    - 返回 qrels、逐 gold source 的覆盖诊断（C-G2 输入）与每题 relevant 数。
    """

    # 预计算每文档每 chunk 的归一化文本：按 chunk 自身 document_id 分组
    #（入参 dict 的 key 是语料文件名，与 gold 的生产 document_id 键空间不同）。
    norm_text: dict[str, list[tuple[str, str]]] = {}
    for chunks in chunks_by_doc.values():
        for c in chunks:
            norm_text.setdefault(c["document_id"], []).append(
                (c["chunk_id"], normalize_text(c["text"])))
    qrels: dict[str, list[str]] = {}
    coverage: list[dict[str, Any]] = []
    num_relevant: dict[str, int] = {}
    for q in questions:
        qid = q["question_id"]
        relevant: list[str] = []
        for src_idx, src in enumerate(q.get("gold_sources", [])):
            doc_id = src.get("document_id")
            anchor = src.get("evidence_anchor") or ""
            anchor_norm = normalize_text(anchor)
            entry: dict[str, Any] = {
                "question_id": qid,
                "source_index": src_idx,
                "document_id": doc_id,
                "anchor": anchor,
                "matched_chunk_ids": [],
                "covered": False,
            }
            if doc_id in norm_text and anchor_norm:
                for cid, ctext in norm_text[doc_id]:
                    if anchor_norm in ctext:
                        entry["matched_chunk_ids"].append(cid)
            entry["covered"] = bool(entry["matched_chunk_ids"])
            if entry["covered"]:
                for cid in entry["matched_chunk_ids"]:
                    if cid not in relevant:
                        relevant.append(cid)
            coverage.append(entry)
        qrels[qid] = relevant
        num_relevant[qid] = len(relevant)
    covered_count = sum(1 for c in coverage if c["covered"])
    return {
        "qrels": qrels,
        "coverage": coverage,
        "coverage_rate": covered_count / len(coverage) if coverage else 0.0,
        "uncovered": [c for c in coverage if not c["covered"]],
        "num_relevant": num_relevant,
        "gold_source_count": len(coverage),
    }


def check_token_budget(chunks: list[dict[str, Any]], target: int = 512) -> dict[str, Any]:
    """C-G5：每 chunk token_count ≤ target（词级拆分粒度近似导致的违例计数）。"""

    violations = [
        {"chunk_id": c["chunk_id"], "token_count": c["token_count"]}
        for c in chunks if c["token_count"] > target
    ]
    return {"violations": violations, "violation_count": len(violations), "pass": not violations}


def check_provenance(
    chunks_by_doc: dict[str, list[dict[str, Any]]], docs: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """C-G4：每 chunk 的 element_ids ⊆ 该文档 B2_fused element_id 集合（无伪造/悬空来源）。"""

    violations: list[dict[str, Any]] = []
    for doc_id, chunks in chunks_by_doc.items():
        valid = {el.get("element_id") for el in docs[doc_id].get("elements", [])}
        for c in chunks:
            dangling = [oid for oid in c["element_ids"] if oid not in valid]
            if dangling:
                violations.append({"chunk_id": c["chunk_id"], "dangling_owner_ids": dangling})
    return {"violations": violations, "violation_count": len(violations), "pass": not violations}
