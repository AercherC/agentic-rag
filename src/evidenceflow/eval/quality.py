"""对显式人工/校准标注计算质量指标，不用非空文本推断正确或拒答。"""

from typing import Literal

from pydantic import Field

from evidenceflow.domain.evaluation_v2 import ContractModel
from evidenceflow.eval.v2 import ranking_metrics


class CitationJudgment(ContractModel):
    """一个 Claim 与一个引用之间的校验结果。"""

    label_valid: bool = Field(description="标签是否来自实际 Context。")
    supports_claim: bool = Field(description="标注者判断引用是否支持对应 Claim。")


class ClaimJudgment(ContractModel):
    """逐 Claim 真值判断；正确性和 Context 支持性独立标注。"""

    correct: bool = Field(description="是否符合参考真值。")
    grounded: bool = Field(description="是否由实际 Context 支持。")
    requires_citation: bool = Field(default=True, description="该 Claim 是否需要引用。")
    citations: list[CitationJudgment] = Field(default_factory=list)


def answer_metrics(claims: list[ClaimJudgment], reference_claim_covered: list[bool], *,
                   annotation_source: str) -> dict:
    """汇总完整标注集；缺少分母时返回 null，不伪造已测分数。"""
    if not annotation_source.strip():
        raise ValueError("annotation provenance is required")
    def ratio(numerator, denominator):
        return numerator / denominator if denominator else None
    precision = ratio(sum(c.correct for c in claims), len(claims))
    recall = ratio(sum(reference_claim_covered), len(reference_claim_covered))
    citations = [citation for claim in claims for citation in claim.citations]
    needed = [claim for claim in claims if claim.requires_citation]
    supported = lambda c: any(x.label_valid and x.supports_claim for x in c.citations)
    return {
        "annotation_source": annotation_source,
        "claim_precision": precision, "claim_recall": recall,
        "claim_f1": (2 * precision * recall / (precision + recall) if precision + recall else 0.)
                    if precision is not None and recall is not None else None,
        "groundedness": ratio(sum(c.grounded for c in claims), len(claims)),
        "answer_completeness": recall,
        "citation_label_validity": ratio(sum(c.label_valid for c in citations), len(citations)),
        "citation_support_precision": ratio(sum(c.label_valid and c.supports_claim for c in citations), len(citations)),
        "citation_completeness": ratio(sum(supported(c) for c in needed), len(needed)),
        "missing_citation_claim_rate": ratio(sum(not c.citations for c in needed), len(needed)),
        "unsupported_cited_claim_rate": ratio(sum(not supported(c) for c in claims if c.citations), sum(bool(c.citations) for c in claims)),
    }


def abstention_metrics(rows: list[tuple[Literal["answerable", "unanswerable"],
                                       Literal["ANSWERED", "INSUFFICIENT_CONTEXT", "ERROR"]]]) -> dict:
    """错误单独计数，拒答必须由显式状态给出。"""
    if any(a not in {"answerable", "unanswerable"} or s not in {"ANSWERED", "INSUFFICIENT_CONTEXT", "ERROR"} for a, s in rows):
        raise ValueError("unknown answerability/status")
    valid = [(a, s) for a, s in rows if s != "ERROR"]
    def rate(a, s):
        selected = [status for label, status in valid if label == a]
        return sum(status == s for status in selected) / len(selected) if selected else None
    return {"question_count": len(rows), "evaluated_count": len(valid),
            "error_count": len(rows) - len(valid),
            "unanswerable_false_answer_rate": rate("unanswerable", "ANSWERED"),
            "answerable_false_abstention_rate": rate("answerable", "INSUFFICIENT_CONTEXT")}


def context_metrics(chunk_ids: list[str], groups: list[list[list[str]]], *,
                    generation_token_count: int, token_estimator: str) -> dict:
    """评估实际纳入的 Context；Token 数由生成模型计数器或具名估算器提供。"""
    if type(generation_token_count) is not int or generation_token_count < 0 or not token_estimator.strip():
        raise ValueError("generation token count and estimator are required")
    coverage = ranking_metrics(chunk_ids, groups, (max(1, len(chunk_ids)),))
    known = {cid for group in groups for bundle in group for cid in bundle}
    # 冗余仅定义为同组已覆盖后又出现的单块替代证据；不惩罚同文档本身。
    duplicate = 0
    seen = set()
    for cid in chunk_ids:
        matching = {i for i, group in enumerate(groups) if [cid] in group}
        duplicate += bool(matching and matching <= seen)
        seen.update(matching)
    return {"context_evidence_recall": coverage[f"evidence_group_recall_at_{max(1, len(chunk_ids))}"],
            "context_known_evidence_precision": sum(cid in known for cid in chunk_ids) / len(chunk_ids) if chunk_ids else None,
            "same_group_redundancy": duplicate / len(chunk_ids) if chunk_ids else 0.,
            "generation_token_count": generation_token_count, "token_estimator": token_estimator}
