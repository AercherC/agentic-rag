"""已冻结的 Phase 12 RAG metrics 历史脚本。

设计定位
--------
本脚本只做**确定性计算**：读取
  - `claim_annotations.json`（人工标注，含 Claim / Grounding / Citation / Abstention 判定）
  - `runtime_outputs.json`（冻结的 Runtime 输出）
  - `generation_eval_dataset.json`（Generation Gold：required_claims / answerability）
然后聚合出 Step 6-13 的指标与产物。

它**不调用任何 LLM**，因此不存在「同一模型既生成又自评分」的问题（计划 §20）。
若标注缺失，脚本直接报错退出，绝不使用启发式规则「凑」指标（计划 §4）。

产物
----
  per_question_results.json
  aggregate_metrics.json
  failure_taxonomy.json
  retrieval_generation_matrix.json
  execution_result.json
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

BASE = Path("evals/evaluation_v3/phase12_rag_quality_v1")

# 计划 §12 的分层 Failure Taxonomy（不含笼统的 "RAG failed"）
FAILURE_TYPES = [
    "RETRIEVAL_FAILURE",
    "CONTEXT_SELECTION_FAILURE",
    "ANSWER_MISSING_CLAIM",
    "ANSWER_WRONG_CLAIM",
    "UNSUPPORTED_CLAIM",
    "CITATION_MISSING",
    "CITATION_WRONG_SOURCE",
    "CITATION_INVALID_LABEL",
    "FALSE_ABSTENTION",
    "FALSE_ANSWER",
    "GENERATION_CONTRACT_ERROR",
    "LLM_REQUEST_ERROR",
]

# 判定集合（计划 §19）
GROUNDING_VALUES = {"SUPPORTED", "PARTIALLY_SUPPORTED", "UNSUPPORTED", "NOT_APPLICABLE"}
CLAIM_JUDGMENTS = {"CORRECT", "INCORRECT", "MISSING"}
CITATION_JUDGMENTS = {"SUPPORTS", "DOES_NOT_SUPPORT", "INVALID"}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(payload: Any, path: Path) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _ratio(num: int, den: int) -> float | None:
    """分母为 0 时返回 None（表示 NOT_APPLICABLE），不把空集当成 0 分。"""
    return (num / den) if den else None


def _mean(values: list[float]) -> float | None:
    return (sum(values) / len(values)) if values else None


def _f1(p: float | None, r: float | None) -> float | None:
    if p is None or r is None or (p + r) == 0:
        return None
    return 2 * p * r / (p + r)


# --------------------------------------------------------------------------
# 校验：标注必须完整覆盖所有题目与所有引用，否则拒绝计算
# --------------------------------------------------------------------------
def validate_annotations(ann: dict, outputs: dict, dataset: dict) -> list[str]:
    problems: list[str] = []
    ann_q = ann.get("questions", {})
    out_by_q = {o["question_id"]: o for o in outputs["outputs"]}
    ds_by_q = {q["question_id"]: q for q in dataset["questions"]}

    missing = sorted(set(ds_by_q) - set(ann_q))
    if missing:
        problems.append(f"缺少题目标注: {missing}")
    extra = sorted(set(ann_q) - set(ds_by_q))
    if extra:
        problems.append(f"标注了不存在的题目: {extra}")

    for qid, a in ann_q.items():
        if qid not in ds_by_q:
            continue
        if a.get("abstention_judgment") not in {
            "CORRECT_ANSWER",
            "FALSE_ABSTENTION",
            "CORRECT_ABSTENTION",
            "FALSE_ANSWER",
            "NOT_APPLICABLE_CONTRACT_ERROR",
        }:
            problems.append(f"{qid}: abstention_judgment 非法或缺失")
        for c in a.get("answer_claims", []):
            g = c.get("grounding")
            if g not in GROUNDING_VALUES:
                problems.append(f"{qid}: claim#{c.get('index')} grounding 非法或缺失")
            if c.get("is_factual") and c.get("judgment") not in CLAIM_JUDGMENTS:
                problems.append(f"{qid}: claim#{c.get('index')} judgment 非法或缺失")
            for cit in c.get("citations", []):
                if cit.get("judgment") not in CITATION_JUDGMENTS:
                    problems.append(f"{qid}: claim#{c.get('index')} citation {cit.get('label')} judgment 非法")
        # required claim 覆盖判定：必须覆盖该题全部 claim（required + optional）
        all_ids = {c["claim_id"] for c in ds_by_q[qid].get("required_claims", [])}
        rstat = a.get("required_claim_status", {})
        if all_ids and set(rstat) != all_ids:
            problems.append(f"{qid}: required_claim_status 键 {sorted(rstat)} != Gold {sorted(all_ids)}")
        if all_ids and any(v not in CLAIM_JUDGMENTS for v in rstat.values()):
            problems.append(f"{qid}: required_claim_status 存在非法判定")

        # 结构一致性：answer 结构指标应能与 runtime 输出对齐
        o = out_by_q.get(qid)
        if o is None:
            problems.append(f"{qid}: runtime_outputs 缺失该题")
            continue
        if o.get("ok") and len(a.get("answer_claims", [])) == 0 and o.get("answer_text"):
            # 允许极短答案无 claim，但需显式说明
            if not a.get("no_factual_claim_reason"):
                problems.append(f"{qid}: 答案非空但没有任何标注 claim，且未说明原因")
    return problems


# --------------------------------------------------------------------------
# 单题计算
# --------------------------------------------------------------------------
def compute_per_question(
    qid: str,
    ans: dict,
    out: dict,
    ds_q: dict,
    ctx_label_to_chunk: dict[str, str],
) -> dict:
    all_claims = ds_q.get("required_claims", [])
    required = [c for c in all_claims if c.get("required", True)]
    optional = [c for c in all_claims if not c.get("required", True)]
    required_ids = [c["claim_id"] for c in required]
    optional_ids = [c["claim_id"] for c in optional]
    rstat = ans.get("required_claim_status", {})

    answer_claims = ans.get("answer_claims", [])
    factual = [c for c in answer_claims if c.get("is_factual")]

    # --- Step 6 Answer Correctness ---
    correct_factual = [c for c in factual if c.get("judgment") == "CORRECT"]
    covered_required = [cid for cid in required_ids if rstat.get(cid) == "CORRECT"]
    covered_optional = [cid for cid in optional_ids if rstat.get(cid) == "CORRECT"]
    claim_precision = _ratio(len(correct_factual), len(factual))
    claim_recall = _ratio(len(covered_required), len(required_ids)) if required_ids else None
    claim_f1 = _f1(claim_precision, claim_recall)

    # --- Step 7 Groundedness ---
    grounded = [c for c in factual if c.get("grounding") == "SUPPORTED"]
    partial = [c for c in factual if c.get("grounding") == "PARTIALLY_SUPPORTED"]
    unsupported = [c for c in factual if c.get("grounding") == "UNSUPPORTED"]
    groundedness = _ratio(len(grounded), len(factual))
    unsupported_rate = _ratio(len(unsupported), len(factual))

    # --- Step 8/9 Citation ---
    all_cits = [cit for c in factual for cit in c.get("citations", [])]
    supporting = [cit for cit in all_cits if cit.get("judgment") == "SUPPORTS"]
    wrong_source = [cit for cit in all_cits if cit.get("judgment") == "DOES_NOT_SUPPORT"]
    invalid = [cit for cit in all_cits if cit.get("judgment") == "INVALID"]
    citation_precision = _ratio(len(supporting), len(all_cits))

    # Citation Completeness：分母 = 「应当被证据支持的事实性 Claim」
    # 定义：判定为 CORRECT 的事实性 claim（它们本来就该被正确引用）。
    # 错误/无支撑 claim 不进分母，其问题已由 precision / groundedness 惩罚。
    should_cite = [c for c in factual if c.get("judgment") == "CORRECT"]
    with_support = [c for c in should_cite if any(x.get("judgment") == "SUPPORTS" for x in c.get("citations", []))]
    citation_completeness = _ratio(len(with_support), len(should_cite))

    valid_label_rate = _ratio(len(all_cits) - len(invalid), len(all_cits))
    invalid_label_rate = _ratio(len(invalid), len(all_cits))

    # --- Step 11 结构指标（基于 Citation Parser 结果，非支持性判断） ---
    cited_labels = out.get("cited_labels", [])
    cited_chunk_ids = [s["chunk_id"] for s in out.get("cited_sources", [])]
    ctx_ids = set(out.get("context_chunk_ids", []))
    # 解析成功但引用了不在 Context 里的 chunk → 结构性错误
    labels_not_in_context = [
        s["source_label"] for s in out.get("cited_sources", []) if s["chunk_id"] not in ctx_ids
    ]
    answered_without_citation = 1 if (out.get("answer_text") and not cited_labels) else 0

    # --- Step 10 Abstention ---
    # 契约错误（GENERATION_CONTRACT_ERROR）未产出可用答案，不是一次「拒答决策」，
    # 因此单独记为 unusable，不进入 abstention 混淆矩阵（避免把技术失败与行为失败混为一谈）。
    aj = ans.get("abstention_judgment")
    conf = {
        "CORRECT_ANSWER": "TN",
        "FALSE_ABSTENTION": "FP",
        "FALSE_ANSWER": "FN",
        "CORRECT_ABSTENTION": "TP",
        "NOT_APPLICABLE_CONTRACT_ERROR": None,
    }[aj]

    # --- Step 12 Failure Attribution ---
    failures: list[str] = []
    gold_ans = ds_q["answerability"]

    # 检索层证据是否可达：即使 LLM 契约失败也要评估，避免故障层被掩盖
    gold_chunks = {e for c in required for e in c.get("acceptable_evidence", [])}
    retrieved_ids = set(out.get("retrieved_chunk_ids", []))
    gold_in_retrieved = bool(gold_chunks & retrieved_ids)
    gold_in_context = bool(gold_chunks & ctx_ids)
    if gold_ans == "ANSWERABLE":
        if not gold_in_retrieved:
            failures.append("RETRIEVAL_FAILURE")
        elif not gold_in_context:
            failures.append("CONTEXT_SELECTION_FAILURE")

    if not out.get("ok"):
        err = out.get("error", "")
        failures.append("GENERATION_CONTRACT_ERROR" if "Error" in err and "Request" not in err else "LLM_REQUEST_ERROR")
    else:
        if gold_ans == "ANSWERABLE":
            if any(rstat.get(cid) == "MISSING" for cid in required_ids):
                failures.append("ANSWER_MISSING_CLAIM")
            if any(c.get("judgment") == "INCORRECT" for c in factual):
                failures.append("ANSWER_WRONG_CLAIM")
        if unsupported:
            failures.append("UNSUPPORTED_CLAIM")
        if invalid:
            failures.append("CITATION_INVALID_LABEL")
        if wrong_source:
            failures.append("CITATION_WRONG_SOURCE")
        if citation_completeness is not None and citation_completeness < 1.0:
            failures.append("CITATION_MISSING")
        if aj == "FALSE_ABSTENTION":
            failures.append("FALSE_ABSTENTION")
        if aj == "FALSE_ANSWER":
            failures.append("FALSE_ANSWER")

    return {
        "question_id": qid,
        "query": ds_q["query"],
        "answerability_gold": gold_ans,
        "generated_status": out.get("generated_status"),
        "abstention_judgment": aj,
        "confusion": conf,
        "required_claims_total": len(required_ids),
        "required_claims_correct": len(covered_required),
        "optional_claims_total": len(optional_ids),
        "optional_claims_correct": len(covered_optional),
        "required_claim_status": rstat,
        "answer_factual_claim_count": len(factual),
        "answer_factual_claims_correct": len(correct_factual),
        "grounding": {
            "supported": len(grounded),
            "partially_supported": len(partial),
            "unsupported": len(unsupported),
        },
        "citations_total": len(all_cits),
        "citations_supporting": len(supporting),
        "citations_wrong_source": len(wrong_source),
        "citations_invalid": len(invalid),
        "metrics": {
            "claim_precision": claim_precision,
            "claim_recall": claim_recall,
            "claim_f1": claim_f1,
            "groundedness": groundedness,
            "unsupported_claim_rate": unsupported_rate,
            "citation_precision": citation_precision,
            "citation_completeness": citation_completeness,
            "valid_citation_label_rate": valid_label_rate,
            "invalid_citation_label_rate": invalid_label_rate,
        },
        "structure": {
            "cited_labels": cited_labels,
            "cited_chunk_ids": cited_chunk_ids,
            "labels_not_in_context": labels_not_in_context,
            "answered_without_citation": answered_without_citation,
        },
        "layers": {
            "gold_evidence_in_retrieved": gold_in_retrieved,
            "gold_evidence_in_context": gold_in_context,
            "context_included_count": out.get("context_included_count"),
            "context_total_tokens": out.get("context_total_tokens"),
        },
        "failure_types": sorted(set(failures)),
        "annotator_note": ans.get("notes"),
    }


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------
def cmd_compute() -> int:
    ann_path = BASE / "claim_annotations.json"
    if not ann_path.exists():
        print("[compute] 缺少 claim_annotations.json —— 人工标注未完成，拒绝用启发式规则凑指标（§4）。")
        return 2
    ann = load_json(ann_path)
    outputs = load_json(BASE / "runtime_outputs.json")
    dataset = load_json(BASE / "generation_eval_dataset.json")

    problems = validate_annotations(ann, outputs, dataset)
    if problems:
        print("[compute] 标注校验失败：")
        for p in problems:
            print("  -", p)
        return 2

    ds_by_q = {q["question_id"]: q for q in dataset["questions"]}
    out_by_q = {o["question_id"]: o for o in outputs["outputs"]}
    ann_q = ann["questions"]

    per_q = [
        compute_per_question(qid, ann_q[qid], out_by_q[qid], ds_by_q[qid], {})
        for qid in ds_by_q
    ]

    # ---------------- 聚合 ----------------
    def collect(metric: str, predicate=lambda r: True) -> list[float]:
        vals = []
        for r in per_q:
            if not predicate(r):
                continue
            v = r["metrics"].get(metric)
            if v is not None:
                vals.append(v)
        return vals

    answerable = lambda r: r["answerability_gold"] == "ANSWERABLE"

    cp_all, cr_all = collect("claim_precision"), collect("claim_recall")
    macro_p, macro_r = _mean(cp_all), _mean(cr_all)
    # 微平均：所有 claim 合并后整体计算，避免短答案题目权重被放大
    tot_factual = sum(r["answer_factual_claim_count"] for r in per_q)
    tot_correct_factual = sum(r["answer_factual_claims_correct"] for r in per_q)
    tot_req = sum(r["required_claims_total"] for r in per_q)
    tot_req_correct = sum(r["required_claims_correct"] for r in per_q)
    micro_p = _ratio(tot_correct_factual, tot_factual)
    micro_r = _ratio(tot_req_correct, tot_req)
    tot_opt = sum(r["optional_claims_total"] for r in per_q)
    tot_opt_correct = sum(r["optional_claims_correct"] for r in per_q)

    agg = {
        "question_count": len(per_q),
        "answerable_count": sum(1 for r in per_q if r["answerability_gold"] == "ANSWERABLE"),
        "insufficient_context_count": sum(1 for r in per_q if r["answerability_gold"] == "INSUFFICIENT_CONTEXT"),
        "answer": {
            "claim_precision_macro": macro_p,
            "claim_recall_macro": macro_r,
            "claim_f1_macro": _f1(macro_p, macro_r),
            "claim_precision_micro": micro_p,
            "claim_recall_micro": micro_r,
            "claim_f1_micro": _f1(micro_p, micro_r),
            "claim_precision_answerable_only": _mean(collect("claim_precision", answerable)),
            "claim_recall_answerable_only": _mean(collect("claim_recall", answerable)),
            "required_claims_total": tot_req,
            "required_claims_correct": tot_req_correct,
            "optional_claims_total": tot_opt,
            "optional_claims_correct": tot_opt_correct,
            "optional_claim_recall": _ratio(tot_opt_correct, tot_opt),
        },
        "grounding": {
            "groundedness_macro": _mean(collect("groundedness")),
            "unsupported_claim_rate_macro": _mean(collect("unsupported_claim_rate")),
            "supported_claims_total": sum(r["grounding"]["supported"] for r in per_q),
            "partially_supported_total": sum(r["grounding"]["partially_supported"] for r in per_q),
            "unsupported_total": sum(r["grounding"]["unsupported"] for r in per_q),
            "factual_claims_total": tot_factual,
        },
        "citation": {
            "citation_precision_macro": _mean(collect("citation_precision")),
            "citation_completeness_macro": _mean(collect("citation_completeness")),
            "valid_citation_label_rate_macro": _mean(collect("valid_citation_label_rate")),
            "invalid_citation_label_rate_macro": _mean(collect("invalid_citation_label_rate")),
            "citations_total": sum(r["citations_total"] for r in per_q),
            "citations_supporting": sum(r["citations_supporting"] for r in per_q),
            "citations_wrong_source": sum(r["citations_wrong_source"] for r in per_q),
            "citations_invalid": sum(r["citations_invalid"] for r in per_q),
        },
    }

    # 微平均 Grounding / Citation（以 claim / citation 为单位）
    agg["grounding"]["groundedness_micro"] = _ratio(
        agg["grounding"]["supported_claims_total"], tot_factual
    )
    agg["grounding"]["unsupported_claim_rate_micro"] = _ratio(
        agg["grounding"]["unsupported_total"], tot_factual
    )
    agg["citation"]["citation_precision_micro"] = _ratio(
        agg["citation"]["citations_supporting"], agg["citation"]["citations_total"]
    )
    agg["citation"]["invalid_citation_label_rate_micro"] = _ratio(
        agg["citation"]["citations_invalid"], agg["citation"]["citations_total"]
    )

    # ---------------- Abstention（正类 = INSUFFICIENT_CONTEXT） ----------------
    tp = sum(1 for r in per_q if r["confusion"] == "TP")
    fp = sum(1 for r in per_q if r["confusion"] == "FP")
    fn = sum(1 for r in per_q if r["confusion"] == "FN")
    tn = sum(1 for r in per_q if r["confusion"] == "TN")
    prec = _ratio(tp, tp + fp)
    rec = _ratio(tp, tp + fn)
    agg["abstention"] = {
        "positive_class": "INSUFFICIENT_CONTEXT",
        "confusion": {"TP": tp, "FP": fp, "FN": fn, "TN": tn},
        "unusable_output_count": sum(1 for r in per_q if r["confusion"] is None),
        "accuracy": _ratio(tp + tn, tp + tn + fp + fn),
        "precision": prec,
        "recall": rec,
        "f1": _f1(prec, rec),
        "false_answer_rate": _ratio(fn, fn + tn),      # 不可答却作答
        "false_abstention_rate": _ratio(fp, fp + tn),  # Gold 可答却拒答
    }

    # ---------------- Step 11 结构指标 ----------------
    # 注意：AnswerStatus 的 .value 为小写（answered / insufficient_context）
    answered = [r for r in per_q if (r["generated_status"] or "").lower() == "answered"]
    insuf = [r for r in per_q if (r["generated_status"] or "").lower() == "insufficient_context"]
    agg["citation_structure"] = {
        "answered_count": len(answered),
        "answered_without_citation_count": sum(r["structure"]["answered_without_citation"] for r in per_q),
        "insufficient_context_count": len(insuf),
        "labels_not_in_context_count": sum(len(r["structure"]["labels_not_in_context"]) for r in per_q),
        "total_cited_labels": sum(len(r["structure"]["cited_labels"]) for r in per_q),
    }

    # ---------------- Step 12 Failure Taxonomy ----------------
    tax_counts = {t: 0 for t in FAILURE_TYPES}
    tax_entries = []
    for r in per_q:
        for t in r["failure_types"]:
            tax_counts[t] += 1
        tax_entries.append({
            "question_id": r["question_id"],
            "answerability": r["answerability_gold"],
            "failure_types": r["failure_types"],
        })
    failure_taxonomy = {
        "phase": "phase12_rag_quality_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "COMPLETE",
        "counts_by_type": tax_counts,
        "questions_with_any_failure": sum(1 for r in per_q if r["failure_types"]),
        "questions_clean": sum(1 for r in per_q if not r["failure_types"]),
        "entries": tax_entries,
    }

    # ---------------- Step 13 Retrieval × Generation 矩阵 ----------------
    matrix_rows = []
    layer_counts = {"RETRIEVAL": 0, "CONTEXT": 0, "GENERATION": 0, "ABSTENTION": 0, "CITATION": 0, "NONE": 0}
    for r in per_q:
        gold_ans = r["answerability_gold"]
        m = r["metrics"]
        if gold_ans == "ANSWERABLE" and not r["layers"]["gold_evidence_in_retrieved"]:
            layer = "RETRIEVAL"
        elif gold_ans == "ANSWERABLE" and not r["layers"]["gold_evidence_in_context"]:
            layer = "CONTEXT"
        elif r["failure_types"] and any(
            t in {"GENERATION_CONTRACT_ERROR", "LLM_REQUEST_ERROR"} for t in r["failure_types"]
        ):
            layer = "GENERATION"
        elif m["claim_recall"] is not None and m["claim_recall"] < 1.0:
            layer = "GENERATION"
        elif r["abstention_judgment"] in {"FALSE_ABSTENTION", "FALSE_ANSWER"}:
            layer = "ABSTENTION"
        elif m["citation_completeness"] is not None and m["citation_completeness"] < 1.0:
            layer = "CITATION"
        else:
            layer = "NONE"
        layer_counts[layer] += 1
        matrix_rows.append({
            "question_id": r["question_id"],
            "answerability": gold_ans,
            "retrieval_success": r["layers"]["gold_evidence_in_retrieved"],
            "context_evidence_present": r["layers"]["gold_evidence_in_context"],
            "answer_correct": (r["metrics"]["claim_recall"] == 1.0) if r["metrics"]["claim_recall"] is not None else None,
            "grounded": (r["metrics"]["groundedness"] == 1.0) if r["metrics"]["groundedness"] is not None else None,
            "citation_correct": (r["metrics"]["citation_precision"] == 1.0)
            if r["metrics"]["citation_precision"] is not None else None,
            "primary_failure_layer": layer,
        })
    matrix = {
        "phase": "phase12_rag_quality_v1",
        "status": "COMPLETE",
        "layer_counts": layer_counts,
        "rows": matrix_rows,
    }

    # ---------------- 写盘 ----------------
    # Citation / Abstention 标注投影：单一真源仍是 claim_annotations.json，
    # 这里导出两个视图以满足计划 §24 的产物清单（避免维护三份互相矛盾的标注）。
    citation_view = {
        "schema_version": "1.0",
        "dataset_id": dataset["dataset_id"],
        "derived_from": "claim_annotations.json",
        "annotator": ann.get("annotator"),
        "questions": {
            qid: {
                "question_id": qid,
                "citations": [
                    {"claim_index": c.get("index"), "label": cit.get("label"),
                     "judgment": cit.get("judgment"), "supports_claim": cit.get("judgment") == "SUPPORTS"}
                    for c in ann_q[qid].get("answer_claims", [])
                    for cit in c.get("citations", [])
                ],
            }
            for qid in ds_by_q
        },
    }
    abstention_view = {
        "schema_version": "1.0",
        "dataset_id": dataset["dataset_id"],
        "derived_from": "claim_annotations.json",
        "annotator": ann.get("annotator"),
        "questions": {
            qid: {
                "question_id": qid,
                "answerability_gold": ds_by_q[qid]["answerability"],
                "generated_status": out_by_q[qid].get("generated_status"),
                "abstention_judgment": ann_q[qid].get("abstention_judgment"),
                "confusion": next(r["confusion"] for r in per_q if r["question_id"] == qid),
            }
            for qid in ds_by_q
        },
    }
    save_json(citation_view, BASE / "citation_annotations.json")
    save_json(abstention_view, BASE / "abstention_annotations.json")

    save_json({"phase": "phase12_rag_quality_v1", "generated_at_utc": datetime.now(timezone.utc).isoformat(),
               "annotator": ann.get("annotator"), "per_question": per_q}, BASE / "per_question_results.json")
    save_json({"phase": "phase12_rag_quality_v1", "generated_at_utc": datetime.now(timezone.utc).isoformat(),
               "aggregate": agg}, BASE / "aggregate_metrics.json")
    save_json(failure_taxonomy, BASE / "failure_taxonomy.json")
    save_json(matrix, BASE / "retrieval_generation_matrix.json")

    print("[compute] per_question =", len(per_q))
    print("[compute] Claim P/R/F1 (macro) = "
          f"{_fmt(macro_p)} / {_fmt(macro_r)} / {_fmt(_f1(macro_p, macro_r))}")
    print("[compute] Groundedness (macro) =", _fmt(agg['grounding']['groundedness_macro']))
    print("[compute] Citation P / Completeness (macro) = "
          f"{_fmt(agg['citation']['citation_precision_macro'])} / "
          f"{_fmt(agg['citation']['citation_completeness_macro'])}")
    print("[compute] Abstention confusion =", agg["abstention"]["confusion"])
    print("[compute] failure layer counts =", layer_counts)
    return 0


def _fmt(v: float | None) -> str:
    return "N/A" if v is None else f"{v:.4f}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 12 指标聚合引擎")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("compute", help="从人工标注计算全部指标与产物").set_defaults(func=lambda a: cmd_compute())
    parsed = parser.parse_args()
    return parsed.func(parsed)


if __name__ == "__main__":
    raise SystemExit(main())
