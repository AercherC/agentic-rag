"""已冻结的 Phase 12 fallback metrics 历史脚本。

读取 runtime_outputs_fallback.json（0.5B 生成的答案）与既有 Gold（required_claims / answerability），
用**确定性规则**做 Claim 抽取与指标聚合。不调用任何 LLM，不构成「模型自评分」。

所有产物标记 FALLBACK_MODEL，明确不代表生产 Runtime 答案质量。
"""
from __future__ import annotations

import json
import re
from pathlib import Path

BASE = Path("evals/evaluation_v3/phase12_rag_quality_v1")

_ENGLISH = re.compile(r"[a-z0-9]+", re.I)
_CJK = re.compile(r"[一-鿿]")


def _tokens(text: str) -> set[str]:
    text = text.lower()
    toks = set(_ENGLISH.findall(text)) | set(_CJK.findall(text))
    # 过滤纯标点/空格
    return {t for t in toks if t.strip()}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = a & b
    union = a | b
    return len(inter) / len(union)


_SENT_SPLIT = re.compile(r"(?<=[。.!?！？;；\n])")
_META = re.compile(r"^(here|the following|based on|according to|note|请注意|以下是|根据)", re.I)


def extract_claims(answer_text: str) -> list[str]:
    """启发式：按句切分，保留含事实 token 的句子作为候选 claim。"""
    if not answer_text or not answer_text.strip():
        return []
    parts = [p.strip() for p in _SENT_SPLIT.split(answer_text) if p.strip()]
    claims = []
    for p in parts:
        # 去掉明显的 [S1] 引用标记后判断是否有事实内容
        body = re.sub(r"\[S\d+\]", "", p).strip()
        toks = _tokens(body)
        if len(body) < 8 or len(toks) < 3:
            continue
        if _META.match(body):
            continue
        claims.append(p)
    return claims


_REQUIRED_BY_Q = None
_REF_BY_Q = None


def _load_gold():
    global _REQUIRED_BY_Q, _REF_BY_Q
    rc = json.load(open(BASE / "required_claims.json", encoding="utf-8"))
    _REQUIRED_BY_Q = {q["question_id"]: q["required_claims"] for q in rc["questions"]}
    ref = json.load(open(BASE / "reference_answers.json", encoding="utf-8"))
    _REF_BY_Q = {q["question_id"]: q["reference_answer"] for q in ref["questions"]}


def _match_claim(gen_claim: str, req_claims: list[dict]) -> str | None:
    """返回最匹配的 required claim_id（Jaccard>阈值），否则 None。"""
    best, best_id = 0.0, None
    gt = _tokens(gen_claim)
    for rc in req_claims:
        j = _jaccard(gt, _tokens(rc["text"]))
        if j > best:
            best, best_id = j, rc["claim_id"]
    return best_id if best >= 0.2 else None


def cmd_metrics() -> None:
    _load_gold()
    dataset = json.load(open(BASE / "generation_eval_dataset.json", encoding="utf-8"))
    ds_by_q = {q["question_id"]: q for q in dataset["questions"]}
    runtime = json.load(open(BASE / "runtime_outputs_fallback.json", encoding="utf-8"))
    outs = {o["question_id"]: o for o in runtime["outputs"]}

    per_q = []
    agg = {
        "claim_recall": [], "claim_precision": [], "claim_f1": [],
        "groundedness": [], "citation_precision": [], "citation_completeness": [],
        "valid_citation_rate": [], "invalid_citation_rate": [],
    }
    absten = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    failures = {}

    for qid, q in ds_by_q.items():
        o = outs[qid]
        req = _REQUIRED_BY_Q[qid]
        req_required = [c for c in req if c.get("required", True)]
        ctx_ids = set(o["context_chunk_ids"])
        answer_text = o["answer_text"]
        valid_cit = o["valid_citations"]
        invalid_cit = o["invalid_citations"]
        pred_status = o["predicted_status"]
        gold_ans = q["answerability"]

        gen_claims = extract_claims(answer_text)
        n_gen = len(gen_claims)
        covered_req = set()
        matched_gen = 0
        grounded_gen = 0
        claims_with_cit = 0
        for gc in gen_claims:
            rid = _match_claim(gc, req)
            if rid:
                matched_gen += 1
                covered_req.add(rid)
                # grounded：该 required claim 的可接受证据是否在 context 中
                acc = next(c["acceptable_evidence"] for c in req if c["claim_id"] == rid)
                if any(a in ctx_ids for a in acc):
                    grounded_gen += 1
            if valid_cit:  # 启发式：有合法引用即认为该 claim 被引用覆盖
                claims_with_cit += 1

        n_req = len(req_required) or 1
        claim_recall = len(covered_req & {c["claim_id"] for c in req_required}) / n_req
        claim_prec = matched_gen / n_gen if n_gen else 0.0
        claim_f1 = (
            2 * claim_prec * claim_recall / (claim_prec + claim_recall)
            if (claim_prec + claim_recall) else 0.0
        )
        grounded = grounded_gen / n_gen if n_gen else 0.0
        cit_total = len(valid_cit) + len(invalid_cit)
        valid_rate = len(valid_cit) / cit_total if cit_total else None
        invalid_rate = len(invalid_cit) / cit_total if cit_total else None
        cit_prec = claims_with_cit / n_gen if n_gen else 0.0
        cit_comp = claims_with_cit / n_gen if n_gen else 0.0

        for k, v in [
            ("claim_recall", claim_recall), ("claim_precision", claim_prec),
            ("claim_f1", claim_f1), ("groundedness", grounded),
            ("citation_precision", cit_prec), ("citation_completeness", cit_comp),
        ]:
            agg[k].append(v)
        if valid_rate is not None:
            agg["valid_citation_rate"].append(valid_rate)
            agg["invalid_citation_rate"].append(invalid_rate)

        # abstention 2x2（正类 INSUFFICIENT_CONTEXT）
        pred_ins = pred_status == "INSUFFICIENT_CONTEXT"
        gold_ins = gold_ans == "INSUFFICIENT_CONTEXT"
        if pred_ins and gold_ins:
            absten["tp"] += 1
        elif pred_ins and not gold_ins:
            absten["fp"] += 1
        elif not pred_ins and gold_ins:
            absten["fn"] += 1
        else:
            absten["tn"] += 1

        # failure taxonomy
        ft = []
        if gold_ans == "ANSWERABLE" and pred_ins:
            ft.append("FALSE_ABSTENTION")
        if gold_ans == "INSUFFICIENT_CONTEXT" and not pred_ins:
            ft.append("FALSE_ANSWER")
        if gold_ans == "ANSWERABLE" and not pred_ins and claim_recall == 0 and n_gen:
            ft.append("ANSWER_MISSING_CLAIM")
        if gold_ans == "ANSWERABLE" and not pred_ins and not valid_cit and n_gen:
            ft.append("CITATION_MISSING")
        if invalid_cit:
            ft.append("CITATION_INVALID_LABEL")
        if not ft:
            ft.append("NO_FAILURE_DETECTED")

        per_q.append({
            "question_id": qid,
            "answerability_gold": gold_ans,
            "query_language": q.get("query_language"),
            "predicted_status": pred_status,
            "generated_claim_count": n_gen,
            "required_claim_count": len(req_required),
            "claim_recall": round(claim_recall, 4),
            "claim_precision": round(claim_prec, 4),
            "claim_f1": round(claim_f1, 4),
            "groundedness": round(grounded, 4),
            "valid_citations": valid_cit,
            "invalid_citations": invalid_cit,
            "citation_precision": round(cit_prec, 4),
            "citation_completeness": round(cit_comp, 4),
            "failure_types": ft,
        })
        failures[qid] = ft

    # 聚合
    def _mean(xs):
        return round(sum(xs) / len(xs), 4) if xs else None

    aggregate = {k: _mean(v) for k, v in agg.items()}
    aggregate["question_count"] = len(per_q)
    aggregate["answerable_count"] = sum(1 for q in ds_by_q.values() if q["answerability"] == "ANSWERABLE")
    aggregate["insufficient_count"] = sum(1 for q in ds_by_q.values() if q["answerability"] == "INSUFFICIENT_CONTEXT")

    # abstention 指标
    tp, fp, fn, tn = absten["tp"], absten["fp"], absten["fn"], absten["tn"]
    acc = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) else 0
    prec = tp / (tp + fp) if (tp + fp) else 0
    rec = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
    false_answer_rate = fp / (fp + tn) if (fp + tn) else 0  # 把可答判为不可答视为 false answer(拒答误判)
    false_abstention_rate = fn / (fn + tp) if (fn + tp) else 0
    aggregate["abstention"] = {
        "accuracy": round(acc, 4), "precision": round(prec, 4),
        "recall": round(rec, 4), "f1": round(f1, 4),
        "false_answer_rate": round(false_answer_rate, 4),
        "false_abstention_rate": round(false_abstention_rate, 4),
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
    }

    # failure taxonomy 计数
    from collections import Counter
    fc = Counter()
    for fts in failures.values():
        for f in fts:
            fc[f] += 1
    failure_taxonomy = {
        "phase": "phase12_rag_quality_v1",
        "status": "FALLBACK_HEURISTIC",
        "note": "基于 0.5B 自由文本 + 规则抽取；failure type 为启发式推断，非人工标注",
        "counts_by_type": dict(fc),
        "per_question": [{"question_id": k, "failure_types": v} for k, v in failures.items()],
    }

    # retrieval_generation_matrix 完成版（answer/grounded/citation 用启发式）
    matrix_rows = []
    for pq in per_q:
        qid = pq["question_id"]
        o = outs[qid]
        ctx_ids = set(o["context_chunk_ids"])
        gold_ids = set(ds_by_q[qid]["acceptable_source_ids"])
        hit = len(gold_ids & ctx_ids)
        ac = pq["answer_correct_heuristic"] = (
            "CORRECT" if pq["claim_recall"] >= 0.5
            else ("MISSING" if pq["claim_recall"] == 0 else "INCORRECT")
        )
        gd = "GROUNDED" if pq["groundedness"] >= 0.99 else ("UNGROUNDED" if pq["groundedness"] == 0 else "PARTIAL")
        cc = (
            "CORRECT" if pq["valid_citations"] and not pq["invalid_citations"]
            else ("WRONG_SOURCE" if pq["invalid_citations"] else "MISSING")
        )
        matrix_rows.append({
            "question_id": qid,
            "answerability": pq["answerability_gold"],
            "retrieval_success": len(o["retrieved_chunk_ids"]) > 0,
            "context_evidence_present": hit > 0,
            "retrieval_verdict": "EVIDENCE_RETRIEVED" if hit > 0 else "EVIDENCE_NOT_RETRIEVED",
            "context_verdict": "IN_CONTEXT" if hit > 0 else "NOT_IN_CONTEXT",
            "answer_correct": ac,
            "grounded": gd,
            "citation_correct": cc,
            "fallback_heuristic": True,
        })

    # 写产物
    json.dump(per_q, open(BASE / "per_question_results_fallback.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    json.dump(aggregate, open(BASE / "aggregate_metrics_fallback.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    json.dump(failure_taxonomy, open(BASE / "failure_taxonomy_fallback.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    json.dump({"phase": "phase12_rag_quality_v1", "status": "FALLBACK_HEURISTIC",
               "rows": matrix_rows},
              open(BASE / "retrieval_generation_matrix_fallback.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

    # 报告
    report = _build_report(aggregate, runtime, failure_taxonomy)
    open(BASE / "report_fallback.md", "w", encoding="utf-8").write(report)

    print("wrote per_question_results_fallback / aggregate_metrics_fallback / "
          "failure_taxonomy_fallback / retrieval_generation_matrix_fallback / report_fallback.md")
    print("Claim F1 mean:", aggregate["claim_f1"], "| Groundedness:", aggregate["groundedness"],
          "| Abstention F1:", aggregate["abstention"]["f1"])


def _build_report(agg: dict, runtime: dict, ft: dict) -> str:
    a = agg["abstention"]
    lines = [
        "# Phase 12 本地 0.5B 冒烟测试报告（FALLBACK / 非生产质量）\n",
        "> 警告：本报告的指标由 **Qwen1.5-0.5B-Chat**（0.5B 小模型）+ **规则化 Claim 抽取** 得出，",
        "> **不代表生产 Runtime（Qwen3.5-35B-A3B）的答案质量**。仅用于验证 Phase 12 评测管线可端到端执行。\n",
        "## Phase 12 Gate (FALLBACK)",
        "- 生成模型：本地 Qwen1.5-0.5B-Chat（替换生产 LLM，符合异常报告恢复路径）",
        "- 检索/Context：复用冻结 Runtime（Dense→Reranker→Top5→Context）",
        "- 指标方法：确定性规则（非 LLM-as-a-Judge，非人工标注）\n",
        "## 数据集",
        f"- Questions: {agg['question_count']}",
        f"- Answerable: {agg['answerable_count']}",
        f"- Insufficient-context: {agg['insufficient_count']}\n",
        "## Answer（启发式）",
        f"- Claim Precision: {agg['claim_precision']}",
        f"- Claim Recall: {agg['claim_recall']}",
        f"- Claim F1: {agg['claim_f1']}\n",
        "## Grounding（启发式）",
        f"- Groundedness: {agg['groundedness']}",
        f"- Unsupported Claim Rate: {round(1 - (agg['groundedness'] or 0), 4)}\n",
        "## Citation（启发式）",
        f"- Citation Precision: {agg['citation_precision']}",
        f"- Citation Completeness: {agg['citation_completeness']}",
        f"- Valid Citation Rate: {agg['valid_citation_rate']}",
        f"- Invalid Citation Rate: {agg['invalid_citation_rate']}\n",
        "## Abstention（二分类，正类 INSUFFICIENT_CONTEXT）",
        f"- Accuracy: {a['accuracy']}",
        f"- Precision: {a['precision']}",
        f"- Recall: {a['recall']}",
        f"- F1: {a['f1']}",
        f"- False Answer Rate: {a['false_answer_rate']}",
        f"- False Abstention Rate: {a['false_abstention_rate']}",
        f"- Confusion: {a['confusion']}\n",
        "## Failure Breakdown（启发式）",
    ]
    for k, v in sorted(ft["counts_by_type"].items(), key=lambda x: -x[1]):
        lines.append(f"- {k}: {v}")
    lines += [
        "\n## Runtime Modified / Gold Modified / Chroma Modified",
        "- Runtime Modified: NO（仅替换生成器模型，检索/Context/Citation 解析未改）",
        "- Retrieval Gold Modified: NO",
        "- Chroma Modified: NO",
        f"- Tests: 418 passed / 17 warnings（冻结基线，详见 anomaly_report.md）\n",
        "## EvidenceFlow V1 Core",
        "- NOT_YET_COMPLETE（基于 0.5B 的冒烟测试不构成生产质量判定；需生产 LLM 复跑）",
        "\n## Recommended Next Phase",
        "- 恢复生产 LLM（充值 ModelScope 或提供 OpenAI-compatible 凭据）后，按 run_phase12_rag_quality.py 复跑得到生产质量指标。",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    cmd_metrics()
