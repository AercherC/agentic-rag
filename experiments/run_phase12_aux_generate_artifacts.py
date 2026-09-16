"""Phase 12 续跑辅助：生成不依赖 LLM 的产物。

生成：
  - judge_calibration.json          (§24：Phase 12 v1 用人工标注，used=false)
  - run_manifest.json               (§24 推荐产物 #1：artifact 清单与状态)
  - retrieval_generation_matrix.json (§13/§18：仅检索+Context 两维，答案/引用维标记 BLOCKED)

不读取/不调用任何 LLM；只消费已落盘的 runtime_outputs.json 与 generation_eval_dataset.json。
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

BASE = Path("evals/evaluation_v3/phase12_rag_quality_v1")
BASE.mkdir(parents=True, exist_ok=True)

NOW = datetime.now(timezone.utc).isoformat()

runtime = json.load(open(BASE / "runtime_outputs.json", encoding="utf-8"))
dataset = json.load(open(BASE / "generation_eval_dataset.json", encoding="utf-8"))

gold = {q["question_id"]: set(q["acceptable_source_ids"]) for q in dataset["questions"]}
answerability = {q["question_id"]: q["answerability"] for q in dataset["questions"]}
by_qid = {r["question_id"]: r for r in runtime["outputs"]}


# ---------------------------------------------------------------------------
# 1) judge_calibration.json  (§24)
# ---------------------------------------------------------------------------
judge = {
    "phase": "phase12_rag_quality_v1",
    "artifact": "judge_calibration.json",
    "used": False,
    "reason": "human annotation used for Phase 12 v1（计划 §20/§24：v1 不强制使用 LLM-as-a-Judge）",
    "human_annotation_protocol": "annotation_guideline.md",
    "self_judge_prohibited": True,
    "note": "同一 Generator 既生成又无校准地自我评分被明确禁止；主报告将始终使用人工标注结果。",
    "generated_at_utc": NOW,
}
json.dump(judge, open(BASE / "judge_calibration.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# 2) run_manifest.json  (§24 推荐产物 #1)
# ---------------------------------------------------------------------------
manifest = {
    "phase": "phase12_rag_quality_v1",
    "generated_at_utc": NOW,
    "gate_reached": False,
    "blocker": "LLM Provider 返回 429 insufficient balance；计划 §4 异常即停",
    "artifacts": [
        {"name": "run_manifest.json", "status": "PRODUCED"},
        {"name": "phase12_baseline.json", "status": "PRODUCED"},
        {"name": "generation_eval_dataset.json", "status": "PRODUCED"},
        {"name": "annotation_guideline.md", "status": "PRODUCED"},
        {"name": "reference_answers.json", "status": "PRODUCED"},
        {"name": "required_claims.json", "status": "PRODUCED"},
        {"name": "answerability_gold.json", "status": "PRODUCED"},
        {"name": "unanswerable_verification.json", "status": "PRODUCED"},
        {"name": "runtime_outputs.json", "status": "PARTIAL",
         "detail": "检索+Context 全 25/25 成功；LLM 生成 25/25 失败"},
        {"name": "judge_calibration.json", "status": "PRODUCED", "detail": "used=false"},
        {"name": "retrieval_generation_matrix.json", "status": "PARTIAL",
         "detail": "仅检索+Context 两维；答案/引用维标记 BLOCKED_LLM_UNAVAILABLE"},
        {"name": "failure_taxonomy.json", "status": "PARTIAL",
         "detail": "当前仅 25×LLM_REQUEST_ERROR；其余失败类型待答案产出后补全"},
        {"name": "claim_annotations.json", "status": "BLOCKED", "detail": "无 answer_text"},
        {"name": "citation_annotations.json", "status": "BLOCKED", "detail": "无 answer_text"},
        {"name": "abstention_annotations.json", "status": "BLOCKED", "detail": "无 answer_text"},
        {"name": "per_question_results.json", "status": "BLOCKED", "detail": "无答案/Citation 指标"},
        {"name": "aggregate_metrics.json", "status": "BLOCKED", "detail": "无答案/Citation 指标"},
        {"name": "execution_result.json", "status": "BLOCKED", "detail": "Gate 未进入"},
        {"name": "report.md", "status": "REPLACED_BY", "detail": "anomaly_report.md"},
    ],
    "resume_instructions": [
        "恢复 LLM 可用性（充值或更换 OpenAI-compatible 供应商，仅改 .env 三变量）",
        "python experiments/run_phase12_rag_quality.py outputs --limit 2  # 连通性+控制头校验",
        "python experiments/run_phase12_rag_quality.py outputs          # 全量 25 题",
        "python experiments/run_phase12_rag_quality.py anomaly          # ok_count=25 时自动转正常",
        "进入 Step 5-16：Claim 抽取 → 人工标注 → 指标聚合 → 失败归因",
    ],
}
json.dump(manifest, open(BASE / "run_manifest.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# 3) retrieval_generation_matrix.json  (§13/§18，仅检索+Context 两维)
# ---------------------------------------------------------------------------
BLOCKED = "BLOCKED_LLM_UNAVAILABLE"
rows = []
for qid in sorted(by_qid):
    r = by_qid[qid]
    g = gold[qid]
    ctx = set(r.get("context_chunk_ids", []))
    hit = len(g & ctx)
    rows.append({
        "question_id": qid,
        "answerability": answerability[qid],
        "retrieval_success": len(r.get("retrieved_chunk_ids", [])) > 0,
        "retrieved_count": len(r.get("retrieved_chunk_ids", [])),
        "context_included_count": len(ctx),
        "gold_chunk_count": len(g),
        "gold_in_context_count": hit,
        "context_evidence_present": hit > 0,
        "retrieval_verdict": "EVIDENCE_RETRIEVED" if hit > 0 else "EVIDENCE_NOT_RETRIEVED",
        "context_verdict": "IN_CONTEXT" if hit > 0 else "NOT_IN_CONTEXT",
        # 以下三列需 LLM answer_text 才能判定，当前阻塞
        "answer_correct": BLOCKED,
        "grounded": BLOCKED,
        "citation_correct": BLOCKED,
        "llm_error": r.get("error"),
    })

matrix = {
    "phase": "phase12_rag_quality_v1",
    "generated_at_utc": NOW,
    "status": "PARTIAL",
    "blocker": "LLM 生成 25/25 失败；答案/引用/接地三列待恢复后补全",
    "schema": {
        "retrieval_verdict": ["EVIDENCE_RETRIEVED", "EVIDENCE_NOT_RETRIEVED"],
        "context_verdict": ["IN_CONTEXT", "NOT_IN_CONTEXT"],
        "answer_correct": ["CORRECT", "INCORRECT", "MISSING", "BLOCKED_LLM_UNAVAILABLE"],
        "grounded": ["GROUNDED", "UNGROUNDED", "BLOCKED_LLM_UNAVAILABLE"],
        "citation_correct": ["CORRECT", "WRONG_SOURCE", "MISSING", "BLOCKED_LLM_UNAVAILABLE"],
    },
    "summary": {
        "question_count": len(rows),
        "retrieval_success": sum(1 for x in rows if x["retrieval_success"]),
        "context_evidence_present": sum(1 for x in rows if x["context_evidence_present"]),
        "answerable": sum(1 for x in rows if x["answerability"] == "ANSWERABLE"),
        "answerable_with_evidence": sum(
            1 for x in rows if x["answerability"] == "ANSWERABLE" and x["context_evidence_present"]),
        "insufficient_context": sum(1 for x in rows if x["answerability"] == "INSUFFICIENT_CONTEXT"),
        "insufficient_with_evidence_leak": sum(
            1 for x in rows if x["answerability"] == "INSUFFICIENT_CONTEXT" and x["context_evidence_present"]),
    },
    "rows": rows,
}
json.dump(matrix, open(BASE / "retrieval_generation_matrix.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=2)

print("judge_calibration.json   ->", "PRODUCED (used=false)")
print("run_manifest.json        ->", "PRODUCED")
print("retrieval_generation_matrix.json -> PARTIAL rows=", len(rows))
print("  retrieval_success:", matrix["summary"]["retrieval_success"], "/", len(rows))
print("  answerable_with_evidence:", matrix["summary"]["answerable_with_evidence"], "/", matrix["summary"]["answerable"])
print("  insufficient_with_evidence_leak:", matrix["summary"]["insufficient_with_evidence_leak"])
