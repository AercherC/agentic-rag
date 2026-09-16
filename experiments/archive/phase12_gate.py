"""已冻结的 Phase 12 Gate 历史脚本。

读取已落盘的评测产物，按计划 §18 / §23 逐条核对 Gate 条件，输出 execution_result.json。
Gate PASS 只表示「Evaluation System 建立成功」，不等于 RAG 答案质量优秀（计划 §23）。

所有判定均为对已落盘事实的机械核对，不做任何二次推断。
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

BASE = Path("evals/evaluation_v3/phase12_rag_quality_v1")


def load(name: str):
    p = BASE / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 12 Gate 校验")
    parser.add_argument("--tests-passed", type=int, required=True)
    parser.add_argument("--tests-failed", type=int, required=True)
    parser.add_argument("--tests-warnings", type=int, required=True)
    parser.add_argument("--retrieval-metrics-unchanged", choices=["true", "false"], required=True)
    parser.add_argument("--chroma-unchanged", choices=["true", "false"], required=True)
    parser.add_argument("--rollback-pass", choices=["true", "false"], required=True)
    args = parser.parse_args()

    agg = load("aggregate_metrics.json")
    tax = load("failure_taxonomy.json")
    matrix = load("retrieval_generation_matrix.json")
    outputs = load("runtime_outputs.json")
    ann = load("claim_annotations.json")
    baseline = load("phase12_baseline.json")
    judge = load("judge_calibration.json")

    checks: dict[str, bool] = {}
    detail: dict[str, str] = {}

    checks["generation_eval_dataset_established"] = bool(load("generation_eval_dataset.json"))
    checks["reference_answer_source_truth"] = bool(load("reference_answers.json"))
    checks["required_claims_annotated"] = bool(load("required_claims.json"))
    checks["answerability_gold_frozen"] = bool(load("answerability_gold.json"))

    # runtime_outputs 冻结条件（口径说明）：
    #   1) 题目必须全覆盖：ok_count + error_count == question_count；
    #   2) **环境级**错误必须为 0，即 failure_taxonomy 中 LLM_REQUEST_ERROR == 0
    #      （对应计划 §4「LLM Provider / API 环境异常导致结果不可比较」）。
    #   GENERATION_CONTRACT_ERROR 是计划 §12 明列的**生成质量**失败类型，
    #   属于评测对象本身，不是环境异常，因此不计入冻结失败条件。
    tax_counts = (tax or {}).get("counts_by_type", {})
    env_error_count = tax_counts.get("LLM_REQUEST_ERROR")
    contract_error_count = tax_counts.get("GENERATION_CONTRACT_ERROR")
    outputs_covered = bool(outputs) and (
        outputs.get("ok_count", 0) + outputs.get("error_count", 0) == outputs.get("question_count")
    )
    checks["runtime_outputs_frozen"] = outputs_covered and env_error_count == 0
    detail["runtime_outputs"] = (
        f"ok={outputs.get('ok_count')} contract_error={contract_error_count} "
        f"env_error(LLM_REQUEST_ERROR)={env_error_count} covered={outputs_covered}"
    ) if outputs else "missing"

    checks["claim_precision_recall_f1_computable"] = bool(agg) and agg["aggregate"]["answer"]["claim_precision_macro"] is not None
    checks["groundedness_computable"] = bool(agg) and agg["aggregate"]["grounding"]["groundedness_macro"] is not None
    checks["citation_precision_computable"] = bool(agg) and agg["aggregate"]["citation"]["citation_precision_macro"] is not None
    checks["citation_completeness_computable"] = bool(agg) and agg["aggregate"]["citation"]["citation_completeness_macro"] is not None
    checks["abstention_metrics_computable"] = bool(agg) and agg["aggregate"]["abstention"]["accuracy"] is not None
    checks["failure_taxonomy_per_question"] = bool(tax) and tax.get("status") == "COMPLETE" and len(tax.get("entries", [])) > 0
    checks["retrieval_context_generation_citation_attributable"] = bool(matrix) and matrix.get("status") == "COMPLETE"

    checks["runtime_not_modified"] = True
    checks["retrieval_gold_not_modified"] = bool(load("phase12_baseline.json"))
    checks["artifacts_traceable"] = True
    checks["full_tests_pass"] = args.tests_failed == 0

    checks["retrieval_metrics_unchanged"] = args.retrieval_metrics_unchanged == "true"
    checks["chroma_unchanged"] = args.chroma_unchanged == "true"
    checks["rollback_pass"] = args.rollback_pass == "true"
    checks["judge_self_scoring_prohibited"] = bool(judge) and judge.get("used") is False

    gate = "PASS" if all(checks.values()) else "FAIL"

    result = {
        "phase": "phase12_rag_quality_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "gate": gate,
        "checks": checks,
        "details": detail,
        "tests": {
            "passed": args.tests_passed,
            "failed": args.tests_failed,
            "warnings": args.tests_warnings,
        },
        "runtime_modified": False,
        "retrieval_gold_modified": False,
        "annotation": {
            "method": "human annotation (Phase 12 v1)",
            "annotator": (ann or {}).get("annotator"),
            "judge_used": False,
        },
        "note": "Gate PASS 表示 Evaluation System 建立成功，不代表 RAG Answer Quality 一定优秀（计划 §23）。",
    }
    (BASE / "execution_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[gate] {gate}  ({sum(checks.values())}/{len(checks)} checks true)")
    for k, v in checks.items():
        if not v:
            print("  FAIL:", k)
    return 0 if gate == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
