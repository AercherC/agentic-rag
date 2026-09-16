#!/usr/bin/env python
"""已冻结的 Phase 11B manifest 历史脚本。

本脚本只做**分类与落盘**，不删除任何东西；删除动作由
phase11b_cleanup_files.py 与手工代码编辑完成，并在此处登记。
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

OUT = Path("evals/evaluation_v3/phase11b_cleanup_v1")

# ---------------------------------------------------------------- Step 5
CONFIG_INVENTORY = [
    ("llm_api_key", "PRODUCTION", "KEEP", "Runtime LLM Client 必需凭证；快照已脱敏。"),
    ("llm_base_url", "PRODUCTION", "KEEP", "LLM Endpoint。"),
    ("llm_model", "PRODUCTION", "KEEP", "LLM 模型名。"),
    ("llm_temperature", "PRODUCTION", "KEEP", "生成温度，默认 0。"),
    ("llm_timeout_seconds", "PRODUCTION", "KEEP", "LLM 请求超时。"),
    ("llm_max_attempts", "PRODUCTION", "KEEP", "瞬态错误重试上限。"),
    ("embedding_model", "PRODUCTION", "KEEP", "BGE-M3 Dense 唯一 Embedding。"),
    ("source_docs_path", "PRODUCTION", "KEEP", "摄取根目录，决定 document_id。"),
    ("chunk_size", "PRODUCTION", "KEEP", "Chunk Token 上限（冻结 512）。"),
    ("chunk_overlap", "PRODUCTION", "KEEP", "Chunk 重叠（冻结 64）。"),
    ("max_context_tokens", "PRODUCTION", "KEEP", "Context Budget。"),
    ("vector_db_path", "PRODUCTION", "KEEP", "Chroma 持久化目录。"),
    ("collection_name", "PRODUCTION", "KEEP", "冻结 Collection 名。"),
    ("reranker_enabled", "PRODUCTION", "KEEP", "默认 True；false 即配置级 rollback 开关。"),
    ("reranker_model", "PRODUCTION", "KEEP", "正式 Cross-Encoder。"),
    ("reranker_input_k", "PRODUCTION", "KEEP", "ANN Top20 候选深度。"),
    ("reranker_final_k", "PRODUCTION", "KEEP", "最终 Top5。"),
    ("reranker_batch_size", "PRODUCTION", "KEEP", "Phase 10 已验证的 CPU batch。"),
    ("reranker_max_length", "PRODUCTION", "KEEP", "Query-Passage 截断长度。"),
    ("reranker_device", "PRODUCTION", "KEEP", "Phase 9 只晋升 cpu。"),
    ("reranker_dtype", "PRODUCTION", "KEEP", "Phase 9 只晋升 float32。"),
    ("multilingual_candidate_top_k", "ROLLBACK", "KEEP", "RERANKER_ENABLED=false 时每路 Dense 深度。"),
    ("multilingual_candidate_pool_k", "ROLLBACK", "KEEP", "rollback 双路并集候选池。"),
    ("multilingual_final_top_k", "ROLLBACK", "KEEP", "rollback 最终返回深度。"),
    ("multilingual_rrf_k", "ROLLBACK", "KEEP", "Weighted RRF 平滑常数。"),
    ("multilingual_original_weight", "ROLLBACK", "KEEP", "Original 分支 RRF 权重。"),
    ("multilingual_translation_weight", "ROLLBACK", "KEEP", "Translation 分支 RRF 权重。"),
    ("multilingual_hybrid_enabled", "FALLBACK", "KEEP", "§17.3.1 Hybrid 开关，默认 False。"),
    ("multilingual_hybrid_bm25_k1", "FALLBACK", "KEEP", "Hybrid BM25 k1，与离线 Gate 一致。"),
    ("multilingual_hybrid_bm25_b", "FALLBACK", "KEEP", "Hybrid BM25 b，与离线 Gate 一致。"),
    ("multilingual_hybrid_rrf_k", "FALLBACK", "KEEP", "Hybrid 末轮 Equal RRF k。"),
    (
        "top_k",
        "UNUSED",
        "KEEP",
        "Runtime 从不消费（主路径用 reranker_input_k / multilingual_candidate_top_k）；"
        "但 tests/test_config.py 有断言依赖，按『五问』规则不得删除，标记遗留。",
    ),
]

# ---------------------------------------------------------------- Step 4
EVAL_CLEANUP = [
    {
        "path": "src/evidenceflow/eval/ir_metrics.py",
        "symbols": ["precision_at_k", "recall_at_k", "reciprocal_rank_at_k",
                    "average_precision_at_k", "ndcg_at_k", "evaluate_ir_run"],
        "layer": "STANDARD",
        "action": "KEEP",
        "evidence": "标准 IR 指标核心；Phase 11A 迁移后唯一权威实现。",
        "behavior_risk": "none",
    },
    {
        "path": "src/evidenceflow/eval/v3.py",
        "symbols": ["build_qrels", "qrels_hash", "evaluate_v3"],
        "layer": "STANDARD",
        "action": "KEEP",
        "evidence": "Evaluation v3 runner；qrels hash 冻结于 cleanup_baseline。",
        "behavior_risk": "none",
    },
    {
        "path": "src/evidenceflow/eval/mapping.py",
        "symbols": ["normalize_anchor", "_map_alternative", "_occurrences"],
        "layer": "STANDARD",
        "action": "KEEP",
        "evidence": "Gold Mapping / mappability；gold_mapping_v3 由其实例化语义定义。",
        "behavior_risk": "none",
    },
    {
        "path": "src/evidenceflow/eval/stable_mapping.py",
        "symbols": ["_compact"],
        "layer": "STANDARD",
        "action": "KEEP",
        "evidence": "Phase 3.5 Gold Stabilization 复现依赖。",
        "behavior_risk": "none",
    },
    {
        "path": "src/evidenceflow/eval/gold.py",
        "symbols": ["gold_chunk_hit", "gold_chunk_rank", "hits_at_depths",
                    "aggregate_hits_at_depths", "is_hit_at"],
        "layer": "COMPATIBILITY",
        "action": "KEEP",
        "evidence": "legacy HitRate-as-Recall reader；tests/test_hybrid_fusion.py 依赖。",
        "behavior_risk": "none（不参与标准 IR 指标）",
    },
    {
        "path": "src/evidenceflow/eval/metrics.py",
        "symbols": ["is_relevant", "is_relevant_to_any", "calculate_question_metrics",
                    "aggregate_question_metrics"],
        "layer": "COMPATIBILITY",
        "action": "KEEP",
        "evidence": "eval/runner.py（v1 runner）与 tests/test_eval_metrics.py 依赖。",
        "behavior_risk": "none",
    },
    {
        "path": "src/evidenceflow/eval/metrics.py",
        "symbols": ["custom_evidence_group_ndcg", "custom_bundle_ndcg", "legacy_research_metrics"],
        "layer": "DIAGNOSTIC",
        "action": "KEEP",
        "evidence": "自定义诊断指标；tests/test_evaluation_v2.py 与 research 实验脚本依赖。"
                    "Phase 11A 已明确其非标准 nDCG，仅 diagnostic-only。",
        "behavior_risk": "none（不进标准 gate）",
    },
    {
        "path": "src/evidenceflow/eval/v2.py",
        "symbols": ["ranking_metrics", "evaluate_evidence_coverage", "evaluate_rankings"],
        "layer": "COMPATIBILITY",
        "action": "KEEP",
        "evidence": "eval/v3.py 的 diagnostic block 与 tests/test_gold_stabilization.py 依赖；"
                    "CompleteEvidenceRR 兼容实现所在。",
        "behavior_risk": "none",
    },
    {
        "path": "src/evidenceflow/eval/quality.py",
        "symbols": ["CitationJudgment", "ClaimJudgment", "answer_metrics",
                    "abstention_metrics", "context_metrics"],
        "layer": "DIAGNOSTIC",
        "action": "KEEP",
        "evidence": "校准标注质量指标；tests/test_evaluation_v2.py 依赖。",
        "behavior_risk": "none",
    },
    {
        "path": "src/evidenceflow/eval/runner.py",
        "symbols": ["EvaluationRunner", "EvaluationReport", "save_evaluation_report"],
        "layer": "COMPATIBILITY",
        "action": "KEEP",
        "evidence": "v1 Evaluation Runner；tests/test_eval_runner.py 依赖，历史 artifact reader。",
        "behavior_risk": "none",
    },
    {
        "path": "src/evidenceflow/eval/comparison.py",
        "symbols": ["paired comparison utilities"],
        "layer": "STANDARD",
        "action": "KEEP",
        "evidence": "Phase 6/7/9 paired comparison 复现依赖。",
        "behavior_risk": "none",
    },
    {
        "path": "src/evidenceflow/eval/lexical.py",
        "symbols": ["BM25Index", "BM25RetrievedChunk"],
        "layer": "COMPATIBILITY_ONLY",
        "action": "KEEP",
        "evidence": "re-export 兼容层；experiments 与历史脚本按旧路径 import retrieval.bm25。",
        "behavior_risk": "none（删除会破坏旧实验复现）",
    },
    {
        "path": "src/evidenceflow/eval/hybrid.py",
        "symbols": ["HybridRRFChunk", "RankedChunkCandidate", "equal_ranked_rrf_fusion"],
        "layer": "COMPATIBILITY_ONLY",
        "action": "KEEP",
        "evidence": "re-export 兼容层；离线 hybrid 实验入口。",
        "behavior_risk": "none（删除会破坏旧实验复现）",
    },
    {
        "path": "src/evidenceflow/domain/evaluation.py",
        "symbols": ["EvaluationSource", "EvaluationQuestion"],
        "layer": "COMPATIBILITY",
        "action": "KEEP",
        "evidence": "v1 Dataset 合同；eval/dataset.py loader 仍消费。",
        "behavior_risk": "none",
    },
    {
        "path": "src/evidenceflow/domain/evaluation_v2.py",
        "symbols": ["MappingArtifact", "EvaluationDatasetV2", "EvidenceGroup"],
        "layer": "STANDARD",
        "action": "KEEP",
        "evidence": "qrels / Gold Mapping 的权威域模型。",
        "behavior_risk": "none",
    },
    {
        "path": "src/evidenceflow/eval/dataset.py",
        "symbols": ["load_evaluation_v2"],
        "layer": "DEAD",
        "action": "DELETE",
        "evidence": "静态扫描 src/tests/experiments 全部 0 引用（reachability_manifest 符号级计数），"
                    "且不在 eval/__init__ 的 __all__ 中；v2 数据集一律经 EvaluationDatasetV2.model_validate 读取。",
        "behavior_risk": "none；Runtime 与标准 IR 均不经过该函数",
    },
]

# ---------------------------------------------------------------- Step 7
LOG_CLEANUP = {
    "artifact_type": "phase11b_log_cleanup_manifest",
    "created_at": datetime.now(timezone.utc).isoformat(),
    "audit_scope": ["src/**/*.py", "运行时 INFO 字段", "临时/调试日志"],
    "findings": [
        {
            "path": "src/evidenceflow/retrieval/reranker.py",
            "line": 310,
            "level": "INFO",
            "fields": ["retrieval_strategy", "candidate_count", "reranker_enabled",
                       "reranker_input_count", "reranker_output_count", "reranker_latency",
                       "reranker_truncated_pair_count", "fallback_branch_used"],
            "assessment": "COMPLIANT",
            "action": "KEEP",
            "reason": "唯一一条 Runtime INFO；字段集合与 Step 7 建议保留项完全一致，"
                      "不含 Chunk 正文 / Prompt / embedding / 逐 candidate score dump。",
        },
        {
            "path": "src/evidenceflow/__main__.py",
            "line": 107,
            "level": "CLI_STDOUT",
            "fields": ["ingestion summary", "rag answer", "error type prefix"],
            "assessment": "COMPLIANT",
            "action": "KEEP",
            "reason": "CLI 用户输出而非应用日志；错误分支只打印异常类型与消息，不含敏感配置或正文。",
        },
    ],
    "sensitive_leak_check": {
        "chunk_text_logged": False,
        "prompt_logged": False,
        "embedding_logged": False,
        "api_key_logged": False,
        "per_candidate_score_dump": False,
    },
    "actions_taken": [],
    "conclusion": "日志层无需修改；无重复 traceback、无 batch 级重复日志、无 tokenizer debug 输出。",
}

# ---------------------------------------------------------------- Step 9
ARTIFACT_INVENTORY = [
    ("evals/evaluation_v2/phase6_retrieval_baselines_v1", "CANONICAL", "KEEP",
     "Phase 6 retrieval baseline 与 paired comparison。"),
    ("evals/evaluation_v2/phase7_reranker_validation_v1", "CANONICAL", "KEEP",
     "Phase 7 reranker validation（c0 ranking）。"),
    ("evals/evaluation_v2/phase8_runtime_readiness_v1", "CANONICAL", "KEEP",
     "Phase 8 runtime readiness。"),
    ("evals/evaluation_v2/phase9_runtime_promotion_v1", "CANONICAL", "KEEP",
     "Phase 9 runtime promotion，含 runtime_after.json（ranking 参照冻结 artifact）。"),
    ("evals/evaluation_v2/phase10_reranker_performance_v1", "CANONICAL", "KEEP",
     "Phase 10 性能与质量-延迟 Pareto。"),
    ("evals/evaluation_v2/phase35_gold_stabilization_v1", "CANONICAL", "KEEP",
     "Gold Mapping v3 与变更日志。"),
    ("evals/evaluation_v2/phase3_gold_audit_v1", "HISTORICAL_EVIDENCE", "KEEP",
     "Gold audit v2 与历史 ranking replay。"),
    ("evals/evaluation_v2/phase5_candidate_contract_v2", "HISTORICAL_EVIDENCE", "KEEP",
     "Candidate Contract v2。"),
    ("evals/evaluation_v2/cleanup_manifest_v2.json", "HISTORICAL_EVIDENCE", "KEEP",
     "Phase 11 之前一次 cleanup 的历史清单。"),
    ("evals/evaluation_v2/phase1_runtime_snapshot_v1.json", "HISTORICAL_EVIDENCE", "KEEP",
     "早期 runtime snapshot。"),
    ("evals/evaluation_v2/phase2_contract_validation_v1.json", "HISTORICAL_EVIDENCE", "KEEP",
     "早期 contract validation。"),
    ("evals/evaluation_v3/standard_ir_v1", "CANONICAL", "KEEP",
     "Phase 11A 标准 IR 指标、qrels、metric contract。"),
    ("evals/research_retrieval", "HISTORICAL_EVIDENCE", "KEEP",
     "research 6 个受控实验的离线结果。"),
    ("evals/research_ingestion", "HISTORICAL_EVIDENCE", "KEEP",
     "research corpus 摄取审计。"),
    ("evals/{diagnostics,experiments,fine_grained,holdout,query_variants,rag,rag_unanswerable,results}",
     "TEMPORARY", "DELETE",
     "全部为空目录（find -empty 核实且无任何代码引用），属历史 eval 版本的空骨架。"),
    ("tmp/{inspect_artifacts,inspect_deep,inspect_depths,inspect_p7,recompute_probe}.py",
     "TEMPORARY", "DELETE", "一次性 debug probe，无 import / test / artifact 引用。"),
    ("tmp/hybrid_retrieval_run.log, tmp/query_rewrite_run.log", "TEMPORARY", "DELETE",
     "实验运行日志副本，无引用。"),
    ("tmp/pdfs", "DEBUG_ONLY", "KEEP", "Gold 审计 PDF 截图；虽无代码引用，但属人工审计证据，"
                                        "不可由任何脚本确定性重建。"),
    ("tmp/generation_eval_result.json", "TEMPORARY", "KEEP",
     "debug gen_eval 的一次性输出；依赖 LLM 不可确定性重建，保留以便对照。"),
    ("experiments/debug/{inspect_retrieval,inspect_vector_store,chunk_id_verify,"
     "audit_research_ingestion,gen_eval}.py", "DEBUG_ONLY", "KEEP",
     "tests/test_debug_paths.py 与 test_inspect_*.py 直接 import，删除会破坏测试。"),
    ("experiments/debug/gen_flowchart.py", "DEBUG_ONLY", "KEEP",
     "docs/evidenceflow_flowchart.html 的生成源，属文档可复现性依赖。"),
    ("__pycache__/** 与 .pyc", "TEMPORARY", "DELETE", "字节码缓存，可确定性重建。"),
    (".pytest_cache", "TEMPORARY", "DELETE", "pytest 缓存，可确定性重建。"),
]

# ---------------------------------------------------------------- Step 3
RUNTIME_PATH_CLASSIFICATION = [
    {
        "area": "Translation",
        "symbol": "evidenceflow.translation.base.TranslationProvider",
        "before_role": "FALLBACK",
        "after_role": "FALLBACK_ONLY",
        "action": "KEEP",
        "reason": "合同保留；rollback 路径需要。",
        "dependency_evidence": "translation/llm.py 实现；runtime.py 仅在 reranker_enabled=false 分支装配。",
    },
    {
        "area": "Translation",
        "symbol": "evidenceflow.translation.llm.LLMTranslationProvider",
        "before_role": "FALLBACK",
        "after_role": "FALLBACK_ONLY",
        "action": "KEEP",
        "reason": "唯一实现，rollback 装配必需。",
        "dependency_evidence": "runtime.py:108；tests/test_translation.py。",
    },
    {
        "area": "BM25",
        "symbol": "evidenceflow.retrieval.bm25.BM25Index",
        "before_role": "FALLBACK",
        "after_role": "FALLBACK_ONLY",
        "action": "KEEP",
        "reason": "§17.3.1 正式 module；Hybrid 与离线实验共用。",
        "dependency_evidence": "runtime.py:_build_bm25_index；tests/test_lexical.py；"
                               "experiments/run_research_hybrid_*.py。",
    },
    {
        "area": "BM25",
        "symbol": "evidenceflow.eval.lexical (re-export)",
        "before_role": "COMPATIBILITY",
        "after_role": "COMPATIBILITY_ONLY",
        "action": "KEEP",
        "reason": "旧实验脚本按旧路径 import；删除即破坏历史复现。",
        "dependency_evidence": "eval/__init__.py 导出；experiments 历史脚本。",
    },
    {
        "area": "Hybrid",
        "symbol": "evidenceflow.retrieval.hybrid_fusion.equal_ranked_rrf_fusion",
        "before_role": "FALLBACK",
        "after_role": "KEEP_CONDITIONAL",
        "action": "KEEP",
        "reason": "multilingual_hybrid_enabled 时启用；默认关闭不影响主路径。",
        "dependency_evidence": "retrieval/multilingual.py；tests/test_hybrid_fusion.py。",
    },
    {
        "area": "Hybrid",
        "symbol": "evidenceflow.eval.hybrid (re-export)",
        "before_role": "COMPATIBILITY",
        "after_role": "COMPATIBILITY_ONLY",
        "action": "KEEP",
        "reason": "离线 hybrid 实验入口。",
        "dependency_evidence": "eval/__init__.py 导出。",
    },
    {
        "area": "RRF",
        "symbol": "evidenceflow.retrieval.fusion.weighted_rrf_fusion",
        "before_role": "ROLLBACK",
        "after_role": "ROLLBACK_ONLY",
        "action": "KEEP",
        "reason": "rollback 双路 candidate selection 与历史实验复现；不参与当前默认主路径。",
        "dependency_evidence": "retrieval/multilingual.py:134；tests/test_retrieval_fusion.py。",
    },
    {
        "area": "Runtime",
        "symbol": "evidenceflow.runtime.build_rag_service",
        "before_role": "PRODUCTION",
        "after_role": "PRODUCTION",
        "action": "KEEP",
        "reason": "唯一 composition root；已核实 CLI 不复制业务逻辑，不存在第二套 pipeline。",
        "dependency_evidence": "__main__.py 延迟导入；tests/test_runtime.py。",
    },
    {
        "area": "Eval",
        "symbol": "evidenceflow.eval.* → production 反向依赖",
        "before_role": "N/A",
        "after_role": "N/A",
        "action": "KEEP",
        "reason": "已核实无反向依赖：eval 不被 runtime/retrieval 导入"
                  "（tests/test_architecture_boundaries.py 守护）。",
        "dependency_evidence": "reachability_manifest：eval.* 全部 EXPERIMENT_ONLY / COMPATIBILITY_ONLY。",
    },
]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat()

    config = {
        "artifact_type": "phase11b_config_inventory",
        "created_at": now,
        "classification_rule": "PRODUCTION 保留；ROLLBACK/FALLBACK 保留并注明用途；"
                               "EXPERIMENT 视 Runtime 是否消费迁移；UNUSED 需五问全否才可删。",
        "fields": [
            {"field": name, "classification": cls, "action": action, "reason": reason}
            for name, cls, action, reason in CONFIG_INVENTORY
        ],
        "counts": {
            cls: sum(1 for _, c, _, _ in CONFIG_INVENTORY if c == cls)
            for cls in ("PRODUCTION", "ROLLBACK", "FALLBACK", "EXPERIMENT", "DEPRECATED", "UNUSED")
        },
    }

    eval_manifest = {
        "artifact_type": "phase11b_eval_cleanup_manifest",
        "created_at": now,
        "principle": "先证明无当前 import/test/runner 依赖，再删除；历史 JSON / report 一律不删。",
        "items": EVAL_CLEANUP,
        "deleted": [i for i in EVAL_CLEANUP if i["action"] == "DELETE"],
        "kept_layers": {
            "STANDARD": sum(1 for i in EVAL_CLEANUP if i["layer"] == "STANDARD"),
            "COMPATIBILITY": sum(1 for i in EVAL_CLEANUP if i["layer"] == "COMPATIBILITY"),
            "COMPATIBILITY_ONLY": sum(1 for i in EVAL_CLEANUP if i["layer"] == "COMPATIBILITY_ONLY"),
            "DIAGNOSTIC": sum(1 for i in EVAL_CLEANUP if i["layer"] == "DIAGNOSTIC"),
            "DEAD": sum(1 for i in EVAL_CLEANUP if i["layer"] == "DEAD"),
        },
    }

    artifacts = {
        "artifact_type": "phase11b_artifact_inventory",
        "created_at": now,
        "deleted_before_hash_recorded_in": "deleted_files.json",
        "items": [
            {"path": path, "classification": cls, "action": action, "reason": reason}
            for path, cls, action, reason in ARTIFACT_INVENTORY
        ],
    }

    for name, payload in (
        ("config_inventory.json", config),
        ("eval_cleanup_manifest.json", eval_manifest),
        ("log_cleanup_manifest.json", LOG_CLEANUP),
        ("artifact_inventory.json", artifacts),
    ):
        (OUT / name).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"written -> {OUT / name}")

    # 总清单最后单独生成（在代码删除与回归之后），此处先落 Step 3 分类。
    (OUT / "runtime_path_classification.json").write_text(
        json.dumps(
            {
                "artifact_type": "phase11b_runtime_path_classification",
                "created_at": now,
                "items": RUNTIME_PATH_CLASSIFICATION,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"written -> {OUT / 'runtime_path_classification.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
