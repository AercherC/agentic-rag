# Evaluation Phase History

主 README 按能力层组织；本文件保留 Phase 编号维度的历史记录与产物索引。
所有 hash 均以各产物目录内 manifest 为准，与冻结基线一致。

## Evaluation v2（Phase 3–10）

| Phase | 主题 | 产物目录 |
|---|---|---|
| Phase 3 | Gold audit | `evals/evaluation_v2/phase3_gold_audit_v1/` |
| Phase 3.5 | Gold stabilization | `evals/evaluation_v2/phase35_gold_stabilization_v1/` + `docs/PHASE35_GOLD_STABILIZATION.md` |
| Phase 5 | Candidate contract v2 | `evals/evaluation_v2/phase5_candidate_contract_v2/` |
| Phase 6 | Retrieval baselines | `evals/evaluation_v2/phase6_retrieval_baselines_v1/` |
| Phase 7 | Reranker validation | `evals/evaluation_v2/phase7_reranker_validation_v1/` |
| Phase 8 | Runtime readiness | `evals/evaluation_v2/phase8_runtime_readiness_v1/` |
| Phase 9 | Runtime promotion | `evals/evaluation_v2/phase9_runtime_promotion_v1/` |
| Phase 10 | Reranker performance | `evals/evaluation_v2/phase10_reranker_performance_v1/` |

早期设计文档：`docs/EVALUATION_V2_PHASES_0_3.md`。

## Evaluation v3（Phase 11–14）

| Phase | 主题 | 产物目录 |
|---|---|---|
| Phase 11A | 标准 IR 指标冻结 | `evals/evaluation_v3/standard_ir_v1/` |
| Phase 11B | 行为不变清理 | `evals/evaluation_v3/phase11b_cleanup_v1/` |
| Phase 11C | 技术债关闭 | `evals/evaluation_v3/phase11c_technical_debt_closure_v1/` |
| Phase 12 | RAG quality（answer/citation） | `evals/evaluation_v3/phase12_rag_quality_v1/` |
| Phase 13 KB | KB chunk audit | `evals/evaluation_v3/phase13_kb_chunk_audit_v1/` |
| Phase 13B | Controlled rebuild | `evals/evaluation_v3/phase13b_controlled_rebuild_v1/` |
| Phase 13C-A/B | PDF parser / cleaning fusion 基准 | `evals/evaluation_v3/phase13c_scientific_ingestion_benchmark/` |
| Phase 14A | DOCX/XLSX 多格式基准（已归档） | `evals/evaluation_v3/phase14a_multiformat_ingestion_benchmark/` |

## Research 检索线

语料构建 / ingestion audit：`evals/research_ingestion/`；
dense baseline、candidate augmentation、query rewrite、references-aware、
hybrid 实验与失败诊断：`evals/research_retrieval/`。

## 2026-09-15 项目清理（Project Cleanup）

非破坏性范围收口：Phase 14A 归档为 COMPLETE_CONTROLLED_EXPERIMENT、
可再生缓存清理、报告去重、README 能力层化。产物：`artifacts/cleanup/`。

| Artifact | 内容 |
|---|---|
| `pre_cleanup_manifest.json` | 清理前全库冻结清单（1221 files / sha256） |
| `cleanup_inventory.json/.md` | 全文件 A-H 分类清单 |
| `cleanup_plan.json` / `cleanup_dry_run.md` | 清理计划与 dry-run |
| `runtime_reachability.json` | AST 依赖扫描（src→phase14a = 0） |
| `chroma_collection_inventory.json` | 已删 Phase14 chroma 的 collection 清单（8 collections） |
| `cleanup_execution_log.json` | 实际删除日志（20.3MB） |
| `dedup_report.json` | 报告去重记录 |
| `obsolete_code_candidates.md` | 过时代码候选评审（本轮零删除） |
| `gate_verification.json` | Gate 1/3/5 验证结果 |
| `phase_directory_mapping.json` | Phase 编号 → 语义目录映射 |
| `EVIDENCEFLOW_CLEANUP_STATUS.md` | 最终验收报告 |

## Phase 11B/11C 原始 artifact 表（历史存档）

Phase 11B（行为不变清理）与 Phase 11C（技术债关闭）的逐 artifact 表已在
2026-09-15 从主 README 移入本文件。产物目录内各 manifest 保持自解释；
技术债状态：`tests_debug_dependency` = CLOSED，`settings_top_k` = KEEP，
`documentation_fingerprint` = CLOSED，`eval_directory_split` = ACCEPTED_DEBT，
`tmp_retained_files` = KEEP。
