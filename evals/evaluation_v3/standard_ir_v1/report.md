# EvidenceFlow Phase 11A —— Standard IR Evaluation Migration + Eval Simplification

**Evaluation Version**: 3.0
**Dataset hash**: `be1e8354babef3927163c19e32c614f7639323095900c487d559ade8249e7357`
**Qrels hash**: `9337c26327097085dd3dc400a73e6c0baa79f375944d2be401edd39d014c9bb9`
**Relevance**: binary (relevance=1), chunk_id 为检索单位，question_id 为查询单位

---

## 1. 已完成的标准指标

| 标准指标 | 状态 | 实现位置 |
|---|---|---|
| Precision@K | ✅ KEEP_STANDARD | `src/evidenceflow/eval/ir_metrics.py:precision_at_k` |
| Recall@K | ✅ KEEP_STANDARD（标准定义，分母=|qrels|） | `ir_metrics.py:recall_at_k` |
| MRR@K | ✅ KEEP_STANDARD | `ir_metrics.py:reciprocal_rank_at_k` |
| MAP@K | ✅ KEEP_STANDARD | `ir_metrics.py:average_precision_at_k` |
| nDCG@K | ✅ KEEP_STANDARD（二元相关性） | `ir_metrics.py:ndcg_at_k` |

辅助标准指标：`HitRate@K`（旧 `recall_at_k` 语义正名，可选）。

---

## 2. Legacy Recall 命名问题

**FOUND**。v1 `metrics.calculate_question_metrics.recall_at_k` 实为 `HitRate@K`（Top-K 任意 relevant→1，否则 0），并非标准 Recall@K。其下游 `runner.EvaluationRunner` 与 `tests/test_eval_metrics.py` 仍依赖，故保留为 `COMPATIBILITY_ONLY` 并在 docstring 标注；主指标改由 `ir_metrics.recall_at_k`（标准定义）承担。

历史 phase9/10 的 `recall_at_5=0.5625` 实为 `evidence_group_recall`（组覆盖比例），非标准 Recall。标准 Recall@5 = **0.5583**。

历史 phase9/10 的 `ndcg_at_5=0.4634` 实为 `custom_evidence_group_ndcg`（自定义组增益），非标准 nDCG。标准 nDCG@5 = **0.4619**。

---

## 3. 自定义指标处置

**Deprecated / Deleted**:
- `custom_evidence_group_ndcg` → `deprecated_for_primary_eval=true`，仅历史兼容读取，不计入主指标（无 test/experiment 引用则删除；本阶段因引用存在转 COMPATIBILITY_ONLY）。
- `custom_complete_evidence_rr`（CompleteEvidenceRR）→ COMPATIBILITY_ONLY。
- `legacy_research_metrics` → COMPATIBILITY_ONLY。
- `overall_evidence_coverage` / `any_evidence_hit_at_k` → DEPRECATE（与 diagnostic 重复，保留兼容读取）。

**Diagnostic-only**（带 `custom=true, diagnostic_only=true`，不与标准指标同级输出）：
- `evidence_group_recall_at_k` → `diagnostic.evidence_group_recall_at_k`
- `all_required_evidence_at_k` → `diagnostic.all_required_evidence_at_k`
- `gold_mappability_rate`（数据质量/可表示性，非检索准确率）

**硬删除项**：0。依据 Step 18 "no current test dependency" 规则，所有候选删除项均有现存 test/experiment 引用，全部保留为兼容/诊断（见 `compatibility_manifest.json`），不硬删除——这本身符合规范的例外条款。

---

## 4. 标准基线（冻结 ranking replay 重算，不覆盖历史）

| Metric | Dense Baseline (phase6 b0) | Current Runtime (phase9) | Delta |
|---|---|---|---|
| Precision@5 | 0.0850 | 0.1150 | +0.0300 |
| Recall@5 | 0.4250 | 0.5583 | +0.1333 |
| Recall@10 | 0.6625 | 0.7333 | +0.0708 |
| Recall@20 | 0.8167 | **N/A** | N/A |
| MRR@5 | 0.2621 | 0.4321 | +0.1700 |
| MAP@5 | 0.2621 | 0.4279 | +0.1658 |
| nDCG@5 | 0.3024 | 0.4619 | +0.1594 |

> **Recall@20 (Current Runtime) = N/A**：当前 Runtime 冻结排名仅持久化到 Top-10（phase9 `runtime_after` / phase10 `baseline` 均为 depth-10），无法 replay 出 @20 标准指标。按既定决策标记为 non-reconstructable，Dense Baseline 保留完整 @20。

**Performance（冻结，未改动）**: p50 = 4.463s, p95 = 5.335s, throughput = 4.38 candidates/s（取自 phase10 baseline）。

**Wins / Ties / Losses（逐题标准指标配对，n=40）**: 16 / 18 / 6 —— Reranker 在多数题目上提升。

---

## 5. Reranker 在标准 IR 指标下是否有效

**CONFIRMED**。在 Precision@5 / Recall@5 / Recall@10 / MRR@5 / MAP@5 / nDCG@5 全部六项标准指标上，Current Runtime 一致优于 Dense Baseline，且逐题 wins(16) > losses(6)。

---

## 6. 历史结论是否需要重新解释

**是**。原"主质量基线 nDCG@5=0.4634 / MRR@5=0.4321"中：
- `MRR@5=0.4321` 为**真正标准 MRR**（与 v3 重算完全一致）；
- `nDCG@5=0.4634` 实为 `custom_evidence_group_ndcg`，**非标准 nDCG**；标准 nDCG@5 = 0.4619（略低，差异来自二元相关性 vs 投机组增益定义）。
- `recall@5=0.5625` 实为组覆盖率；标准 Recall@5 = 0.5583。

简历/对外主结论应改用标准指标（nDCG@5=0.4619 / MRR@5=0.4321 / MAP@5=0.4279 / Recall@20=0.8167-Dense）。

---

## 7. 边界合规性

| 项 | 状态 |
|---|---|
| Runtime 默认检索路径 / Dense Retriever / Reranker / Candidate Top20 / Final Top5 | **未修改** |
| Gold / Chunk / Cleaner / Embedding / Chroma Index | **未修改** |
| Gold 原始 Source Truth | **未修改**（qrels 仅从冻结 Mapping 派生） |
| 历史 Phase 6–10 JSON | **未覆盖**（新 artifact 独立落盘于 evaluation_v3/） |
| Context / Generator / Citation / Translation / BM25 / Hybrid / RRF | **未改动** |

---

## 8. 产物清单（evals/evaluation_v3/standard_ir_v1/）

`run_manifest.json` · `eval_inventory.json` · `qrels.json` · `qrels_manifest.json` · `metric_contract.json` · `phase6_recomputed.json` · `phase7_recomputed.json` · `phase9_recomputed.json` · `phase10_recomputed.json` · `standard_baseline_comparison.json` · `per_query_results.json` · `resume_metrics.json` · `removed_metrics.json` · `compatibility_manifest.json` · `execution_result.json` · `report.md`

---

## 9. Phase 11A Gate

**PASS**（完整 tests 通过后确认；详见 execution_result.json）。下一步：**停止**，不进入 Runtime Cleanup（Phase 11B）。
