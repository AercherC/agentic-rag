# Phase 13A Anomaly Report

- raised_at_utc: 2026-09-10T12:35:30.860239+00:00
- stage: finalize
- clause: §4 停止条件：361 chunk 身份在审计过程中发生变化
- detail: {"chunk_count": 361, "sorted_chunk_id_hash": "d81f9bc529b008e8d0cad65a19f33330c81291458341d69362bbc5c7bf31b0e0", "local_recompute_matches_store": true}

按计划 §4（STOP-ON-ANOMALY）立即停止，不进入 Controlled Rebuild。

---

## 解除记录（同日，保留原文以形成完整审计轨迹）

**结论：误报，非真实异常。根因是审计脚本自身的取值错误。**

- `CorpusAudit.verify_identity()` 返回的键是 `local_recompute_matches_store` 与
  `sorted_chunk_id_hash`；而 `step_finalize()` 首版写成 `identity.get("matches")`，
  该键不存在 → 取到 `None` → 被当成"身份不一致"而抛出 `AuditAnomaly`。
- 判断依据就在该异常的 `detail` 里：`chunk_count = 361`、
  `local_recompute_matches_store = true`、
  `sorted_chunk_id_hash = d81f9bc529b008e8d0cad65a19f33330c81291458341d69362bbc5c7bf31b0e0`
  —— 与 Step 0 冻结值**完全一致**，即当时并不存在身份变化。

**修正内容**

1. `step_finalize()` 改为显式断言 `local_recompute_matches_store`，
   并**新增**一项比对：`identity["sorted_chunk_id_hash] == baseline.json 的冻结 hash`，
   任一不符才抛异常。
2. 顺带加严 Step 0：不再把 6 项标准 IR 指标作为常量抄进 artifact，而是
   **回读 `evals/evaluation_v3/standard_ir_v1/resume_metrics.json` 的 `current_runtime` 逐项比对**
   （容差 `METRIC_TOLERANCE = 5e-7`），并断言 `Recall@20` 仍为 `null`。
   重跑后 6/6 `matches = true`。

**复验结果**

- 重跑 `finalize` 通过；11 个产物的 sha256 已写入 `run_manifest.json`，
  并与磁盘内容逐项核对一致。
- 重跑 `baseline` / `decide` / `finalize` 全链通过，`PROCEED_TO_CONTROLLED_REBUILD = YES` 不变。
- 全量测试 471 passed / 0 failed。

**本阶段 §4 停止条款的真实触发次数 = 0。**
