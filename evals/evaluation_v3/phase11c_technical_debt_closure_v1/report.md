# Phase 11C：Technical Debt Closure + Evaluation Conformance Check

- **Gate：PASS**
- 产物目录：`evals/evaluation_v3/phase11c_technical_debt_closure_v1/`
- 本阶段**不引入新算法、不新增第三方依赖、不改变 Runtime 行为**。

---

## 1. 背景与目标

Phase 9 / 10 / 11A / 11B 已证明：默认 Runtime 唯一、标准 IR 指标稳定、Cleanup 前后 ranking 完全一致、
Chroma 身份一致、Rollback 可用、408 tests 全通过。因此 11C 不再做大规模清理，只关闭 11B 已确认的
剩余小型技术债，然后冻结当前 RAG Runtime 与 Evaluation 作为稳定基线。

执行顺序：Baseline → debug dependency → top_k audit → doc/fingerprint reconciliation →
metric conformance → regression → stop。

---

## 2. Step 0 — 冻结 11C Baseline（`baseline.json`）

在任何代码修改之前采集：

| 项 | 值 |
|---|---|
| Full tests | 408 passed / 0 failed / 17 warnings |
| Precision@5 / Recall@5 / Recall@10 | 0.1150 / 0.5583 / 0.7333 |
| MRR@5 / MAP@5 / nDCG@5 | 0.4321 / 0.4279 / 0.4619 |
| qrels hash | `9337c26327097085dd3dc400a73e6c0baa79f375944d2be401edd39d014c9bb9`（40Q / 45 relevant chunks） |
| Phase 9 `runtime_after` ranking hash | `ffc5cff422eb581888a4a5ada9fadf0ee884ff69d8613ae1ca16d9dbf26a3e85` |
| Chroma | `ef_research_bge_m3_c512_o64_v1`，361 chunks，`d81f9bc529b008e8d0cad65a19f33330c81291458341d69362bbc5c7bf31b0e0` |
| dataset hash | `be1e8354babef3927163c19e32c614f7639323095900c487d559ade8249e7357` |
| Runtime composition fingerprint | `1d932659bb5774ae…` |
| Rollback config fingerprint | `c3b3db3b5ea185e7…` |
| 文件清单 hash | `9f95bcb6abf4a384…` |
| README.md / PROJECT_SPEC.md hash | `85df837c…` / `a209511e…` |

> 说明：Runtime composition fingerprint = 检索相关 Settings 子集 + 装配模块源码 hash，
> 只 hash 配置与源码、不加载模型，避免把环境差异混入。

---

## 3. Step 1 — 解耦 tests → experiments/debug（`debug_dependency_migration.json`）

**11B 记录表述更正**：不是"6 个 debug 脚本被 tests import"，而是 **2 个 debug 脚本中的 6 个符号**
（`inspect_retrieval.py` 4 个 + `inspect_vector_store.py` 2 个）。

按规则逐符号分类：全部为 **PRODUCTION_REUSABLE** —— 纯函数、无模型加载 / IO / 全局状态，
且 tests 与 experiments 双方长期复用同一份实现。因此迁移到 `src/evidenceflow/inspection.py`，
迁移后依赖方向收敛为：

```text
src/evidenceflow/inspection.py
        ↑                ↑
      tests        experiments/debug
```

| 符号 | 迁移前 | 迁移后 |
|---|---|---|
| `content_preview` / `format_page` / `format_retrieval_result` / `format_multilingual_result` | `experiments.debug.inspect_retrieval` | `evidenceflow.inspection` |
| `select_records` / `format_storage_record` | `experiments.debug.inspect_vector_store` | `evidenceflow.inspection` |

- 迁移后 `tests → experiments.debug` 的 import 数 = **0**（`decoupled = true`）。
- 新增守护测试 `test_tests_do_not_import_experiments_debug`，防止依赖回流。
- 未删除任何 debug 脚本：它们仍被交互式巡检使用，且 `inspect_vector_store` 现只从 src 取依赖。
- 本阶段**不处理** `tests → experiments.*(非 debug)` 的 7 个既有实验脚本单测（Phase 6/8/9 与 research 实验），
  超出本阶段范围，记录为 out-of-scope。

**改动文件**：新增 `src/evidenceflow/inspection.py`；改写 `experiments/debug/inspect_retrieval.py`、
`experiments/debug/inspect_vector_store.py`；`tests/test_inspect_retrieval.py`、
`tests/test_inspect_vector_store.py` 仅改 import；`tests/test_architecture_boundaries.py` 新增 1 条断言。

迁移后清理：两个 debug 脚本不再直接引用 `content_preview` / `format_page`
（二者现在只由 `inspection.py` 内部调用），已删除对应 import，保持 11B 「未使用 import = 0」的标准。


---

## 4. Step 2 — `Settings.top_k` 审计（`dead_config_audit.json`）

结论：**KEEP**（发现真实消费者，按规则停止删除）。

| 检查项 | 结果 |
|---|---|
| Runtime 是否读取 | 否（PRODUCTION 命中 0 处） |
| CLI 是否读取 | 否 |
| `.env.example` 是否公开 | 是 |
| README 是否公开为当前配置 | 否 |
| Rollback / fallback 路径是否读取 | 否 |
| **experiments 是否读取** | **是**（`experiments/debug/inspect_retrieval.py:59 / :80） |
| historical artifact reader 是否读取 | 否 |
| Settings snapshot 是否要求该字段存在 | 是 |

真实消费者：`inspect_retrieval.py` 把 `settings.top_k` 作为 Inspector 的 `DenseRetriever` 默认深度，
并作为 `--top-k` 的缺省值。这是 experiments 层的合法使用，因此：

- 保留 `Settings.top_k`，保留 `tests/test_config.py` 中的断言，不修改 `.env.example` / README。
- **11B 分类更正**：`UNUSED` → `EXPERIMENT_ONLY`（11B 只扫描了 Runtime 与 tests，未扫 `experiments/`），
  已在 `PROJECT_SPEC.md` §37.5 标注更正。

---

## 5. Step 3 — README / PROJECT_SPEC / fingerprint 一致性核对
（`documentation_fingerprint_reconciliation.json`）

**问题**：11B `execution_result.json` 的 `runtime_fingerprint_changed_files` 只有
`[PROJECT_SPEC.md, src/evidenceflow/eval/dataset.py]`，但 `runtime_fingerprint_note` 却写了"PROJECT_SPEC/README 文档"。

**结论：表述不准确，不是数据冲突。**

- `runtime_fingerprint` 采集范围 = `PROJECT_SPEC.md` + `src/evidenceflow/**/*.py`，**不含 README.md**。
- 因此 README 在 11B 被重写，也不会出现在 `changed_files` 里。
- mtime 证据：11B baseline 冻结于 `2026-09-08T15:20:34Z`（本地 23:20:34），
  README / PROJECT_SPEC 的 mtime 分别为 23:41 / 23:44 —— 文档改动发生在 baseline 冻结之后，
  属 11B Step 12「documentation cleanup」的预期动作。
- `changed_files` 是**scope 内差异**，`note` 是**本次改动的自然语言概述**，两者集合不同。
- 伪路径检查：仓库根目录只有 `README.md` 与 `PROJECT_SPEC.md` 两个真实文档，
  不存在任何带前缀的同名文档；`changed_files` 条目均为普通相对路径。

**处理**：只在 11B `report.md` 追加「Phase 11C 校正说明」章节，
**不修改 11B 的任何实验数据 JSON**。

---

## 6. Step 4 — 标准 IR 指标语义一致性 Cross-check（`metric_conformance.json`）

- 参考库探测：`pytrec_eval` / `beir` / `ranx` / `ir_measures` / `trectools` **全部不可用**
  → 采用**方案 B**，不新增大型依赖。
- 方法：在脚本内按教科书定义**独立实现**一份参考实现（二元相关性、首次出现去重、
  nDCG gain=1 / discount=`log2(rank+1)`），生成冻结 TREC-style fixture，
  再与 `src/evidenceflow/eval/ir_metrics.py` 逐指标比对。
- Fixture：4 个 query，覆盖正常命中、**零命中**、**重复 chunk_id 去重**、**无 qrels 需排除**四类；
  深度 1/3/5/10，指标 Precision / Recall / MRR / MAP / nDCG。
- 结果：**120 项比对，0 处不一致**（容差 `1e-12`）。

Fixture 抽查（与手工推导一致）：

| Query | P@5 | R@5 | R@10 | MRR@5 | MAP@5 | nDCG@5 |
|---|---:|---:|---:|---:|---:|---:|
| q1（`[c2,c1,c4,c3,c6,c5]`） | 0.400000 | 0.666667 | 1.000000 | 0.500000 | 0.333333 | 0.498189 |
| q2（零命中） | 0 | 0 | 0 | 0 | 0 | 0 |
| q3（含重复 `cA`） | 0.400000 | 1.000000 | 1.000000 | 1.000000 | 0.750000 | 0.877215 |

新增单元测试 `tests/test_ir_metric_conformance.py`（9 条），把 fixture 变成长期回归资产。

---

## 7. Step 5 — Eval 物理目录拆分：NO ACTION（ACCEPTED_DEBT）

standard / diagnostic / compatibility 三层 **contract 已能区分**，拆物理目录不提升质量，
且会造成大量 import churn。记录为 **ACCEPTED_DEBT**，不再推进。

---

## 8. Step 6 — tmp/ 保留项确认（`tmp_retention.json`）

| 路径 | Owner | Purpose | Last consumer | 可重建性 | 结论 |
|---|---|---|---|---|---|
| `tmp/pdfs`（70 文件） | Gold 审计 | 人工审计 Gold 标注的 PDF 截图证据 | 人工复核 | 不可确定性重建 | KEEP |
| `tmp/generation_eval_result.json`（11,233 B） | `experiments/debug/gen_eval.py` | 一次性 Generation 评估输出 | `gen_eval.py`（写入方） | 依赖真实 LLM 调用，不可重建 | KEEP |

只确认 owner / purpose / last_consumer / reproducibility_value，**不强删**。

---

## 9. Step 7 — Full Regression

```
418 passed, 17 warnings in 25.68s
```

- 0 failed；warnings 数量与构成与基线一致（17，全部来自 chromadb 弃用告警），**无新增**。
- 数量 408 → 418：Step 1 依赖守护 +1，Step 4 指标一致性 +9。
  **未删除任何测试**，production / fallback / eval 覆盖均无下降。

---

## 10. Step 8 — Evaluation v3 Regression（`runtime_regression.json`）

重算冻结 40Q ranking（不重跑检索，输入为 `phase9_runtime_promotion_v1/runtime_after.json`）：

| Metric | 目标值 | 重算值 | abs diff |
|---|---:|---:|---:|
| Precision@5 | 0.115000 | 0.11500000000000002 | 1.39e-17 |
| Recall@5 | 0.558333 | 0.5583333333333333 | 3.33e-07 |
| Recall@10 | 0.733333 | 0.7333333333333333 | 3.33e-07 |
| MRR@5 | 0.432083 | 0.4320833333333334 | 3.33e-07 |
| MAP@5 | 0.427917 | 0.42791666666666667 | 3.33e-07 |
| nDCG@5 | 0.461851 | 0.46185080710419263 | 1.93e-07 |

- 逐题差异：**0**（40/40 完全一致）
- ranking IDs：完全一致（`ranking_ids_sha256 = 648474e70dd77df5…`）
- qrels hash 不变；dataset hash 不变
- **Recall@20：继续 N/A**（冻结排名仅 Top-10，不重新发明）

---

## 11. Step 9 — Chroma / Runtime / Rollback 回归

- Chroma：**361 chunks**，identity hash 与基线一致（`identity_unchanged = true`）。
- Runtime composition fingerprint：与基线一致（`unchanged = true`）；Rollback 配置 fingerprint 一致。
- 装配验证（`rollback_validation.json`）：

| 配置 | Retriever |
|---|---|
| `RERANKER_ENABLED=true` | `DenseRerankerRetriever` |
| `RERANKER_ENABLED=false` | `MultilingualRetriever` |
| `RERANKER_ENABLED=false` + `MULTILINGUAL_HYBRID_ENABLED=false` | `MultilingualRetriever` |
| `RERANKER_ENABLED=false` + `MULTILINGUAL_HYBRID_ENABLED=true` | `MultilingualRetriever` |

- Translation / BM25 / Hybrid / RRF 实现仍存在（`LLMTranslationProvider`、`BM25Index`、
  `equal_ranked_rrf_fusion`、`weighted_rrf_fusion`、`MultilingualRetriever` 全部可导入）。
- **Rollback：PASS**。

---

## 12. Step 10 — 最终技术债清单（`technical_debt_closure.json`）

| ID | Status | 说明 |
|---|---|---|
| `tests_debug_dependency` | **CLOSED** | 6 个纯函数迁入 `src/evidenceflow/inspection.py`；tests → experiments.debug = 0 |
| `settings_top_k` | **KEEP** | experiments 层真实消费者；11B 的 UNUSED 分类已更正为 EXPERIMENT_ONLY |
| `documentation_fingerprint` | **CLOSED** | fingerprint scope 不含 README；11B note 表述已校正，历史数据未改 |
| `eval_directory_split` | **ACCEPTED_DEBT** | contract 已分层，物理不拆；NO ACTION |
| `tmp_retained_files` | **KEEP** | 不可确定性重建，只确认不强删 |

---

## 13. Gate

| Check | 结果 |
|---|---|
| tests → experiments/debug 依赖已解耦 | PASS |
| Settings.top_k 已处理（KEEP 并更正分类） | PASS |
| README / fingerprint 记录一致性已解释 | PASS |
| Standard IR metric conformance | PASS（120/120） |
| Evaluation v3 指标完全不变 | PASS |
| per-query difference = 0 | PASS |
| Chroma identity 不变 | PASS |
| Runtime composition 不变 | PASS |
| rollback PASS | PASS |
| full tests PASS | PASS（418 / 0 / 17 warnings） |
| Eval 目录拆分记录为 ACCEPTED_DEBT | PASS |
| 无新算法 / 新依赖链 / 新 Runtime 行为 | PASS |

**Phase 11C Gate = PASS。**

---

## 14. 「是否达到标准」的口径

按本阶段第 19 节的两层判断：

- **A. Evaluation Methodology Standard = STANDARDIZED**：已具备 qrels +
  Precision / Recall / MRR / MAP / nDCG + 冻结 baseline + paired comparison +
  historical artifact + regression，并通过独立参考实现的语义一致性 cross-check。
- **B. Retrieval Quality Benchmark Standard**：当前为**私有 40Q / 361-chunk 数据集**，
  只能在同数据集内声明 `internal benchmark improvement`；
  **不得**把 `nDCG@5 = 0.461851` 与其他公开 benchmark 数字横向比较。

---

## 15. 停止点

Phase 11C PASS 后停止，**Cleanup 系列正式结束**。

后续 Eval 目录拆分、新 Reranker、GPU、Quantization、Context Optimization、
Dynamic Fallback、Agent / Memory 均须作为**新的产品/研究目标单独立项**。
