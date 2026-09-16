# Phase 11B — Runtime Stabilization + Cleanup

**Gate: PASS**

本阶段只做**行为不变的清理**，不引入任何新算法或新 Runtime 行为。

---

## 1. 清理前的基线（Step 0）

| 项目 | 值 |
|---|---|
| Tests | 408 passed / 0 failed / 17 warnings |
| Chroma collection | `ef_research_bge_m3_c512_o64_v1` |
| Chunk count | 361 |
| sorted chunk_id identity hash | `d81f9bc5…31b0e0` |
| qrels hash | `9337c263…c9bb9`（40 题 / 45 relevant chunks） |
| git | 不可用（项目目录无 `.git`，无法采集 status/diff） |

冻结标准 IR 基线：Precision@5 = 0.1150、Recall@5 = 0.5583、Recall@10 = 0.7333、MRR@5 = 0.4321、MAP@5 = 0.4279、nDCG@5 = 0.4619；Recall@20 继续 N/A。

## 2. Production Reachability Audit（Step 1 / 11）

从 `evidenceflow.__main__` / `evidenceflow.runtime` 做 AST import 闭包，63 个模块分类：

| 分类 | 数量 |
|---|---:|
| PRODUCTION_REACHABLE | 43 |
| EXPERIMENT_ONLY | 13 |
| FALLBACK_REACHABLE | 5 |
| COMPATIBILITY_ONLY | 2 |
| UNREACHABLE | 0 |

符号级扫描（src / tests / experiments 三处引用计数）发现的“零引用公共符号”共 1 个：
`evidenceflow.eval.dataset.load_evaluation_v2`（其余零引用项均为模块内使用的私有 helper 或 Protocol 类型注解）。

未使用 import 静态检测（自写 AST 检查器）：**0 处**。

## 3. Runtime 主路径（Step 2）

主路径唯一，无第二套 pipeline：

```text
Query → DenseRetriever → Chroma ANN Top20
      → DenseRerankerRetriever → BGE Reranker → Top5
      → ContextBuilder → Generator → Citation → RAGAnswer
```

- `runtime.build_rag_service` 是唯一 Composition Root；
- CLI（`__main__.py`）只做参数解析与渲染，业务装配全部延迟导入 `runtime`；
- `evidenceflow.eval.*` 不被 `runtime` / `retrieval` 反向依赖（`test_architecture_boundaries.py` 守护）。

## 4. Translation / BM25 / Hybrid / RRF（Step 3）

| 能力 | 分类 | 动作 |
|---|---|---|
| `TranslationProvider` / `LLMTranslationProvider` | FALLBACK_ONLY | KEEP |
| `BM25Index`（retrieval/bm25.py） | FALLBACK_ONLY | KEEP |
| `eval.lexical`（BM25 re-export） | COMPATIBILITY_ONLY | KEEP |
| `equal_ranked_rrf_fusion`（Hybrid） | KEEP_CONDITIONAL | KEEP |
| `eval.hybrid`（Equal RRF re-export） | COMPATIBILITY_ONLY | KEEP |
| `weighted_rrf_fusion`（RRF） | ROLLBACK_ONLY | KEEP |
| `MultilingualRetriever` | ROLLBACK | KEEP |

删除项：无。默认主路径不含上述任一分支（Hybrid 默认关闭，多语链路仅在 `RERANKER_ENABLED=false` 装配）。

## 5. Eval / Settings / Domain / Logging（Step 4–7）

- **Eval**：标准层（`ir_metrics.py`、`v3.py`、mapping/stable_mapping）KEEP；兼容层（v1 runner、`metrics.is_relevant`、`gold.hits_at_depths`、`v2.ranking_metrics`、两个 re-export）KEEP；诊断层（`quality`、`custom_*_ndcg`）KEEP 且明确 diagnostic-only；死代码 `load_evaluation_v2` DELETE。
- **Settings**：31 个字段分类 —— PRODUCTION 21、ROLLBACK 6、FALLBACK 4、UNUSED 1（`top_k`，Runtime 从不消费但 `tests/test_config.py` 有断言依赖，按规则保留并标记遗留）。
- **Domain Model**：cosine / RRF / reranker score / final rank 继续语义分离，未做模型合并或字段删除。
- **Logging**：全仓仅 1 条 Runtime INFO（`retrieval/reranker.py:310`），字段为 `retrieval_strategy / candidate_count / reranker_enabled / reranker_input_count / reranker_output_count / reranker_latency / reranker_truncated_pair_count / fallback_branch_used`，无 Chunk 正文 / Prompt / embedding / 逐 candidate dump / 敏感配置 → 无需修改。

## 6. 文件与 Artifact 清理（Step 8 / 9）

删除 125 条目录项（含复核补删的 `evals/rag`、`evals/rag/results`）：

| 类型 | 数量 |
|---|---:|
| 字节码文件（`*.pyc`） | 87 |
| 缓存目录（`__pycache__` ×16 + `.pytest_cache`） | 17 |
| 临时文件（debug probe ×5 + 实验日志 ×2） | 7 |
| 空目录（历史 eval 骨架） | 14 |
| 目录项合计 | 125 |

每个被删文件已在 `deleted_files.json` 记录 path / sha256 / size / reason。**未删除**任何 Gold、qrels、Mapping、Chroma 或 Phase 6–11A 关键 report。

## 7. 测试（Step 10 / 16）

未删除任何测试。408 → 408 passed，warnings 17 → 17，失败 0。
理由：候选删除项均承载 production / fallback / compatibility / historical 守护价值，删除会降低覆盖能力（详见 `removed_tests.json`）。

## 8. 回归结果（Step 13–15）

| 检查 | 结果 |
|---|---|
| 标准 IR 六项指标（冻结值比对） | 全部一致 |
| Evaluation v3 真机重算（40 题） | 全部一致 |
| 逐题 standard metrics 差异 | 0 |
| ranking 源 artifact hash | 未变（`ffc5cff4…4a3e85`） |
| Chroma chunk_count | 361（未变） |
| Chroma sorted chunk_id hash | `d81f9bc5…31b0e0`（未变） |
| Recall@20 | 继续 N/A |
| Rollback：`RERANKER_ENABLED=false` | PASS（装配 `MultilingualRetriever`） |
| Rollback：默认 `RERANKER_ENABLED=true` | PASS（装配 `DenseRerankerRetriever`） |
| Hybrid 条件路径开 / 关 | 两种配置均装配成功 |
| Translation / BM25 / Hybrid / RRF 实现存在性 | 全部为 true |

重算值：

| Metric | Frozen | Recomputed |
|---|---:|---:|
| Precision@5 | 0.1150 | 0.115000 |
| Recall@5 | 0.5583 | 0.558333 |
| Recall@10 | 0.7333 | 0.733333 |
| MRR@5 | 0.4321 | 0.432083 |
| MAP@5 | 0.4279 | 0.427917 |
| nDCG@5 | 0.4619 | 0.461851 |

## 9. 文档（Step 12）

- `README.md`：架构图改为 Phase 11B 后的三段式（Ingestion / Retrieval / Generation），Rollback 辅助路径单独表格，指标统一为标准 IR，新增 Phase 11B 产物索引。
- `PROJECT_SPEC.md`：§3 增加指向 §37 的“当前正式架构”说明，历史多语架构标记为 Historical / Legacy / Fallback；新增 §37 记录本阶段结论。

## 10. 剩余技术债

1. `Settings.top_k` 无 Runtime 消费，仅测试断言依赖；建议后续字段与断言一并下线。
2. Eval 层 v1 runner / legacy HitRate reader / custom diagnostic 指标虽已分层，但未拆分目录。
3. `tmp/pdfs`、`tmp/generation_eval_result.json` 属不可确定性重建的一次性审计证据，仍留在 `tmp/`。
4. `experiments/debug/` 下 6 个脚本被 `tests/` 直接 import，debug 与测试边界仍耦合。

## 11. 停止点

Phase 11B PASS 后停止，未启动 Lightweight Reranker、GPU、Quantization、Context Optimization、Dynamic Fallback 或任何 Agent / Memory 新功能。

---

## 12. Phase 11C 校正说明（2026-09-09 追加，不修改上方原始数据）

1. **fingerprint note 表述校正**：本节上方引用的 `runtime_fingerprint_note`（"仅 PROJECT_SPEC/README 文档与已删除死代码的 dataset.py"）
   把 README 与 `runtime_fingerprint_changed_files` 混为一谈。实际
   `runtime_fingerprint` 的采集范围只有 `PROJECT_SPEC.md` + `src/evidenceflow/**/*.py`，**不含 README.md**，
   因此 README 在 11B 被重写也不会出现在 changed_files 中。changed_files 是"scope 内差异"，
   note 是"本次改动的自然语言概述"，两者集合不同，不是数据冲突。详见
   `evals/evaluation_v3/phase11c_technical_debt_closure_v1/documentation_fingerprint_reconciliation.json`。

2. **`Settings.top_k` 分类更正**：11B 仅扫描了 Runtime 与 tests，将其判为 UNUSED。
   11C 补扫 `experiments/` 后发现 `experiments/debug/inspect_retrieval.py` 真实读取
   `settings.top_k`（Inspector 默认检索深度），因此更正为 **EXPERIMENT_ONLY**，字段保留不再下线。

3. **"6 个 debug 脚本被 tests import" 表述更正**：实际是 2 个 debug 脚本
   （`inspect_retrieval.py`、`inspect_vector_store.py`）中的 **6 个符号**被 tests 引用；
   11C 已将这 6 个纯函数上提到 `src/evidenceflow/inspection.py`，依赖方向收敛为 tests → src、experiments → src。
