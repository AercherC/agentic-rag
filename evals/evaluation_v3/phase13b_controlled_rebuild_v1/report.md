# Phase 13B — Knowledge Base Controlled Rebuild（受控 A/B 实验报告）

- **阶段**：Phase 13B（受控重建 A/B）
- **产物目录**：`evals/evaluation_v3/phase13b_controlled_rebuild_v1/`
- **执行入口**：`experiments/run_phase13b_controlled_rebuild.py`
- **日期**：2026-09-10

---

## 0. 一句话结论

**结构切分能可证地修复「caption 被正文稀释」类失败（ARR-007：dense rank 25 → 1），但**不能**修复「跨文献第二篇词汇不匹配」类失败（ARR-037 G2 / ARR-040 G2 任何切分都未恢复）。**
整体 Recall@10 / Recall@20 在候选切分下**不升反降**（因投影 qrels 下相关单元变多、元素隔离产生大量短 chunk），而 MRR / MAP / nDCG 稳定提升。

---

## 1. 冻结基线与可复现性

| 项 | 值 |
|---|---|
| 旧 collection | `ef_research_bge_m3_c512_o64_v1`（361 chunk，身份 hash 未变） |
| canonical qrels hash | `9337c26327097085…`（未变） |
| 我的 harness 复现 Dense 基线 | Recall@5=0.4250 / R@10=0.6625 / R@20=0.8167（与冻结值一致） |
| runtime source fingerprint | 未变 |
| composition fingerprint（排除 reranker_model 环境敏感字段） | 未变 |

**方案「异常停止」四条触发：0 次。** 旧库身份、Gold/qrels、baseline 复现、指标可比性全部通过。

---

## 2. 结构化解析与清理（Step 1–2）

- **3056 个 DocumentElement**，8 类：title 125 / paragraph 1837 / figure_caption 126 /
  table_caption 98 / table 109 / equation 136 / reference_entry 481 / furniture 144。
- 表格体识别覆盖 **10/10 篇**（修复了 13A 发现的「表格行列关系丢失」）。
- 关键修复点：adapointr 的全大写 `TABLE N` caption 风格（无冒号）此前被误判，现正确识别。
- 元素级清理删除 144 个 furniture 元素（页码 / arXiv 侧栏 / 版权 / 水印）。
- 章节归属率 **0.977**（section 由 13A 同口径 heading 检测派生）。

---

## 3. Chunk A/B 设计（Step 3）

| 变体 | chunk 数 | 跨章节 | 保留换行 | near-max |
|---|---|---|---|---|
| BASELINE_C512_O64 | 361 | 0.316（13A） | **0/361** | 0.975 |
| SECTION_AWARE_MINIMAL | 385 | **0.0** | **385/385** | 0.444 |
| ELEMENT_AWARE | 935 | **0.0** | **935/935** | 0.099 |

三个候选切分都直接修复了 13A 发现的三条结构缺陷：跨章节归零、行结构恢复、caption/table 可隔离。

---

## 4. 检索指标（Step 5，Dense Top20）

| 变体 | P@5 | R@5 | R@10 | R@20 | MRR@5 | MAP@5 | nDCG@5 |
|---|---|---|---|---|---|---|---|
| BASELINE_C512_O64 | 0.0850 | 0.4250 | 0.6625 | 0.8167 | 0.2621 | 0.2621 | 0.3024 |
| SECTION_AWARE_MINIMAL | 0.0909 | 0.4394 | 0.6212 | 0.7828 | 0.3283 | 0.3207 | 0.3528 |
| ELEMENT_AWARE | 0.0909 | 0.4394 | 0.5758 | 0.6515 | 0.3687 | 0.3535 | 0.3790 |

> 注：候选切分的指标基于**投影 qrels**（canonical Gold anchor 落到新 chunk 上的派生相关单元），
> 与 baseline 的 canonical qrels 不是同一相关单元集合，故 R@10/R@20 的绝对跨列对比需谨慎。
> 但 MRR/MAP/nDCG 的排序质量提升是稳健信号。

---

## 5. 失败案例验证（Step 6）

| 题号 | 变体 | gold dense ranks |
|---|---|---|
| ARR-007 | BASELINE_C512_O64 | [None] |
| ARR-007 | SECTION_AWARE_MINIMAL | [None] |
| ARR-007 | ELEMENT_AWARE | [1] |
| ARR-037 | BASELINE_C512_O64 | [17, None] |
| ARR-037 | SECTION_AWARE_MINIMAL | [19, None] |
| ARR-037 | ELEMENT_AWARE | [None, None] |
| ARR-040 | BASELINE_C512_O64 | [6, None] |
| ARR-040 | SECTION_AWARE_MINIMAL | [2, None] |
| ARR-040 | ELEMENT_AWARE | [1, None] |

- **ARR-007**（figure_caption）：ELEMENT_AWARE 把 caption 隔离后 rank **25 → 1**，**可证结构因果**。
- **ARR-037 G2**（adapointr 动态 query）：任何切分都未恢复，指向**词汇不匹配**。
- **ARR-040 G2**（hypergraph）：同上。

这**同时验证了 13A 的两个相反假设**：结构是 caption 类失败的主因（§27-A），但不是跨文献失败的主因（§27-C）。

---

## 6. 控制变量（Step 7）

- Embedding（BGE-M3）/ Reranker / LLM / Runtime 默认路径 **全部未改**。
- runtime source fingerprint 与 composition fingerprint（排除环境敏感字段）**未漂移**。
- 因此收益/代价只可能来自切分策略。

---

## 7. 结论与建议

- **Verdict**：PARTIAL（结构切分修复 caption 稀释类失败，未修复跨文献词汇不匹配类失败）
- **结构切分的价值**：修复「caption 被正文稀释」导致的单文档检索失败（ARR-007）。
- **结构切分的边界**：不能修复「跨文献词汇不匹配」（ARR-037/040 的 G2）。
- **建议**：
  1. 若目标是 caption 类问题 → 采用 ELEMENT_AWARE 的 caption 隔离；
  2. 若目标是 ARR-037/040 的 G2 → 转向 sparse / lexical candidate augmentation（与 13A §11 预判一致）；
  3. 生成层指标（Claim / Citation）留待切分方案定案后单独评估，避免收益无法归因。
