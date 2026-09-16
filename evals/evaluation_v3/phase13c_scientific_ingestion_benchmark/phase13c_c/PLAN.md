# Phase 13C-C — PDF Chunk Benchmark 方案（v1，执行前冻结）

> 创建于 2026-09-15，先于任何执行动作落盘。核心问题（清理方案 §33）：
> **PDF cleaning/fusion 的结构提升（13C-B B2_fused）是否真正转化成 retrieval 提升？**

## 1. 目的

在冻结的 B2_fused ScientificDocument 上对比 4 种 chunk 策略的 Dense 检索质量，
隔离 chunk 变量（固定 tokenizer、embedding、Top20、qrels 投影规则），产出
是否进入 Sparse/BM25 candidate augmentation 的数据驱动决策。

## 2. 冻结输入（先复制后运行）

| 输入 | 来源 | 冻结位置 |
|---|---|---|
| B2_fused ScientificDocument × 10 篇真实论文 | `AppData/Local/Temp/evidenceflow_phase13c/cleaned/B2_fused/`（**易失，2026-09-15 确认仍在**） | `phase13c_c/frozen_inputs/b2_fused/`（复制 + sha256） |
| 40 题 Research Retrieval Dataset | `evals/research_retrieval/research_questions_v1.json`（dataset_hash=be1e8354…） | 原地引用，不复制不改 |
| 13C-B fusion_index.json | `phase13c_b/cleaned/fusion_index.json` | 原地引用（B2_fused 摘要一致性校验） |
| 生产 baseline 排名参照 | `standard_ir_v1/per_query_results.json`（phase10） | 只读参照，不参与 bootstrap |

T01–T12 模板文档不属于本 Phase（受控模板评测已在 13C-B 完成）。

## 3. Arm 定义（4 臂）

统一约束：tokenizer=BGE-M3（离线本地）、`target=512`、`add_special_tokens=False`、
计数与 14A-C 相同（拼接后精确计数）；chunk_id 含 arm 策略名（防跨臂碰撞）。

| Arm | 策略 | 说明 |
|---|---|---|
| PC0 | fixed_token C512/O64 | 基线：B2_fused 元素按 reading_order 串接（segment 间 `\n\n`），固定窗口滑切 |
| PC1 | section-aware C512/O0 | 按 `section_path` 分组，组内 pack 元素，**section 边界不跨**；超 512 的 section 内部再滑切 |
| PC2 | element-aware C512/O0 | 表格（table 元素序列化）与 figure caption、formula 元素各成独立 chunk（不足 512 不合并跨类型）；body 段落组内 token pack |
| PC3 | section-aware C512/O64 | 同 PC1 但组内滑切带 O64 overlap |

O0 的处理：单 section 超 512 时仍必须滑切（窗口 512 步长 512），section 间零重叠。

## 4. 检索评测协议

- Query：40 题统一用 `english_query` 字段（中→英不引入翻译变量；翻译问题在
  ARR 数据集已标注 language_group，分析时分层看但不作 arm）。
- Embedding：BGE-M3（`.cache/models` 或 HF offline，query 与 chunk 同模型同 normalize）。
- 每臂一个临时 Chroma collection（`ef_13cc_pc0..3`），cosine，取 Top20。
- **不接 Reranker**（隔离 chunk 变量；Top5 指标直接从 Top20 排名截取）。

## 5. qrels 投影（per-arm，anchor 严格制）

- relevant(chunk) = 该 chunk 的 owner elements 的 normalized/line_preserved 文本
  包含 gold `evidence_anchor`（归一化：NFKC 连字折叠 + casefold 大小写折叠 +
  空白折叠；冒烟测试实证 caption 大小写变体与 ﬁ/ﬂ 连字会阻断朴素子串匹配）。
  归一化只影响匹配，不改变二元相关语义。
- 多 gold source 任一命中即 relevant；二元相关，无分级。
- **页码不做 relevant 依据**（仅记录 chunk page 范围与 gold page 的重叠作为诊断）。
- 每臂独立投影（不同切分产生不同 chunk 集合）；`num_relevant` 按臂记录并汇报差异。

## 6. 指标与统计

- 主指标：nDCG@5；辅助：P@5、R@5、R@10、MRR@5、MAP@5（canonical 实现
  `src/evidenceflow/eval/ir_metrics.py`，禁止重写）。
- 统计：PC1/PC2/PC3 vs PC0 的 paired bootstrap（10,000 次重采样、seed=20260915、
  逐题指标差），报告 95% CI 与 p（双测）。
- 逐题 delta 表 + 三个 focus 题专项（见 §8）。

## 7. 硬门（gate_definitions_version = phase13c-c-gates-v1.0）

| Gate | 定义 | 判定 |
|---|---|---|
| C-G1 | 冻结输入：frozen_inputs 的 sha256 与 manifest 一致；40 题 dataset_hash 与冻结值一致 | 全等 PASS |
| C-G2 | anchor 覆盖：每臂每条 gold anchor（40 题全部 gold source）至少被该臂 ≥1 chunk 包含 | 每臂 coverage=1.0；某臂 <1.0 时该臂 FAIL 并输出丢失 anchor 清单（结构丢证据即硬伤） |
| C-G3 | 确定性：任意一臂重复切分两次，chunk_id 序列全等 | 全等 PASS |
| C-G4 | provenance：每 chunk owners ⊆ 该文档 B2_fused `element_id` 集合；伪造/悬空 owner=0 | 计数=0 PASS |
| C-G5 | token 预算：每 chunk token_count ≤ 512（独立 chunk 型保护元素不足不合并） | 违例=0 PASS |
| C-G6 | 协议一致：4 臂共用同一 embedding 模型/hash、同一 Top20 深度、同一投影规则代码路径 | 校验记录 PASS |
| C-G7 | 统计可复现：固定 seed 下 bootstrap 重复两次结果全等 | 全等 PASS |

测量语义修订必须升 gate_definitions 版本，阈值一个不动。

## 8. 决策规则（数据驱动，不预设结论）

1. **主问题**：若 PC1/PC2/PC3 相对 PC0 的 nDCG@5 提升 bootstrap 显著（CI 不含 0），
   结构清洗→检索转化成立；若全部不显著，结论为「13C-B 结构收益未转化到 chunk 级
   Dense 检索」，如实记录。
2. **ARR-007**（gold anchor="Figure 2: PCN architecture"，figure_caption 类）：
   PC2（caption 独立 chunk）是否 Top10 命中；对比其在 PC0 的名次。
3. **ARR-037 / ARR-040**（生产冻结排名 Top20 全 miss）：任一 PC 臂是否进入 Top20；
   若 4 臂仍全部 miss → 按 §33 进入 Sparse/BM25 candidate augmentation 评估，
   **不再回头改 parser**。
4. 任何结论必须能从 metrics JSON 复算；报告禁止硬编码数字。

## 9. 产物与代码位置

- 产物：`evals/evaluation_v3/phase13c_scientific_ingestion_benchmark/phase13c_c/`
  （frozen_inputs / chunks / embeddings(保留+hash 锚定) / chroma(临时，可删) /
  retrieval / metrics / gates / reports / manifest.json / status.json）
- 代码：`experiments/phase13c/chunk13cc.py`、`experiments/phase13c/metrics13cc.py`、
  `experiments/phase13c/runners/run_phase13c_c.py`；
  复用 `experiments/phase14a` 的纯函数（make_chunk / pack_units / token_len /
  project_qrels 思路 / paired_bootstrap）——**import 不复制、不改归档代码**。
- `src/evidenceflow/**` 零改动。

## 10. 明确不做

- 不修改生产 chunker / Runtime / Collection
- 不跑 LLM、不接 reranker、不做 query rewrite
- 不动 Gold、qrels 冻结文件、standard_ir_v1
- 不因过门修改 Gold；环境阻塞记 BLOCKED ≠ FAIL
