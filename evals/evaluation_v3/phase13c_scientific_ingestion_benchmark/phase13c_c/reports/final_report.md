# Phase 13C-C PDF Chunk Benchmark — Final Report

- 生成时间：2026-09-15T07:53:54.155323+00:00
- verdict：**PASS**（gate_definitions phase13c-c-gates-v1.0）
- 协议：B2_fused × 10 文档、40 题、BGE-M3 Dense Top20、无 reranker、anchor 严格制 qrels

## Arm 结构统计

| arm | 策略 | chunks | token_mean | token_max |
|---|---|---:|---:|---:|
| PC0 | fixed_token | 436 | 439.0 | 512 |
| PC1 | section_aware | 960 | 187.8 | 512 |
| PC2 | element_aware | 795 | 226.8 | 512 |
| PC3 | section_aware_overlap | 973 | 192.4 | 512 |

## 标准 IR（aggregate，40 题）

| arm | P@5 | R@5 | R@10 | MRR@5 | MAP@5 | nDCG@5 | Hit@20 |
|---|---:|---:|---:|---:|---:|---:|---:|
| PC0 | 0.2450 | 0.4101 | 0.6561 | 0.4558 | 0.2937 | 0.4142 | 0.9500 |
| PC1 | 0.2400 | 0.4669 | 0.6050 | 0.4946 | 0.3347 | 0.4480 | 0.8750 |
| PC2 | 0.2000 | 0.4199 | 0.6097 | 0.4517 | 0.2712 | 0.3859 | 0.8500 |
| PC3 | 0.2400 | 0.4627 | 0.5905 | 0.4883 | 0.3223 | 0.4406 | 0.8750 |

## Bootstrap（arm vs PC0，n=10000，seed=20260915）

| metric | arm | mean Δ | CI95 | P(Δ>0) | significant |
|---|---|---:|---|---:|---|
| ndcg_at_5 | PC1 | +0.0338 | [-0.0724, 0.1442] | 0.727 | False |
| ndcg_at_5 | PC2 | -0.0283 | [-0.1291, 0.0758] | 0.301 | False |
| ndcg_at_5 | PC3 | +0.0264 | [-0.0745, 0.1280] | 0.695 | False |
| recall_at_5 | PC1 | +0.0569 | [-0.0625, 0.1812] | 0.814 | False |
| recall_at_5 | PC2 | +0.0099 | [-0.1205, 0.1437] | 0.560 | False |
| recall_at_5 | PC3 | +0.0527 | [-0.0666, 0.1782] | 0.796 | False |
| recall_at_10 | PC1 | -0.0511 | [-0.1639, 0.0623] | 0.181 | False |
| recall_at_10 | PC2 | -0.0464 | [-0.1633, 0.0686] | 0.215 | False |
| recall_at_10 | PC3 | -0.0657 | [-0.1607, 0.0235] | 0.073 | False |
| mrr_at_5 | PC1 | +0.0387 | [-0.0863, 0.1700] | 0.719 | False |
| mrr_at_5 | PC2 | -0.0042 | [-0.1192, 0.1142] | 0.479 | False |
| mrr_at_5 | PC3 | +0.0325 | [-0.0892, 0.1554] | 0.696 | False |

## Focus Questions

- **ARR-007** — PC0: rank=None top20=False | PC1: rank=None top20=False | PC2: rank=1 top20=True | PC3: rank=None top20=False （生产 phase10：R@10=0.0，R@20=None）
- **ARR-037** — PC0: rank=6 top20=True | PC1: rank=7 top20=True | PC2: rank=7 top20=True | PC3: rank=8 top20=True （生产 phase10：R@10=0.0，R@20=None）
- **ARR-040** — PC0: rank=1 top20=True | PC1: rank=3 top20=True | PC2: rank=3 top20=True | PC3: rank=3 top20=True （生产 phase10：R@10=0.0，R@20=None）

## 硬门

- C_G1_inputs: True
- C_G2_coverage: {'PC0': True, 'PC1': True, 'PC2': True, 'PC3': True}
- C_G3_determinism: {'PC0': True, 'PC1': True, 'PC2': True, 'PC3': True}
- C_G4_provenance: {'PC0': True, 'PC1': True, 'PC2': True, 'PC3': True}
- C_G5_token_budget: {'PC0': True, 'PC1': True, 'PC2': True, 'PC3': True}
- C_G6_protocol_consistency: True
- C_G7_bootstrap_determinism: True

## 决策（数据驱动）

- 13C-B 结构收益未转化为 chunk 级 Dense 检索显著提升（nDCG@5 CI 均含 0），如实记录。
- ARR-037：至少一个 PC 臂进入 Top20。
- ARR-040：至少一个 PC 臂进入 Top20。

## 补充判读（结果收口分析，2026-09-15）

1. **主问题（结构收益是否转化）**：PC1/PC3 相对 PC0 在 nDCG@5 / R@5 / MRR@5 / MAP@5
   上方向一致为正（+0.026~+0.057），但在 40 题规模下 paired bootstrap 全部不显著
   （CI 均含 0）→ 诚实结论：**13C-B 的结构收益在 chunk 级 Dense 检索上表现为一致但
   不显著的正向趋势**，不构成"显著转化"的证据，也不构成否定。
2. **PC2（element-aware）整体为负**（nDCG@5 -0.028，P@5 最低），但在 **ARR-007
   （figure_caption 类证据）上是唯一命中臂且 rank=1**：caption 独立成 chunk 的个案
   价值明确，而正文打包过碎带来整体损失。caption/table 类独立 + body 保持 section
   聚合的混合策略是后续候选，但**按项目原则不凭单点个案改主路径**，仅记录。
3. **ARR-037 / ARR-040**：四臂全部 Top10 命中。注意生产 phase10 参照值（R@10=0.0）
   与本 benchmark 不可直接对比——qrels 投影口径（anchor 严格制 vs 页码）与检索路径
   （dense-only vs 翻译 RRF）均不同。本结果能支撑的结论是：**gold 证据在 B2_fused
   chunk 空间中是 dense 可召回的**，生产层 miss 更可能来自旧 parser 文本质量与排序
   路径，而非证据缺失。按方案 §33，下一步评估 Sparse/BM25 candidate augmentation
   时应基于 B2_fused 文本空间重建候选层，而不是回改 parser。
4. **深截断上 PC0 占优**（R@10 0.6561、Hit@20 0.9500 均最高）：粗粒度大 chunk 在
   Top20 覆盖面上占优，与 14A-C「小语料下粗 chunk 占优」同构；chunk 数量差异
   （436 vs 960/973）使 Top20 的物理覆盖密度不同，跨臂比较时须注意该混淆。
5. **per-arm qrels 固有注意**：每臂独立投影导致 num_relevant 逐臂不同
   （如 ARR-040：29~37），aggregate 指标跨臂可比性有限；逐题 delta 与 bootstrap
   在共有 eligible query 上配对，口径已对齐，但绝对值解读需谨慎。

## 后续动作建议（供决策，不自动执行）

- 主路径 chunk 策略维持 PC0（fixed C512/O64）不变：无显著证据支持切换。
- ARR-037/040 方向：基于 B2_fused 文本空间的 BM25 candidate augmentation 评估
  （清理方案 §33 预定路径）。
- 若后续进入混合 chunk 策略实验（caption/table 独立 + section 聚合），须新开
  Phase 并冻结门定义，不在 13C-C 上追加。
