# Phase 13E Hybrid Chunk Benchmark — Final Report

- 生成时间：2026-09-15T08:26:52.632337+00:00
- verdict：**PASS**（gate_definitions phase13e-gates-v1.0）
- 协议：同 13C-C（BGE-M3 Dense Top20、无 reranker、anchor 严格制）；standalone=caption/table/formula 独立 chunk

## Arm 结构统计

| arm | 策略 | chunks | token_mean | token_max | 来源 |
|---|---|---:|---:|---:|---|
| PH1 | standalone+section O0 | 1219 | 147.9 | 512 | 本轮 |
| PH2 | standalone+section O64 | 1230 | 151.7 | 512 | 本轮 |
| PH3 | standalone+fixed O64 | 816 | 232.7 | 512 | 本轮 |
| PC0 | fixed C512/O64 | 436 | — | — | 13C-C 冻结 |
| PC1 | section O0 | 960 | — | — | 13C-C 冻结 |
| PC2 | element-aware | 795 | — | — | 13C-C 冻结 |

## 标准 IR（aggregate）

| arm | P@5 | R@5 | R@10 | MRR@5 | MAP@5 | nDCG@5 | Hit@20 |
|---|---:|---:|---:|---:|---:|---:|---:|
| PH1 | 0.2350 | 0.5085 | 0.6223 | 0.5704 | 0.3719 | 0.4903 | 0.8500 |
| PH2 | 0.2350 | 0.5124 | 0.6222 | 0.5746 | 0.3729 | 0.4907 | 0.8500 |
| PH3 | 0.2350 | 0.4805 | 0.6124 | 0.4946 | 0.3192 | 0.4485 | 0.9000 |
| PC0 | 0.2450 | 0.4101 | 0.6561 | 0.4558 | 0.2937 | 0.4142 | 0.9500 |
| PC1 | 0.2400 | 0.4669 | 0.6050 | 0.4946 | 0.3347 | 0.4480 | 0.8750 |
| PC2 | 0.2000 | 0.4199 | 0.6097 | 0.4517 | 0.2712 | 0.3859 | 0.8500 |

## Bootstrap（n=10000，seed=20260915）

| metric | arm | vs | mean Δ | CI95 | P(Δ>0) | significant |
|---|---|---|---:|---|---:|---|
| ndcg_at_5 | PH1 | PC0 | +0.0761 | [-0.0358, +0.1922] | 0.908 | False |
| ndcg_at_5 | PH1 | PC1 | +0.0423 | [-0.0498, +0.1300] | 0.828 | False |
| ndcg_at_5 | PH2 | PC0 | +0.0766 | [-0.0355, +0.1920] | 0.909 | False |
| ndcg_at_5 | PH2 | PC1 | +0.0427 | [-0.0494, +0.1302] | 0.835 | False |
| ndcg_at_5 | PH3 | PC0 | +0.0343 | [-0.0725, +0.1456] | 0.735 | False |
| ndcg_at_5 | PH3 | PC1 | +0.0005 | [-0.1267, +0.1283] | 0.503 | False |
| recall_at_5 | PH1 | PC0 | +0.0984 | [-0.0380, +0.2418] | 0.916 | False |
| recall_at_5 | PH1 | PC1 | +0.0416 | [-0.0543, +0.1465] | 0.793 | False |
| recall_at_5 | PH2 | PC0 | +0.1023 | [-0.0335, +0.2468] | 0.923 | False |
| recall_at_5 | PH2 | PC1 | +0.0455 | [-0.0502, +0.1495] | 0.819 | False |
| recall_at_5 | PH3 | PC0 | +0.0705 | [-0.0525, +0.2006] | 0.870 | False |
| recall_at_5 | PH3 | PC1 | +0.0136 | [-0.1238, +0.1530] | 0.577 | False |
| recall_at_10 | PH1 | PC0 | -0.0338 | [-0.1372, +0.0733] | 0.261 | False |
| recall_at_10 | PH1 | PC1 | +0.0173 | [-0.0949, +0.1302] | 0.616 | False |
| recall_at_10 | PH2 | PC0 | -0.0339 | [-0.1373, +0.0731] | 0.261 | False |
| recall_at_10 | PH2 | PC1 | +0.0172 | [-0.0951, +0.1301] | 0.616 | False |
| recall_at_10 | PH3 | PC0 | -0.0437 | [-0.1462, +0.0540] | 0.196 | False |
| recall_at_10 | PH3 | PC1 | +0.0074 | [-0.1314, +0.1453] | 0.546 | False |
| mrr_at_5 | PH1 | PC0 | +0.1146 | [-0.0192, +0.2533] | 0.954 | False |
| mrr_at_5 | PH1 | PC1 | +0.0758 | [-0.0262, +0.1771] | 0.931 | False |
| mrr_at_5 | PH2 | PC0 | +0.1187 | [-0.0129, +0.2550] | 0.961 | False |
| mrr_at_5 | PH2 | PC1 | +0.0800 | [-0.0221, +0.1821] | 0.941 | False |
| mrr_at_5 | PH3 | PC0 | +0.0387 | [-0.0879, +0.1729] | 0.721 | False |
| mrr_at_5 | PH3 | PC1 | +0.0000 | [-0.1542, +0.1508] | 0.502 | False |

## Focus

- **ARR-007** — PH1: rank=2 | PH2: rank=2 | PH3: rank=1 | PC0: miss | PC1: miss | PC2: rank=1
- **ARR-022** — PH1: rank=1 | PH2: rank=1 | PH3: rank=1 | PC0: rank=17 | PC1: rank=1 | PC2: rank=4
- **ARR-037** — PH1: rank=8 | PH2: rank=9 | PH3: rank=8 | PC0: rank=6 | PC1: rank=7 | PC2: rank=7
- **ARR-040** — PH1: rank=3 | PH2: rank=3 | PH3: rank=2 | PC0: rank=1 | PC1: rank=3 | PC2: rank=3

## 硬门

- E_G1_inputs: True
- E_G2_coverage: {'PH1': True, 'PH2': True, 'PH3': True}
- E_G3_determinism: {'PH1': True, 'PH2': True, 'PH3': True}
- E_G4_provenance: {'PH1': True, 'PH2': True, 'PH3': True}
- E_G5_token_budget: {'PH1': True, 'PH2': True, 'PH3': True}
- E_G6_protocol: True
- E_G7_bootstrap_determinism: True

## 补充判读（按 PLAN §6 决策规则，数据驱动）

### 1. 主问题（E-G 决策规则 1）

- PH1/PH2 相对 PC0：方向一致正收益且幅度超过 13C-C 中 PC1 vs PC0 的增量
  （nDCG@5 +0.076 vs +0.034；R@5 +0.098/+0.102 vs +0.057），但 95% CI 均含 0，
  **未达显著**。
- 相对 PC1：+0.042 nDCG@5（P=0.83），方向正、幅度减半，同样不显著。
- **结论**：观察性正收益成立（"standalone+section 聚合"是六个臂中 R@5/MRR@5/nDCG@5
  最高的策略族），但按冻结规则不构成主路径变更依据；主路径维持 PC0。

### 2. ARR-007（决策规则 2）— 已恢复

- PC0/PC1：miss（top20 无相关）→ PH1 rank=2、PH2 rank=2、PH3 rank=1（PC2 rank=1）。
- 三个新臂全部 recovered，与 13D「瓶颈在 chunk 边界而非 ranker」的诊断闭环：
  caption/table 独立出块即修复该证据命中。
- 附带：ARR-022 由 PC0 的 rank=17 提升到 PH1/PH2 的 rank=1。

### 3. 归因分离（决策规则 3）

| 对照 | 隔离的组件 | nDCG@5 | R@5 | MRR@5 | 结论 |
|---|---|---:|---:|---:|---|
| PH3 vs PC0 | standalone（体同 fixed O64） | +0.034 | +0.070 | +0.039 | standalone 单独即恢复 caption/table 证据 |
| PH1 vs PH3 | section 聚合（均含 standalone） | +0.042 | +0.028 | +0.076 | section 聚合在 standalone 之上额外贡献排序质量 |
| PH1 vs PH2 | O64 overlap（section 体） | +0.0004 | -0.004 | -0.004 | overlap 无可测增益，PH1 更省（1219 vs 1230 chunks） |

- 两组件贡献相当且方向一致；若未来开 Runtime 晋升 Phase，PH1 为首选混合臂
  （性能与 PH2 持平、索引成本低）。

### 4. 回归监控（决策规则 4）

- 相对 PC1：R@10 +0.017、Hit@20 0.85 vs 0.875 — 无明显回落，旧风险未复现。
- 相对 PC0：R@10 -0.034（P=0.26）、Hit@20 0.85 vs 0.95 — 方向性轻微代价但
  在噪声范围内；细粒度化（436→1219 chunks）使 top20 竞争者变多，属预期权衡。

### 5. 总判定

- 7 门全过，verdict=PASS；混合策略（standalone + section 聚合）在本 40 题小语料
  上为方向一致、幅度可观但未显著的正收益，且精确修复了 caption/table 证据瓶颈。
- 不改主路径（PC0）；不因结果好直接切主路径（PLAN §8）；晋升须另开 Runtime Phase
  并在更大语料上复验显著性。
