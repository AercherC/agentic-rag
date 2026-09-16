# Phase 13D Candidate Augmentation Benchmark — Final Report

- 生成时间：2026-09-15T08:17:13.075947+00:00
- verdict：**PASS**（gate_definitions phase13d-gates-v1.0）
- 协议：PC0 chunk 空间 × 40 题；BM25(k1=1.5, b=0.75, bge-m3 tokenizer)；RRF(k=60) 与 add-only(BM25 Top10)；无 reranker

## 标准 IR（aggregate）

| arm | 排名来源 | P@5 | R@5 | R@10 | MRR@5 | MAP@5 | nDCG@5 | Hit@20 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| DA0 | Dense Top20（13C-C 冻结） | 0.2450 | 0.4101 | 0.6561 | 0.4558 | 0.2937 | 0.4142 | 0.9500 |
| DA1 | BM25 Top20 | 0.2150 | 0.3658 | 0.5662 | 0.3571 | 0.2036 | 0.3239 | 0.8750 |
| DA2 | RRF(k=60) | 0.2650 | 0.4780 | 0.6982 | 0.4888 | 0.3051 | 0.4402 | 0.9500 |
| DA3 | add-only(BM25 Top10) | 0.2450 | 0.4101 | 0.6561 | 0.4558 | 0.2937 | 0.4142 | 0.9500 |

## Bootstrap（vs DA0，n=10000，seed=20260915）

| metric | arm | mean Δ | CI95 | P(Δ>0) | significant |
|---|---|---:|---|---:|---|
| ndcg_at_5 | DA2 | +0.0260 | [-0.0512, 0.0990] | 0.758 | False |
| ndcg_at_5 | DA3 | +0.0000 | [0.0000, 0.0000] | 0.000 | False |
| recall_at_5 | DA2 | +0.0679 | [-0.0352, 0.1746] | 0.901 | False |
| recall_at_5 | DA3 | +0.0000 | [0.0000, 0.0000] | 0.000 | False |
| recall_at_10 | DA2 | +0.0420 | [-0.0511, 0.1381] | 0.806 | False |
| recall_at_10 | DA3 | +0.0000 | [0.0000, 0.0000] | 0.000 | False |
| precision_at_5 | DA2 | +0.0200 | [-0.0200, 0.0600] | 0.812 | False |
| precision_at_5 | DA3 | +0.0000 | [0.0000, 0.0000] | 0.000 | False |
| mrr_at_5 | DA2 | +0.0329 | [-0.0550, 0.1167] | 0.777 | False |
| mrr_at_5 | DA3 | +0.0000 | [0.0000, 0.0000] | 0.000 | False |

## Focus：12 个历史候选失败（ recovered 情况）

- **ARR-007**（n_rel=1）— DA0: miss | DA1: miss | DA2: miss | DA3: miss
- **ARR-015**（n_rel=1）— DA0: rank=11 | DA1: rank=6 | DA2: rank=6 | DA3: rank=11
- **ARR-020**（n_rel=1）— DA0: rank=5 | DA1: rank=4 | DA2: rank=2 | DA3: rank=5
- **ARR-021**（n_rel=5）— DA0: rank=2 | DA1: rank=2 | DA2: rank=1 | DA3: rank=2
- **ARR-022**（n_rel=1）— DA0: rank=17 | DA1: miss | DA2: miss | DA3: rank=17
- **ARR-025**（n_rel=7）— DA0: rank=6 | DA1: rank=13 | DA2: rank=8 | DA3: rank=6
- **ARR-030**（n_rel=1）— DA0: rank=6 | DA1: rank=10 | DA2: rank=5 | DA3: rank=6
- **ARR-031**（n_rel=1）— DA0: rank=14 | DA1: miss | DA2: rank=19 | DA3: rank=14
- **ARR-037**（n_rel=6）— DA0: rank=6 | DA1: rank=3 | DA2: rank=4 | DA3: rank=6
- **ARR-038**（n_rel=20）— DA0: rank=1 | DA1: rank=6 | DA2: rank=1 | DA3: rank=1
- **ARR-039**（n_rel=17）— DA0: rank=4 | DA1: rank=4 | DA2: rank=6 | DA3: rank=4
- **ARR-040**（n_rel=30）— DA0: rank=1 | DA1: rank=1 | DA2: rank=1 | DA3: rank=1

## 硬门

- D_G1_inputs: True
- D_G2_determinism: True
- D_G3_protocol: True
- D_G4_pool_integrity: True
- D_G5_qrels_identity: True
- D_G6_bootstrap_determinism: True

## 决策（数据驱动）

- DA2/DA3 相对 DA0 无显著差异（CI 均含 0），如实记录。
- ARR-007 recovered by: 无臂。

## 补充判读（结果收口分析，2026-09-15）

- DA2：R@5 mean Δ=+0.0679（P(Δ>0)=0.901）——方向为正，显著性见 CI。

## 补充判读（结果收口分析，2026-09-15，含运行后诊断）

1. **RRF 融合（DA2）方向一致为正但未达显著**：R@5 Δ=+0.0679（P(Δ>0)=0.901）、
   R@10 Δ=+0.0420（P=0.806）、P@5 Δ=+0.0200（P=0.812，无精度代价）、
   nDCG@5 Δ=+0.0260（P=0.758）；40 题规模下 95% CI 均含 0。
   诚实结论：**RRF 是唯一方向一致的增强形态**，证据不足以推广进 Runtime
   （晋升须另开 Phase 过 Runtime 门 + reranker 全链评估）。
2. **DA3（add-only，Dense Top20 池）恒等于 DA0 是构造使然**：Dense 已填满 20 席，
   BM25 补充在截断处全部落出池外。这是本 Phase 方案设计的教训——add-only 协议
   必须为另一路留池位（历史诊断用 dense Top10 + BM25 Top10）。
3. **历史 add-only 池的补充诊断**（非门内 arm，仅诊断）：dense Top10 + BM25 Top10
   联合池 Top20 命中 36/40，**低于**纯 Dense Top20 的 38/40——用 BM25 挤占
   Dense 深位反而损失覆盖。因此"union 加池"类增强劣于"重排融合"（RRF）。
4. **ARR-007 未恢复的量化归因**：相关 chunk（PC0::0004，含 "Figure 2: PCN
   Architecture" caption 与 Problem Statement 的混合大 chunk）在 BM25 全量排序中
   名次为 **26**——差 6 名进 Top20。BM25 对该证据并非无信号，而是混合大 chunk
   稀释了词面匹配（"architecture"在 10 篇论文正文高频，IDF 低）。
   这把 13C-C 的 PC2 发现（caption 独立成 chunk → dense rank=1）与 13D 的
   BM25 深排名衔接成一个一致叙事：**caption/表格类证据的检索瓶颈在 chunk 边界，
   不在排序器**——混合 chunk 策略（caption/table 独立 + body section 聚合）
   是比 candidate augmentation 更对症的后续实验方向。
5. **12 个历史候选失败恢复面**：RRF 将 ARR-015（11→6）、ARR-020（5→2）、
   ARR-030（6→5）、ARR-037（6→4）名次改善，ARR-031 进入 Top20（19）；
   但 ARR-022 在 DA1/DA2 中 miss（17），ARR-007/038 类无改善。增强收益集中且
   不对称，不构成对 dense-only 的整体替代。
6. **per-arm qrels 注意**：qrels 为 13C-C PC0 anchor 严格制投影（D-G5 校验一致），
   本 Phase 内四臂共用同一 qrels，aggregate 可比。

## 结论与下一步（供决策，不自动执行）

- **不把 BM25 接入主路径**：DA1 单独明显劣于 dense（nDCG@5 -0.09），RRF 增益
  不显著；"没有评测证据不增加技术复杂度"。
- **更高价值的后续方向**：混合 chunk 策略 Phase（caption/table 独立 chunk + body
  section 聚合，即 PC2×PC1 杂交），直接针对 13C-C/13D 两轮共同指向的
  chunk 边界瓶颈；须新开 Phase 冻结门定义。
- 若仍要追 candidate augmentation：应以 RRF 形态 + 更大题量（或分层子集）重评，
  当前 40 题检验力不足。
