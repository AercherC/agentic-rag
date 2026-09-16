# Phase 13E — Hybrid Chunk Benchmark 方案（v1，执行前冻结）

> 创建于 2026-09-15。动机：13C-C 显示 caption 独立 chunk 使 ARR-007 dense rank=1
> （但 PC2 整体为负），13D 显示 ARR-007 相关 chunk 在 BM25 深排名 26——两轮共同
> 指向 **caption/表格类证据的瓶颈在 chunk 边界**。本 Phase 回答：
> **"caption/table/formula 独立 chunk + body section 聚合"的混合策略能否同时保住
> PC1 的整体收益与 PC2 的 caption 个案收益？**

## 1. 冻结输入（复用 13C-C，参照臂零重算）

| 输入 | 来源 | 用途 |
|---|---|---|
| B2_fused ×10 | `phase13c_c/frozen_inputs/b2_fused/` | 新臂切分输入 |
| PC0/PC1/PC2 chunks + runs | `phase13c_c/chunks/*.jsonl`、`retrieval/*_run.json` | 参照臂（hash 对 13C-C manifest） |
| PC0/PC1/PC2 逐题 IR | `phase13c_c/metrics/ir_per_query.json` | bootstrap 参照值 |
| 40 题 | `evals/research_retrieval/research_questions_v1.json` | 同 13C-C（english_query） |

## 2. Arm 定义（3 新臂 + 3 参照臂）

统一约束同 13C-C：target=512、bge-m3 tokenizer、add_special_tokens=False、
chunk_id 含 arm 策略名；独立 chunk 语义与 PC2 一致（table/
table_caption/figure_caption/equation 各自成 chunk 并**从 body 流中移除**，不重复）。

| Arm | standalone | body 策略 | 说明 |
|---|---|---|---|
| PH1 | 同 PC2 | section-aware O0 | PC2×PC1 杂交主臂 |
| PH2 | 同 PC2 | section-aware O64 | 杂交 + overlap |
| PH3 | 同 PC2 | fixed-token C512/O64 | 归因臂：区分"caption 独立"与"section 聚合"各自贡献 |
| PC0/PC1/PC2 | — | — | 13C-C 冻结结果直接作参照，不重算 |

## 3. 检索评测协议

与 13C-C 完全一致：BGE-M3 CLS 池化 + L2 归一化、每臂临时 Chroma（cosine）、
Dense Top20、无 reranker、anchor 严格制 qrels（NFKC+casefold+空白折叠，
逐臂投影）。

## 4. 指标与统计

- 标准 IR：P@5、R@5、R@10、MRR@5、MAP@5、nDCG@5、Hit@20（canonical 实现）。
- 配对 bootstrap（n=10000、seed=20260915、×2 全等）：
  PH1/PH2/PH3 vs **PC0**（生产对齐基线）与 vs **PC1**（section 参照），
  指标 nDCG@5 / R@5 / R@10 / MRR@5。
- Focus：ARR-007（必须恢复目标）、ARR-022（13D 观察到的回归风险案例）、
  ARR-037/ARR-040（持续跟踪）。

## 5. 硬门（gate_definitions_version = phase13e-gates-v1.0）

| Gate | 定义 | 判定 |
|---|---|---|
| E-G1 | 冻结输入/参照臂文件 hash 与 13C-C manifest 一致 | 全等 PASS |
| E-G2 | anchor 覆盖：每个新臂 gold anchor 覆盖率=1.0 | 每臂全过 |
| E-G3 | 确定性：新臂切分两次 chunk_id+text 全等 | 全等 PASS |
| E-G4 | provenance：新臂 owners ⊆ B2_fused element_ids | 违例=0 PASS |
| E-G5 | token 预算：新臂 chunk token ≤512 | 违例=0 PASS |
| E-G6 | 协议一致：同 embedding/Top20/qrels 规则；参照臂 qrels 重投影与 13C-C num_relevant 逐题一致 | 校验 PASS |
| E-G7 | 统计可复现：bootstrap ×2 全等 | 全等 PASS |

测量语义修订必须升 gate_definitions 版本，阈值一个不动。

## 6. 决策规则（数据驱动）

1. **主问题**：PH1/PH2 相对 PC0 与 PC1 的 nDCG@5/R@5 bootstrap 是否显著为正
   （或至少方向一致且幅度大于 PC1 vs PC0 的增量）？
2. **ARR-007**：任一新臂是否 recovered（对照 PC1/PC0 的 miss）？
3. **归因**：PH3 vs PH1 分离"standalone"与"section 聚合"的贡献；
   PH1 vs PH2 分离 overlap 的贡献。
4. **回归监控**：新臂相对 PC1 是否出现 R@10/Hit@20 明显回落
   （13C-C 中细粒度臂深截断回落的旧风险）。
5. 任何结论可从 metrics JSON 复算；不预设结论；单点个案不改主路径。

## 7. 产物与代码

- 产物：`evals/evaluation_v3/phase13e_hybrid_chunk_benchmark/`
  （chunks/embeddings(保留+hash)/chroma(临时不入 manifest)/retrieval/metrics/
  gates/reports/status.json/manifest.json）
- 代码：`experiments/phase13c/chunk13cc.py` 增补 PH1/PH2/PH3 策略（活跃模块）；
  `experiments/phase13c/runners/run_phase13e_hybrid.py`。
- `src/evidenceflow/**` 零改动；无新增依赖。

## 8. 明确不做

- 不改生产 Runtime/Chunker；不跑 LLM/reranker；不动 Gold/qrels 冻结文件；
- 不因结果好直接切主路径（晋升须另开 Runtime Phase）；
- 环境阻塞记 BLOCKED ≠ FAIL。
