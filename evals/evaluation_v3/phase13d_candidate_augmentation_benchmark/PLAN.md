# Phase 13D — Candidate Augmentation Benchmark 方案（v1，执行前冻结）

> 创建于 2026-09-15。问题承接 13C-C：gold 证据在 B2_fused PC0 chunk 空间 dense
> 大体可召回，但存在 dense miss 的证据类型（figure_caption 类，代表 ARR-007；
> 历史诊断中 BM25 曾恢复 12 个候选失败中的 6 个）。本 Phase 回答：
> **在 B2_fused PC0 chunk 空间上，BM25 candidate augmentation 能否以可接受的
> 精度代价恢复 dense miss？**

## 1. 冻结输入（全部复用 13C-C 冻结产物，零重新嵌入）

| 输入 | 来源 | 校验 |
|---|---|---|
| PC0 chunks（436，fixed C512/O64） | `phase13c_c/chunks/PC0.jsonl` | sha256 对 13C-C manifest |
| Dense Top20 run | `phase13c_c/retrieval/PC0_run.json` | sha256 对 13C-C manifest |
| anchor 严格制 qrels（PC0） | `phase13c_c/metrics/qrels_projection.json` | sha256 + 装载后与 13C-C 文件逐条一致 |
| 40 题（含中文原查询） | `evals/research_retrieval/research_questions_v1.json` | dataset_hash=be1e8354… |

## 2. Arm 定义（4 臂，统一 Top20 深度、无 reranker）

| Arm | 排名来源 | 说明 |
|---|---|---|
| DA0 | Dense Top20 | 13C-C PC0 冻结 run 直接复用（参照基线） |
| DA1 | BM25 Top20 | 生产实现 `evidenceflow.retrieval.bm25.BM25Index`，k1=1.5、b=0.75、bge-m3 tokenizer 分词（与历史 ARR 诊断同口径）；查询用 40 题原始 `query`（bm25_query=original_only 惯例，中英混合由 tokenizer 天然处理） |
| DA2 | RRF(DA0, DA1)，k=60 | score(c)=Σ 1/(60+rank)；同分按 chunk_id 字典序；截 Top20 |
| DA3 | add-only union | Dense Top20 保序在前，BM25 Top10 中未出现的 chunk 按其 BM25 名次追加（历史诊断 add_only 协议复刻）；池深 20 |

chunk 空间：**仅 PC0**（与生产主路径 chunk 策略对齐，单变量评估 augmentation）。
BM25 输入的 DocumentChunk 由 PC0 chunks 构造：file_name/content_hash 取语料
manifest `papers[]` 契约；metadata 携带 owners（provenance 不断）。

## 3. 指标

- 标准 IR（canonical `src/evidenceflow/eval/ir_metrics.py`）：P@5、R@5、R@10、
  MRR@5、MAP@5、nDCG@5、Hit@20。
- 配对 bootstrap（n=10000、seed=20260915、×2 全等）：DA2/DA3 vs DA0，
  指标 nDCG@5 / R@5 / R@10 / P@5 / MRR@5。
- Focus：12 个历史候选失败（ARR-007/015/020/021/022/025/030/031/037/038/039/040）
  逐臂 recovered（Top10/Top20）名单；ARR-007 单列（本 Phase 的动机案例）。

## 4. 硬门（gate_definitions_version = phase13d-gates-v1.0）

| Gate | 定义 | 判定 |
|---|---|---|
| D-G1 | 冻结输入 hash：PC0 chunks / dense run / qrels 均与 13C-C manifest 一致 | 全等 PASS |
| D-G2 | 确定性：BM25 两次检索名次全等；RRF 两次融合名次全等 | 全等 PASS |
| D-G3 | 协议一致：四臂同 Top20 深度、同 qrels、同 tokenizer；BM25 参数=k1 1.5/b 0.75 | 校验 PASS |
| D-G4 | 池完整性：任一臂最终 Top20 无重复 chunk_id | 违例=0 PASS |
| D-G5 | qrels 一致性：运行时 qrels 与 13C-C 冻结 PC0 qrels 逐题逐 chunk 全等 | 全等 PASS |
| D-G6 | 统计可复现：bootstrap 两次全等 | 全等 PASS |

测量语义修订必须升 gate_definitions 版本，阈值一个不动。

## 5. 决策规则（数据驱动）

1. **augmentation 有效性**：DA2 或 DA3 相对 DA0 的 R@5/R@10 bootstrap 显著为正
   且 P@5 不显著为负 → 支持"引入 BM25 候选层"进入下一步（正式 hybrid 评估须另开
   Phase 并过 Runtime 门）；均不显著 → 如实记录"当前证据不足"。
2. **ARR-007**：DA1/DA2/DA3 是否 recovered（Top10/Top20）；预期 BM25 对
   "Figure 2: PCN Architecture" 类精确词面证据有优势——验证之。
3. **DA2 vs DA3**：RRF 重排与 add-only 的 IR 差异，决定未来融合策略形态。
4. 12 个历史候选失败的恢复面（recovered ids 对比历史诊断的 bm25_top20_recovered）。
5. 不预设结论；报告数字必须可从 metrics JSON 复算。

## 6. 产物与代码

- 产物：`evals/evaluation_v3/phase13d_candidate_augmentation_benchmark/`
  （chunks_link（引用而非复制，hash 校验）/ runs / metrics / gates / reports /
  status.json / manifest.json）
- 代码：`experiments/phase13c/runners/run_phase13d_augmentation.py`；
  复用生产 `retrieval/bm25.py`、canonical IR、13C-C 的 bootstrap 纯函数。
- `src/evidenceflow/**` 零改动；无新增依赖（rank_bm25 未安装且不需要）。

## 7. 明确不做

- 不改 Runtime/默认路径；不跑 LLM/reranker；不动 Gold/qrels；
- 不因结果好就"顺手"把 BM25 接进主路径（那需要另开 Phase 过 Runtime 门）；
- 环境阻塞记 BLOCKED ≠ FAIL。
