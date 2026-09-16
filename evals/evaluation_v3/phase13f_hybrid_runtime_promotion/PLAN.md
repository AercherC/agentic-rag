# Phase 13F — Hybrid Chunk Runtime Promotion（混合切分 Runtime 晋升）

- 冻结时间：2026-09-15（任何实现代码之前）
- gate_definitions_version：`phase13f-gates-v1.0`
- 决策来源：13E PASS（混合策略方向一致正收益 + ARR-007 三新臂全恢复）；
  用户明确确认「移植 13A/13B 解析器（推荐）」+「hybrid 设为默认（完整晋升）」。

## 1. 目标

把 13E 验证的 PH1 混合切分策略（standalone caption/table/equation 独立出块 +
body 按 section 聚合 O0）晋升为生产 Runtime 默认路径：

- src/evidenceflow/** 新增结构解析能力（13A 已 Gate 纯函数 + 13B block 级元素化移植）；
- Chunker 新增 hybrid 策略；Settings 新增 `chunk_strategy`（默认 hybrid）；
- 以离线 Gate + 一致性验证收口，591 基线测试不得回退。

## 2. 已知偏差（执行前如实冻结）

13E benchmark 的 PH1 chunk 基于 **B2_fused**（P0 正文 + **P1=MinerU** 结构融合，
table 用 MinerU serialized_markdown）。生产 Runtime 不能引入 MinerU 外部依赖，
本阶段生产结构解析 = **纯 P0（13A/13B 移植）**：

- standalone 类型判定、section 归属、heading 检测与 B2_fused 同源（均出自 13A 函数）；
- **table 文本不同**：生产用 block 原文（13B `_is_table_body` 判定体），非 MinerU markdown；
- 因此「runtime chunk ≡ 13E PH1 chunk」逐字节门**构造上不可达**，一致性门降级为：
  - F-G2：runtime 结构解析 ≡ 13B Step1 冻结产物 `structured_elements.json`
    （同为 P0-only，代码忠实移植时应逐元素一致——这是本阶段最强的等同门）；
  - F-G7：端到端 IR 复验（方向门，见 §4）。

## 3. 实现范围

| 文件 | 变更 |
|---|---|
| `src/evidenceflow/ingestion/structure.py` | 新增：13A detect_headings / resolve_section / detect_document_furniture / page_level_furniture_spans + 13B parse_page_blocks / _classify_block_text / _is_table_body / build_structured_document / furniture 移除 |
| `src/evidenceflow/ingestion/chunker.py` | 新增 `chunk_document_structured`（PH1 语义：standalone 独立 + body section 聚合 O0；`\n\n` 串接；精确拼接后计数；硬预算 ≤ chunk_size；超长 unit 词级拆分） |
| `src/evidenceflow/config.py` | 新增 `chunk_strategy`（"hybrid"|"fixed"，默认 "hybrid"）；collection 守卫（hybrid 不得指向已知 fixed 身份 collection） |
| `src/evidenceflow/ingestion/service.py` | 策略分发：hybrid + PDF → 结构解析→混合切分；md/txt 与 fixed → 原路径 |
| `src/evidenceflow/runtime.py` | IngestionService 注入 chunk_strategy |
| `.env.example` | 补 CHUNK_STRATEGY 与 collection 说明 |
| tests | structure/chunker_hybrid/config 守卫新测试；既有 591 不得修改断言放宽 |
| `experiments/phase13c/runners/run_phase13f_runtime_promotion.py` | 离线 Gate runner |

固定路径（chunk_document 滑动窗口）**零修改**，作为回滚通道保留。

## 4. 门定义（F-G1..F-G8，冻结不变）

| gate | 定义 | 判据 |
|---|---|---|
| F-G1 | 冻结输入完整性 | 13C-C/13E 参照产物 hash 与各自 manifest 全等；10 篇语料 PDF hash 与 13B manifest 全等 |
| F-G2 | 结构等同 | runtime 解析 10 篇 PDF 的元素集与 13B `structured_elements.json` 冻结产物逐元素一致（element_id / text / element_type / page / section / subsection / section_path / bbox 全等）；10/10 文档全过 |
| F-G3 | 切分确定性 | hybrid 切分 ×2：chunk_id/content/token_count 序列全等（每文档） |
| F-G4 | token 预算 | 全部 hybrid chunk token_count ≤ chunk_size（512）；违例=0 |
| F-G5 | 溯源 | 每 chunk metadata.element_ids 非空且 ⊆ 该文档 furniture 移除后元素集；page_start/page_end 符合 Domain 校验 |
| F-G6 | 协议一致 | 新 collection（名称不含 c512_o64 固定身份）；BGE-M3 add_special_tokens=False 口径；Top20 检索 |
| F-G7 | 端到端 IR 复验 | 40 题 anchor 严格制评测 runtime hybrid：R@5、nDCG@5 相对冻结 PC0（13C-C 逐题 per-query 复用）paired bootstrap 方向不为负（mean Δ ≥ 0）；与 13E PH1 聚合值偏差记录并归因（P0-only table 文本差异） |
| F-G8 | 测试基线 | 全量 pytest：591 基线 + 新增测试全绿；src 变更仅限 §3 清单 |

测量语义修订必须升 gate_definitions 版本，阈值一个不动。
任何 FAIL：逐门归因后停止，不得为过门修改 Gold、冻结产物或门定义。

## 5. 明确不做

- 不引入 MinerU 或任何外部解析依赖；不修改 13C-C/13E 冻结产物；
- 不删改 fixed 路径与旧 collection（回滚通道）；不跑 LLM/reranker；
- 不因结果好额外扩大 src 改动面（每次只实现本 PLAN §3 清单）。

## 6. 产物

- 代码：§3 清单；
- 证据：`evals/evaluation_v3/phase13f_hybrid_runtime_promotion/`
  （gates/、metrics/、reports/、status.json、manifest.json、本 PLAN）；
- 新 collection：`ef_research_bge_m3_hybrid_c512_v1`（重建后成为默认主路径身份；
  旧 `ef_research_bge_m3_c512_o64_v1` 保留为回滚）。
