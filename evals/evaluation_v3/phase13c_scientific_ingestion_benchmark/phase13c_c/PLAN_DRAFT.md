# Phase 13C-C — Chunk Benchmark 执行计划（草案 v1，待用户审阅）

> 状态：**草案**。所有 ⚠️ 标记处需要用户确认后才能冻结为正式计划。
> 定位：13C 家族第三步。输入 = 13C-B 冻结的 B2_fused ScientificDocument；
> 输出 = 13C-D（或未来 runtime promotion）的 chunking 候选与证据。Sandbox-Only。

---

## 1. 一句话定位

本项目不是为了证明"结构化切分比 C512/O64 好"——13B 已经证明了它能修复 caption 稀释、
也证明了它修不了词汇不匹配。13C-C 要回答的是：
**13C-B 清洗后的 ScientificDocument（参考文献独立、公式双表示、figure 结构化、
类型修正）作为切分输入，是否改变 chunk 层的检索质量与结构保真；并把"清洗的贡献"
与"切分策略的贡献"用控制变量分开。**

## 2. 输入冻结（Step 0）

| 输入 | 来源 | 冻结方式 |
|---|---|---|
| B2_fused ScientificDocument ×10（真实语料） | 13C-B `cleaned/B2_fused/<doc>.json` | 逐文件 sha256 + digest |
| B0_p0_cleaned ×10（控制组输入） | 13C-B `cleaned/B0_p0_cleaned/<doc>.json` | 同上 |
| 问题集 45 条 | `evaluation_v2/phase35_gold_stabilization_v1/research_questions_v2_gold_stable.json` | dataset id `research_questions_v2_gold_stable_v1` |
| canonical qrels | `evaluation_v3/standard_ir_v1/qrels.json`（sha256 `9337c263…`，**只读**） | 哈希比对 |
| 44 canonical anchors | 13C-B `frozen_inputs.json` | 数量 + sha256 + ids |
| 基线 collection 身份 | `ef_research_bge_m3_c512_o64_v1`（361 chunks，`d81f9bc5…`） | **只读，任何情况下不写** |
| 13B 冻结 dense 基线 | harness R@5=0.425 / R@10=0.6625 / R@20=0.8167 | 常量比对 |
| 13B/13C-B 全部冻结产物哈希 | 两期 manifest | 一致性校验 |

**禁止重解析**：13C-C 不调用任何 parser，不重新清洗；一切从 13C-B 的落盘产物出发。

## 3. 必须回答的五个问题

- **Q1 输入质量传导**：同一 C512/O64 切分器，吃 B0 与吃 B2 产出的 chunk 在
  检索指标与结构保真上是否不同？差异能否全部归因到 13C-B 的清洗改动
  （reference 独立 / 公式规范化 / 类型修正 / 装饰移除），逐项给出证据？
- **Q2 策略贡献分解**：在 B2 输入上，Section-aware / Element-aware 相对 C512/O64 的
  增益是否复现 13B 结论（caption 隔离修复 ARR-007 类失败）？
- **Q3 已知不可解项复核**：ARR-037/040 G2（词汇不匹配）在 B2 输入 + 任何切分下
  是否仍然不可解？预期"是"——若意外变可解，必须先归因（很可能是测量巧合）再下结论。
- **Q4 结构元素在 chunk 层的保真**：caption→chunk 隔离率、reference 不污染正文
  chunk、公式双表示在 chunk 文本中不互相覆盖、chunk→element_id→page 追溯率。
- **Q5 选型**：哪个 arm（或哪个 arm 的哪些参数）作为 13C-D / runtime promotion 的
  候选？允许 NO QUALIFIED CANDIDATE。

## 4. Arm 设计（控制变量矩阵）

复用 13B 已验证的切分实现（`run_phase13b_controlled_rebuild.py` 的
`chunk_section_aware` / `chunk_element_aware` / official C512/O64 对照实现），
**只换输入，不改切分算法**。chunk_id 沿用实验命名（含 strategy 前缀，避免与
baseline chunk 空间碰撞）。

| arm | 切分策略 | 输入 | 作用 |
|---|---|---|---|
| **C0-b2** | official C512/O64 | B2 | 主对照：与 13B 冻结 dense 基线同策略，输入换成 B2 |
| **C0-b0** | official C512/O64 | B0 | 输入消融控制组：C0-b2 vs C0-b0 = **清洗贡献** |
| **C1-b2** | SECTION_AWARE_MINIMAL（O0） | B2 | 复现 13B Candidate A |
| **C2-b2** | ELEMENT_AWARE（O0） | B2 | 复现 13B Candidate B（caption/table/equation 隔离） |
| **C3-b2** | SECTION_AWARE + O64 | B2 | 复现 13B Candidate A′（overlap 对照） |
| ⚠️ C4-b2 | Parent/Child（小 chunk 检索、父块供 generation） | B2 | **是否纳入本阶段由用户决定**；若纳入，检索评估只评子块，父块仅记录映射 |

⚠️ 待确认：C0-b0 消融是否纳入正式 arm（我的建议：纳入，它是 Q1 的唯一手段，
成本 = 一次额外 embedding + 检索）。

## 5. 检索评测协议

- **检索**：BGE-M3 dense（离线，`.cache/huggingface`），Top20；与 13B harness 同参数。
- **Embedding 服务**：沿用 13B 先例新建**候选 Chroma collection**（命名
  `phase13c_c_<arm>`），基线 collection 只读。⚠️ 备选：本地 numpy 余弦索引
  （避免 Chroma 依赖，但与 13B 协议不可比——建议沿用 Chroma）。
- **qrels**：canonical qrels 不改。候选 chunk 空间不同 → 用 13B 的
  **anchor projection** 方法派生候选 qrels（`(document_id, page, anchor, occurrence)`
  → 候选 char_spans 覆盖判定，复用 `_anchor_projection_source` 的字段与逻辑）；
  同时报告与切分无关的 **GoldSpanRecall@k** 作严格对照。
- **指标**：R@5/R@10/R@20、MRR@10、nDCG@10（候选 qrels 上）+ GoldSpanRecall@10/20
  + 44 canonical anchors 在 Top20 的可检索性（对照 13C-B 的 44/44）。
- **Reranker**：⚠️ 默认不进主指标（与 13B dense 口径保持可比）；可选 sidecar
  报告 B2 输入下 Top5 rerank 的变化，由用户决定。

## 6. 结构保真指标（chunk 层，13C-B 的延伸）

| 指标 | 定义 | 陷阱预防 |
|---|---|---|
| caption_isolation_rate | figure/table caption 独立成 chunk 或与宿主图元同 chunk 的比例 | 分母只算 GT caption；"caption 与正文混排"才算失分 |
| reference_pollution | 含 reference 文本的正文 chunk 占比 | 只统计 B2 中 `retrieval_role=reference` 的元素文本；正文引用标注 `[5]` 不算 |
| equation_integrity | 公式 chunk 的 raw/normalized 双表示共存率 | 两侧形态计数先自证，一侧为 0 即陷阱 |
| chunk_provenance_coverage | chunk 带 element_ids + page 的比例 | 分母 = 全部 chunk；None ≠ 0 |
| token_distribution | 各 arm 的 chunk 数 / token 分布 | 与 13B chunk_variants 同格式，便于对照 |

## 7. 硬门（草案 G1–G6）

- **G1 确定性**：同一 arm 重跑 chunk+embedding+检索，排序结果逐位一致。
- **G2 沙盒完整性**：基线 Chroma / `src/evidenceflow` / 源 PDF / 13A/13B/13C-B 冻结产物
  全部不变；P1 `-m txt` OCR 疑问不影响本阶段（不调用 parser）。
- **G3 qrels 投影完整**：45 条 qrels 全部可投影到每个候选 arm（投不进的必须逐条
  列出原因，不允许静默丢弃）。
- **G4 C0-b2 偏差归因**：C0-b2 与 13B 冻结 dense 基线的指标差必须给出
  chunk 级归因清单（哪些 chunk 因清洗而变化）。允许指标变化本身，不允许不可解释。
- **G5 无隐藏外部 LLM**；embedding/reranker 仅本地离线模型。
- **G6 呈现合同**：无 arm 被静默剔除；`None ≠ 0`；`metric_range_violations` 零越界。

决策顺序沿用 §15 风格：Hard Gates → 检索质量 → 结构保真 → 成本，不做加权总分。

## 8. 停止条件（STOP-ON-ANOMALY）

1. 基线 Chroma / src / 源 PDF / 任何冻结产物哈希变化；
2. qrels 文件被修改（哪怕 1 字节）；
3. anchor projection 出现无法解释的批量投不进（>10%）；
4. 候选 collection 意外写入基线 collection；
5. embedding 不可复现（同输入两次向量不一致）；
6. 指标出现不可能值且 `metric_range_violations` 未捕获（守卫失效即停）。

## 9. 产物目录（草案）

```
evals/evaluation_v3/phase13c_scientific_ingestion_benchmark/phase13c_c/
├── frozen_inputs.json / gate_definitions.json / manifest.json / progress_checkpoint.json
├── chunks/<arm>/<doc>.json            # 每个 chunk: chunk_id, element_ids, pages, char_spans, strategy
├── qrels_projected/<arm>.json         # anchor projection 派生 qrels + 投影失败清单
├── retrieval_runs/<arm>.json          # 逐 query Top20
├── metrics/<arm>.json + _family_*.json
├── ablation.json                      # C0-b0 vs C0-b2 的 chunk 级差异归因清单
├── reports/{chunk_comparison.md, final_recommendation.md}
└── selected_chunking.json
```

## 10. 测试清单（草案，13 个）

`tests/test_phase13c_chunk_benchmark.py`：
1. `test_chunk_id_uniqueness_and_strategy_prefix`
2. `test_char_spans_cover_chunk_text`（chunk 文本可回溯到 B2 元素）
3. `test_anchor_projection_full_coverage`（45 条全投影）
4. `test_anchor_projection_rejects_silent_drop`（mutation：投不进被丢必须报警）
5. `test_gold_span_recall_ignores_chunking`
6. `test_c0_b2_vs_frozen_baseline_deviation_attributed`
7. `test_caption_isolation_denominator`（分母只算 GT caption——mutation：混入正文引用不算）
8. `test_reference_pollution_excludes_callouts`
9. `test_equation_dual_representation_in_chunk`
10. `test_provenance_coverage_none_not_zero`
11. `test_deterministic_retrieval_order`
12. `test_blocked_or_pending_not_rendered_fail`
13. `test_metric_range_violations`

## 11. 执行步骤（Step 0–8）

```
Step 0  冻结输入与基线身份          （13C-B 产物哈希 + qrels + questions + anchors）
Step 1  冻结 Gate 定义与 arm 矩阵   （含用户对 ⚠️ 项的裁决）
Step 2  Chunk 构建五/六 arm         （复用 13B 切分器，输入换 B2/B0）
Step 3  Embedding + 候选 collection （BGE-M3 离线；新建 phase13c_c_* collection）
Step 4  Anchor projection 派生 qrels（45 条全覆盖 + GoldSpanRecall）
Step 5  Dense 检索评测              （Top20，逐 query 落盘）
Step 6  结构保真指标 + 消融归因     （C0-b0 vs C0-b2）
Step 7  硬门判定 + 选型
Step 8  报告 + manifest + 测试回归
```

## 12. 明确不做（本阶段边界）

- Sparse / lexical / hybrid candidate augmentation（ARR-037/040 G2 的解法）——单独立项；
- Runtime promotion、修改 `src/evidenceflow`、重建基线 collection；
- Query expansion、reranker 调参、generation 层评测（13C-D 之后）；
- 重新解析 / 重新清洗（一切以 13C-B 冻结产物为输入）。

## 13. ⚠️ 待用户确认清单

1. C4（Parent/Child）是否纳入本阶段？
2. C0-b0 输入消融是否作为正式 arm？（建议：是）
3. 候选向量索引：新建 Chroma collection（沿用 13B 协议）还是本地 numpy 索引？（建议：Chroma）
4. Reranker sidecar 是否需要？（建议：本阶段不做，保持与 13B dense 口径可比）
5. 报告口径：是否同时给出 13B 旧输入（A/B）与本轮 B2 输入的对照表？
   （建议：是，一张表即可，不需要重跑 13B）
