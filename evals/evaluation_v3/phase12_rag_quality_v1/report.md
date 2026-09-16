# Phase 12：End-to-End RAG Quality Evaluation

- **Gate：PASS（20/20 checks）**
- 产物目录：`evals/evaluation_v3/phase12_rag_quality_v1/`
- 本阶段是 **Evaluation Phase**：**未改动任何 Runtime 组件**（Retriever / Reranker / ContextBuilder /
  Generator / CitationParser / Chroma / qrels 全部冻结）。
- ⚠️ **Gate PASS 仅表示「评测体系建立成功」，不等于 RAG 答案质量优秀**（计划 §23）。

---

## 1. 背景与目标

Phase 9–11C 已把检索链路的 IR 指标冻结为稳定基线，但那只回答「召回得好不好」，
不回答「**答得对不对、有没有依据、引用是否可信、该拒答时是否拒答**」。

Phase 12 的目标是补齐后半段：

| 维度 | 指标 |
|---|---|
| Answer Correctness | Claim Precision / Recall / F1 |
| Groundedness | Groundedness、Unsupported Claim Rate |
| Citation Correctness | Citation Precision、Completeness |
| Abstention | 拒答混淆矩阵（TP/FP/FN/TN）、False Answer Rate |

**硬约束**：不改检索/生成代码；遇 LLM 环境异常即停并落盘检查点；不得用 LLM 自评充当主真值。

---

## 2. Step 0–3 — 评测集与真值构建（不依赖 LLM）

| 产物 | 内容 |
|---|---|
| `phase12_baseline.json` | 冻结基线：composition fingerprint `a6545c19…`、`runtime_source_fingerprint`、361 chunks、标准 IR 指标 |
| `generation_eval_dataset.json` | **25 题** = 20 道派生自 canonical 40Q + **5 道新增不可答题 `GEN-U01..05`** |
| `reference_answers.json` | 20 道可答题的参考答案，逐条由 Gold Chunk 正文派生 |
| `required_claims.json` | **60 条 required claims** + 3 条 optional claims |
| `answerability_gold.json` | 每题的可答性真值（ANSWERABLE / INSUFFICIENT_CONTEXT） |
| `unanswerable_verification.json` | 5 道不可答题的关键术语在 **361 chunk 上 0 命中**，`all_passed = true` |

要点：
- canonical 40Q **未被修改**，新增题使用独立 ID 空间（`GEN-U*`）。
- 不可答题的「证据确实不存在」经过**程序化复验**，不是人工断言。

---

## 3. Step 4 — Runtime Outputs 冻结（`runtime_outputs.json`）

| 项 | 值 |
|---|---|
| LLM | `Qwen/Qwen3.5-35B-A3B`（ModelScope OpenAI-compatible） |
| 题目数 | 25 |
| **成功** | **22** |
| 环境级错误（`LLM_REQUEST_ERROR`） | **0** |
| 生成契约错误（`GENERATION_CONTRACT_ERROR`） | **3**（ARR-007 / GEN-U03 / GEN-U04） |
| 生成结果分布 | answered = 18，insufficient_context = 4 |

**关于 3 条契约错误**：`src/evidenceflow/generation/generator.py:60-61` 对 `STATUS:` / `ANSWER:` 控制头
做**严格解析**，LLM 未遵守即抛 `ValueError`。这属于计划 §12 明列的**生成质量**失败类型，
是评测对象本身，**不是环境异常**，因此不触发 §4 停止条款。

> 契约错误率 = 3/25 = **12%**，是本次评测暴露的最突出的生成侧稳定性问题。

---

## 4. Step 5–13 — 人工标注与指标聚合

**标注方法**：**人工逐条判读**（`annotator = "human (careful manual reading of each answer against its frozen BuiltContext)"`），
**不使用 LLM 自评**（`judge_calibration.json` 记录 `used = false`）。

### 4.1 Answer Correctness（`aggregate_metrics.json`）

| 指标 | Macro | Micro |
|---|---:|---:|
| Claim Precision | **1.0000** | **1.0000** |
| Claim Recall | 0.7708 | 0.7333 |
| Claim F1 | **0.8706** | 0.8462 |

- required claims：**44 / 60** 正确；optional claims：1 / 3 正确。
- **Precision = 1.0**：模型没有编造任何不被真值接受的 claim。
- **Recall = 0.77**：缺口 16 条，**100% 集中在两类**（精确到题）：

| 缺口来源 | 题目 | required 缺失 |
|---|---|---:|
| `RETRIEVAL` 层（Gold 未进 Context） | ARR-007(3/3)、ARR-037(3/4)、ARR-040(2/5) | **8** |
| `GENERATION` 层（证据在但答案未覆盖） | ARR-039(3/5)、ARR-016(2/3)、ARR-012(1/3)、ARR-019(1/2)、ARR-029(1/3) | **8** |
| `CITATION` / `ABSTENTION` / `NONE` 层 | — | **0** |

> 关键：缺口在两类之间**完全对称（各 8 条）**，且全部是「**答案缺席**」而非「答案出错」。

### 4.2 Groundedness

| 指标 | 值 |
|---|---:|
| Groundedness（macro / micro） | **1.0000 / 1.0000** |
| Unsupported Claim Rate | **0.0000** |
| factual claims 总数 | 76（supported 76 / partially 0 / unsupported 0） |

所有输出的 factual claim 均可在其 Context 中找到支撑，**无幻觉**。

### 4.3 Citation Correctness

| 指标 | 值 |
|---|---:|
| Citation Precision（macro / micro） | **1.0000 / 1.0000** |
| Citation Completeness（macro） | **0.9065** |
| 引用总数 | 89（supporting 89 / wrong_source 0 / invalid 0） |
| Invalid Citation Label Rate | **0.0000** |
| answered_without_citation | **0** |
| labels_not_in_context | **0** |

- **引用零错误**：不存在引错来源或引用不存在的 `S*` 标签。
- Completeness = 0.91：**有陈述未挂引用**（漏引），是「缺引用」而非「错引用」。

### 4.4 Abstention

> 判定基数为 **22**（排除 3 道契约错误题——它们未产出可判定的输出，`unusable_output_count = 3`）。

| 混淆矩阵（正类 = INSUFFICIENT_CONTEXT） | 值 |
|---|---:|
| TP / FP / FN / TN | 3 / 1 / 0 / 18 |
| Accuracy | 0.9545 |
| Precision / Recall / F1 | 0.75 / **1.00** / 0.8571 |
| False Answer Rate | **0.0000** |
| False Abstention Rate | 0.0526 |

- **False Answer Rate = 0**：所有真正不可答的题都正确拒答（Recall = 1.0）。
- 仅 1 例 **False Abstention**（可答题却拒答）——保守倾向，安全方向上偏保守。

---

## 5. Step 11–12 — 失败归因

### 5.1 Failure Taxonomy（`failure_taxonomy.json`）

| 失败类型 | 计数 |
|---|---:|
| `RETRIEVAL_FAILURE` | 3 |
| `ANSWER_MISSING_CLAIM` | 7 |
| `CITATION_MISSING` | 7 |
| `FALSE_ABSTENTION` | 1 |
| `GENERATION_CONTRACT_ERROR` | 3 |
| `CONTEXT_SELECTION_FAILURE` | 0 |
| `ANSWER_WRONG_CLAIM` / `UNSUPPORTED_CLAIM` / `CITATION_WRONG_SOURCE` / `CITATION_INVALID_LABEL` / `FALSE_ANSWER` / `LLM_REQUEST_ERROR` | **0** |

- 25 题中 **15 题存在至少一类失败，10 题完全干净**。
- **`RETRIEVAL_FAILURE = 3`** 恰为 ARR-007 / ARR-037 / ARR-040 —— 与 Phase 9 记录的 **R@10 = 0** 完全一致，
  即这些题的失败**根源在检索**，不是生成。
- 失败集中在「**漏**」（缺 claim / 缺引用）而非「**错**」（错 claim / 错引用）：
  这说明当前系统的风险画像偏 **保守**。

### 5.2 Retrieval × Generation 诊断矩阵（`retrieval_generation_matrix.json`，status = COMPLETE）

对每题定位 `primary_failure_layer`（首个出错的链路环节），得到真实分层：

| primary_failure_layer | 题数 | 含义 |
|---|---:|---|
| `RETRIEVAL` | **3** | Gold 未进 Context（R@10 = 0），检索瓶颈，生成无法补救。题目：ARR-007、ARR-037、ARR-040 |
| `CONTEXT` | 0 | Context 选择未造成额外损失 |
| `GENERATION` | **7** | 证据已在 Context 内但答案未覆盖：5 道 claim 缺失（ARR-012/016/019/029/039）+ 2 道契约错误（GEN-U03、GEN-U04） |
| `ABSTENTION` | 1 | 可答题却拒答（False Abstention，ARR-030） |
| `CITATION` | **4** | 答案正确但引用不完整（漏引）：ARR-002、ARR-010、ARR-014、ARR-031 |
| `NONE` | **10** | 全链路通过 |

补充说明：
- **ARR-007 归 `RETRIEVAL`**（检索失败在前），它同时是 3 道契约错误题之一——`primary_failure_layer`
  只记录**首个**出错环节，故不重复计数。
- **`GEN-U01/02/05` 归 `NONE`**：它们是**不可答题**，检索未命中 Gold 属**预期行为**
  （真值本就是 INSUFFICIENT_CONTEXT），因此不计为失败 —— 这验证了归因逻辑能区分
  「证据不存在」与「证据未被检索到」。

> 结论：失败**主要发生在 GENERATION 层（7/25）**，其次 CITATION（4）与 RETRIEVAL（3）；
> CONTEXT 层零损失。且所有失败均不来自「编造」——Answer Precision 与 Groundedness 全为 1.0，
> 即系统的风险画像偏 **保守**：宁可少说、少引，也不说错、不引错。

---

## 6. Step 14 — Full Regression

```
418 passed, 17 warnings in 31.11s
```

- 0 failed；warnings = 17（全部来自 chromadb 弃用告警），与 Phase 11C 基线**完全一致，无新增**。
- 未删除任何测试。

### Evaluation v3 Regression（`runtime_regression.json`，gate = PASS）

| Metric | 冻结值 | 重算值 | match |
|---|---:|---:|---|
| Precision@5 | 0.115000 | 0.115000 | ✓ |
| Recall@5 | 0.558333 | 0.558333 | ✓ |
| Recall@10 | 0.733333 | 0.733333 | ✓ |
| MRR@5 | 0.432083 | 0.432083 | ✓ |
| MAP@5 | 0.427917 | 0.427917 | ✓ |
| nDCG@5 | 0.461851 | 0.461851 | ✓ |

- 逐题差异：**0**（40/40 完全一致）
- ranking hash：`ffc5cff422eb5818…` 未变
- Chroma：361 chunks，identity hash `d81f9bc529b008e8…` 未变
- `runtime_source_fingerprint`（`PROJECT_SPEC.md` + `src/evidenceflow/**/*.py`）：未变 → **源码零改动**
- **Recall@20：继续 N/A**（冻结排名仅 Top-10，不重新发明）

### Rollback 装配验证（`rollback_validation.json`）

| 配置 | 实际装配 Retriever |
|---|---|
| `RERANKER_ENABLED=true` | `DenseRerankerRetriever` |
| `RERANKER_ENABLED=false` | `MultilingualRetriever` |
| `+ MULTILINGUAL_HYBRID_ENABLED=false` | `MultilingualRetriever` |
| `+ MULTILINGUAL_HYBRID_ENABLED=true` | `MultilingualRetriever` |

**8/8 检查全部 PASS。**

---

## 7. 已知限制（必须随结论一起引用）

1. **composition fingerprint 含环境敏感字段**：`settings_subset.reranker_model` 受 `RERANKER_MODEL`
   环境变量影响。基线在默认值 `BAAI/bge-reranker-base` 下冻结，而本地离线装配必须指向
   `.cache/models/bge-reranker-base`（HF hub 缓存为未完成下载）。二者为**同一模型**，差异仅部署路径。
   复核采用「对齐基线采集环境」方式，已在 `runtime_regression.json` 的 `env_normalization` 逐字记录。
   → 记录为 **KNOWN_ENVIRONMENT_SENSITIVITY**（不再本阶段修代码）。
2. **语料全英文**：10 篇论文，`chinese_corpus = 0`；cross-lingual 仅单向（中文查询 → 英文语料）。
3. **不可答题为新增题目**：与 canonical 40Q 不同源，检索层无历史基线可比。
4. **样本规模 25 题**：属 **internal benchmark**，**不得**与公开 benchmark 数字横向比较。

---

## 8. Gate

| Check | 结果 |
|---|---|
| generation_eval_dataset 已建立 | PASS |
| reference answer 来自 source truth | PASS |
| required claims 已标注 | PASS |
| answerability gold 已冻结 | PASS |
| runtime outputs 已冻结（25 题全覆盖 + 环境错误 0） | PASS |
| Claim P/R/F1 可计算 | PASS |
| Groundedness 可计算 | PASS |
| Citation Precision / Completeness 可计算 | PASS |
| Abstention 指标可计算 | PASS |
| Failure taxonomy 逐题 | PASS |
| 检索×生成×引用可归因 | PASS |
| Runtime 未修改 | PASS |
| Retrieval Gold 未修改 | PASS |
| Artifacts 可追溯 | PASS |
| Full tests 通过（418 / 0 / 17） | PASS |
| 标准 IR 指标不变 | PASS |
| Chroma 不变 | PASS |
| Rollback PASS | PASS |
| 禁止 LLM 自评（judge used = false） | PASS |

**Phase 12 Gate = PASS（20 / 20）。**

---

## 9. 结论与口径

### 9.1 质量画像（25 题 internal benchmark）

```
Answer Precision  1.00  ← 不编造
Groundedness      1.00  ← 有依据
Citation Precision 1.00  ← 不错引
False Answer Rate 0.00  ← 不硬答

Answer Recall     0.77  ← 覆盖不足（主要缺口）
Citation Complete 0.91  ← 漏引
Contract Error    0.12  ← 契约稳定性
```

**一句话**：当前 RAG 是「**保守型**」系统 —— 不说错、不乱引、不乱答，但**会有遗漏**，
且每 25 题约 3 题因未遵守生成控制头而无法产出可用答案。

### 9.2 两层判断口径

- **A. Evaluation Methodology Standard = STANDARDIZED**
  已具备 qrels + 标准 IR 指标 + 冻结 baseline + paired comparison + historical artifact + regression，
  Phase 12 又补齐了答案级质量维度（Claim / Groundedness / Citation / Abstention）。
- **B. RAG Quality Benchmark Standard = PRIVATE**
  私有 25 题 / 361-chunk 数据集，只能在**同数据集内**声明改进；**不得**外推。

---

## 10. 停止点

Phase 12 PASS 后停止。

以下均须**单独立项**，不得在本阶段顺手实施：
- 降低生成契约错误率（prompt 工程 / 输出约束 / 重试策略）
- 提升 claim 覆盖（检索深度、多文档聚合、Context 选择）
- 补齐漏引（引用标注策略）
- 修复 `reranker_model` 环境敏感性（配置归一化）
- Answer 质量优化、新 Reranker、Context Optimization、Agent / Memory

---

## 11. 收口校验（2026-09-10）

Phase 12 结论落盘后，对产物做了一轮独立复核，三项全部通过。

### 11.1 内部一致性

| 校验项 | 结果 |
|---|---|
| 标注覆盖率 | `claim_annotations` 25 题 / `citation_annotations` 25 题 / `abstention_annotations` 25 题，**无遗漏** |
| claim 标注为空的 5 题 | **GEN-U01…U05**，其 `required_count = 0` —— 标注为空是**正确行为**，不是缺口 |
| 非空 claim 标注 20 题 | = 19 道成功题 + **ARR-007**（可答题但生成失败，3 条 claim 全记 `MISSING`），与 `RETRIEVAL` 层缺失 8 条精确吻合 |
| 拒答矩阵自洽 | TP+FP = 4 = `insufficient_context` 数；FN+TN = 18 = `answered` 数 |
| 契约错误题处理 | 3 题（ARR-007 / GEN-U03 / GEN-U04）在 `per_question_results` 中均留痕，未被静默计入分母 |

### 11.2 可复现性

重跑 `phase12_metrics.py compute`：

- 三项核心数字完全复现：Claim P/R/F1 = 1.0000 / 0.7708 / 0.8706、Groundedness = 1.0000、
  Citation P/Completeness = 1.0000 / 0.9065、混淆矩阵 = {TP:3, FP:1, FN:0, TN:18}、
  失败分层 = {RETRIEVAL:3, CONTEXT:0, GENERATION:7, ABSTENTION:1, CITATION:4, NONE:10}。
- **正文逐字节确定**：剔除时间戳字段后，`aggregate_metrics.json` / `per_question_results.json` /
  `failure_taxonomy.json` 的规范化 hash 在重跑前后完全一致；`retrieval_generation_matrix.json`
  因不含时间戳，**整个文件 hash 不变**。

### 11.3 阻塞态残留扫描

全目录扫描 `insufficient balance` / `429` / `BLOCKED` / `LLMRequestError` / `异常即停`，命中项**全部**为刻意保留的审计轨迹：

| 文件 | 命中性质 |
|---|---|
| `anomaly_report.md` | 审计轨迹本体 + 文末 §9 解除记录（**刻意保留**） |
| `run_manifest.json` / `progress_checkpoint.json` | `blocker_history`（**刻意保留**） |
| `report.md` | 仅 §1 复述计划原条款 |
| `runtime_outputs.json` | **误报** —— 命中位于 sha256 十六进制串中，非真实引用 |

### 11.4 关于 PROJECT_SPEC.md 未立 Phase 12 章节（有意决定）

`PROJECT_SPEC.md` 目前记录至 §37.8（Phase 11C），**尚未包含 Phase 12 章节**。
这是**有意的决定，不是遗漏**：

- Phase 12 的 `_runtime_fingerprint()` 采集范围 = `PROJECT_SPEC.md` + `src/evidenceflow/**/*.py`。
  给 SPEC 追加章节会改变该指纹，使 `runtime_regression.json` 记录的源码指纹失配。
- 权衡后决定**保持 SPEC 不变**，以维持 Phase 12 冻结指纹**可随时复跑验证**。
- Phase 12 的收口结论由本 `report.md` 承载。
- **后续若要补 SPEC 章节**：须同时补一份「仅文档变更」对账说明（逐文件 hash 对比，
  证明 `src/**/*.py` 全部未变），并把回归的指纹校验从「整体 hash 相等」升级为逐文件比对，
  以区分「代码变更」与「文档变更」。这与 Phase 11C
  `documentation_fingerprint_reconciliation.json` 的处理原则一致。
