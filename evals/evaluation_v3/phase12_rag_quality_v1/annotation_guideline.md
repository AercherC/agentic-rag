# Phase 12 Generation Evaluation 人工标注协议

- dataset_id: `rag_generation_eval_v1`
- schema_version: `1.0`
- 适用阶段：Step 5（Claim Extraction）— Step 12（Failure Taxonomy）
- 标注者：Phase 12 v1 使用人工标注（Agent 按本协议逐条判定），不使用 LLM-as-a-Judge

---

## 1. 标注输入

对每一题，标注者同时可见：

| 输入 | 来源 | 说明 |
|---|---|---|
| `query` | `generation_eval_dataset.json` | 原始用户查询 |
| `answerability` gold | `answerability_gold.json` | ANSWERABLE / INSUFFICIENT_CONTEXT |
| `reference_answer` | `reference_answers.json` | 仅 ANSWERABLE 题有，来自 Gold Chunk 明写事实 |
| `required_claims` | `required_claims.json` | 可独立判定的事实单元 + acceptable_evidence(chunk_id) |
| `context_source_labels` / `context_chunk_ids` | `runtime_outputs.json` | 实际送入 LLM 的 Context 条目 |
| `answer_text` | `runtime_outputs.json` | 模型生成的用户可见回答 |
| `cited_sources` | `runtime_outputs.json` | 模型实际引用的 Source（label → chunk_id） |

**Groundedness 判定的唯一依据是实际 BuiltContext**（`context_chunk_ids`），
不是"知识库某处存在该事实"。即使 Gold 证据在库内，只要没进 Context，也算 UNSUPPORTED。

---

## 2. Answer Claim 抽取（Step 5）

把 `answer_text` 拆成事实性 Claim：

1. 一条 Claim = 一个可独立判断的事实；
2. 不抽取纯连接性/修辞性语句（"综上所述"、"该方法效果很好" 若无具体事实支撑则记为 NON_FACTUAL）；
3. 保留模型原句片段作为 `claim_text`，不做规范化改写，便于复核；
4. 若回答为 `INSUFFICIENT_CONTEXT`，则 factual claim 数为 0，只做 Abstention 判定；
5. 抽取结果写入 `claim_annotations.json`，每条带 `claim_id`、`claim_text`、`citations`（该 Claim 上出现的 source label）。

---

## 3. Claim 正确性判定（Step 6 用）

对每个 **模型生成的 Claim**，标注：

| 标签 | 含义 |
|---|---|
| `CORRECT` | 与 `required_claims` 中某条语义一致，或被 acceptable_evidence 支持 |
| `INCORRECT` | 与 Gold 事实矛盾，或引入 Context 不支持的具体事实 |
| `UNVERIFIABLE` | 无法判断真假（视为不正确计入 Precision，但不计入 Groundedness 分母争议） |

对每个 **Gold required claim**（`required=true`），标注是否被回答覆盖：

| 标签 | 含义 |
|---|---|
| `COVERED` | 回答中至少一条 Claim 与之语义一致 |
| `MISSING` | 回答完全未提及 |
| `CONTRADICTED` | 回答给出相反事实 |

聚合：

```
Claim Recall    = COVERED(required claims) / all required claims
Claim Precision = correct factual claims / all factual claims in answer
Claim F1        = 2PR / (P + R)
```

---

## 4. Faithfulness / Groundedness 判定（Step 7）

对每个模型生成的 **事实性 Claim** 标注 Context 支持度：

| 标签 | 含义 |
|---|---|
| `SUPPORTED` | 该 Claim 可由实际 BuiltContext 中的某个 Chunk 直接支持 |
| `PARTIALLY_SUPPORTED` | Context 支持其中一部分，但 Claim 有超出 Context 的推断/数值 |
| `UNSUPPORTED` | Context 中没有任何 Chunk 支持该 Claim |
| `NOT_APPLICABLE` | 非事实性语句，不计入分母 |

聚合：

```
Groundedness           = SUPPORTED / (all factual claims)
Unsupported Claim Rate = (UNSUPPORTED + PARTIALLY_SUPPORTED) / (all factual claims)
```

**注意**：`PARTIALLY_SUPPORTED` 不计入 Grounded。

---

## 5. Citation 正确性判定（Step 8）

对 **每个（Claim, cited source）二元组** 单独判定：

| 标签 | 含义 |
|---|---|
| `SUPPORTS` | 该 source 对应的 chunk 确实支持该 Claim |
| `DOES_NOT_SUPPORT` | source 存在但不支持该 Claim |
| `INVALID` | source label 不在 Context allowlist，或引用了不存在的标签 |

规则：

- 一个 Claim 引用多个 Source 时，**逐个独立判定**，不因其中一个正确而判全部正确；
- 支持判定基于 `cited_sources[].chunk_id` 是否落在该 Claim 的 acceptable_evidence 内，
  若落在外则需人工复核该 chunk 是否仍支持（acceptable_evidence 为 Gold 最小集合，可能偏严）；
- 聚合：

```
Citation Precision = SUPPORTING citations / all citations attached to factual claims
```

---

## 6. Citation 完整性判定（Step 9）

对每个 **应引用的 factual claim**（= 需要证据支撑的 Claim）判定：

```
Citation Completeness = 至少有 1 个 SUPPORTS citation 的 factual claims
                        / 应引用的 factual claims
```

与 Citation Precision 的区别：Precision 看"引得对不对"，Completeness 看"该引的是不是都引了"。

---

## 7. Abstention 判定（Step 10）

Positive class = `INSUFFICIENT_CONTEXT`（即"正确拒答"为正类）。

| 组合 | 标签 |
|---|---|
| Gold INSUFFICIENT + 系统拒答 | `CORRECT_ABSTENTION` |
| Gold INSUFFICIENT + 系统作答 | `FALSE_ANSWER` |
| Gold ANSWERABLE + 系统拒答 | `FALSE_ABSTENTION` |
| Gold ANSWERABLE + 系统作答 | `CORRECT_ANSWER`（正确性另由 Claim 指标衡量） |

```
Abstention Accuracy = (CORRECT_ABSTENTION + CORRECT_ANSWER) / all
Abstention Precision= CORRECT_ABSTENTION / (CORRECT_ABSTENTION + FALSE_ABSTENTION)
Abstention Recall   = CORRECT_ABSTENTION / (CORRECT_ABSTENTION + FALSE_ANSWER)
False Answer Rate   = FALSE_ANSWER / all INSUFFICIENT gold
False Abstention Rate = FALSE_ABSTENTION / all ANSWERABLE gold
```

---

## 8. 失败分类（Step 12）

按 **首要失败原因** 给每题打一个标签（可加次要标签）：

| 标签 | 触发条件 |
|---|---|
| `RETRIEVAL_FAILURE` | Gold Chunk 未出现在 Top-5 Context |
| `CONTEXT_SELECTION_FAILURE` | Gold Chunk 被检索到但因预算被跳过，未进 Context |
| `ANSWER_MISSING_CLAIM` | 证据在 Context 中，但回答遗漏 required claim |
| `ANSWER_WRONG_CLAIM` | 回答给出与证据矛盾的事实 |
| `UNSUPPORTED_CLAIM` | 回答包含 Context 不支持的事实性 Claim |
| `CITATION_MISSING` | factual claim 未附任何 citation |
| `CITATION_WRONG_SOURCE` | citation 指向的 chunk 不支持对应 Claim |
| `CITATION_INVALID_LABEL` | 出现 Context allowlist 外的标签 |
| `FALSE_ABSTENTION` | 证据充足却拒答 |
| `FALSE_ANSWER` | 证据不足却作答 |
| `GENERATION_CONTRACT_ERROR` | 输出不满足 rag-generator-v2 控制头协议 |
| `LLM_REQUEST_ERROR` | LLM 请求失败（本轮全部题目命中此项） |

**禁止**只写 "RAG failed"。

---

## 9. 通用规则

1. 无法判断时标注 `UNKNOWN`，**UNKNOWN 不得自动算正确**；
2. 每条标注都要有 `evidence_ref`（chunk_id 或"无"）与 `note`；
3. 标注不得参考其他题目的结果以保持独立；
4. 本协议变更需同步更新 `schema_version`。

---

## 10. Judge Calibration（Step 15）

Phase 12 v1 **不使用** LLM-as-a-Judge，`judge_calibration.json` 记录 `used = false`。
原因：25 题规模可由人工完整标注，避免"同一模型自生成自评分"的偏差。
后续若引入 Judge，必须先在本人工 Gold 上校准并报告 Accuracy/Precision/Recall/F1/agreement，
一致性不足则 `Judge = NOT_PROMOTED`，主报告仍以人工结果为准。
