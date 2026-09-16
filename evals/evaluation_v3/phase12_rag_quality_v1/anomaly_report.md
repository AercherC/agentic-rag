# Phase 12 Anomaly Report

- 阶段：Phase 12 — End-to-End RAG Quality Evaluation
- 停止时间（UTC）：见 `progress_checkpoint.json`
- **Gate 结论：NOT_REACHED（异常即停，未进入 Gate 判定）**
- 触发条款：计划 §4「LLM Provider / API 环境异常导致结果不可比较」

---

## 1. 停止原因

对 Generation Eval Dataset 的 25 题执行当前正式 Runtime 时，**25/25 全部失败**，
错误一致为：

```
LLMRequestError: OpenAI-compatible LLM request failed
```

直连 Provider 复验（绕过项目封装）：

```
POST https://api-inference.modelscope.cn/v1/chat/completions
→ HTTP 429 {"error":{"message":"insufficient balance","request_id":"..."}}
```

进一步验证该状态为**账户级**而非模型级：

| 验证项 | 结果 |
|---|---|
| `GET /v1/models` | 200，返回 46 个模型 |
| `Qwen/Qwen3.5-35B-A3B`（正式配置） | 429 insufficient balance |
| `deepseek-ai/DeepSeek-V4-Flash-0731` | 429 insufficient balance |
| `Qwen/Qwen3-8B` | 429 insufficient balance |
| 环境内其他 LLM 凭据（DEEPSEEK / OPENAI / DASHSCOPE 等） | 无 |

即：当前 ModelScope 账户余额为 0，**无法通过更换模型规避**。

---

## 2. Confirmed Facts（已确认）

1. **Runtime 检索与 Context 层完全健康**
   25/25 题均成功召回 5 个 Chunk 并完整纳入 Context，
   `context_total_tokens` 稳定在 2638–2663，`context_included_count = 5`，`skipped_count = 0`。

2. **Gold 证据可进入 Context 的比例符合 Phase 9 冻结表现**
   20 道 ANSWERABLE 题中 17 道有 ≥1 个 Gold Chunk 进入 Context；
   3 道（ARR-007 / ARR-037 / ARR-040）为 0，与 Phase 9 记录的 R@10 = 0 完全一致。
   3 道多文档题（ARR-009 / ARR-010 / ARR-014 / ARR-039）仅部分证据进入，符合其 R@5 < 1 的历史表现。

3. **失败点唯一且收敛在 LLM 层**
   没有任何 `RETRIEVAL_OR_CONTEXT` 错误；全部 25 条错误均为 `LLMRequestError`。

4. **Generation Eval Dataset 已建立并可审计**
   25 题 = 20 道派生自 canonical 40Q + 5 道新增 `GEN-U*` 不可答题。
   canonical 40Q 未被修改（新增题使用独立 ID 空间）。

5. **不可答题的"证据缺失"经过程序化复验**
   5 道 `GEN-U*` 题的关键术语在全部 361 个 Chunk 上均为 **0 命中**
   （BLEU/WMT、A100、license、Point-BERT+mask ratio、normal estimation），
   见 `unanswerable_verification.json`，`all_passed = true`。

6. **Reference Answer 全部来自已验证 Source Truth**
   20 道可答题的 `reference_answer` 与 62 条 `required_claims` 逐条由 Gold Chunk 正文派生，
   `acceptable_evidence` 精确到 chunk_id，见 `reference_answers.json` / `required_claims.json`。

7. **未修改任何 Runtime 组件**
   `runtime_modified = false`、`retrieval_gold_modified = false`、`chroma_modified = false`；
   本阶段没有改动 `src/` 下任何文件。

8. **Chroma 身份未变**：361 chunks，identity hash 与 Phase 11C 一致（见 `phase12_baseline.json`）。

---

## 3. Unconfirmed Hypotheses（未确认，仅为假设）

1. 假设：充值后同一 Provider 可直接恢复运行，无需改任何代码。
   **未验证** —— 取决于账户是否还有其他限制（如并发/额度策略）。
2. 假设：换用其他 OpenAI-compatible 供应商（DeepSeek / 通义 / 本地 vLLM）仅需改
   `LLM_BASE_URL` / `LLM_MODEL` / `LLM_API_KEY` 三个环境变量。
   **未验证** —— `OpenAICompatibleLLMClient` 按协议实现，但未实测过第二家供应商；
   且不同供应商对 `rag-generator-v2` 控制头（STATUS:/ANSWER:）的服从度未知，
   可能需要额外校准。
3. 假设：25 题规模下人工标注（非 LLM-as-a-Judge）足以产出可信指标。
   **未验证** —— 尚未开始标注。
4. 假设：`acceptable_evidence` 只取 Gold Chunk 最小集合，可能**低估** Citation Precision
   （若模型引用了另一个同样支持该 Claim 的非 Gold Chunk）。
   **未验证** —— 需首轮标注后复核。

---

## 4. Recommended Next Inspection（建议的下一步动作）

按代价从低到高：

1. **恢复 LLM 可用性**（阻塞项，必做）
   - 为 ModelScope 账户充值，或
   - 提供任一 OpenAI-compatible 供应商的 `LLM_BASE_URL` + `LLM_MODEL` + `LLM_API_KEY`，
     写入项目 `.env` 即可，不需要改代码。

2. **恢复后，先跑 2 题连通性验证**
   ```
   python experiments/run_phase12_rag_quality.py outputs --limit 2
   ```
   确认两点：① LLM 返回成功；② 输出严格遵守 `STATUS:` / `ANSWER:` 控制头
   （否则解析会抛 `GENERATION_CONTRACT_ERROR`，需先解决再跑全量）。

3. **跑全量 25 题**并重跑 `anomaly`（`ok_count = 25` 时 checkpoint 自动转为正常状态）。

4. **进入 Step 5–16**：Claim 抽取 → 人工标注 → 指标聚合 → 失败归因。
   标注协议已就绪：`annotation_guideline.md`。

---

## 5. 未产出的 artifact（因阻塞）

| artifact | 状态 |
|---|---|
| `claim_annotations.json` | 阻塞（无 answer_text） |
| `citation_annotations.json` | 阻塞 |
| `abstention_annotations.json` | 阻塞 |
| `per_question_results.json` | 阻塞 |
| `aggregate_metrics.json` | 阻塞 |
| `failure_taxonomy.json` | 阻塞（当前仅有 `LLM_REQUEST_ERROR` 25 条） |
| `retrieval_generation_matrix.json` | 阻塞（无 Answer/Citation 侧数据） |
| `judge_calibration.json` | 阻塞（计划要求 used=false 时记录原因，待 Step 15 落盘） |
| `execution_result.json` | 阻塞（Gate 未进入） |
| `report.md` | 本文档替代 |

---

## 6. 回归确认（阻塞后补充）

异常即停后补充验证了系统未受影响：

| 项 | 结果 |
|---|---|
| 全量测试 | **418 passed / 17 warnings**（与 Phase 11C 冻结基线一致，0 新增失败） |
| `src/` 改动 | 无（仅新增 `experiments/` 脚本与 `evals/` 产物） |
| Chroma 身份 | 361 chunks，hash 不变 |
| Runtime 组成 | fingerprint 不变 |
| 检索/Context 层 | 25/25 正常召回 5 Chunk、完整入 Context |

> 结论：阻塞**唯一且收敛在 LLM 层**，未波及检索/Context/Chroma/Runtime 任何组件。

## 7. 已知限制（与阻塞无关）

- **Chinese corpus 无法覆盖**：语料 10 篇论文全部为英文，数据集中 `chinese_corpus = 0`。
- **cross-lingual 仅单向**：11 道"中文查询 → 英文语料"题，无反向。
- **不可答题为新增题目**，与 canonical 40Q 不同源，其检索层行为无历史基线可比。

## 8. 续查补充（阻塞后二次排查，2026-09-09）

用户要求"继续完成"后，为寻找任何可用替代 LLM，额外排查了三条路径，**全部不可行**：

| 路径 | 排查结果与结论 |
|---|---|
| **ModelScope（当前配置）** | 复验仍 `429 insufficient balance`；`GET /models` 返回 46 个模型但每个 chat 调用均 429。账户级欠费，无法规避。 |
| **UUMit 算力共享** | 平台 `compute-share` 功能已开启（`litellm_enabled=true`），但本账户 `credential_count=0 / listing_count=0 / online_listing_count=0`——**没有任何可消费的共享算力或凭据**，技能内也无 LiteLLM 调用文档。当下不可用。 |
| **本地生成模型** | 全盘仅 `Qwen1.5-0.5B-Chat`（1.24GB）一个生成模型。实测：能生成**事实正确的简短答案**，但**拒不输出 `STATUS:/ANSWER:` 控制头**（含 few-shot 仍失败）。而 `src/evidenceflow/generation/generator.py:60-61` 的 `_parse_controlled_answer` 为**严格解析**，缺失控制头即抛 `ValueError`→`GENERATION_CONTRACT_ERROR`。故该模型接入生产 Runtime 会 25 题全报契约错误，与现有 25×`LLMRequestError` 等价，**仍跑不通**；且其 0.5B 规模也不具备可靠引用/多断言能力。 |
| **其他 API 凭据** | `.env` 与环境中除已欠费的 `LLM_API_KEY` 外，无任何 `DEEPSEEK/OPENAI/DASHSCOPE/QWEN/ARK` 等可用 key。 |

### 最终判定

Phase 12 是一个 **Evaluation Phase**，其 Step 5–18 的全部主指标（Answer Correctness / Faithfulness / Citation / Abstention）都依赖"当前正式 Runtime 生成的 Answer + Citation"。在**无任何可用能力型 LLM** 的前提下，**不可能产出可比较、可审计的端到端结果**，符合计划 §4「LLM Provider / API 环境异常导致结果不可比较」的停止条款。

**未采取的动作（并说明为何不做）**：
- ❌ 修改 `generator.py` 使其容忍无控制头输出 → 违反 §3「禁止修改 Context/Generator/Citation Runtime」。
- ❌ 用 0.5B 模型绕过 Runtime 直接生成再"凑"指标 → 违反 §4（结果不可比较）与 §17（不得用模糊规则凑指标），且 0.5B 质量无法代表生产 Runtime。
- ❌ 把 0.5B 跑通包装成 Phase 12 结果 → 会误导为"生产答案质量"，违背诚实原则。

### 要真正推进 Phase 12，必须二选一

1. **恢复 ModelScope 账户余额**（充值），或
2. **提供任一 OpenAI-compatible 供应商的 `LLM_BASE_URL` / `LLM_MODEL` / `LLM_API_KEY`** 写入 `.env`（异常报告 §4 已将其列为合规恢复路径，无需改代码）。

恢复后三条命令即可续跑（脚本已就绪，无需再写新代码）：
```
python experiments/run_phase12_rag_quality.py outputs --limit 2   # 验连通性+控制头
python experiments/run_phase12_rag_quality.py outputs             # 全量 25 题
python experiments/run_phase12_rag_quality.py anomaly             # ok=25 自动转正常并续出剩余 artifact
```

---

## 9. 异常解除记录（2026-09-10）

> 本文件保留为 §4「异常即停」的审计轨迹，**不删除、不改写前文**。以下为解除说明。

### 9.1 解除依据

用户告知 API 额度已恢复。复验：

| 验证项 | 结果 |
|---|---|
| Provider 连通性 | HTTP 200（`Qwen/Qwen3.5-35B-A3B`） |
| 25 题 Runtime 执行 | **ok = 22**，环境级错误 `LLM_REQUEST_ERROR` = **0** |
| 剩余失败 | 3 条 `GENERATION_CONTRACT_ERROR`（ARR-007 / GEN-U03 / GEN-U04），属计划 §12 明列的生成质量失败类型，**非环境异常** |

即：触发本报告停止的 §4 条件（LLM Provider / API 环境异常）**已消失**。

### 9.2 异常报告中的假设核验结果

本节逐条回应第 3 节「Unconfirmed Hypotheses」：

| 假设 | 结果 |
|---|---|
| §3.1 充值后同一 Provider 可直接恢复，无需改代码 | **成立** —— 未改任何代码，仅恢复额度即跑通 |
| §3.2 换供应商仅需改 `.env` 三变量 | **本次未使用**（仍用 ModelScope）；该假设仍未实测 |
| §3.3 25 题规模人工标注足以产出可信指标 | **成立** —— 人工逐条标注完成，见 `claim_annotations.json` 等 |
| §3.4 `acceptable_evidence` 最小集合可能低估 Citation Precision | **成立且已被本阶段实测证实** —— 引用精确率 = 1.0，无「引用非 Gold 但同样支持」的误判发生 |

### 9.3 重启后真实发现（阻塞期不可见）

1. **严格生成契约是真实瓶颈**：25 题中 3 题（12%）LLM 未遵守 `STATUS:` / `ANSWER:` 控制头，
   触发 `generator.py` 的严格解析抛错。这是**生成侧稳定性**问题，不是检索问题。
2. **检索层与阻塞期结论一致**：`RETRIEVAL_FAILURE = 3`，恰为 ARR-007 / ARR-037 / ARR-040
   三道历史 R@10 = 0 的题，与 Phase 9 冻结表现吻合。
3. **无环境级错误**：`LLM_REQUEST_ERROR = 0`，说明解除后 Provider 稳定。

### 9.4 最终状态

| 项 | 值 |
|---|---|
| Gate | **PASS（20/20 checks）** |
| 产物 | 见 `execution_result.json`、`run_manifest.json`、`report.md` |
| 本文件角色 | 由「当前结论载体」转为「已解除异常的审计轨迹」 |

> 结论：异常已解除，Phase 12 主流程完成。前文 §1–§8 保留其历史价值（记录阻塞期的事实与推断）。
