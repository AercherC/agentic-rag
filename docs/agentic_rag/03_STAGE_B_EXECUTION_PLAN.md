# EvidenceFlow 阶段 B 执行计划：评测先行

- 版本：v2（2026-10-15）。§9 的 D1–D6 已由用户确认
- v2：B0 完成；按 D6 加入 answer_keys / key-fact recall 与 B5b
- 执行者：Claude Code 子代理（Opus 5.5，按步骤难度选 low / medium），在本机 `agent-gpu` Conda 环境中执行
- 设计与验收：主会话（每步结束后在主会话审查 diff 与产物）
- 依据：`docs/agentic_rag/01_ARCHITECTURE_REVIEW.md` §7.7、§7.8、§10；`PROJECT_SPEC.md` v0.27；`artifacts/stage_a/STAGE_A_SUMMARY.md`
- 起点：tag `stage-a-done`（`1516cd9`）

---

## 0. 执行者必读

### 0.1 目标
阶段 B **不增加任何 Agent 能力，也不改变任何线上回答**。它只做四件事：
1. 给线上链路加上 **RunTrace**，能看清一次提问每一步发生了什么、花了多少 token 和时间；
2. 建立**端到端评测集 `rag_e2e_v1`**：在现有 40 题基础上补齐必需论断（required claims），新增多文档/对比题和不可答题，并给每题标注期望状态；
3. 建立**可复现的端到端评测流水线**：采集（collect）→ 评委打分（judge）→ 确定性计分（score）→ 配对比较（compare）；
4. **检验 LLM 评委**（构造负例检出率、复测一致率），并产出 RAGService 的**端到端基线 E0**。本阶段不做人工标注（用户决定 D2），所以 E0 的**主指标全部是确定性指标**，评委给出的指标一律标为“诊断（未经人工校准）”。

完成后，阶段 C/D/E 每加一个能力，都能和 E0 逐题配对比较，判断它是否真的有用。

### 0.2 开工前先读
1. `AGENTS.md`（规则、§7 汇报格式）、`CLAUDE.md`。
2. `PROJECT_SPEC.md` v0.27：§1、§3、§8–§10、§12、§13、§14.1。
3. 本计划 §1（现状事实）和 §2（开源方案对照）。设计理由都在那里，执行时不要重新发明。
4. Phase 12 产物：`evals/evaluation_v3/phase12_rag_quality_v1/` 下的 `report.md`、`annotation_guideline.md`、`generation_eval_dataset.json`、`claim_annotations.json`。
5. 可复用的在线脚本基础设施：`experiments/stage_a/online_common.py`、`run_generation_smoke.py`（请求间隔、429 重试、JSONL 续跑、身份校验）。
6. 每改一个模块之前，先读该模块、直接依赖它的模块和对应测试。

### 0.3 硬性规则
- **按 B0 → B7 顺序执行（B5b 在 B5 与 B6 之间），每步 Gate 通过才进入下一步。** Gate 不通过就 STOP：诊断写进 `artifacts/stage_b/B<n>_report.md`，然后汇报。不要放宽 Gate，不要修改 Gold、qrels 或任何已有的 `evals/` 冻结产物。
- 每步结束提交一次，提交信息 `stage-b(B<n>): <摘要>`，署名行 `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`。不提交 `.env`、`Claude outputs/`。不 push。
- **线上行为不变**：除 B2 的 trace 挂钩外，不改 `retrieval/`、`context/`、`generation/`、`citation/` 的行为。B2 的 Gate 会逐位验证这一点。
- 不新增第三方依赖。RAGAS、RAGChecker、ARES、DeepEval、LangSmith、Langfuse、OpenTelemetry 都**只借鉴设计，不安装**。
- 不实现阶段 C–E 的任何内容：Planner、Grader、补检、PARTIAL、文档过滤、逐句引用校验都不做。
- 涉及合同的改动（新 Domain Model、数据集 schema、评委协议、指标定义、Gate 规则、CLI 参数）**先写进 `PROJECT_SPEC.md`**，再改代码。v0.27 已经没有 §S，阶段 B 的合同直接写进对应章节（RunTrace 进 §12 与新增小节，评测进 §13），并在 §16 历史索引中注明“Stage B 新增”。
- 数值类 Gate（检索回放）一律 `MODEL_DEVICE=cpu`。
- 真实 LLM 调用只在本计划明确允许的步骤（B2 的 1 次冒烟、B5、B5b、B6、B7）进行，调用前先估算次数并写进报告；单元测试只用 Fake/Stub。
- 不确定就停下来问，不要猜。

### 0.4 环境准备（PowerShell）
```powershell
conda activate agent-gpu
cd D:\Agent_learn\evidenceFlow
$env:HF_HOME = (Resolve-Path .cache/huggingface)
$env:TRANSFORMERS_OFFLINE = '1'
$env:HF_HUB_OFFLINE = '1'
$env:RERANKER_MODEL = (Resolve-Path .cache/models/bge-reranker-base)
$env:MODEL_DEVICE = 'cpu'
```
如果 `conda activate` 在当前 shell 不生效，直接用 `D:\miniconda\envs\agent-gpu\python.exe`。所有产物写到 `artifacts/stage_b/`；新数据集写到新目录 `evals/rag_e2e_v1/`，B3 结束后冻结。

---

## 1. 现状事实（主会话已核实，执行时复核）

### 1.1 已有的评测资产
| 资产 | 位置 | 状态 |
|---|---|---|
| canonical 40 题 | `evals/evaluation_v2/phase35_gold_stabilization_v1/research_questions_v2_gold_stable.json` | 40 题全部可答；20 中文 + 20 英文；36 题 1 个证据组、4 题 2 个证据组；证据是**文本锚点 + 定位信息**，不是 chunk ID |
| 检索基线 B0' | `artifacts/stage_a/A1_exact_baseline.json` | 冻结，sha256 `422fd629…`；只有 IR 指标，没有生成质量 |
| Phase 12 生成评测集 | `evals/evaluation_v3/phase12_rag_quality_v1/` | 25 题 = 20 题来自 40 题 + 5 道不可答题 `GEN-U01..05`；60 条 required claims |
| Phase 12 标注 | 同上 `claim_annotations.json` 等 | 针对**旧 361-chunk 索引 + v2 生成器 + Qwen** 的输出 |
| 指标代码 | `eval/ir_metrics.py`、`eval/quality.py`、`eval/comparison.py`、`eval/stable_mapping.py` | canonical；`comparison.py` 已有按题族 bootstrap 与 Holm 校正 |

### 1.2 主会话核查中发现、直接影响本计划设计的四个问题
1. **Phase 12 的“人工标注”实际上是 Agent 标的。** `annotation_guideline.md` 写着“Phase 12 v1 使用人工标注（Agent 按本协议逐条判定）”。用 LLM 标出来的数据去“校准 LLM 评委”是循环论证。用户决定本阶段不做人工标注（D2），因此**评委指标不能作为主指标**，只能作为经过机器检验的诊断指标。Phase 12 标注只作为辅助一致性参考，报告中一律写成“agent-annotated”。真正由人确认的真值只有一处：B3 中用户逐题审阅过的题目、期望状态、required claims 和证据锚点，所以主指标都建立在它们之上。
2. **Phase 12 标注几乎没有负例。** 80 条回答论断中 76 条 CORRECT、0 条 INCORRECT；93 条引用全部 SUPPORTS；只有 required claim 覆盖（45 COVERED / 18 MISSING）是均衡的。在这种数据上，一个永远回答“对”的评委也能拿到很高的一致率。**所以不能只看一致率**：要同时看 Cohen's κ，并用人工构造的负例检验评委能不能发现错误（§5 B6）。
3. **Phase 12 的证据是旧索引的 chunk ID。** required claims 的 `acceptable_evidence` 指向 361-chunk 索引，不能直接用于当前 817-chunk 索引。本阶段把每条 claim 的证据改成**文本锚点**（与 40 题相同的协议），再用 `stable_mapping` 映射到当前 chunk。这样以后重建索引，评测集也不用重做。
4. **Phase 12 的输出是可以复原的。** `runtime_outputs.json` 保存了回答全文、上下文 chunk ID 和 `context_text_sha256`；旧索引 `ef_research_bge_m3_c512_o64_v1` 仍在 Chroma 中（A4 有意保留）。所以可以只读复原当时的上下文全文，并用 sha256 验证，作为评委的第二份检验材料。

### 1.3 当前生成器与成本参考
- 生成器 `rag-generator-v3`，DeepSeek `deepseek-flash`，`json_object` 模式，thinking 关闭（用户 2026-10-14 决定）。
- A2 冒烟：25 次调用，输入约 4.7 万 token、输出约 2.7 千 token，单题中位数约 6 秒。
- 阶段 A 遗留、本阶段要处理的：`eval/quality.py` 的 `abstention_metrics` 只认大写状态；“status=insufficient_context 但正文带了部分有引用的回答”怎么计分（§9 决策 D4）。
- 勘误：架构审查 §10 阶段 B Gate 中的“Judge 一致率 ≥ 0.85”已由 D2 取代：评委只作诊断，阶段 B 改为负例检验 + 复测。

---

## 2. 开源方案对照：借鉴什么、不借鉴什么

主会话直接读了以下项目的源码（commit 记录在括号内），结论如下。

| 项目 | 读过的位置 | 借鉴 | 不借鉴及原因 |
|---|---|---|---|
| **RAGChecker**（amazon-science） | `ragchecker/computation.py` | 论断级的**诊断分解**：用三组蕴含关系（标准论断⊆检索块、标准论断⊆回答、回答论断⊆检索块）分别算出检索侧的 claim recall、context precision 和生成侧的 context utilization、幻觉率、噪声敏感度。它能把“没检索到”和“检索到了却没用上”分开，正好对应 Phase 12 的 RETRIEVAL / GENERATION 两层失败。 | 不装包（依赖 spaCy 与外部 checker 模型）。**“标准论断⊆检索块”我们不用 LLM 判**：每条 claim 有人工确认的证据锚点，映射成 chunk ID 后用集合运算就能确定性算出。 |
| **RAGAS**（explodinggradients，298b682） | `src/ragas/metrics/_faithfulness.py` | 忠实度的两步法：先把回答拆成**不含代词、可独立理解**的陈述，再对每条陈述相对上下文给出 0/1 判定和理由。评委 prompt 的结构照此设计。 | 不作为依赖（架构审查 §7.8）；不用它的 answer relevance、answer similarity 这类无参考打分。 |
| **ARES**（stanford-futuredata） | `README.md`（PPI 部分） | **预测驱动推断（PPI）**：少量人工标注 + 评委在全集上的判断 → 纠偏后的估计和置信区间。前提是人工标注与被评估的输出来自同一分布。**本阶段不做人工标注，所以不实现 PPI**；如果阶段 D 之前补做人工标注（见 §9 的说明），再按它的思路加上纠偏估计。另一个借鉴点：ARES 程序化地构造负例（换成无关段落、改出错误回答）来训练评委；本阶段 B6 把同样的构造思路用来**检验**评委。 | 不微调评委模型，不做它的合成问题生成流水线。 |
| **SuperMew**（icey1287，e2ebabe） | `backend/evaluation/rag.py`、`.github/workflows/rag-eval.yml` | ① **采集与计分分离**：昂贵的在线采集产出冻结的 observations，计分是纯函数，可以重复跑，CI 里重算报告并 `cmp` 逐字节比对；② observations 用 **dataset fingerprint** 绑定数据集，防止拿错题集计分；③ **Gate 策略文件**：每个指标声明方向、最小值/最大值、允许的最大退化、是否必需；④ `RagExpectedBehavior`：每题写明期望结果和“是否允许拒答”；⑤ 题目带 **tags** 做分片统计，带 `critical` 标记的题不允许退化；⑥ 没有可靠实现的指标**显式标记为不可用**，不伪造分数。 | 不搬它的平台层（PostgreSQL、Redis、worker、HITL）。它的评委指标目前标记为不可用，所以评委这一块要自己设计。 |
| **PaperQA2**（Future-House，57e89f7） | `src/paperqa/agents/models.py`、`tools.py` | 把“不确定 / 信息不足”当作合法输出（`AgentStatus.UNSURE`、`has_successful_answer`）。LitQA2 评测同时报告**准确率（答对 / 全部）**和**精确率（答对 / 作答）**：拒答多会拉低准确率、抬高精确率。我们也同时报告这两个数，避免单看一个数被“多拒答”或“乱作答”骗过。 | LitQA2 是选择题，评分代码已不在这个仓库；我们的题是开放式回答，只借鉴“两个数一起报”的思路。 |
| **OpenInference**（Arize-ai） | `spec/semantic_conventions.md` | RunTrace 的字段命名参考它的约定：span 类型 RETRIEVER / RERANKER / LLM，`llm.token_count.prompt` / `completion`，`retrieval.documents`，`reranker.input_documents` / `output_documents` / `top_k`。事件名仍用项目已有的 `docs/EVIDENCEFLOW_UI_RUNTIME_CONTRACT_v0.md` 中的 taxonomy。 | 不引入 OpenTelemetry SDK 或任何采集服务（架构审查 §7.7）。 |

**评委的已知偏差**（来自 LLM-as-a-judge 的通行结论，本阶段的设计都针对它们）：
- 自我偏好：同一模型评自己的输出会偏宽松。Phase 12 自己也写过 `self_judge_prohibited: true`。→ 用户已决定评委用**另一家厂商的模型**（D1）。
- 不确定性：temperature 0 也不保证每次输出一样。→ B6 测评委自身的复测一致率。
- 偏向冗长、偏向“看起来有引用”的回答。→ 评委不打总分，只做逐条的二元判定，聚合由确定性代码完成。

---

## 3. 本阶段的评测设计（先读懂再执行）

### 3.1 指标分层：能确定性算的绝不用 LLM
| 层 | 指标 | 计算方式 | 是否需要评委 | E0 中的地位 |
|---|---|---|---|---|
| 检索 | B0' 的 P/R/MRR/MAP/nDCG@5、Recall@20 | `eval/ir_metrics.py`，不变 | 否 | 主指标 |
| 上下文 | **Context AllRequired**（所有必需证据组都进了上下文）、**context claim recall**（标准论断的证据有多少进了上下文） | 证据锚点映射成 chunk ID 后做集合运算（对应 RAGChecker 的 claim_recall） | 否 | **主指标**（阶段 C 的 Gate 就用它） |
| 状态 | 期望状态混淆矩阵、不可答题错误作答率、可答题误拒答率、“状态与正文不一致”题数（D4） | `RAGAnswer.status` 对比 `expected_status` | 否 | **主指标** |
| 关键要素 | **`key_fact_recall`**（可答题中 key_matchable required claim 的 `answer_keys` 被回答命中的比例；macro 为主，另报 micro） | `key_match` 确定性子串匹配（SPEC §13.1(d)，借鉴 ALCE `compute_str_em`） | 否 | **主指标**（阶段 D 的 primary 依据） |
| 状态 + 关键要素 | 准确率与精确率（PaperQA2 口径）：“答对” = 实际 answered 且全部 key_matchable required claim 命中；准确率 = 答对 / 可答题，精确率 = 答对 / 实际作答的可答题 | 状态比对 + `key_match` | 否 | **主指标** |
| 受控上下文（生成器层，B5b） | `ablation_false_answer_rate`（删掉证据后仍作答的比例）、`counterfactual_faithfulness_rate`（数字被改后回答跟随上下文的比例） | 确定性构造变体，标签由构造决定（借鉴 RGB negative rejection / counterfactual robustness） | 否 | **主指标** |
| 引用格式 | 回答中合法引用数、引用到的来源是否属于证据组 | 确定性比对 | 否 | 主指标 |
| 回答 | claim recall（标准论断被回答覆盖的比例） | 评委逐条判定 COVERED / MISSING / CONTRADICTED | 是 | 诊断（未经人工校准） |
| 回答 | groundedness / 无依据论断率（回答论断能否由实际上下文支持） | 评委先拆陈述，再逐条判定 SUPPORTED / UNSUPPORTED | 是 | 诊断（未经人工校准） |
| 回答 | 引用支持精确率（引用的那条来源是否支持它所挂的论断） | 评委逐条判定 | 是 | 诊断（未经人工校准） |
| 诊断 | context utilization（证据进了上下文、回答却没覆盖的比例）、失败层归因（RETRIEVAL / CONTEXT / GENERATION / CITATION / STATUS） | 由上面各层结果组合 | 部分 | 诊断 |
| 成本 | LLM 调用次数、输入/输出 token、端到端与各阶段 p50/p95 延迟 | 来自 RunTrace | 否 | 主指标（只记录，不设合格线） |

### 3.2 评委的工作方式
- 评委只输出**结构化的逐条二元判定**（JSON，`extra="forbid"` 校验，失败修复一次），从不输出总分。
- 每次评委调用落盘：prompt 版本、模型、temperature、原始回复、解析结果、token 用量。
- 评委看到的上下文就是生成器当时实际看到的 `BuiltContext`（由 observation 中的 chunk ID 复原并用 sha256 校验），不是“知识库里某处有”。
- 评委 prompt 不包含被评系统的名字或版本，避免暗示。

### 3.3 一题一份 observation，计分与采集分离
```
dataset (rag_e2e_v1, 冻结, 带 fingerprint)
   │  collect（在线，调用生成器）
   ▼
observations.jsonl（回答、状态、上下文 chunk ID + 文本 sha256、RunTrace、错误）
   │  judge（在线，调用评委）
   ▼
judgments.jsonl（逐条判定 + 原始回复）
   │
   ▼
score（纯函数，离线，可重复；不调用任何模型）
   ▼
                  report.json / report.md  ── compare（配对 bootstrap + Holm）──► 与基线对比
```
- `score` 对同样的输入必须产出**逐字节相同**的报告（Gate 用 `cmp` 验证，借鉴 SuperMew）。
- 每个文件头部记录 dataset fingerprint、代码 commit、配置快照；fingerprint 不一致时拒绝计分。
- 在线步骤沿用 `online_common.py` 的策略：请求间隔、429 重试、JSONL 续跑、续跑前校验身份。

---

## B0 起点确认与小修

**目的：** 在干净、可验证的起点上开始，并顺手收掉阶段 A 留下的两个小问题。

**步骤**
1. 确认 `HEAD` 是 `stage-a-done` 或其后只有文档提交；打 tag `stage-b-start`。
2. CPU 跑一次 `experiments/stage_a/replay_runtime_retrieval.py` 完整回放，用 `a3_compare_replay.py` 确认与 B0' 逐题一致。
3. 小修：`scripts/verify_active_index.py` 打开 collection 时改用 `create_if_missing=False`（目前 collection 不存在时会建一个空的）；补测试。
4. **评委配置**（D1：另一家厂商的模型）：由**用户**在 `.env` 中加入 `EVAL_JUDGE_BASE_URL`、`EVAL_JUDGE_MODEL`、`EVAL_JUDGE_API_KEY`（执行者不读、不打印、不修改 `.env`）。执行者只做一次连通性探测（1 次请求），确认 `json_object` 或 `json_schema` 哪种结构化模式可用（沿用 `probe_structured_output.py` 的探测思路），把模式、模型名（不含 key）写进报告。评委模型与生成器同属一家厂商时 STOP。

**Gate B0**
- pytest 全部通过，记录总数 N_B0（基线 507 + 本步新增）。
- 回放与 B0' 逐题一致：Top5、Top20 ID 与分数、精排分数。
- `index_state.json` sha256 仍为 `5fbad4f0…8809`；`evals/` 无改动。

**提交：** `stage-b(B0): baseline check, verify_active_index read-only fix`

---

## B1 合同先行（只改文档）

**目的：** 把阶段 B 的所有合同先写进 SPEC，后面的代码以它为准。

**写进 `PROJECT_SPEC.md` 的内容（每条 3–10 行）**
1. **RunTrace**（新小节，挂在 §12 之后或作为 §12.5）：Domain Model 字段、事件名（复用 UI 合同 v0：`run.started`、`retrieval.dense.started/completed`、`rerank.started/completed`、`context.started/completed`、`generation.started/completed`、`citation_validation.completed`、`run.completed`、`run.failed`）、每个事件的 payload 白名单（ID、计数、分数、耗时、token 用量；**不含**上下文全文、Prompt 全文、API Key、模型思考内容）、`--trace PATH` 的 CLI 行为、“trace 不得改变回答”的不变式。
2. **评测集 schema `rag_e2e_v1`**（§13 新增小节）：`EvalCase` 字段（见 B3）、`expected_status`、`acceptable_abstention`、`tags`、`critical`、证据组、`required_claims`（每条带证据锚点）、`unanswerable_kind`、fingerprint 计算方式。
3. **评委协议**（§13）：三类判定任务的输入/输出 schema、prompt 版本号、修复规则、落盘字段、模型与 temperature 来源。
4. **指标定义**（§13）：§3.1 表中每个指标的分子分母和空分母处理（返回 null，不伪造）；部分回答计分规则（§9 D4 的结果）；`abstention_metrics` 改用 `AnswerStatus` 小写取值。
5. **Gate 策略文件格式**（§13）：方向、min/max、`max_regression`、`required`、`critical` 题不退化。
6. **评委检验合格线与指标地位**（§13）：见 B6；写明“评委指标未经人工校准，只作诊断，不作为任何阶段的晋升 Gate”。
7. **answer_keys 与 key_match**（§13.1(a)(c)(d)(e)）：claim 的 `answer_keys`（组间“与”、组内“或”，空列表即 `key_matchable=false`）、B3 的三条确定性校验、`key_match` 的规范化与 ASCII 边界规则、`key_fact_recall` 与“答对”的定义。
8. **mapping 文件**（§13.1(b)）：证据映射与题集分离，`mapping_<collection>.json` 绑定 `index_signature`、`collection_name`、`dataset_fingerprint`，任一不一致拒绝计分。
9. **B5b 受控上下文**（§13.1(f)）：`evidence_ablation` 与 `counterfactual_number` 的构造、有效性检查、预期判定、指标与产物格式。

**Gate B1**
- 只改了 `PROJECT_SPEC.md`、本计划（勘误与 v2），另新增 `artifacts/stage_b/B1_report.md`；pytest 不受影响。
- 主会话审查通过后才进入 B2（这一步是设计审查点）。

**提交：** `stage-b(B1): stage B contracts in PROJECT_SPEC`

---

## B2 RunTrace（只观测，不改行为）

**目的：** 看清一次提问每一步发生了什么。后面的评测流水线直接读 trace，不再自己埋点。

**代码改动**
1. `src/evidenceflow/domain/trace.py`：`RunTrace`、`TraceEvent`（Pydantic，`extra="forbid"`）。
2. 一个最小的 `TraceRecorder` 协议 + 默认的 no-op 实现；`RAGService`、`DenseRerankerRetriever`、`Generator` 通过**可选注入**的 recorder 记录事件。未注入时行为与现在完全一致，没有额外开销。
3. token 用量取自已有的 `Generator.last_diagnostics`；检索与重排序记录候选 chunk ID、分数、耗时；不记录正文。
4. `runtime.build_rag_service(settings, *, trace_recorder=None)`；CLI `ask` 增加 `--trace PATH`，把 RunTrace 写成 JSON（UTF-8、键排序）。
5. 不引入任何新依赖；不写日志之外的任何持久化。

**Gate B2**
- pytest 全部通过；新增测试覆盖：事件顺序、payload 白名单（断言不含正文、Prompt、Key）、失败时写出 `run.failed`、未注入 recorder 时调用路径与原来一致。
- **行为不变**：
  - CPU 完整回放（走 `build_retriever`）与 B0' 逐题一致；
  - 用 Fake LLM 对同一组问题分别开、关 trace 跑 `RAGService.answer`，`RAGAnswer` 逐字段相等。
- `python -m evidenceflow ask "<问题>" --trace artifacts/stage_b/B2_trace_sample.json` 真实运行 1 次（1 次 LLM 请求），trace 文件通过 schema 校验，并在报告中贴出事件列表。

**提交：** `stage-b(B2): RunTrace with optional recorder and ask --trace`

---

## B3 评测集 rag_e2e_v1

**目的：** 一份能同时评检索、上下文、状态和回答质量的题集，证据全部用文本锚点表示，与索引版本无关。

**题目构成（目标约 70 题）**
| 来源 | 数量 | 说明 |
|---|---|---|
| canonical 40 题 | 40 | 题干、证据组不变；为没有 required claims 的 20 题补写 claims |
| Phase 12 不可答题 `GEN-U01..05` | 5 | 沿用 |
| 新增多文档 / 对比题 | 约 15 | 例如两篇论文的方法、指标或局限对比；每题至少 2 个证据组，分属不同论文 |
| 新增不可答题 | 约 10 | 分三类：**语料外事实**（论文根本没涉及）、**近似但缺细节**（话题在语料里，但问的具体细节不存在）、**错误前提**（问题里的说法与论文矛盾）。近似类最能检验系统，至少占一半 |

中英文大致各半；新题不做中英平行版本。

**每题字段（`EvalCase`）**：`case_id`（新题用 `E2E-M*` / `E2E-U*` 独立 ID 空间）、`family_id`、`query`、`query_language`、`question_type`、`tags`（如 `multi_doc`、`comparison`、`unanswerable:near_miss`、`cross_lingual`）、`critical`、`expected_status`、`acceptable_abstention`、`required_evidence_groups`（锚点协议同 40 题）、`required_claims`（每条：`claim_id`、`text`、`required`、`evidence_anchors`、`answer_keys`，语义见 SPEC §13.1(a)）、`forbidden_claims`（错误前提题必填）、`unanswerable_kind`、`provenance`（谁起草、谁确认）。

**步骤**
1. **迁移**：把 40 题和 5 道不可答题转成新 schema。Phase 12 已有的 20 题 claims：在旧索引（`c512_o64_v1`，只读 `get_collection`）中找到 `acceptable_evidence` 对应的 chunk 正文，从中摘出支撑该 claim 的原句作为锚点，再到源 PDF 文本中验证并补全定位信息。
2. **起草新内容**（按 §9 决策 D3）：执行者从论文原文起草新题、claims 和锚点，**每条都标 `provenance.drafted_by = "agent"`**，同时为每条 claim 起草 `answer_keys`（无法做关键要素匹配的留空，即 `key_matchable=false`），生成一份待审清单 `artifacts/stage_b/B3_review_sheet.md`（每题一节：题干、期望状态、claims、**每条 claim 的 answer_keys**、锚点原文摘录、所在论文与页码）。
3. **用户审阅**：用户逐题选择接受 / 修改 / 删除，执行者按结果落盘，并把 `provenance.approved_by` 改为 `"user"`。**未经用户确认的题不得进入冻结版。**
4. **程序化校验**（全部通过才冻结）：
   - 每个锚点用 `stable_mapping` 在源 PDF 中定位成功，并映射到当前 active 索引的 chunk；无法映射的标为 UNMAPPABLE 并在报告中列出（不计入上下文类指标的分母，但保留在题集中）；
   - 不可答题：核心术语在 817 个 chunk 中检索为 0 命中，或（近似类）人工确认缺失的细节不在任何 chunk 中，写进 `B3_unanswerable_verification.json`；
   - 错误前提题：`forbidden_claims` 与论文原文矛盾，附原文锚点；
   - answer_keys 按 SPEC §13.1(c) 三条规则校验：① 每组至少一个别名出现在该 claim 锚点的 locator 页源文本中；② 任何别名不命中题干；③ 别名非空、规范化后长度 ≥ 2、组内去重；
   - 证据映射单独成文件 `evals/rag_e2e_v1/mapping_ef_research_bge_m3_hybrid_c512_v2.json`（SPEC §13.1(b)），题集本身只存锚点；
   - schema 校验、ID 唯一、fingerprint 计算并写入。
5. 冻结到 `evals/rag_e2e_v1/`：`dataset.json`、`dataset.sha256`、`mapping_ef_research_bge_m3_hybrid_c512_v2.json`、`README.md`（来源、规模、分布、已知局限）。

**Gate B3**
- 所有题 `approved_by = "user"`；锚点映射成功率与 UNMAPPABLE 清单写进报告。
- 可答题每题至少 1 条 required claim；多文档题至少 2 个证据组分属不同论文。
- 不可答题验证全部通过；不可答题中近似类占比 ≥ 50%。
- answer_keys 校验全部通过；报告给出可答题中 key_matchable required claim 的比例（不设硬下限，低于 60% 须说明原因）。
- 已有 `evals/` 文件无改动（`git diff` 为空）；pytest 通过（含新 schema 的测试）。

**提交：** `stage-b(B3): rag_e2e_v1 dataset (user-approved, anchor-based claims)`

---

## B4 评测流水线（离线实现）

**目的：** 实现 §3.3 的四个步骤。全部用 Fake 测试，这一步不发真实请求。

**代码位置**
- `src/evidenceflow/eval/e2e/`：`dataset.py`（加载与 fingerprint）、`observation.py`、`judge.py`（三类判定的 prompt、schema、解析与修复）、`perturb.py`（B6 的确定性负例构造）、`scoring.py`（纯函数，§3.1 全部指标与失败层归因）、`gates.py`（Gate 策略与判定）、`key_match.py`（关键要素匹配，SPEC §13.1(d)）、`controlled.py`（B5b 变体构造与有效性检查，SPEC §13.1(f)）。
- `scripts/eval_e2e.py`：子命令 `collect`、`judge`、`score`、`compare`、`perturb`、`controlled`。
- `eval/quality.py`：`abstention_metrics` 改用小写 `AnswerStatus`；按 D4 实现部分回答规则；保持已有测试通过，新增测试覆盖。
- 架构边界不变：`runtime` / `retrieval` 不得 import `eval`（`tests/test_architecture_boundaries.py` 守护）。

**要点**
- `collect`：通过 `runtime.build_rag_service(..., trace_recorder=...)` 跑每一题，记录回答、状态、上下文 chunk ID 与文本 sha256、RunTrace、错误类型（`GenerationContractError` / `CitationValidationError` / `LLMRequestError` 分开计数，不算作“答错”）。
- `judge`：逐题调用评委；评委配置独立于生成器（新增 `EVAL_JUDGE_*` 环境变量，只在评测脚本中读取，**不进入全局 Settings**，遵守“实验参数不得进入全局 Settings”）。
- `score`：只读输入文件，不调用任何模型；输出 JSON（键排序）与 Markdown；包含整体、按 tags 分片、按失败层归因、逐题明细；空分母为 null。
- `compare`：复用 `eval/comparison.py`（按题族 bootstrap、随机化检验、Holm 校正），预先声明要比较的指标集合。
- `perturb`：对 observation 做受控破坏（见 B6），用固定随机种子，同样输入产出同样负例；每个负例记录破坏类型和预期判定。
- `controlled`：读取 B5 observations，按 SPEC §13.1(f) 构造 `evidence_ablation` / `counterfactual_number` 变体并调用生成器（在 B5b 运行）；构造与有效性检查是纯函数，用 Fake 测试。
- 报告中每个指标带 `status` 字段：`primary` / `diagnostic_uncalibrated` / `unavailable`（借鉴 SuperMew：没有可靠实现就显式标为不可用）。

**Gate B4**
- pytest 全部通过；新增测试至少覆盖：每个指标的确定性 fixture（含空分母）、失败层归因、fingerprint 不一致时拒绝、`score` 对同一输入两次输出逐字节相同、评委回复格式错误时修复一次后仍失败的处理、负例构造的确定性与每类破坏的正确性、Gate 判定（含 `critical` 题退化、`diagnostic_uncalibrated` 指标不得参与 Gate）。
- 另需覆盖：`key_match` 边界规则（ASCII 边界、CJK 不要求边界、引用标签去除、组内别名“或”、组间“与”）；消融有效性检查（锚点原文残留、要素残留、空上下文 `trivially_empty`）；数字替换规则（小数进位、整数 +1、< 10 的整数不选、新值冲突重试、3 次后 `INVALID`）；以上构造对同一输入结果确定。
- 不发任何真实请求（测试中断言 LLM client 为 Fake）。

**提交：** `stage-b(B4): offline e2e evaluation pipeline (collect/judge/score/compare)`

---

## B5 采集 RAGService 输出（在线）

**目的：** 用当前线上链路对 `rag_e2e_v1` 全部题目跑一遍，产出冻结的 observations。它同时是 B6 评委检验的材料和 B7 基线的输入。

**步骤**
1. 调用前在报告中写明预计请求数（约 70 次生成调用，含修复最多约 140 次）。
2. `python scripts/eval_e2e.py collect --dataset evals/rag_e2e_v1/dataset.json --out artifacts/stage_b/B5_observations.jsonl`，CPU 检索。
3. 网络类错误按 `online_common.py` 策略重试；重跑后仍失败的题记为 `LLMRequestError`，不删除。

**Gate B5**
- 每题都有一条 observation（成功或带错误类型）；`LLMRequestError` ≤ 2 题，否则 STOP。
- 每条 observation 的上下文 chunk ID 都能在 active 索引中找到，sha256 可复原。
- RunTrace 全部通过 schema 校验；报告给出 token 总量、p50/p95 延迟。

**提交：** `stage-b(B5): collect RAGService observations on rag_e2e_v1`

---

## B5b 受控上下文检验（在线，生成器层）

**目的：** 不依赖人工标注和评委，用确定性构造的上下文变体检验生成器：证据被删掉后是否仍然作答（借鉴 RGB negative rejection），数字被改掉后回答是跟随上下文还是凭模型自身知识（借鉴 RGB counterfactual robustness）。合同见 SPEC §13.1(f)。这是生成器层检验，不是端到端。

**步骤**
1. 先离线构造全部变体，打印 `evidence_ablation` / `counterfactual_number` 的有效、`INVALID`、`trivially_empty` 数量和预计请求数，写进报告，再发请求。
2. `python scripts/eval_e2e.py controlled --observations artifacts/stage_b/B5_observations.jsonl --out artifacts/stage_b/B5b_controlled.jsonl`。上下文由 chunk ID 从 active 索引只读复原并用 sha256 校验，不经过检索，不需要 CPU 检索。
3. 网络类错误按 `online_common.py` 策略重试；仍失败的记 `LLMRequestError`，不删除。

**Gate B5b**
- 有效非空的消融变体 ≥ 20、数字反事实有效变体 ≥ 8；不足时 STOP，并在报告中给出分布（各类不合格原因计数）。
- 每个变体都有确定的预期判定和有效性记录；`INVALID` 清单写进报告。
- `LLMRequestError` ≤ 2；真实请求数 ≤ 步骤 1 估算的 1.5 倍。

**提交：** `stage-b(B5b): controlled-context generator checks`

---

## B6 评委检验（无人工标注）

**目的：** 在没有人工标注的前提下，尽量证明评委“能发现错误、判定稳定”，并量化它的偏差方向。通过检验的评委指标仍然只是**诊断指标**：它们可以帮助定位问题，但不能作为任何阶段的晋升 Gate。

**检验材料**
1. **构造负例（主要检验）**：先用评委判定 B5 的原始 observations，再对其中被判为正确的样本做受控破坏（`scripts/eval_e2e.py perturb`，固定种子），每类约 20 例：
   - 把某条论断的引用换成同一上下文中另一个不相关的来源 → 预期“引用不支持”；
   - 把论断中的数字或方法名改成上下文中不存在的值 → 预期“无依据”；
   - 插入一条来自另一篇论文、与本题上下文无关的陈述 → 预期“无依据”；
   - 删掉回答中覆盖某条 required claim 的句子 → 预期该 claim“MISSING”。
   负例与未破坏的原样本混在一起、打乱顺序交给评委，评委不知道哪些被改过。破坏是程序化的，所以每个负例的“正确答案”是确定的，不依赖任何人或模型的判断。
2. **复测一致性**：同一批输入让评委隔一段时间再判一次，统计两次判定一致的比例。
3. **确定性交叉核对**：claim recall 判定与 B3 的证据锚点交叉检查。证据**没有**进入上下文、而评委判为 COVERED 的论断，逐条列出（这类判定要么是模型凭自身知识作答，要么是评委误判），写进报告供主会话抽查。另做评委 COVERED 与 `key_match` 命中的逐条对照：key_match 命中、评委却判 MISSING 的，以及 key_match 未命中、评委却判 COVERED 的，都逐条列出。
4. **Phase 12 旧输出（辅助参考）**：从 `c512_o64_v1` 只读复原 Phase 12 的上下文并用 `context_text_sha256` 校验；让评委判定，与 Phase 12 的 agent 标注对比。报告中注明“agent-annotated，仅供参考”，不进入合格判定。

**合格线（每个判定任务分别判断）**
| 检查 | 合格线 |
|---|---|
| 构造负例的检出率 | ≥ 0.9 |
| 未破坏样本的误报率 | ≤ 0.1 |
| 复测一致率 | ≥ 0.95 |
| 与 Phase 12 agent 标注的一致率与 κ | 只报告，不设合格线 |

**结果处理**
- 三类判定任务各自标记为“通过机器检验”或“未通过”。通过的在 E0 报告中显示为 `diagnostic_uncalibrated`；未通过的显示为 `unavailable`，并写明原因。
- **claim recall 判定未通过时 STOP**：它是阶段 D Gate 依赖的指标，需要主会话和用户决定是换评委模型、改 prompt（改完必须用**新生成的**负例重新检验，不能在同一批负例上反复调），还是在阶段 D 之前补做人工标注。
- 记录评委的偏差方向：对未破坏样本偏严（误报多）还是对负例偏宽（漏检多）。
- prompt 一旦用于检验就冻结版本号；之后任何改动都必须重新检验。

**Gate B6**
- 负例按类型数量达标；每个负例都有确定的预期判定。
- 各项检查结果、每类破坏的检出率、混淆矩阵和偏差方向写进 `B6_judge_check.json` 与报告。
- 第 3 项列出的可疑判定清单写进报告。

**提交：** `stage-b(B6): judge checks with synthetic negatives and test-retest`

---

## B7 端到端基线 E0 与阶段收尾

**目的：** 给当前 RAGService 打分，冻结为阶段 C/D/E 的对照基线 E0。

**步骤**
1. `judge`：对 B5 全部 observations 运行评委（约 140 次调用，写入报告；B6 中已经判过的原始样本可以直接复用，不重复调用）。
2. `score`：产出 `artifacts/stage_b/E0_report.json` / `.md`。主指标（§3.1 中标为“主指标”的确定性指标，含 `key_fact_recall`、准确率与精确率）放在最前面，另设“受控上下文（生成器层）”一节读取 `B5b_controlled.jsonl`；评委指标单独一节，标题写明“诊断（未经人工校准）”。
3. 用 `score` 重算一次，`cmp` 逐字节一致。
4. 写 Gate 策略文件 `evals/rag_e2e_v1/gates_v1.json`：阶段 C/D/E 对比时用到的确定性指标的方向、合格线、允许退化和 `critical` 题清单。候选指标包括 `key_fact_recall`、`ablation_false_answer_rate`；初值以 E0 为准，阈值依据架构审查 §10。评委指标不写进 Gate。
5. 更新 `PROJECT_SPEC.md` §13：写入 E0 路径、sha256 和主指标；更新 README 的指标表（增加端到端指标）。
6. 写 `artifacts/stage_b/STAGE_B_SUMMARY.md`：各 Gate 结果、E0 指标、评委检验结论、失败层分布、成本、遗留问题、给阶段 C 的输入（重点：多文档题的 Context AllRequired 与 context claim recall 现状）。遗留问题中必须写明：**阶段 D 以 `key_fact_recall` 为 primary 依据；评委 claim recall 只作诊断。**
7. 打 tag `stage-b-done`（本地）。

**Gate B7**
- E0 报告可逐字节重建；dataset fingerprint 与 observations、judgments 一致。
- 报告包含：整体与分片指标、状态混淆矩阵、`key_fact_recall`、准确率与精确率、受控上下文一节、失败层归因、成本与延迟；每个指标带 `primary` / `diagnostic_uncalibrated` / `unavailable` 标记。
- pytest 全部通过；`index_state.json` 与 `evals/` 已有文件未改动；CPU 回放与 B0' 一致（确认 B2–B7 没有影响线上）。

**提交：** `stage-b(B7): E0 end-to-end baseline, gates_v1, SPEC and summary`，然后 `git tag stage-b-done`。**完成后停止，不要开始阶段 C。**

---

## 6. 交付物

- `evals/rag_e2e_v1/`：`dataset.json`、`dataset.sha256`、`mapping_ef_research_bge_m3_hybrid_c512_v2.json`、`README.md`、`gates_v1.json`
- `artifacts/stage_b/`：`B0_report.md` … `B7_report.md`、`B2_trace_sample.json`、`B3_review_sheet.md`、`B3_unanswerable_verification.json`、`B5_observations.jsonl`、`B5b_report.md`、`B5b_controlled.jsonl`、`B6_negatives.jsonl`、`B6_judge_check.json`、`E0_judgments.jsonl`、`E0_report.json`、`E0_report.md`、`STAGE_B_SUMMARY.md`
- 代码：`domain/trace.py`、`eval/e2e/*`（含 `key_match.py`、`controlled.py`）、`scripts/eval_e2e.py`，以及对应测试

## 7. STOP 条件

出现以下任何情况，立即停止并汇报：
- 任何一步 pytest 失败且无法在本步范围内修复；
- B0 / B2 / B7 的 CPU 回放与 B0' 不一致，或 trace 开关导致 `RAGAnswer` 不同；
- B0 评委模型与生成器属于同一家厂商，或评委连通性探测失败；
- B3 有未经用户确认的题需要进入冻结版，或不可答题验证失败；
- B5 `LLMRequestError` 重跑后仍超过 2 题；
- B5b 有效非空消融变体 < 20 或数字反事实有效变体 < 8，或 `LLMRequestError` 超过 2；
- B6 claim recall 判定未通过机器检验；
- 需要新增依赖，或需要修改 Gold、qrels、已有 `evals/` 冻结产物；
- 真实请求数超过本计划估算的 1.5 倍；
- 本计划与代码实际情况不符且无法判断如何处理。

## 8. 成本估算（执行时以实际为准）

| 步骤 | 调用 | 说明 |
|---|---|---|
| B0 | 1–3 | 评委连通性与结构化模式探测 |
| B2 | 1 | `ask --trace` 真实运行 |
| B5 | 约 70（最多约 140） | 生成，含修复 |
| B5b | 约 60–80（最多约 160） | 受控上下文变体的生成调用，含修复；`INVALID` 与 `trivially_empty` 不调用 |
| B6 | 约 250 | 原始样本判定约 140（B7 复用）、约 80 个负例、复测约 60、Phase 12 辅助约 50 |
| B7 | 约 0–20 | 大部分复用 B6 的原始判定，只补未判过的 |
| 合计 | 约 410–530 | 生成器走 DeepSeek，评委走另一家厂商，两边分别计费；评委 prompt 比生成长 |

**用户的时间**：B0 前在 `.env` 中配置评委（几分钟）；B3 审阅约 50 题的新内容和补写的 claims（预计 2–3 小时）。

## 9. 用户决定（2026-10-15 确认）

| 编号 | 问题 | 决定 | 对计划的影响 |
|---|---|---|---|
| D1 | 评委用什么模型 | **另一家厂商的模型：智谱 `glm-4-flash`**（用户已在 `.env` 配置 `EVAL_JUDGE_*`） | B0 探测连通性与结构化模式；`glm-4-flash` 是轻量模型，B6 若未通过机器检验，由用户决定是否换更强的智谱模型 |
| D2 | 是否做人工标注 | **不做** | 评委指标只作诊断，不进 Gate；不实现 PPI；B6 改为负例检验 + 复测；E0 主指标全部是确定性指标 |
| D3 | 新题和 claims 谁来写 | **执行者起草，用户逐题审阅** | B3 产出待审清单，未经用户确认的题不得冻结 |
| D4 | 拒答状态但正文有部分回答 | **状态按拒答计，正文照常计分，单独统计不一致题数** | 写进 B1 指标定义与 B4 实现 |
| D5 | 题集规模 | 约 70 题（默认） | 结论一律写成“方向一致 / 置信区间”，不写“显著提升” |
| D6 | 无人工标注时，回答质量用什么确定性方法评 | **采纳两项确定性评测**：① 关键要素匹配（claim 的 `answer_keys` + `key_fact_recall`）；② 受控上下文检验生成器（B5b）。依据：ALCE `eval.py::compute_str_em`、RGB `evalue.py::checkanswer/processdata`（negative rejection、counterfactual robustness） | B1 写进 SPEC §13.1(a)–(f)；B3 起草并校验 `answer_keys`，mapping 单独成文件；B4 实现 `key_match.py`、`controlled.py`；新增 B5b；`key_fact_recall` 与“答对”判定为主指标，阶段 D 以 `key_fact_recall` 为 primary 依据 |

**关于 D2 的后续影响**：阶段 D 的 Gate 是“Claim Recall 提升”，按 D6 以确定性的 `key_fact_recall` 为 primary 依据；评委 claim recall 只作诊断旁证（最终以阶段 D 计划为准）。`key_fact_recall` 只覆盖 key_matchable 的 claim；如果阶段 D 需要评估无法做关键要素匹配的 claim，可以届时只针对阶段 D 的对比补做一小批人工标注（约 15–20 题），再按 ARES 的 PPI 思路做纠偏估计。
