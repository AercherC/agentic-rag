# EvidenceFlow 阶段 A 执行计划：基础修复

- 版本：v1（2026-10-14）
- 执行者：Claude Code（Sonnet），在本机 `agent-gpu` Conda 环境中执行
- 验收者：用户，可以交给 Opus 审查
- 背景与依据：`docs/agentic_rag/01_ARCHITECTURE_REVIEW.md`

---

## 0. 执行者必读

### 0.1 目标
阶段 A **不增加任何 Agent 能力**，只做五件事：
1. 把线上检索换成确定性的 Exact Dense；
2. 把生成改成结构化输出，并给 `RAGAnswer` 加上 `status`；
3. 让线上读取的索引与 Phase 15A 的 lifecycle 一致，并禁止直接修改 ACTIVE 索引；
4. 清理废弃文件和孤儿数据；
5. 精简 `PROJECT_SPEC.md`。

完成后，阶段 B（评测）有一个确定、可复现的基线可以对照。

### 0.2 开工前先读
1. `AGENTS.md`：项目规则。注释和 docstring 用中文；测试用 Fake/Stub，不调用真实 LLM；每完成一步按 §7 的格式汇报。
2. `PROJECT_SPEC.md` 的以下部分：顶部 Current Architecture Summary、§24–§28、§39、§41、§42、§43。
3. `docs/agentic_rag/01_ARCHITECTURE_REVIEW.md` 的 §2、§4、§9。
4. 每改一个模块之前，先读该模块、直接依赖它的模块，以及对应的测试。

### 0.3 硬性规则
- **按 A0 → A5 的顺序执行。每一步的 Gate 通过后才能进入下一步。** Gate 不通过就 **STOP**：把诊断写进 `artifacts/stage_a/A<n>_report.md`，然后向用户报告，**不要放宽 Gate，也不要修改 Gold、qrels 或冻结的 `evals/` 产物**。
- 每一步结束都要提交一次 git，提交信息为 `stage-a(A<n>): <摘要>`。**不要提交 `.env`。**
- 不新增第三方依赖。唯一例外是 A1 中可选的 `faiss-cpu`，安装前必须先征得用户同意。
- 不实现阶段 B–E 的任何内容：Planner、Grader、补检、RunTrace、文档过滤、新评测题都不做。
- 涉及合同的改动（数据模型、接口、配置、Generator 协议），在动代码之前先写进 `PROJECT_SPEC.md` 顶部新建的一节 **“§S Stage A 合同变更（待 A5 合并）”**，每条 3–8 行。A5 再把它们并入新的 SPEC。
- **数值 Gate 一律用 CPU 运行**，设置 `MODEL_DEVICE=cpu`。所有冻结基线都是 CPU 跑出来的，GPU 的浮点差异可能改变排序。GPU 结果只记录，不作为 Gate。
- 不确定的地方不要猜，停下来问用户。

### 0.4 每次打开终端时的环境准备（PowerShell）
```powershell
conda activate agent-gpu
cd D:\Agent_learn\evidenceFlow
$env:HF_HOME = (Resolve-Path .cache/huggingface)
$env:TRANSFORMERS_OFFLINE = '1'
$env:HF_HUB_OFFLINE = '1'
# 已知环境敏感点：离线装配时 reranker 要指向本地目录（见 MEMORY.md 与 Phase 12 §7）
$env:RERANKER_MODEL = (Resolve-Path .cache/models/bge-reranker-base)
$env:MODEL_DEVICE = 'cpu'   # 跑数值 Gate 时必须是 cpu
```
所有产物写到 `artifacts/stage_a/`（新建目录）。不要覆盖 `evals/` 下的任何已有文件。

### 0.5 已知事实（审查时确认过，执行时请复核）
- Active collection：`ef_research_bge_m3_hybrid_c512_v2`，10 篇文档、817 个 chunk
  - chunk ID hash：`e87a613ebe65b10262996e83974b55839e0b6c27fbe22053ee7b8826dd1f153d`
  - index signature：`b4e9432c7170615ade2c53803a887fa86d0f793949934a57db421f0da57739e5`
- 冻结的 B0（HNSW Dense20 → Reranker → Top5）：`evals/evaluation_v3/phase15b_hybrid_candidate_v2/production_reranked_baseline.json`
  - P@5 0.2684、R@5 0.5103、MRR@5 0.5175、MAP@5 0.3631、nDCG@5 0.4903
- Chroma 中共有 7 个 collection：
  - `ef_research_bge_m3_c512_o64_v1`（361）
  - `..._c512_o64_refmeta_v1`（361）
  - `ef_research_structure_v1`（385）
  - `ef_research_structure_v1_elemaware`（935）
  - `ef_research_bge_m3_hybrid_c512_v1`（817）
  - `..._hybrid_c512_v2_staging`（817）
  - `..._hybrid_c512_v2`（817）
- `experiments/run_production_reranked_baseline_replay.py` 自己装配组件，**不经过 `runtime.py`**，所以它验证不了线上链路。
- 可复用的比较函数：`experiments/run_phase15b1_dense_candidate_depth.py` 中的 `compare_replay`、`load_frozen_inputs`、`validate_active_index`、`standard_summary`；`experiments/phase15b_active_relevance.py` 中的 `project_active_relevance`。

---

## A0 冻结起点与重建基线

**目的：** 留一个可以回退的起点，并在 `agent-gpu` 环境里亲手确认当前基线，不依赖审查报告转述的数字。

**步骤**
1. 用 `git status` 查看改动。把当前所有修改和未跟踪的正式产物提交为 `stage-a(A0): snapshot before stage A`，包括：
   - `experiments/run_phase15b1_*`、`run_phase15c*`
   - `evals/evaluation_v3/phase15b1_*`、`phase15c*`、`gpu_inference_acceleration_v1`
   - `artifacts/environment/`

   **不要提交：** `1.26`、`src/evidenceflow.zip`、`.env`。

   然后执行 `git tag stage-a-start`。
2. 新建 `CLAUDE.md`，内容只有一行：`本项目的协作规则见 AGENTS.md；当前阶段计划见 docs/agentic_rag/02_STAGE_A_EXECUTION_PLAN.md。` 这样以后的 Claude Code 会话会自动读到规则。
3. 运行全量测试，并记录结果：
   ```powershell
   python -m pytest -q 2>&1 | Tee-Object artifacts/stage_a/A0_pytest.log
   ```
4. 运行索引身份检查，结果保存为 `A0_verify_active_index.txt` 和 `A0_verify_phase15a.txt`：
   ```powershell
   python scripts/verify_active_index.py --collection ef_research_bge_m3_hybrid_c512_v2 --expected-count 817
   python scripts/verify_phase15a.py
   ```
5. 运行冻结 B0 的回放。注意 `--output` 必须指向 stage_a 目录，不能覆盖冻结文件：
   ```powershell
   python experiments/run_production_reranked_baseline_replay.py --output artifacts/stage_a/A0_hnsw_replay.json
   ```
   然后写一段小脚本，用 `compare_replay` 把它与冻结的 `production_reranked_baseline.json` 逐题比较 Top5。
6. 记录环境信息到 `A0_env.json`：`torch.__version__`、`torch.cuda.is_available()`、`chromadb.__version__`、是否能 `import faiss`、Python 版本。

**Gate A0**
- pytest 全部通过。记下测试总数 N0，后面 A4 会用到。
- 索引身份检查全部通过（817 个 chunk、chunk ID hash 一致）。
- HNSW D20 回放与冻结 B0 的 40 题 Top5 逐题一致。
  - 如果不一致：这说明 HNSW 的不确定性在 D20 也出现了。照样记录，**这不阻塞 A1**（A1 正是为了消除它），但要在报告里列出不一致的题号。

**产物：** `artifacts/stage_a/A0_report.md` 和上面的日志。

---

## A1 线上检索切换为 Exact Dense

**目的：** 消除 HNSW 带来的不确定性，让后续所有 A/B 实验可以逐题复现。

**设计（在 §S 中写下合同）**
- Chroma 仍然是唯一的持久化来源。Runtime 启动时，通过 `ChromaVectorStore.iter_all_chunks_with_embeddings()` 在进程内构建 `ExactDenseIndex`。
- `DenseRetriever` 依赖的 `DenseSearch` 协议不变：`ExactDenseIndex.search(query_embedding, top_k)` 本来就符合这个协议。
- **不新增 backend 配置开关。** SPEC §41 规定 Runtime 不维护第二套算法；回滚靠 git tag。
- 启动时要校验：Exact 索引的大小等于 `store.count`，维度为 1024。不满足就报错并拒绝启动。
- 候选深度保持 K=20、Final K 保持 5。**D30 晋升不在阶段 A 的范围内。**
- 后端：`faiss` 能 import 就用 `faiss.IndexFlatIP`，否则用 `numpy.exact`。在 §S 中写明：**`numpy.exact` 是被接受的正式后端**（817 个向量不需要 FAISS），并把这一条作为对 §42 FAISS 相关要求的修订提交用户确认。如果用户坚持要 FAISS，就在征得同意后执行 `conda install -c pytorch faiss-cpu`。

**代码改动**
1. `src/evidenceflow/runtime.py`
   - 新增 `build_retriever(settings) -> DenseRerankerRetriever`。它负责装配 embedding、store、Exact 索引、DenseRetriever 和 reranker，并做上面的启动校验。
   - `build_rag_service` 改为调用 `build_retriever`。这样后面的回放脚本可以不创建 LLM client，只装配检索部分。
   - 加中文注释，说明为什么用 Exact：可复现性，以及 817 个向量的计算成本可以忽略。
2. `src/evidenceflow/retrieval/__init__.py`：确认 `ExactDenseIndex` 已经导出，不需要改。
3. `tests/test_runtime.py`
   - 更新 Fake：FakeStore 增加 `iter_all_chunks_with_embeddings` 和 `count`；monkeypatch `evidenceflow.runtime.ExactDenseIndex`。
   - 新增断言：DenseRetriever 收到的 search 对象是 Exact 索引。
   - 新增用例：大小不一致时 `build_retriever` 抛出异常。

**新增实验脚本**
- `experiments/stage_a/replay_runtime_retrieval.py`
  - 只调用 `evidenceflow.runtime.build_retriever(Settings())` 走线上路径，对冻结的 40 题（`evals/research_retrieval/research_questions_v1.json`）做检索。
  - 输出 JSON，schema 与 `production_reranked_baseline.json` 相同，这样 `compare_replay` 可以直接复用。实现前先读这两个文件的结构。
  - 参数：`--output <path>`，以及 `--candidates-only`（只输出 Dense Top20，不做 rerank，用于快速检查确定性）。

**验证步骤**
1. `python -m pytest -q`，必须全部通过。
2. **候选确定性：** 用 10 个独立进程分别跑 `--candidates-only`，比较 40 题的 Top20 ID 和分数是否全部逐元素一致：
   ```powershell
   1..10 | ForEach-Object { python experiments/stage_a/replay_runtime_retrieval.py --candidates-only --output artifacts/stage_a/A1_cand_run$_.json }
   ```
   再写一个比较脚本，把结果写入 `A1_determinism.json`。
3. **完整链路：** 用两个独立进程各跑一次完整的 rerank 回放，得到 `A1_exact_replay_1.json` 和 `_2.json`，二者 Top5 必须逐题一致。
4. **与 HNSW B0 对比：** 用 `standard_summary` 计算 Exact 回放的 IR@5 指标，并与冻结 B0 逐题比较 Top5，结果写入 `A1_exact_vs_hnsw.json`。内容包括：有差异的题号、差异前后的 Top5、以及 5 项 IR 指标的差值。
5. 计时：分别记录 Exact 索引的构建耗时和单次查询的 dense 耗时。
6. 可选：用 `MODEL_DEVICE=cuda` 跑一次完整回放，记录 Top5 与 CPU 的差异和延迟，写入 `A1_gpu_reference.json`。这一项不作为 Gate。

**Gate A1**
- 10 次独立进程的 Top20 候选 100% 一致；两次完整回放的 Top5 100% 一致。
- 与 HNSW B0 相比，R@5 和 nDCG@5 的下降都不超过 0.01（绝对值）。下降超过这个幅度就 **STOP**，报告差异题，由用户决定。
- 通过后，把 Exact 回放冻结为**新的正式基线**：
  - `artifacts/stage_a/A1_exact_baseline.json`
  - 同时记录它的 sha256
  - 在 §S 中写明“B0' = Exact Dense20 → Reranker → Top5”，并列出 5 项指标

**提交：** `stage-a(A1): production dense search uses ExactDenseIndex`

---

## A2 结构化生成与 `RAGAnswer.status`

**目的：** 消除控制头格式错误（Phase 12 中为 3/25），并让下游能区分“已回答”和“拒答”。

### A2.0 先探测 LLM 服务能力，探测结果决定实现方式
- 新增 `experiments/stage_a/probe_structured_output.py`，使用 `.env` 中配置的 endpoint 和 model（不要打印 key）。
- 用 3 个简单的 RAG 风格请求，分别尝试以下三种方式：
  - (a) `response_format={"type":"json_schema","json_schema":{...,"strict":true}}`
  - (b) `response_format={"type":"json_object"}`
  - (c) 不传 `response_format`，只在 Prompt 里要求输出 JSON
- 每种方式记录：
  - 是否返回 HTTP 错误
  - `content` 能否被 `json.loads` 解析
  - 是否混入 `<think>...</think>`
  - 是否被 ```json 代码块包住
  - `usage` 字段是否存在
- 结果写入 `artifacts/stage_a/A2_probe.json`，并选出可用的最优模式：json_schema 优于 json_object，json_object 优于 prompt_only。
- 如果三种都不可用（例如服务报错或无法连接）：**STOP**，向用户报告。

### A2.1 合同（先写进 §S）
- **Prompt 版本**升为 `rag-generator-v3`。模型输出必须是 JSON：
  ```json
  {"status": "answered" | "insufficient_context", "answer": "<面向用户的回答，含 [S1] 等标签>"}
  ```
  Context 模板、Grounding 规则、Citation 标签语法和不可信数据声明，全部沿用 v2。只有输出协议改变。
- **解析顺序：**
  1. 去掉开头的 `<think>…</think>`；
  2. 去掉外层的 ```json 代码块；
  3. `json.loads`；
  4. 用 Pydantic 模型 `GeneratorOutput`（`extra="forbid"`）校验。
- **修复重试：** 解析或校验失败时，最多重试一次。重试时把原消息、模型的原始回复，以及一条修复指令（“上一条回复不符合 JSON schema，只返回 JSON”）一起发给模型。仍然失败就抛出 `GenerationContractError(ValueError)`。这次修复调用与 LLM Client 内部的瞬态重试是两回事，要分开计数。
- **空 Context 时的行为不变：** 不调用 LLM，直接返回确定性的拒答。
- **`RAGAnswer` 新增必填字段 `status: AnswerStatus`。** 由 `assemble_rag_answer` 从 `GeneratedAnswer.status` 透传。Citation 规则（answered 必须至少有一个引用；任何状态都不允许未知标签）保持不变。
- **LLM Client 协议新增：**
  - `generate_json(messages, *, schema_name: str, json_schema: dict) -> str`，按 A2.0 选定的模式发送请求；
  - `last_usage: dict | None` 属性（`prompt_tokens`、`completion_tokens`），没有 usage 时为 None。
  - 原有的 `generate()` 保留。

### A2.2 代码改动
| 文件 | 改动 |
|---|---|
| `src/evidenceflow/llm/base.py` | Protocol 增加 `generate_json` |
| `src/evidenceflow/llm/openai_compatible.py` | 实现 `generate_json` 和 `last_usage`；沿用现有的瞬态重试与异常分类 |
| `src/evidenceflow/domain/generation.py` | 新增 `GeneratorOutput` |
| `src/evidenceflow/generation/generator.py` | 改为 v3；删除 v2 的控制头解析，**不保留双路径**；新增 `GenerationContractError` |
| `src/evidenceflow/generation/__init__.py`、`llm/__init__.py` | 导出新增的符号 |
| `src/evidenceflow/domain/answer.py` | `RAGAnswer.status` |
| `src/evidenceflow/citation/parser.py` | 组装时透传 status |
| `src/evidenceflow/__main__.py` | 输出时显示状态（例如在 `Answer:` 前加一行 `Status: answered`）；`_render_ask_error` 增加 `GenerationContractError` → `Generation format error` |
| 测试 | 更新以下测试（都直接构造或断言了 `RAGAnswer` 或 v2 格式）：`test_generator.py`、`test_rag_answer_assembly.py`、`test_domain_models.py`、`test_ask_cli.py`、`test_phase9_runtime_promotion.py`、`test_llm_client.py`、`test_rag_service.py` |

**新增测试用例**
- 合法 JSON；
- 带 `<think>` 前缀；
- 带 ```json 代码块；
- 非法 JSON 后修复成功；
- 修复后仍失败时抛出 `GenerationContractError`；
- 未知 status 和多余字段被拒绝；
- 空 Context 不调用 LLM；
- `last_usage` 为 None 时也能正常工作。

所有测试都用 Fake LLM。

### A2.3 真实冒烟（需要联网和 API key）
- 新增 `experiments/stage_a/run_generation_smoke.py`
  - 读取 `evals/evaluation_v3/phase12_rag_quality_v1/generation_eval_dataset.json`（25 题）。
  - 走 `runtime.build_rag_service(Settings())` 的完整线上链路，此时已经是 Exact + v3。
  - 对每题记录：`status`、是否格式失败、是否触发了修复、`CitationValidationError`、`LLMRequestError`、`last_usage`、耗时。
  - 输出写到 `artifacts/stage_a/A2_generation_smoke.json`。**不要覆盖** Phase 12 的 `runtime_outputs.json`。
- 这一步只统计格式和状态，**不评估答案质量**，质量评测属于阶段 B。

**Gate A2**
- pytest 全部通过。
- 冒烟 25 题中，`GenerationContractError` 为 0。
- 触发修复的次数如实记录；如果超过 3 题，在报告中标注风险。
- `LLMRequestError`（网络或服务类错误）单独统计，不计入 Gate。如果超过 2 题，重跑一次；仍然超过就 **STOP**。
- `CitationValidationError` 如实记录。v3 没有改变 Citation 规则，如果出现，列出题号交给用户判断。

**提交：** `stage-a(A2): structured generator v3 and RAGAnswer.status`

---

## A3 索引生命周期接到线上

**目的：** 让 Phase 15A 的 activate/rollback 真正影响线上读取的索引，并防止 ACTIVE 索引被半更新。

### 合同（先写进 §S）
- **新增配置项** `Settings.index_state_path: Path = Path("artifacts/phase15a/index_state.json")`，加上中文 description。
- **Runtime 解析 collection 的规则：**
  1. 如果 `index_state_path` 存在：以 `state.active.collection_name` 为准。如果用户**显式设置了** `COLLECTION_NAME`（用 `"collection_name" in settings.model_fields_set` 判断）而且与 active 不一致，就抛出配置错误并拒绝启动，不要静默选择其中一个。
  2. 如果文件不存在：使用 `Settings.collection_name`。这条路径给测试和全新环境使用。
  3. 打开 collection 后，校验其 metadata 中的 `lifecycle == "ACTIVE"`（与 §39.1 的 15A-0 规则一致）。前提是 metadata 里有这个字段；先用 `scripts/verify_phase15a.py` 确认 v2 的 metadata 里确实有。
- **`Settings.collection_name` 的默认值**改为 `ef_research_bge_m3_hybrid_c512_v2`。
- **`ingest` 的保护：** 如果目标 collection 就是 active pointer 指向的 collection，CLI 拒绝执行，退出码为 2。提示信息说明应该使用 staging 构建脚本。
- **staging 构建脚本** `scripts/build_staging_index.py`：
  - 复用 `StagingBuildCoordinator` + `IngestionService`（作为 builder 回调）+ `validate_collection`；
  - 参数 `--expected-chunk-ids-from-active`（用于全量重建时的一致性验证）、`--activate`（默认不激活）；
  - 不复制 ingestion 逻辑。
- `ingestion/service.py` 中“先删后写”的问题，因为 ACTIVE 已经不允许 ingest，在 staging 的全新 collection 上不再触发。在 §S 中记录这一点，**本阶段不改 IngestionService 的内部逻辑**。

### 代码改动
| 文件 | 改动 |
|---|---|
| `src/evidenceflow/config.py` | 新增 `index_state_path`；修改 `collection_name` 默认值；中文 description |
| `src/evidenceflow/runtime.py` | 新增 `resolve_active_collection(settings) -> str`；`build_retriever` 和 `build_ingestion_service` 都调用它；ingest 的保护逻辑可以放在 `build_ingestion_service` 里，或由 CLI 调用一个校验函数，以简洁为准 |
| `src/evidenceflow/__main__.py` | ingest 遇到保护时的错误提示与退出码 |
| `scripts/build_staging_index.py` | 新增 |
| `.env.example` | `LLM_BASE_URL`、`LLM_MODEL` 与代码默认值一致（ModelScope / Qwen/Qwen3.5-35B-A3B）；`COLLECTION_NAME` 的注释改为“通常不需要设置，以 lifecycle active pointer 为准” |
| `tests/test_runtime.py`、`test_config.py`、`test_ingest_cli.py`、`test_phase15a_lifecycle.py` | 补充用例（见下） |

**新增测试**（全部使用临时目录中的 state 文件和 Fake store）
- 有 state 文件、未显式设置 → 使用 active；
- 显式设置且不一致 → 报错；
- 先 `IndexLifecycle.activate` 再 `rollback` → 解析结果随 pointer 改变；
- 没有 state 文件 → 回落到 Settings；
- metadata 中 lifecycle 不是 ACTIVE → 报错；
- ingest 指向 active → 退出码 2；
- staging 脚本使用 Fake builder 时能完成 SCANNING…READY，且默认不激活。

### 真实验证
1. 用当前 `.env` 和 state 文件运行 `python -m evidenceflow ask "PoinTr 如何表示用于 Transformer 处理的点云？"`，确认能正常回答，并显示 status。
2. `python -m evidenceflow ingest data/source_docs` → 应该被拒绝，退出码为 2。
3. 全量 staging 重建，不激活：
   ```powershell
   python scripts/build_staging_index.py --expected-chunk-ids-from-active
   ```
   - 期望 validation 通过：817 个 chunk，chunk ID 集合与 active 完全一致。这同时验证了 ingestion 的确定性。
   - 完成后通过 Chroma API 删除这个 staging collection，并把 lifecycle 中的 staging 指针清空（使用现有 API，或在报告中说明手动处理的方式）。

   如果 chunk ID 不一致：**STOP**。这说明 ingestion 不确定，需要用户决定。

**Gate A3**
- pytest 全部通过；上面 3 项真实验证全部符合预期。
- 用 A1 的回放脚本再跑一次完整回放，Top5 与 `A1_exact_baseline.json` 逐题一致。

**提交：** `stage-a(A3): runtime follows lifecycle active pointer; guard active ingest`

---

## A4 清理

**目的：** 删除已经确认无用的文件和数据，降低以后的阅读和测试成本。**所有删除都要可恢复：** git 中的文件可以从 tag `stage-a-start` 恢复；不在 git 中的文件先移到仓库外备份。

### A4.1 备份不在 git 中的内容
```powershell
$bk = "D:\Agent_learn\evidenceFlow_backup\stage_a_$(Get-Date -Format yyyyMMdd)"
New-Item -ItemType Directory -Force $bk | Out-Null
Copy-Item -Recurse data\chroma "$bk\chroma"
Move-Item tmp "$bk\tmp"
Move-Item 1.26, src\evidenceflow.zip $bk
```
- 删除 `src/evidenceflow.egg-info`，然后执行 `pip install -e .`，再确认 `python -c "import evidenceflow"` 正常。
- 删除 `tmp/` 之前，先确认 `evals/evaluation_v3/phase12_rag_quality_v1/` 中有 `claim_annotations.json`、`citation_annotations.json`、`abstention_annotations.json`、`runtime_outputs.json`。审查时这几个文件都在。缺任何一个就 **STOP**。

### A4.2 删除孤儿 Chroma collection（只能用 API）
- 新增 `scripts/prune_chroma_collections.py`：
  - 先导出全部 collection 的 `name / id / count / metadata`，写入 `artifacts/stage_a/A4_chroma_before.json`；
  - 用一个**写死的允许删除列表**调用 `client.delete_collection`；
  - 支持 `--dry-run`，默认就是 dry-run，必须加 `--execute` 才真正删除。
- 允许删除的列表：
  - `ef_research_bge_m3_hybrid_c512_v2_staging`
  - `ef_research_bge_m3_c512_o64_refmeta_v1`
  - `ef_research_structure_v1`
  - `ef_research_structure_v1_elemaware`
- **禁止删除**（脚本里要硬编码断言）：
  - `ef_research_bge_m3_hybrid_c512_v2`
  - `ef_research_bge_m3_hybrid_c512_v1`
  - `ef_research_bge_m3_c512_o64_v1`
- 删除前再 grep 一次，确认被删的名字只出现在下面 A4.3 要删的实验脚本，以及 `scripts/repair_active_index_metadata.py` 里。如果在其他地方出现，就把它从删除列表中去掉并报告。
- `scripts/repair_active_index_metadata.py` 是一次性修复脚本，它引用的 staging collection 删掉之后就不能再运行了。在文件头加注释，标注它为历史脚本；或者经用户同意后删除。

### A4.3 删除已结束阶段的实验及其测试（文件都在 git 中）
删除清单：
- `experiments/phase13c/`（包括 `runners/`）
- `experiments/phase14a/`
- `experiments/run_phase13_kb_chunk_audit.py`
- `experiments/run_phase13b_controlled_rebuild.py`
- `experiments/run_phase13c_cleaning_benchmark.py`
- `experiments/run_phase13c_parser_benchmark.py`
- `experiments/build_research_reference_metadata_collection.py`
- `experiments/run_research_references_aware_experiment.py`
- `experiments/archive/`
- `tests/test_phase13_kb_chunk_audit.py`
- `tests/test_phase13b_controlled_rebuild.py`
- `tests/test_phase13c_cleaning_benchmark.py`
- `tests/test_phase13c_parser_benchmark.py`

删除前用 AST 或 grep 再确认一次：剩下的 `src`、`tests`、`scripts`、`experiments` 中，没有任何文件 import 上面这些模块。审查时的 import 依赖图显示没有，但执行时要复核。

**保留：**
- `experiments/run_phase15b1_dense_candidate_depth.py`、`phase15b_active_relevance.py`、`run_phase15c*.py`、`run_production_reranked_baseline_replay.py`、`stabilize_gold_v3.py`、`audit_gold_v2.py`、`prepare_research_corpus.py`、`debug/`
- `evals/` 下的全部冻结产物，一个都不动。

### A4.4 eval 中的 legacy 代码
- 用 grep 确认 `evidenceflow.eval.runner` 的使用者。如果只有 `eval/__init__.py` 和 `tests/test_eval_runner.py`，就删除 `eval/runner.py` 和 `tests/test_eval_runner.py`，并更新 `eval/__init__.py`。
- `eval/metrics.py`、`eval/dataset.py`、`domain/evaluation.py` 仍然被 `eval/v2.py` 和一些实验使用，**本阶段不删**。只在模块 docstring 里加一句“COMPATIBILITY_ONLY：legacy HitRate 语义，新代码用 eval/ir_metrics.py”。

### A4.5 清理过期注释和字段（不改变行为）
- `storage/base.py:47` 附近：删掉 `iter_all_chunks` docstring 中关于 BM25Index 和 §17.3.1 的说法，改成它现在的用途：生命周期校验、Exact 索引构建。
- `retrieval/reranker.py:323`：删除日志 extra 中的 `"fallback_branch_used": False`。同步修改 SPEC §30 中的 logging 清单；如果有测试断言了这个字段，一并更新。
- `retrieval/reranker.py` 模块 docstring：“实验可用的…”改为正式用途的描述。
- `domain/chunk.py:76,80`：去掉 description 中关于 RRF 的说法。
- `.workbuddy/memory/MEMORY.md`：在末尾追加一条 Stage A 的状态说明，修正“项目无 .git”和“C512/O64”这两处过期信息。这个文件不在 git 中，只改文本，不删。

**Gate A4**
- `python -m pytest -q` 全部通过，测试总数等于 N0 减去被删测试文件中的用例数，再加上 A1–A3 新增的用例数。在报告中列出这个算式。
- A1 的回放脚本跑一次完整回放，Top5 与 `A1_exact_baseline.json` 逐题一致。
- `scripts/verify_active_index.py` 检查 v2 仍然通过。
- `python -m evidenceflow ask "..."` 正常。

**提交：** `stage-a(A4): remove finished-phase experiments, orphan collections and stale notes`

---

## A5 精简 PROJECT_SPEC

**目的：** 让 SPEC 回到“当前合同”的定位，历史内容另外存放。

**步骤**
1. 把当前文件原样复制到 `docs/history/PROJECT_SPEC_v0.26_full.md`，在文件头加一行说明：这是历史版本，不再是当前规范。
2. 重写 `PROJECT_SPEC.md` 为 **v0.27**，标题改为 `EvidenceFlow — Current Specification`，目标 300–450 行，包含以下章节：
   1. 当前架构（ingest / ask 两条链路的流程，包括 Exact 和 v3）
   2. 开发环境（`agent-gpu`）与依赖
   3. Domain Models 摘要（Document、DocumentChunk、RetrievedChunk、RerankedRetrievedChunk、BuiltContext、GeneratedAnswer、RAGAnswer 及其 status）
   4. ID 与 hash 规则（原 §6、§10 的要点）
   5. Ingestion 合同（loader、cleaner、hybrid 结构解析与切分、collection 身份隔离）
   6. 索引生命周期（原 §39 + A3 的 active pointer 规则、ACTIVE 保护、staging 构建脚本）
   7. Retrieval 合同（Exact Dense20 → Reranker Top5，确定性排序规则，设备规则即原 §43）
   8. Context Builder 合同
   9. Generation v3 合同（A2）
   10. Citation 合同（原 §25–27）
   11. Settings 一览表（字段、默认值、含义）
   12. CLI 与错误处理（退出码、错误前缀）
   13. Evaluation：canonical 实现（`eval/ir_metrics.py`、`quality.py`、`comparison.py`、`stable_mapping.py`）、标准指标、Gate 规则（原 §20.8.5）、冻结基线（A1 的 B0'，含 sha256）
   14. 明确不在当前范围内的事项（链接到 `docs/agentic_rag/01_ARCHITECTURE_REVIEW.md` §8）
   15. 变更流程（与 AGENTS.md 一致）
   16. 历史索引：旧章节号 → `docs/history/` 中的位置
3. 把 §S 的全部内容合并进相应章节，然后删除 §S。
4. 更新 `README.md`：架构描述中的 “Chroma ANN Top20” 改为 Exact；新增结构化生成与 status；指标表换成 A1 的 B0'；链接到 `docs/agentic_rag/`。
5. `AGENTS.md` 只做最小改动：在“文档优先级”中加入 `docs/agentic_rag/02_STAGE_A_EXECUTION_PLAN.md`，标明它是阶段性计划。

**Gate A5**
- 新的 SPEC 中，凡是与代码直接对应的内容（Settings 字段及默认值、CLI 退出码、Generator 输出 schema、RAGAnswer 字段），都要逐项与代码核对一致，核对表写在 `A5_report.md` 里。
- 全文 grep 不到已退休概念被描述为“当前”的说法：Translation、MultilingualRetriever、BM25 Production、RRF、HNSW 作为线上检索、`STATUS:` 控制头。
- pytest 全部通过。

**提交：** `stage-a(A5): consolidate PROJECT_SPEC v0.27; archive v0.26` 并执行 `git tag stage-a-done`

---

## 6. 阶段 A 完成后的交付物

`artifacts/stage_a/` 下应有：
- `A0_report.md` … `A5_report.md`，每份都按 AGENTS.md §7 的格式写
- `A1_exact_baseline.json`（新的正式基线）及其 sha256
- `A1_determinism.json`、`A1_exact_vs_hnsw.json`
- `A2_probe.json`、`A2_generation_smoke.json`
- `A4_chroma_before.json`
- `STAGE_A_SUMMARY.md`：一页总结，包括：
  - 每个 Gate 的结果
  - 新基线的指标
  - A2 中的修复次数和 token 用量
  - 遗留问题
  - 给阶段 B 的输入：新基线路径、生成冒烟的发现

**完成后停止，不要开始阶段 B。**

## 7. STOP 条件汇总

出现以下任何一种情况，立即停止并向用户报告：
- 任何一步 pytest 失败，并且无法在本步的范围内修复；
- A1：多进程结果不一致，或 R@5 / nDCG@5 的下降超过 0.01；
- A2：三种结构化输出模式都不可用；冒烟中 `GenerationContractError` 大于 0；网络错误重跑后仍然超过 2 题；
- A3：staging 重建的 chunk ID 与 active 不一致；
- A4：备份校验失败；被删对象仍然被其他地方引用；Phase 12 的标注产物不全；
- 需要新增依赖，或需要修改 Gold、qrels、冻结的 `evals/` 产物；
- 本计划与代码的实际情况不符（例如函数名或文件结构不一样），而且无法判断该怎么处理。

## 8. 给执行者的提示

- 冻结的 hash 与路径常量集中在 `experiments/run_phase15b1_dense_candidate_depth.py` 和 `run_production_reranked_baseline_replay.py` 里。复用它们的常量，不要自己重新写一遍 hash。
- CPU 上 reranker 跑一次 40 题完整回放大约需要 5 分钟。A1 的 10 次确定性检查只用 `--candidates-only`，完整回放只跑两次。
- Windows 下写 JSON 时统一使用 `encoding="utf-8"`、`ensure_ascii=False`、`sort_keys=True`，与项目里现有的 `write_json` 保持一致。
- 每一步的汇报要写清楚：改了哪些文件、为什么这样改、跑了哪些测试、结果是什么、还有哪些限制。
