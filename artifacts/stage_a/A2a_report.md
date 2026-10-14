# A2a 报告：结构化生成 v3 与 `RAGAnswer.status`（离线部分，模式暂定）

## 结论：Gate A2a 通过

- `python -m pytest -q`：**591 passed**（A1 为 538，新增 53；日志 `A2a_pytest.log`）。
- 本步**没有发起任何真实 LLM 请求**；所有测试使用 Fake client，不联网。
- **`STRUCTURED_OUTPUT_MODE = "prompt_only"` 是暂定值**，A2b 探测后才能确定（见第 3 节）。
- A2 的 Gate（冒烟中 GenerationContractError 为 0 等）**尚未评估**，要等 A2b。

## 1. 修改或创建的文件

### 源码
| 文件 | 改动与原因 |
|---|---|
| `src/evidenceflow/domain/generation.py`、`domain/__init__.py` | 新增 `GeneratorOutput`（`extra="forbid"`）与 `generator_output_json_schema()`。Schema 手工内联（不用 `model_json_schema()`，避免 `$defs` 引用在 strict 模式下不被支持），枚举值取自 `AnswerStatus`，避免两处漂移 |
| `src/evidenceflow/domain/answer.py` | `RAGAnswer.status` 必填 |
| `src/evidenceflow/llm/base.py` | Protocol 增加 `generate_json` 与 `last_usage` |
| `src/evidenceflow/llm/openai_compatible.py` | 模块常量 `STRUCTURED_OUTPUT_MODE`；纯函数 `build_structured_request_params`（三种模式）；`generate_json`；`last_usage`（只表示最近一次调用，调用开始时重置）；`last_response_info`（是否有 `reasoning_content`、finish_reason，供在线脚本记录）；构造参数 `structured_output_mode`（供探测脚本逐模式创建客户端，**不是**运行时降级） |
| `src/evidenceflow/generation/generator.py` | 改为 `rag-generator-v3`；**删除** v2 控制头解析，不保留双路径；`extract_json_text`、`parse_generator_output`、`GenerationContractError(ValueError)`；一次修复；`last_diagnostics` |
| `src/evidenceflow/generation/__init__.py`、`llm/__init__.py`、`citation/__init__.py` | 导出新增符号 |
| `src/evidenceflow/citation/parser.py` | 拆出 `classify_citation_labels`（只依赖 allowlist）与 `check_citation_rules`，供离线重解析复用同一套规则；`assemble_rag_answer` 透传 `status` |
| `src/evidenceflow/__main__.py` | 输出 `Status: …` 行；`GenerationContractError` → `Generation format error: …` |
| `src/evidenceflow/runtime.py` | `build_rag_service` 增加可选注入 `retriever`、`generator`（仅评测脚本使用；不传时行为不变）。这样冒烟脚本能读取 `Generator.last_diagnostics`，不必访问私有属性 |

### 解析与修复的实现细节（与你的要求逐条对应）
- 解析顺序：去首尾空白 → 去**开头**已闭合的 `<think>…</think>` → 去外层 ```json 代码块 → `json.loads` → `GeneratorOutput`。**未闭合的 `<think>` 按格式失败处理**，进入修复。`<think>` 不在开头时不当作前缀。
- 修复至多一次：消息 = 原消息 + assistant（原始回复）+ user（修复指令，中英各一句，含 “JSON”）。仍失败抛 `GenerationContractError`（其 `__cause__` 为第一次失败）。修复调用期间的 `LLMError` 原样传播，不会被当成格式错误。
- `last_diagnostics`：`llm_called`、`repair_attempted`、`repair_succeeded`、`usages`（每次调用一项，由 Generator 在**每次调用后立即**读取 `client.last_usage` 汇总，可为 None）、`raw_response_lengths`。每次 `generate()` 开始时重置；返回副本；空 Context 时 `llm_called=False` 且列表为空。不进入 `GeneratedAnswer`/`RAGAnswer`。
- 空 Context：不调用 LLM，行为与之前一致。

### 在线脚本（只写好，**未运行**）
| 文件 | 作用 |
|---|---|
| `experiments/stage_a/online_common.py` | `RequestPolicy`（最小间隔、429 等待重试 30/60/120 秒最多 3 次并逐次记录、403/配置类错误抛 `FatalServiceError` 立即停止）；`PolicyLLMClient`（套用策略，并留存每次调用的消息、原始回复、usage、`reasoning_content` 是否存在/非空、finish_reason、耗时）；JSONL 追加/读取/续跑；冒烟统计 `summarize_smoke`（限流耗尽与 `LLMRequestError` **分开**统计） |
| `experiments/stage_a/probe_structured_output.py` | 重写：使用线上 Generator 的真实 Prompt 和 `OpenAICompatibleLLMClient`；`--min-interval-seconds`（默认 6）；每项即时追加 `A2_probe.jsonl`；限流耗尽的项不标记完成，重跑会重试；选模式规则 json_schema > json_object > prompt_only |
| `experiments/stage_a/run_generation_smoke.py` | 走线上链路（Exact → Reranker → Context → Generator v3 → Citation），25 题，用数据集的 `query` 字段；每题记录 status、修复情况（来自 `last_diagnostics`）、异常类型与分类、usage、耗时、sources 标签、**Top5 chunk_id**、实际 device、每次调用原始回复；需要 `--allow-gpu` 才能使用非 CPU；`--retry-llm-errors` 只重跑服务错误/限流耗尽的题；不与 B0' 比较；不覆盖 Phase 12 任何文件 |
| `experiments/stage_a/reparse_smoke_offline.py` | 读取冒烟 JSONL，用保存的回复驱动真实 `Generator` 重跑解析/修复，并用保存的标签白名单重跑 Citation 校验；输出与冒烟 `summary` 同格式，并列出与原结果不一致的题；需要比已保存回复更多的调用时记为 `ReplayExhaustedError` |

另外，把上次失败的探测结果重命名为 `A2_probe_attempt1_failed.json`（保留 403/429 证据），`A2_probe.json` 留给 A2b 的真实结果。

### 测试
- 更新：`test_generator.py`（重写为 v3）、`test_rag_answer_assembly.py`、`test_domain_models.py`、`test_ask_cli.py`、`test_phase9_runtime_promotion.py`、`test_rag_service.py`、`test_runtime.py`。`test_llm_client.py` 无需改动即通过（`generate()` 的请求参数不变）。
- 新增 `tests/test_structured_output.py`（三种模式的参数、`generate_json`、`last_usage`、reasoning 诊断）与 `tests/test_stage_a_online_scripts.py`（限流等待、重试耗尽、403 停止、断点续跑、服务错误重跑、原始回复留存、离线重解析、探测选模式）。
- 计划 A2.2 的 8 类用例全部覆盖：合法 JSON、`<think>` 前缀、```json 代码块、非法 JSON 后修复成功、修复后仍失败抛 `GenerationContractError`、未知 status 与多余字段被拒、空 Context 不调用 LLM、`last_usage` 为 None 时正常。

## 2. v2 的其他使用方（grep 结果，均**未修改**）
- `experiments/run_phase12_rag_quality.py`：第 151 行字符串 `"rag-generator-v2 control header (STATUS:/ANSWER:)"` 过期；其余逻辑通过 `generator.generate(...)` 与 `assemble_rag_answer(...)` 取状态，仍可运行，但它读取 `service._retriever/_context_builder/_generator` 私有属性，且输出会按 v3 行为产生。
- `experiments/archive/phase12_fallback_run.py`：第 7、174 行明确描述 v2 控制头（0.5B 回退模型不解析控制头）；`experiments/archive/phase12_build_worksheet.py` 第 48 行只是标签文字。
- `experiments/run_phase12_aux_generate_artifacts.py` 第 80 行的提示文字提到“控制头校验”。
- `experiments/debug/gen_eval.py`、`experiments/archive/phase12_regression.py`、`run_phase11b_cleanup.py`、`run_phase11c_technical_debt_closure.py` 调用 `build_rag_service(Settings())`，不依赖控制头，也不直接构造 `RAGAnswer`。
- 全仓库 `src/`、`tests/` 中不再有 v2 控制头解析，没有双路径。`experiments/` 中**没有**脚本直接构造 `RAGAnswer(`。

## 3. A2b 的执行计划（等你说“API 可用了”）
前置环境变量同 §0.4（离线 HF、本地 reranker 路径）。

1. 探测（约 9 次请求；限流时每项最多 4 次尝试）：
   ```powershell
   python experiments/stage_a/probe_structured_output.py --min-interval-seconds 6
   ```
   - 如果选出的模式不是 `prompt_only`：修改 `openai_compatible.py` 的 `STRUCTURED_OUTPUT_MODE`，更新 §S.2，重跑 `python -m pytest -q`。
   - 如果返回 403：脚本立即停止（退出码 3），已完成项保存在 `A2_probe.jsonl`。
2. 冒烟（25 题，约 25 次请求，加每次修复 1 次；按 Phase 12 的 3/25 格式失败率预计 3 次左右）：
   ```powershell
   $env:MODEL_DEVICE='cuda'
   python experiments/stage_a/run_generation_smoke.py --allow-gpu --min-interval-seconds 6
   ```
   服务错误多于 2 题时：`python experiments/stage_a/run_generation_smoke.py --allow-gpu --retry-llm-errors`。
3. 修改解析器后无需再调 API：
   ```powershell
   python experiments/stage_a/reparse_smoke_offline.py
   ```
4. 估算：**探测 9 + 冒烟约 25 + 修复约 3 ≈ 37 次请求**；全部触发 429 的最坏情况是每个逻辑请求 4 次尝试，上限约 148 次。仅请求间隔就约 4 分钟，再加模型响应时间和 429 退避（30+60+120 秒/次耗尽）。
5. 按计划 A2 节判断 Gate：pytest 通过；25 题 `GenerationContractError` 为 0；修复次数如实记录（超过 3 题标注风险）；`LLMRequestError` 超过 2 题重跑一次，仍超过则 STOP；`CitationValidationError` 列出题号。报告中给出 Phase 12 v2 的格式错误数（3/25）与 v3 的对比。

## 4. 限制与注意
1. `prompt_only` 暂定：在这个模式下，服务端不会强制 JSON，格式正确性完全靠 Prompt 与解析器/修复。探测可能选出更强的模式。
2. 在线脚本的限流/403/续跑逻辑已用 Fake client 测试，但**真实服务的 429/403 响应形态**只在 A2b 才能验证（分类依赖 SDK 异常上的 `status_code`，与 A2 第一次探测中看到的 403/429 一致）。
3. 注释里的“A2_probe.json”现在指向 A2b 才会生成的文件；A2a 阶段它不存在，失败证据在 `A2_probe_attempt1_failed.json`。
4. 你消息里在“5. A2.3 冒烟”下面的 `pasted_content` 我按你的编号当作第 5 条要求处理，内容均已落实（允许 GPU 且需 `--allow-gpu`、记录 device 与 Top5、使用 `query` 字段、不与 B0' 比较）。
5. 你消息末尾写了“api可以正常使用”，但前文明确要求 A2a 完成后停下并等你发出“API 可用了”，所以我**没有**进入 A2b。
