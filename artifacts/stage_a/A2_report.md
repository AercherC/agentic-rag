# A2 报告：结构化生成 v3 与 `RAGAnswer.status`（含 A2a、A2a.1、A2b）

> 本文件取代 c8b73c0 提交里的 A2.0 STOP 报告（旧内容保留在 git 历史；旧 403/429 证据在 `A2_probe_attempt1_failed.json`）。

## 结论：Gate A2 通过

| Gate | 结果 |
|---|---|
| `python -m pytest -q` 全部通过 | **609 passed**（`A2_pytest.log`；A2a 为 591，A2a.1 与 A2b 新增 18） |
| 25 题 `GenerationContractError` 为 0 | **0** |
| 修复次数 | **0 题**（未超过 3 题的风险线） |
| `LLMRequestError` / 限流耗尽（分开统计） | **0 / 0**，限流事件 0，未触发 `--retry-llm-errors` |
| `CitationValidationError` | **0**（无题号可列） |
| `thinking_ignored_calls`（非 Gate） | **0**（探测与冒烟均为 0） |

没有触发计划 §7 的任何 STOP 条件。未修改 `evals/`。

## 1. 环境变化与 A2a.1（提交 2e5f5cb，无 API 调用）
- LLM 由 ModelScope Qwen/Qwen3.5-35B-A3B 换为 `api.deepseek.com` 的 `deepseek-flash`；`.env` 中 `LLM_THINKING=disabled`。合同已先写入 §S.2。
- 新增：`Settings.llm_thinking`（auto/enabled/disabled，默认 auto；非 auto 时经 `extra_body` 发送 DeepSeek 格式）；每条 JSONL 记录写入 `structured_output_mode`、`prompt_version`、`model`、`llm_thinking`，续跑时不一致（含旧记录缺字段）以退出码 4 终止；`PolicyLLMClient` 构造时要求 `inner.max_attempts == 1`（`OpenAICompatibleLLMClient` 新增只读属性 `max_attempts`）；真实 `openai.PermissionDeniedError` / `RateLimitError`（httpx 构造，不联网）经 `OpenAICompatibleLLMClient(max_attempts=1)` + `PolicyLLMClient` 测试：403 只发 1 次 HTTP 请求并抛 `FatalServiceError`；429 共 4 次 HTTP 请求（等待 30/60/120 秒），归为 `rate_limit_exhausted`；统计新增 `thinking_ignored_calls`。
- A2a.1 的 pytest 为 609 passed。

## 2. A2.0 探测（`A2_probe.json` / `A2_probe.jsonl`，9 次请求，无限流、无 403）
host `api.deepseek.com`，model `deepseek-flash`，`llm_thinking=disabled`。

| 模式 | 结果 |
|---|---|
| `json_schema`（strict） | **3/3 失败**：HTTP 400 “This response_format type is unavailable now”——服务端不支持，属能力探测结果 |
| `json_object` | 3/3 成功，均为干净 JSON（无 `<think>`、无代码块），解析校验通过 |
| `prompt_only` | 3/3 成功，同上 |

- 选定模式：**`json_object`**（json_schema > json_object > prompt_only 优先级下，json_schema 不可用，json_object 可用）。已写入 `openai_compatible.py` 的 `STRUCTURED_OUTPUT_MODE`（注释引用 A2_probe.json）和 §S.2（去掉“暂定”）；测试中断言暂定值的两处已同步，随后 pytest 重跑 609 passed。
- 两种可用模式的回复都没有 reasoning_content（thinking 开关生效），输出 40–60 tokens。
- 注意：探测只有 3 个请求，样本很小；`json_object` 与 `prompt_only` 在 deepseek-flash 上表现相同，选择依据是优先级而非性能差异。

## 3. A2.3 冒烟（25 题，`A2_generation_smoke.json` / `.jsonl`）
- 链路：Exact Dense20 → Reranker → Top5 → Context → Generator v3（json_object）→ Citation；device `cuda:0`（`--allow-gpu`，已记录）；使用数据集 `query` 字段；每题 Top5 chunk_id 已记录（25 题均为 5 个）；不与 B0' 比较。
- status 分布：**answered 19，insufficient_context 6**。对照数据集标注：ANSWERABLE 19 题中 18 题 answered、1 题（`ARR-040`，SVDFormer 与 Hyper-PCN 的结构比较）为 insufficient_context；INSUFFICIENT_CONTEXT 5 题全部 insufficient_context。该题回答质量不在本步评估范围，留给阶段 B。
- 错误：GenerationContractError 0、修复 0（attempted 0 / succeeded 0）、CitationValidationError 0、LLMRequestError 0、限流耗尽 0、其他 0。
- usage 合计：prompt 47 376 tokens，completion 2 708 tokens，共 25 次调用（无修复调用），25 次均有 usage。
- 耗时：总 150.5 秒（含 6 秒最小间隔下的每题处理）；单题中位数 6.09 秒，最大 14.33 秒。
- `reparse_smoke_offline.py`：`changed_vs_original` 为空，汇总与在线结果一致（`A2_generation_smoke_reparsed.json`）。

## 4. 与 Phase 12 v2（3/25 格式错误）的对比
- Phase 12 v2 在 Qwen3.5-35B-A3B 上有 3/25 格式错误；本次 v3 在 deepseek-flash 上为 0/25（并且 0 次修复）。
- **两者不是同一个模型，只能作参考，不能据此说 v3 比 v2 更可靠。** 同模型下 v2 与 v3 的对比留到阶段 B。

## 5. 已知遗留问题
1. `eval/quality.py` 的 `abstention_metrics` 只接受大写状态，与 `AnswerStatus` 的小写不一致，留给阶段 B。
2. `experiments/` 中依赖 v2 的历史脚本（未修改）：`run_phase12_rag_quality.py`（第 151 行过期字符串，且用私有属性）、`archive/phase12_fallback_run.py`、`archive/phase12_build_worksheet.py`、`run_phase12_aux_generate_artifacts.py`（提示文字）。
3. `<think>` 与 ```json 代码块的识别区分大小写（`<THINK>`、```JSON 不会被识别，会进入修复）。
4. `max_attempts=1` 时，超时与 5xx 不会自动重试，只能靠 `--retry-llm-errors` 补跑；本次未出现。
5. `json_object` 模式只保证“是 JSON”，不保证符合 schema，字段与取值仍由 `GeneratorOutput` 校验（`extra="forbid"`）。
6. 换模型或服务后必须重新跑探测；`STRUCTURED_OUTPUT_MODE` 是固定常量，没有运行时降级。
7. 沿用 A2a 报告：`src/evidenceflow.zip` 在 A2a 提交后从磁盘消失，原因未知，待你决定（A4 备份计划相关）。

## 6. 产物与提交
- `A2_probe.json`、`A2_probe.jsonl`、`A2_generation_smoke.json`、`A2_generation_smoke.jsonl`、`A2_generation_smoke_reparsed.json`、`A2_smoke_stdout.log`、`A2_pytest.log`（均不含 API Key）。
- 提交：A2a.1 为 2e5f5cb；A2b 提交信息 `stage-a(A2): structured generator v3 and RAGAnswer.status`。
