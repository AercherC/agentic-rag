# A2 报告：STOP —— LLM 服务不可用，A2.0 探测无法完成

## 结论：触发计划 §7 的 STOP 条件（A2：三种结构化输出模式都不可用）

**这不是模式能力问题，而是服务端拒绝了所有请求，包括最普通的 prompt_only。** 因此无法选出结构化输出模式，A2.1 之后的实现没有开始。

## 1. 探测结果（`A2_probe.json`，只记录 host 与 model，不含 key）
- host：`api-inference.modelscope.cn`；model：`Qwen/Qwen3.5-35B-A3B`；3 种模式 × 3 个请求 = 9 次调用，**9 次全部 HTTP 错误**，无一次拿到可解析的回复。

| 错误 | 次数 | 内容 |
|---|---|---|
| 403 PermissionDeniedError | 6 | “For account security, please complete your profile via email verification. ModelScope Magicubes is only available to verified accounts.” |
| 429 RateLimitError | 3 | “We have to rate limit you for model Qwen/Qwen3.5-35B-A3B.” |

- 403 与 429 在三种模式中交错出现（含 prompt_only），与是否传 `response_format` 无关。因此没有证据表明任何一种模式不受支持，也没有证据表明某种模式可用。
- 所以 `reasoning_content`、`<think>` 闭合、usage、耗时等记录项在本次探测中**没有可用数据**（耗时仅为错误返回耗时，0.1–2.8 s）。

## 2. 我没有做的事
- 没有重试、换模型、换 endpoint 或修改 `.env`；403 是账号认证要求，429 是服务限流，都不应由我绕过。
- 没有继续实现 A2.1 之后的代码：模式常量依赖探测结果，计划要求 STOP 后立即停止。

## 3. 需要你处理（任选其一）
1. 在 ModelScope 账号设置里完成邮箱验证（403 提示的地址：`modelscope.cn/my/settings/account`），并等限流缓解后告诉我，我重跑探测。
2. 提供另一个可用的 OpenAI-compatible endpoint 和 model（由你自己改 `.env`，我不会读取或打印 key）。注意 §S.2 与基线对比都以 Qwen/Qwen3.5-35B-A3B 为前提，换模型会影响阶段 B 的可比性，需要你确认。

## 4. 本步已提交与未提交的内容
已提交（只含文档与探测）：
- `PROJECT_SPEC.md`：§S.1 补充“15C.1 脚本已被取代、不可重跑、A4.3 删除”和“离线实验必须逐条 `embed_query`”；新增 §S.2（rag-generator-v3 合同、解析顺序、修复、`last_diagnostics`；**选定模式一项留空，待探测完成后填写**）。
- `docs/agentic_rag/02_STAGE_A_EXECUTION_PLAN.md`：A4.3 删除清单加入 `run_phase15c1_exact_dense_architecture.py`，保留清单中的 `run_phase15c*.py` 收窄为 `run_phase15c_dense30_promotion.py`。
- `experiments/stage_a/probe_structured_output.py`、`artifacts/stage_a/A2_probe.json`、本报告。

**未提交**（工作区保留）：`src/evidenceflow/domain/generation.py` 与 `domain/__init__.py` 中新增的 `GeneratorOutput` 和 `generator_output_json_schema()`。这是探测脚本需要的 schema 来源，已用 python 确认能生成预期 schema（`additionalProperties: false`，status/answer 均 required），但还没有对应测试，也没有运行 pytest；等 A2 继续时一并完成。

## 5. 与 Phase 12 v2 的格式错误对比
无法给出 v3 数据（没有成功的生成）。v2 基线仍是 Phase 12 的 3/25 格式失败。

## 6. 测试
本步未运行 pytest（没有改动已提交的生产代码）。
