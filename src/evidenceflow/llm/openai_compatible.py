"""基于官方 OpenAI SDK 的通用 Chat Completions Adapter。"""

from __future__ import annotations

from collections.abc import Sequence
from math import isfinite
from typing import Any

import httpx
from openai import OpenAI

from evidenceflow.domain.generation import LLMMessage
from evidenceflow.llm.base import LLMConfigurationError, LLMRequestError
from evidenceflow.llm.diagnostics import classify_llm_failure

# 结构化输出模式：json_schema（strict）> json_object > prompt_only。
# 暂定值（Stage A §S.2）：prompt_only 被所有 OpenAI-compatible 服务支持，且解析器已能处理
# <think> 与 ```json 代码块；待 A2b 真实探测（artifacts/stage_a/A2_probe.json）后确定。
# 这里只是固定常量，不做运行时模式降级或自动探测。
STRUCTURED_OUTPUT_MODE = "prompt_only"
STRUCTURED_OUTPUT_MODES = ("json_schema", "json_object", "prompt_only")

# thinking 开关（DeepSeek 等）：auto = 不发送，沿用服务端默认；enabled/disabled 显式开或关。
THINKING_MODES = ("auto", "enabled", "disabled")


def build_structured_request_params(
    mode: str,
    *,
    schema_name: str,
    json_schema: dict,
    messages: Sequence[LLMMessage],
) -> dict[str, Any]:
    """按模式生成附加到 Chat Completions 请求的参数（纯函数，无网络）。

    - json_schema：``response_format`` 携带 strict JSON Schema；
    - json_object：仅声明 JSON 对象；OpenAI 协议要求 Prompt 中出现 "JSON" 一词，
      否则服务端会拒绝，所以这里提前校验以获得明确错误；
    - prompt_only：不传 ``response_format``，完全靠 Prompt 约束。
    """

    if mode not in STRUCTURED_OUTPUT_MODES:
        raise LLMConfigurationError(f"unknown structured output mode: {mode}")
    if mode == "json_schema":
        return {
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": schema_name, "strict": True, "schema": json_schema},
            }
        }
    if mode == "json_object":
        if not any("json" in message.content.lower() for message in messages):
            raise LLMConfigurationError('json_object mode requires the word "JSON" in the prompt')
        return {"response_format": {"type": "json_object"}}
    return {}


class OpenAICompatibleLLMClient:
    """将内部消息转换为 OpenAI-compatible Chat Completions 请求。"""

    def __init__(
        self,
        *,
        api_key: str | None,
        base_url: str,
        model: str,
        temperature: float = 0.0,
        timeout_seconds: float = 60.0,
        max_attempts: int = 2,
        structured_output_mode: str = STRUCTURED_OUTPUT_MODE,
        thinking: str = "auto",
        _client: Any | None = None,
    ) -> None:
        """验证配置并建立无自动重试、带显式超时的同步 SDK Client。"""

        if api_key is None or not api_key.strip():
            raise LLMConfigurationError("LLM API key is required")
        if not base_url.strip():
            raise LLMConfigurationError("LLM base URL must not be blank")
        if not model.strip():
            raise LLMConfigurationError("LLM model must not be blank")
        if (
            isinstance(temperature, bool)
            or not isinstance(temperature, (int, float))
            or not isfinite(temperature)
            or not 0 <= temperature <= 2
        ):
            raise LLMConfigurationError("LLM temperature must be between 0 and 2")
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not isfinite(timeout_seconds)
            or timeout_seconds <= 0
        ):
            raise LLMConfigurationError("LLM timeout must be positive")
        if isinstance(max_attempts, bool) or max_attempts not in {1, 2}:
            raise LLMConfigurationError("LLM max attempts must be 1 or 2")

        if structured_output_mode not in STRUCTURED_OUTPUT_MODES:
            raise LLMConfigurationError("unknown structured output mode")
        if thinking not in THINKING_MODES:
            raise LLMConfigurationError("unknown thinking mode")
        self._structured_output_mode = structured_output_mode
        self._thinking = thinking
        # 只表示最近一次调用；调用开始时重置，由上层（Generator）负责汇总多次调用。
        self._last_usage: dict | None = None
        # 诊断用：最近一次响应是否带 reasoning_content、结束原因；不进入业务合同。
        self.last_response_info: dict[str, Any] = {}
        self._model = model
        self._temperature = float(temperature)
        self._max_attempts = max_attempts
        self._client = _client or OpenAI(
            api_key=api_key,
            base_url=base_url,
            # 显式禁用系统代理：Clash 等本地代理未运行时不再截断 TLS 握手。
            http_client=httpx.Client(trust_env=False, timeout=float(timeout_seconds)),
            # 本阶段错误直接向上报告，不引入隐藏的 SDK 自动重试与额外等待。
            max_retries=0,
        )

    @property
    def max_attempts(self) -> int:
        """单次调用的最大尝试次数（在线脚本要求为 1，由脚本策略负责重试）。"""

        return self._max_attempts

    @property
    def last_usage(self) -> dict | None:
        """最近一次调用的 token 用量；服务未返回 usage 时为 None。"""

        return self._last_usage

    def generate(self, messages: Sequence[LLMMessage]) -> str:
        """调用 Chat Completions，并把所有供应商/传输失败统一为 LLMRequestError。"""

        return self._complete(messages, {})

    def generate_json(
        self,
        messages: Sequence[LLMMessage],
        *,
        schema_name: str,
        json_schema: dict,
    ) -> str:
        """按构造时固定的结构化输出模式请求 JSON，原样返回模型文本（解析由调用方负责）。"""

        extra = build_structured_request_params(
            self._structured_output_mode,
            schema_name=schema_name,
            json_schema=json_schema,
            messages=messages,
        )
        return self._complete(messages, extra)

    def _complete(self, messages: Sequence[LLMMessage], extra: dict[str, Any]) -> str:
        """共用的请求与瞬态重试逻辑；记录最近一次调用的 usage 与响应诊断。"""

        if not messages:
            raise LLMRequestError("LLM messages must not be empty")
        request_messages = [message.model_dump() for message in messages]
        request_extra = dict(extra)
        if self._thinking != "auto":
            # DeepSeek 的 thinking 开关（OpenAI 格式）必须走 extra_body 透传。
            request_extra["extra_body"] = {"thinking": {"type": self._thinking}}
        self._last_usage = None
        self.last_response_info = {}
        content: Any = None
        for attempt in range(1, self._max_attempts + 1):
            try:
                response = self._client.chat.completions.create(
                    model=self._model,
                    messages=request_messages,
                    temperature=self._temperature,
                    **request_extra,
                )
                # Chat Completion 正常路径要求至少一个 Choice 且 content 是非空纯文本。
                choice = response.choices[0]
                content = choice.message.content
                self._record_response(response, choice)
                break
            except Exception as exc:
                classification = classify_llm_failure(exc)
                retryable = classification in {
                    "timeout",
                    "network_connection",
                    "rate_limit",
                    "provider_5xx",
                }
                # 只对已证实可能瞬态的类别补一次尝试；配置及普通 4xx 立即失败。
                if attempt >= self._max_attempts or not retryable:
                    raise LLMRequestError("OpenAI-compatible LLM request failed") from exc
        if not isinstance(content, str) or not content.strip():
            raise LLMRequestError("OpenAI-compatible LLM returned empty text")
        return content

    def _record_response(self, response: Any, choice: Any) -> None:
        """提取 usage 与 reasoning_content 诊断；任何字段缺失都按“没有”处理。"""

        usage = getattr(response, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", None)
        completion_tokens = getattr(usage, "completion_tokens", None)
        self._last_usage = (
            {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens}
            if usage is not None
            else None
        )
        message = choice.message
        reasoning = getattr(message, "reasoning_content", None)
        extra_fields = getattr(message, "model_extra", None)
        if reasoning is None and isinstance(extra_fields, dict):
            reasoning = extra_fields.get("reasoning_content")
        self.last_response_info = {
            "finish_reason": getattr(choice, "finish_reason", None),
            "reasoning_content_present": reasoning is not None,
            "reasoning_content_non_empty": bool(reasoning),
        }
