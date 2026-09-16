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

    def generate(self, messages: Sequence[LLMMessage]) -> str:
        """调用 Chat Completions，并把所有供应商/传输失败统一为 LLMRequestError。"""

        if not messages:
            raise LLMRequestError("LLM messages must not be empty")
        request_messages = [message.model_dump() for message in messages]
        content: Any = None
        for attempt in range(1, self._max_attempts + 1):
            try:
                response = self._client.chat.completions.create(
                    model=self._model,
                    messages=request_messages,
                    temperature=self._temperature,
                )
                # Chat Completion 正常路径要求至少一个 Choice 且 content 是非空纯文本。
                content = response.choices[0].message.content
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
