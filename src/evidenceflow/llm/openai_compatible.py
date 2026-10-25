"""A general Chat Completions adapter built on the official OpenAI SDK."""

from __future__ import annotations

from collections.abc import Sequence
from math import isfinite
from typing import Any

import httpx
from openai import OpenAI

from evidenceflow.concurrency import PerThread
from evidenceflow.domain.generation import LLMMessage
from evidenceflow.llm.base import LLMConfigurationError, LLMRequestError
from evidenceflow.llm.diagnostics import classify_llm_failure

# Structured output mode: json_schema (strict) > json_object > prompt_only.
# Chosen value (Stage A §S.2): based on artifacts/stage_a/A2_probe.json - deepseek-flash returns
# 400 for json_schema (response_format type unavailable), while json_object and prompt_only both
# pass; following the json_schema > json_object > prompt_only priority we select json_object (which
# requires the Prompt to contain "JSON").
# This is a fixed constant only: there is no runtime mode downgrade or automatic probing; after
# switching model/service the probe must be re-run.
STRUCTURED_OUTPUT_MODE = "json_object"
STRUCTURED_OUTPUT_MODES = ("json_schema", "json_object", "prompt_only")

# thinking switch (DeepSeek etc.): auto = do not send it, keep the server default; enabled/disabled
# explicitly turn it on or off.
THINKING_MODES = ("auto", "enabled", "disabled")


def build_structured_request_params(
    mode: str,
    *,
    schema_name: str,
    json_schema: dict,
    messages: Sequence[LLMMessage],
) -> dict[str, Any]:
    """Build the parameters added to a Chat Completions request for the given mode (pure; no network).

    - json_schema: ``response_format`` carries a strict JSON Schema;
    - json_object: declares a JSON object only; the OpenAI protocol requires the word "JSON" to
      appear in the Prompt, otherwise the server rejects the request, so it is validated here up
      front to produce a clear error;
    - prompt_only: no ``response_format`` is sent and the Prompt alone constrains the output.
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
    """Turn internal messages into OpenAI-compatible Chat Completions requests."""

    # usage and response diagnostics only describe "the most recent call on this thread"; they are
    # isolated per thread so concurrent requests do not overwrite each other.
    _last_usage = PerThread(lambda: None)
    last_response_info = PerThread(dict)

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
        trust_env: bool = False,
        _client: Any | None = None,
    ) -> None:
        """Validate the configuration and build a synchronous SDK client with no automatic retry and an explicit timeout."""

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
        # Refers to the most recent call only; reset when a call starts, and the layer above
        # (Generator) is responsible for aggregating multiple calls.
        self._last_usage: dict | None = None
        # Diagnostics: whether the latest response carried reasoning_content, plus the finish
        # reason; these do not enter the business contract.
        self.last_response_info: dict[str, Any] = {}
        self._model = model
        self._temperature = float(temperature)
        self._max_attempts = max_attempts
        self._client = _client or OpenAI(
            api_key=api_key,
            base_url=base_url,
            # System proxies are ignored by default: when a local proxy such as Clash is not
            # running, the TLS handshake used to be truncated. Set LLM_TRUST_ENV=true to opt in
            # when outbound traffic must go through a proxy.
            http_client=httpx.Client(trust_env=bool(trust_env), timeout=float(timeout_seconds)),
            # Errors are reported upwards directly at this stage, with no hidden SDK automatic
            # retries or extra waiting.
            max_retries=0,
        )

    @property
    def max_attempts(self) -> int:
        """Maximum attempts for a single call (online scripts require 1, with retry handled by the script policy)."""

        return self._max_attempts

    @property
    def last_usage(self) -> dict | None:
        """Token usage of the current thread's most recent call; None when the service returned no usage."""

        return self._last_usage

    def generate(self, messages: Sequence[LLMMessage]) -> str:
        """Call Chat Completions, normalising every provider/transport failure into LLMRequestError."""

        return self._complete(messages, {})

    def generate_json(
        self,
        messages: Sequence[LLMMessage],
        *,
        schema_name: str,
        json_schema: dict,
    ) -> str:
        """Request JSON using the structured output mode fixed at construction time and return the model text verbatim (parsing is the caller's job)."""

        extra = build_structured_request_params(
            self._structured_output_mode,
            schema_name=schema_name,
            json_schema=json_schema,
            messages=messages,
        )
        return self._complete(messages, extra)

    def _complete(self, messages: Sequence[LLMMessage], extra: dict[str, Any]) -> str:
        """Shared request and transient-retry logic; records the latest call's usage and response diagnostics."""

        if not messages:
            raise LLMRequestError("LLM messages must not be empty")
        request_messages = [message.model_dump() for message in messages]
        request_extra = dict(extra)
        if self._thinking != "auto":
            # DeepSeek's thinking switch (OpenAI format) must be passed through extra_body.
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
                # The normal Chat Completion path requires at least one Choice with non-empty
                # plain-text content.
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
                # Retry once only for categories proven to be potentially transient; configuration
                # errors and ordinary 4xx fail immediately.
                if attempt >= self._max_attempts or not retryable:
                    raise LLMRequestError("OpenAI-compatible LLM request failed") from exc
        if not isinstance(content, str) or not content.strip():
            raise LLMRequestError("OpenAI-compatible LLM returned empty text")
        return content

    def _record_response(self, response: Any, choice: Any) -> None:
        """Extract usage and reasoning_content diagnostics; any missing field is treated as "absent"."""

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
