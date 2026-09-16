"""测试统一 LLM 合同与 OpenAI-compatible Chat Adapter。"""

from types import SimpleNamespace

import httpx
import pytest
from pydantic import ValidationError

from evidenceflow.domain import LLMMessage
from evidenceflow.llm import (
    LLMClient,
    LLMConfigurationError,
    LLMRequestError,
    OpenAICompatibleLLMClient,
)


class FakeCompletions:
    """记录 Chat 请求并返回可配置的 SDK 形状响应。"""

    def __init__(self, content: object = "Grounded answer [S1].", error: Exception | None = None) -> None:
        self.content = content
        self.error = error
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> object:
        """模拟 `client.chat.completions.create`。"""

        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self.content))]
        )


class SequencedCompletions(FakeCompletions):
    """按顺序返回异常或响应，用于验证一次性瞬态重试。"""

    def __init__(self, outcomes: list[object]) -> None:
        super().__init__()
        self.outcomes = outcomes

    def create(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=outcome))]
        )


def make_sdk_client(completions: FakeCompletions) -> object:
    """构造最小 SDK Client 对象树。"""

    return SimpleNamespace(chat=SimpleNamespace(completions=completions))


def test_llm_message_validates_role_and_non_blank_content() -> None:
    assert LLMMessage(role="system", content="rules").role == "system"
    with pytest.raises(ValidationError):
        LLMMessage(role="tool", content="invalid")
    with pytest.raises(ValidationError):
        LLMMessage(role="user", content="   ")


def test_openai_adapter_satisfies_protocol_and_preserves_raw_text() -> None:
    completions = FakeCompletions(content="  exact model text [S1].  ")
    client = OpenAICompatibleLLMClient(
        api_key="test-only-key",
        base_url="https://example.test/v1",
        model="test-model",
        temperature=0.0,
        timeout_seconds=10,
        _client=make_sdk_client(completions),
    )
    messages = [LLMMessage(role="user", content="question")]

    result = client.generate(messages)

    assert isinstance(client, LLMClient)
    assert result == "  exact model text [S1].  "
    assert completions.calls == [
        {
            "model": "test-model",
            "messages": [{"role": "user", "content": "question"}],
            "temperature": 0.0,
        }
    ]


@pytest.mark.parametrize("api_key", [None, "", "   "])
def test_openai_adapter_fails_fast_without_api_key(api_key: str | None) -> None:
    with pytest.raises(LLMConfigurationError, match="API key"):
        OpenAICompatibleLLMClient(
            api_key=api_key,
            base_url="https://example.test/v1",
            model="test-model",
            _client=make_sdk_client(FakeCompletions()),
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("base_url", " ", "base URL"),
        ("model", "", "model"),
        ("temperature", float("nan"), "temperature"),
        ("temperature", 3, "temperature"),
        ("timeout_seconds", 0, "timeout"),
        ("timeout_seconds", float("inf"), "timeout"),
    ],
)
def test_openai_adapter_rejects_invalid_runtime_configuration(
    field: str, value: object, message: str
) -> None:
    kwargs: dict[str, object] = {
        "api_key": "test-only-key",
        "base_url": "https://example.test/v1",
        "model": "test-model",
        "temperature": 0.0,
        "timeout_seconds": 10,
        "_client": make_sdk_client(FakeCompletions()),
    }
    kwargs[field] = value
    with pytest.raises(LLMConfigurationError, match=message):
        OpenAICompatibleLLMClient(**kwargs)  # type: ignore[arg-type]


def test_openai_adapter_wraps_sdk_failure_without_exposing_secret() -> None:
    completions = FakeCompletions(error=TimeoutError("provider timed out"))
    client = OpenAICompatibleLLMClient(
        api_key="test-only-secret",
        base_url="https://example.test/v1",
        model="test-model",
        max_attempts=1,
        _client=make_sdk_client(completions),
    )

    with pytest.raises(LLMRequestError, match="request failed") as raised:
        client.generate([LLMMessage(role="user", content="question")])

    assert isinstance(raised.value.__cause__, TimeoutError)
    assert "test-only-secret" not in str(raised.value)


def test_openai_adapter_retries_timeout_once_then_succeeds() -> None:
    completions = SequencedCompletions([TimeoutError("timed out"), "answer"])
    client = OpenAICompatibleLLMClient(
        api_key="test-only-key",
        base_url="https://example.test/v1",
        model="test-model",
        max_attempts=2,
        _client=make_sdk_client(completions),
    )

    assert client.generate([LLMMessage(role="user", content="question")]) == "answer"
    assert len(completions.calls) == 2


def test_openai_adapter_does_not_retry_non_transient_error() -> None:
    completions = SequencedCompletions([ValueError("bad request"), "unused"])
    client = OpenAICompatibleLLMClient(
        api_key="test-only-key",
        base_url="https://example.test/v1",
        model="test-model",
        max_attempts=2,
        _client=make_sdk_client(completions),
    )

    with pytest.raises(LLMRequestError):
        client.generate([LLMMessage(role="user", content="question")])
    assert len(completions.calls) == 1


@pytest.mark.parametrize("content", [None, "", "   ", ["not", "text"]])
def test_openai_adapter_rejects_missing_or_empty_text(content: object) -> None:
    client = OpenAICompatibleLLMClient(
        api_key="test-only-key",
        base_url="https://example.test/v1",
        model="test-model",
        _client=make_sdk_client(FakeCompletions(content=content)),
    )

    with pytest.raises(LLMRequestError, match="empty text"):
        client.generate([LLMMessage(role="user", content="question")])


def test_explicit_constructor_passes_timeout_and_disables_sdk_retries(monkeypatch) -> None:
    """正式 Client 构造必须显式传入 Endpoint、禁用系统代理、显式超时且关闭隐藏自动重试。"""

    captured: dict[str, object] = {}

    def fake_openai(**kwargs: object) -> object:
        captured.update(kwargs)
        return make_sdk_client(FakeCompletions())

    monkeypatch.setattr("evidenceflow.llm.openai_compatible.OpenAI", fake_openai)
    client = OpenAICompatibleLLMClient(
        api_key="test-only-key",
        base_url="https://example.test/v1",
        model="test-model",
        temperature=0.2,
        timeout_seconds=12.5,
    )
    client.generate([LLMMessage(role="user", content="question")])

    assert captured["api_key"] == "test-only-key"
    assert captured["base_url"] == "https://example.test/v1"
    assert captured["max_retries"] == 0

    http_client = captured["http_client"]
    assert isinstance(http_client, httpx.Client)
    assert http_client.trust_env is False
    assert http_client.timeout == httpx.Timeout(12.5)
