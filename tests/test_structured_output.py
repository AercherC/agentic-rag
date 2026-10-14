"""测试结构化输出：三种请求模式的纯函数构造、generate_json 与 last_usage（不联网）。"""

from types import SimpleNamespace

import pytest

from evidenceflow.domain import LLMMessage, generator_output_json_schema
from evidenceflow.llm import (
    STRUCTURED_OUTPUT_MODE,
    STRUCTURED_OUTPUT_MODES,
    LLMClient,
    LLMConfigurationError,
    LLMRequestError,
    OpenAICompatibleLLMClient,
    build_structured_request_params,
)

SCHEMA = generator_output_json_schema()
JSON_MESSAGES = [LLMMessage(role="user", content="Return a JSON object.")]
PLAIN_MESSAGES = [LLMMessage(role="user", content="Say hello.")]


def test_provisional_mode_is_prompt_only_and_all_modes_are_known() -> None:
    assert STRUCTURED_OUTPUT_MODE == "prompt_only"
    assert STRUCTURED_OUTPUT_MODES == ("json_schema", "json_object", "prompt_only")


def test_json_schema_mode_params_use_strict_schema() -> None:
    params = build_structured_request_params(
        "json_schema", schema_name="generator_output", json_schema=SCHEMA, messages=PLAIN_MESSAGES
    )

    assert params == {
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "generator_output", "strict": True, "schema": SCHEMA},
        }
    }
    # 合同：additionalProperties=false，status/answer 均 required。
    assert SCHEMA["additionalProperties"] is False
    assert set(SCHEMA["required"]) == {"status", "answer"}


def test_json_object_mode_params_and_requires_json_word_in_prompt() -> None:
    params = build_structured_request_params(
        "json_object", schema_name="x", json_schema=SCHEMA, messages=JSON_MESSAGES
    )
    assert params == {"response_format": {"type": "json_object"}}
    with pytest.raises(LLMConfigurationError, match="JSON"):
        build_structured_request_params(
            "json_object", schema_name="x", json_schema=SCHEMA, messages=PLAIN_MESSAGES
        )


def test_prompt_only_mode_adds_no_response_format() -> None:
    assert (
        build_structured_request_params(
            "prompt_only", schema_name="x", json_schema=SCHEMA, messages=PLAIN_MESSAGES
        )
        == {}
    )


def test_unknown_mode_is_rejected_everywhere() -> None:
    with pytest.raises(LLMConfigurationError, match="mode"):
        build_structured_request_params(
            "auto", schema_name="x", json_schema=SCHEMA, messages=JSON_MESSAGES
        )
    with pytest.raises(LLMConfigurationError, match="mode"):
        OpenAICompatibleLLMClient(
            api_key="test-only-key",
            base_url="https://example.test/v1",
            model="m",
            structured_output_mode="auto",
            _client=object(),
        )


class RecordingCompletions:
    """返回预置 content / usage，并记录每次请求参数。"""

    def __init__(self, contents: list[object], usage: object = None, message_extra: dict | None = None) -> None:
        self.contents = list(contents)
        self.usage = usage
        self.message_extra = message_extra or {}
        self.calls: list[dict] = []

    def create(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        message = SimpleNamespace(content=self.contents.pop(0), **self.message_extra)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message, finish_reason="stop")], usage=self.usage
        )


def make_client(completions: RecordingCompletions, mode: str) -> OpenAICompatibleLLMClient:
    return OpenAICompatibleLLMClient(
        api_key="test-only-key",
        base_url="https://example.test/v1",
        model="test-model",
        structured_output_mode=mode,
        _client=SimpleNamespace(chat=SimpleNamespace(completions=completions)),
    )


@pytest.mark.parametrize(
    ("mode", "expected_extra"),
    [
        ("json_schema", {"response_format": {"type": "json_schema", "json_schema": {"name": "generator_output", "strict": True, "schema": SCHEMA}}}),
        ("json_object", {"response_format": {"type": "json_object"}}),
        ("prompt_only", {}),
    ],
)
def test_generate_json_sends_mode_specific_request(mode: str, expected_extra: dict) -> None:
    completions = RecordingCompletions(['{"status": "answered", "answer": "x [S1]"}'])
    client = make_client(completions, mode)

    text = client.generate_json(JSON_MESSAGES, schema_name="generator_output", json_schema=SCHEMA)

    assert text == '{"status": "answered", "answer": "x [S1]"}'
    assert completions.calls == [
        {
            "model": "test-model",
            "messages": [{"role": "user", "content": "Return a JSON object."}],
            "temperature": 0.0,
            **expected_extra,
        }
    ]
    assert isinstance(client, LLMClient)


def test_default_client_uses_the_module_constant_mode() -> None:
    completions = RecordingCompletions(["{}"])
    client = OpenAICompatibleLLMClient(
        api_key="test-only-key",
        base_url="https://example.test/v1",
        model="m",
        _client=SimpleNamespace(chat=SimpleNamespace(completions=completions)),
    )
    client.generate_json(JSON_MESSAGES, schema_name="generator_output", json_schema=SCHEMA)
    assert "response_format" not in completions.calls[0]  # 暂定 prompt_only


def test_last_usage_reflects_only_the_latest_call_and_is_none_without_usage() -> None:
    usage = SimpleNamespace(prompt_tokens=12, completion_tokens=3)
    completions = RecordingCompletions(["a", "b"], usage=usage)
    client = make_client(completions, "prompt_only")
    assert client.last_usage is None  # 尚未调用

    client.generate(PLAIN_MESSAGES)
    assert client.last_usage == {"prompt_tokens": 12, "completion_tokens": 3}

    completions.usage = None  # 第二次调用服务没有返回 usage
    client.generate(PLAIN_MESSAGES)
    assert client.last_usage is None


def test_failed_call_resets_last_usage() -> None:
    """请求抛出异常时不能残留上一次调用的 usage。"""

    usage = SimpleNamespace(prompt_tokens=1, completion_tokens=1)
    completions = RecordingCompletions(["a"], usage=usage)
    client = make_client(completions, "prompt_only")
    client.generate(PLAIN_MESSAGES)
    assert client.last_usage is not None

    completions.contents = []  # 下一次 create 内部 pop 空列表 -> 抛 IndexError，被统一包装
    with pytest.raises(LLMRequestError):
        client.generate(PLAIN_MESSAGES)
    assert client.last_usage is None


def test_reasoning_content_presence_is_recorded() -> None:
    present = RecordingCompletions(["x"], message_extra={"reasoning_content": "thinking"})
    client = make_client(present, "prompt_only")
    client.generate(PLAIN_MESSAGES)
    assert client.last_response_info == {
        "finish_reason": "stop",
        "reasoning_content_present": True,
        "reasoning_content_non_empty": True,
    }

    empty = RecordingCompletions(["x"], message_extra={"reasoning_content": ""})
    client = make_client(empty, "prompt_only")
    client.generate(PLAIN_MESSAGES)
    assert client.last_response_info["reasoning_content_present"] is True
    assert client.last_response_info["reasoning_content_non_empty"] is False

    absent = RecordingCompletions(["x"])
    client = make_client(absent, "prompt_only")
    client.generate(PLAIN_MESSAGES)
    assert client.last_response_info["reasoning_content_present"] is False
