"""测试 A2b 在线脚本的关键逻辑：限流等待、403 停止、断点续跑、原始回复留存、离线重解析。

全部使用 Fake client 和注入的 sleep/clock，不联网、不真实等待。
"""

import json
from types import SimpleNamespace

import pytest

from evidenceflow.domain import (
    BuiltContext,
    ContextEntry,
    LLMMessage,
    SourceReference,
)
from evidenceflow.generation import Generator
from evidenceflow.llm import LLMConfigurationError, LLMRequestError, OpenAICompatibleLLMClient
from experiments.stage_a.online_common import (
    FatalServiceError,
    PolicyLLMClient,
    RequestPolicy,
    ResumeIdentityError,
    append_jsonl,
    check_resume_identity,
    failure_class,
    identity_fields,
    thinking_ignored_calls,
    extract_allowed_labels,
    read_jsonl,
    summarize_smoke,
)
from experiments.stage_a.probe_structured_output import (
    REQUESTS,
    aggregate_probe,
    analyze_raw,
    run_probe,
)
from experiments.stage_a.reparse_smoke_offline import reparse_all, reparse_record
from experiments.stage_a.run_generation_smoke import RecordingRetriever, run_smoke

IDENTITY = identity_fields(
    structured_output_mode="prompt_only", prompt_version="rag-generator-v3",
    model="deepseek-flash", llm_thinking="disabled",
)
VALID = json.dumps({"status": "answered", "answer": "Fact [S1]."})


class HttpError(Exception):
    """带 status_code 的 Fake 供应商异常（classify_llm_failure 依赖该属性）。"""

    def __init__(self, status_code: int) -> None:
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code


def wrapped(status_code: int) -> LLMRequestError:
    """模拟 OpenAICompatibleLLMClient 的包装：LLMRequestError 的 __cause__ 是供应商异常。"""

    error = LLMRequestError("OpenAI-compatible LLM request failed")
    error.__cause__ = HttpError(status_code)
    return error


class FakeClock:
    """可控时钟 + 记录 sleep，让节流与退避无需真实等待。"""

    def __init__(self) -> None:
        self.now = 1000.0
        self.sleeps: list[float] = []

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds

    def clock(self) -> float:
        return self.now


def make_policy(clock: FakeClock, **kwargs) -> RequestPolicy:
    return RequestPolicy(sleep=clock.sleep, clock=clock.clock, **kwargs)


# ---- RequestPolicy ---------------------------------------------------------------------


def test_policy_enforces_min_interval_between_requests() -> None:
    clock = FakeClock()
    policy = make_policy(clock, min_interval_seconds=6.0)

    policy.call(lambda: "a")
    clock.now += 2.0  # 距上次只过了 2 秒
    policy.call(lambda: "b")

    assert clock.sleeps == [pytest.approx(4.0)]


def test_policy_waits_and_retries_on_429_then_succeeds_and_records_events() -> None:
    clock = FakeClock()
    policy = make_policy(clock, min_interval_seconds=0.0, backoff_seconds=(30, 60, 120))
    outcomes = [wrapped(429), wrapped(429), "ok"]

    def fn():
        outcome = outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    assert policy.call(fn, label="generate_json") == "ok"
    assert clock.sleeps == [30, 60]
    assert [(e["attempt"], e["wait_seconds"], e["exhausted"]) for e in policy.rate_limit_events] == [
        (1, 30, False),
        (2, 60, False),
    ]


def test_policy_gives_up_after_three_retries_and_marks_exhausted() -> None:
    clock = FakeClock()
    policy = make_policy(clock, min_interval_seconds=0.0)
    calls = []

    def fn():
        calls.append(1)
        raise wrapped(429)

    with pytest.raises(LLMRequestError) as raised:
        policy.call(fn)

    assert len(calls) == 4  # 首次 + 3 次重试
    assert raised.value.rate_limit_exhausted is True
    assert clock.sleeps == [30, 60, 120]
    assert policy.rate_limit_events[-1]["exhausted"] is True


@pytest.mark.parametrize("status", [403, 401])
def test_policy_stops_immediately_on_403_without_retry(status: int) -> None:
    clock = FakeClock()
    policy = make_policy(clock, min_interval_seconds=0.0)
    calls = []

    def fn():
        calls.append(1)
        raise wrapped(status)

    with pytest.raises(FatalServiceError):
        policy.call(fn)

    assert calls == [1] and clock.sleeps == []


def test_policy_does_not_retry_other_errors() -> None:
    clock = FakeClock()
    policy = make_policy(clock, min_interval_seconds=0.0)
    calls = []

    def fn():
        calls.append(1)
        raise wrapped(400)

    with pytest.raises(LLMRequestError):
        policy.call(fn)

    assert calls == [1] and policy.rate_limit_events == []


# ---- PolicyLLMClient：原始回复与诊断留存 ----------------------------------------------------


class FakeInner:
    """按顺序返回回复/异常，并提供 last_usage 与 last_response_info。"""

    max_attempts = 1  # PolicyLLMClient 要求 inner 的 max_attempts == 1

    def __init__(self, *replies) -> None:
        self.replies = list(replies)
        self.last_usage = None
        self.last_response_info: dict = {}

    def generate(self, messages):
        return self.generate_json(messages, schema_name="x", json_schema={})

    def generate_json(self, messages, *, schema_name, json_schema):
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            self.last_usage = None
            raise reply
        self.last_usage = {"prompt_tokens": 5, "completion_tokens": 2}
        self.last_response_info = {
            "finish_reason": "stop",
            "reasoning_content_present": True,
            "reasoning_content_non_empty": False,
        }
        return reply


def messages() -> list[LLMMessage]:
    return [LLMMessage(role="system", content="s"), LLMMessage(role="user", content="u")]


def test_policy_client_saves_raw_reply_usage_and_reasoning_flags() -> None:
    clock = FakeClock()
    client = PolicyLLMClient(FakeInner("raw text"), make_policy(clock, min_interval_seconds=0.0))
    client.begin()

    raw = client.generate_json(messages(), schema_name="x", json_schema={})

    assert raw == "raw text"
    call = client.calls[0]
    assert call["raw"] == "raw text"
    assert call["usage"] == {"prompt_tokens": 5, "completion_tokens": 2}
    assert call["reasoning_content_present"] is True
    assert call["reasoning_content_non_empty"] is False
    assert call["messages"][1] == {"role": "user", "content": "u"}
    assert client.last_usage == {"prompt_tokens": 5, "completion_tokens": 2}


def test_policy_client_records_errors_separately_from_rate_limits() -> None:
    clock = FakeClock()
    policy = make_policy(clock, min_interval_seconds=0.0)
    client = PolicyLLMClient(FakeInner(wrapped(429), "ok", wrapped(400)), policy)
    client.begin()

    assert client.generate_json(messages(), schema_name="x", json_schema={}) == "ok"
    assert len(client.question_rate_limit_events) == 1
    with pytest.raises(LLMRequestError):
        client.generate_json(messages(), schema_name="x", json_schema={})

    assert client.calls[1]["error"] == "request_4xx"
    assert client.last_usage is None
    client.begin()
    assert client.question_rate_limit_events == []  # 新的一题重新计数


# ---- 冒烟：断点续跑 / 403 停止 / 重跑服务错误 ---------------------------------------------


def context(labels=("[S1]",)) -> BuiltContext:
    entries = [
        ContextEntry(
            source_label=label,
            retrieval_rank=i,
            chunk_id=f"c{i}",
            content="evidence",
            source_reference=SourceReference(
                source_label=label, document_id="d", file_name="f.md",
                page_start=None, page_end=None, chunk_id=f"c{i}",
            ),
            token_count=1,
        )
        for i, label in enumerate(labels, start=1)
    ]
    text = "\n\n".join(f"{label}\nFile: f.md\nContent:\nevidence" for label in labels)
    return BuiltContext(
        context_text=text, entries=entries, total_tokens=len(entries),
        candidate_count=len(entries), included_count=len(entries), skipped_count=0,
    )


class FakeService:
    """走真实 Generator 的最小服务；检索结果固定，便于断言 Top5 记录。"""

    def __init__(self, generator, recorder) -> None:
        self.generator = generator
        self.recorder = recorder
        self.answered: list[str] = []

    def answer(self, query: str):
        self.answered.append(query)
        self.recorder.last_ids = ["chunk-a", "chunk-b"]
        generated = self.generator.generate(query, context())
        return SimpleNamespace(
            status=generated.status, answer=generated.answer_text, sources=[SimpleNamespace(source_label="[S1]")]
        )


def build_smoke(replies, tmp_path, **policy_kwargs):
    clock = FakeClock()
    llm = PolicyLLMClient(FakeInner(*replies), make_policy(clock, min_interval_seconds=0.0, **policy_kwargs))
    generator = Generator(llm)
    recorder = RecordingRetriever(SimpleNamespace(retrieve=lambda q: []))
    service = FakeService(generator, recorder)
    return service, generator, llm, recorder, tmp_path / "smoke.jsonl"


ITEMS = [{"question_id": f"Q{i}", "query": f"query {i}"} for i in range(1, 5)]


def run(items, service, generator, llm, recorder, path, **kwargs):
    kwargs.setdefault("identity", IDENTITY)
    return run_smoke(items, service=service, generator=generator, llm=llm, recorder=recorder,
                     device="cpu", jsonl_path=path, **kwargs)


def test_smoke_records_top5_diagnostics_status_and_raw_calls(tmp_path) -> None:
    service, generator, llm, recorder, path = build_smoke([VALID, "bad", VALID], tmp_path)

    records = run(ITEMS[:2], service, generator, llm, recorder, path)

    first, second = records
    assert first["top5_chunk_ids"] == ["chunk-a", "chunk-b"]
    assert first["status"] == "answered" and first["failure_class"] is None
    assert first["diagnostics"]["repair_attempted"] is False
    assert first["allowed_labels"] == ["[S1]"]
    assert first["device"] == "cpu"
    # 第二题触发修复：两次调用的原始回复都被保存（含修复调用）。
    assert [c["raw"] for c in second["calls"]] == ["bad", VALID]
    assert second["diagnostics"]["repair_attempted"] is True and second["diagnostics"]["repair_succeeded"] is True
    assert len(read_jsonl(path)) == 2


def test_smoke_resumes_and_skips_completed_questions(tmp_path) -> None:
    service, generator, llm, recorder, path = build_smoke([VALID, VALID], tmp_path)
    append_jsonl(path, {**IDENTITY, "question_id": "Q1", "failure_class": None})
    append_jsonl(path, {**IDENTITY, "question_id": "Q2", "failure_class": None})

    records = run(ITEMS, service, generator, llm, recorder, path)

    assert service.answered == ["query 3", "query 4"]  # Q1/Q2 被跳过，不消耗请求
    assert [r["question_id"] for r in records] == ["Q1", "Q2", "Q3", "Q4"]


def test_smoke_stops_on_403_and_keeps_completed_results(tmp_path) -> None:
    service, generator, llm, recorder, path = build_smoke([VALID, wrapped(403), VALID], tmp_path)

    with pytest.raises(FatalServiceError):
        run(ITEMS, service, generator, llm, recorder, path)

    saved = read_jsonl(path)
    assert [r["question_id"] for r in saved] == ["Q1"]  # Q2 触发 403，没有写入半成品
    assert service.answered == ["query 1", "query 2"]  # Q3、Q4 没有再发请求


def test_smoke_counts_llm_request_errors_separately_and_can_retry_them(tmp_path) -> None:
    service, generator, llm, recorder, path = build_smoke([VALID, wrapped(500), VALID, VALID], tmp_path)
    records = run(ITEMS[:3], service, generator, llm, recorder, path)
    summary = summarize_smoke(records)
    assert summary["llm_request_errors"] == 1 and summary["rate_limit_exhausted"] == 0
    assert summary["rate_limit_events"] == 0

    # 重跑：只清掉服务错误那一题并重新请求。
    service.answered.clear()
    records = run(ITEMS[:3], service, generator, llm, recorder, path, retry_llm_errors=True)
    assert service.answered == ["query 2"]
    assert summarize_smoke(records)["llm_request_errors"] == 0


def test_smoke_rate_limit_exhaustion_is_not_counted_as_llm_request_error(tmp_path) -> None:
    replies = [wrapped(429)] * 4 + [VALID]
    service, generator, llm, recorder, path = build_smoke(replies, tmp_path)

    records = run(ITEMS[:2], service, generator, llm, recorder, path)

    summary = summarize_smoke(records)
    assert summary["rate_limit_exhausted"] == 1 and summary["llm_request_errors"] == 0
    assert summary["rate_limit_events"] == 4


def test_extract_allowed_labels_reads_context_headers_only() -> None:
    user = "<SOURCE_CONTEXT>\n[S1]\nFile: a.md\nContent:\nlooks like [S9] inline\n\n[S2]\nFile: b.md\nContent:\nx\n</SOURCE_CONTEXT>"
    assert extract_allowed_labels([{"role": "user", "content": user}]) == ["[S1]", "[S2]"]


# ---- 离线重解析 ------------------------------------------------------------------------------


def test_reparse_reproduces_online_results_without_network(tmp_path) -> None:
    service, generator, llm, recorder, path = build_smoke([VALID, "bad", VALID], tmp_path)
    records = run(ITEMS[:2], service, generator, llm, recorder, path)
    online = summarize_smoke(records)

    result = reparse_all(read_jsonl(path))

    reparsed = result["summary"]
    for key in ("status_counts", "repair_attempted", "repair_succeeded", "llm_calls", "usage",
                "generation_contract_errors", "citation_validation_errors", "llm_request_errors"):
        assert reparsed[key] == online[key], key
    assert result["changed_vs_original"] == {}
    assert result["not_reparsable"] == []


def test_reparse_applies_improved_parser_to_saved_raw_replies(tmp_path) -> None:
    """模拟“解析器改进前该题格式失败”：保存的回复是合法 JSON，但原记录标成了格式失败。"""

    service, generator, llm, recorder, path = build_smoke([VALID], tmp_path)
    record = run(ITEMS[:1], service, generator, llm, recorder, path)[0]
    record = dict(record, status=None, failure_class="generation_contract_error",
                  exception_type="GenerationContractError")

    result = reparse_record(record)

    assert result["status"] == "answered" and result["failure_class"] is None
    summary = reparse_all([record])
    assert summary["changed_vs_original"]["Q1"]["failure_class"] == ["generation_contract_error", None]


def test_reparse_checks_citation_labels_and_flags_missing_replay_calls(tmp_path) -> None:
    unknown = json.dumps({"status": "answered", "answer": "Fact [S9]."})
    service, generator, llm, recorder, path = build_smoke([VALID], tmp_path)
    record = run(ITEMS[:1], service, generator, llm, recorder, path)[0]
    cited = dict(record, calls=[dict(record["calls"][0], raw=unknown)])
    assert reparse_record(cited)["failure_class"] == "citation_validation_error"

    # 保存的唯一回复是坏的，而旧记录当时没有修复调用：新逻辑需要第二次调用，无法离线复现。
    bad_only = dict(record, calls=[dict(record["calls"][0], raw="not json")])
    result = reparse_record(bad_only)
    assert result["exception_type"] == "ReplayExhaustedError"

    # 当时没有拿到任何回复（服务错误）的题不可重解析，原分类保持。
    no_raw = dict(record, calls=[{"kind": "generate_json", "error": "provider_5xx"}],
                  failure_class="llm_request_error", exception_type="LLMRequestError")
    assert reparse_record(no_raw)["reparsable"] is False


# ---- 探测脚本 --------------------------------------------------------------------------------


def test_analyze_raw_flags_think_and_fence_and_parse_errors() -> None:
    clean = analyze_raw(VALID)
    assert clean["schema_valid_after_cleanup"] and clean["json_loads_ok_raw"]
    assert clean["has_think_open"] is False and clean["wrapped_in_code_fence"] is False

    fenced = analyze_raw("<think>x</think>\n```json\n" + VALID + "\n```")
    assert fenced["has_think_open"] and fenced["think_closed"] is True
    assert fenced["wrapped_in_code_fence"] is True and fenced["schema_valid_after_cleanup"] is True
    assert fenced["json_loads_ok_raw"] is False

    unclosed = analyze_raw("<think>never closed " + VALID)
    assert unclosed["think_closed"] is False and unclosed["schema_valid_after_cleanup"] is False


PROBE_ID = {"model": "deepseek-flash", "llm_thinking": "disabled"}


def probe_client_factory(per_mode: dict):
    def make_client(mode: str):
        return FakeInner(*per_mode[mode])

    return make_client


def test_probe_selects_best_usable_mode_and_resumes(tmp_path) -> None:
    clock = FakeClock()
    policy = make_policy(clock, min_interval_seconds=0.0)
    jsonl = tmp_path / "probe.jsonl"
    per_mode = {
        "json_schema": [wrapped(400)] * len(REQUESTS),  # 服务不支持 json_schema
        "json_object": [VALID] * len(REQUESTS),
        "prompt_only": [VALID] * len(REQUESTS),
    }

    records = run_probe(probe_client_factory(per_mode), policy, jsonl, **PROBE_ID)
    summary = aggregate_probe(records)

    assert summary["usable"] == {"json_schema": False, "json_object": True, "prompt_only": True}
    assert summary["selected_mode"] == "json_object"  # 优先级：json_object > prompt_only
    assert len(read_jsonl(jsonl)) == 9

    # 续跑：全部已完成，不再创建任何请求。
    again = run_probe(probe_client_factory({m: [] for m in per_mode}), policy, jsonl, **PROBE_ID)
    assert len(again) == 9


def test_probe_stops_on_403_and_keeps_completed_items(tmp_path) -> None:
    clock = FakeClock()
    policy = make_policy(clock, min_interval_seconds=0.0)
    jsonl = tmp_path / "probe.jsonl"
    per_mode = {
        "json_schema": [VALID, wrapped(403)],
        "json_object": [],
        "prompt_only": [],
    }

    with pytest.raises(FatalServiceError):
        run_probe(probe_client_factory(per_mode), policy, jsonl, **PROBE_ID)

    assert len(read_jsonl(jsonl)) == 1


def test_probe_rate_limit_exhaustion_is_retried_on_rerun_not_treated_as_unusable(tmp_path) -> None:
    clock = FakeClock()
    policy = make_policy(clock, min_interval_seconds=0.0)
    jsonl = tmp_path / "probe.jsonl"
    first = {"json_schema": [wrapped(429)] * 4 + [VALID, VALID], "json_object": [VALID] * 3, "prompt_only": [VALID] * 3}

    records = run_probe(probe_client_factory(first), policy, jsonl, **PROBE_ID)
    exhausted = [r for r in records if r.get("rate_limit_exhausted")]
    assert len(exhausted) == 1 and exhausted[0]["completed"] is False
    assert aggregate_probe(records)["usable"]["json_schema"] is False  # 缺一项，不算完整

    # 重跑只补上未完成的那一项。
    second = {"json_schema": [VALID], "json_object": [], "prompt_only": []}
    records = run_probe(probe_client_factory(second), policy, jsonl, **PROBE_ID)
    assert aggregate_probe(records)["selected_mode"] == "json_schema"


# ---- A2a.1：续跑身份保护 / max_attempts 保护 / 真实 SDK 异常 / thinking 统计 ------------------


def test_smoke_records_carry_identity_fields(tmp_path) -> None:
    service, generator, llm, recorder, path = build_smoke([VALID], tmp_path)

    (record,) = run(ITEMS[:1], service, generator, llm, recorder, path)

    for key, value in IDENTITY.items():
        assert record[key] == value
        assert read_jsonl(path)[0][key] == value


@pytest.mark.parametrize(
    "changed", ["structured_output_mode", "prompt_version", "model", "llm_thinking"]
)
def test_smoke_resume_refuses_records_from_a_different_run(tmp_path, changed) -> None:
    service, generator, llm, recorder, path = build_smoke([VALID, VALID], tmp_path)
    run(ITEMS[:1], service, generator, llm, recorder, path)

    other = {**IDENTITY, changed: "something-else"}
    with pytest.raises(ResumeIdentityError, match=changed):
        run(ITEMS[:2], service, generator, llm, recorder, path, identity=other)

    assert len(read_jsonl(path)) == 1  # 旧记录原样保留，没有混入新记录


def test_smoke_resume_refuses_legacy_records_without_identity(tmp_path) -> None:
    service, generator, llm, recorder, path = build_smoke([VALID], tmp_path)
    append_jsonl(path, {"question_id": "Q1", "status": "answered"})

    with pytest.raises(ResumeIdentityError):
        run(ITEMS[:2], service, generator, llm, recorder, path)


def test_check_resume_identity_accepts_matching_records(tmp_path) -> None:
    check_resume_identity([{**IDENTITY}], IDENTITY, tmp_path / "x.jsonl")


def test_probe_resume_refuses_records_from_a_different_model(tmp_path) -> None:
    policy = make_policy(FakeClock(), min_interval_seconds=0.0)
    jsonl = tmp_path / "probe.jsonl"
    per_mode = {m: [VALID] * len(REQUESTS) for m in ("json_schema", "json_object", "prompt_only")}
    run_probe(probe_client_factory(per_mode), policy, jsonl, **PROBE_ID)

    with pytest.raises(ResumeIdentityError, match="model"):
        run_probe(probe_client_factory(per_mode), policy, jsonl, model="other", llm_thinking="disabled")
    with pytest.raises(ResumeIdentityError, match="llm_thinking"):
        run_probe(probe_client_factory(per_mode), policy, jsonl, model="deepseek-flash", llm_thinking="auto")


def test_probe_records_carry_per_mode_identity(tmp_path) -> None:
    policy = make_policy(FakeClock(), min_interval_seconds=0.0)
    per_mode = {m: [VALID] * len(REQUESTS) for m in ("json_schema", "json_object", "prompt_only")}

    records = run_probe(probe_client_factory(per_mode), policy, tmp_path / "p.jsonl", **PROBE_ID)

    assert {r["structured_output_mode"] for r in records} == set(per_mode)
    assert all(r["structured_output_mode"] == r["mode"] for r in records)
    assert all(r["model"] == "deepseek-flash" and r["llm_thinking"] == "disabled" for r in records)


class _MaxAttempts2:
    max_attempts = 2


def test_policy_client_rejects_inner_client_with_internal_retries() -> None:
    policy = make_policy(FakeClock(), min_interval_seconds=0.0)

    with pytest.raises(LLMConfigurationError, match="max_attempts"):
        PolicyLLMClient(_MaxAttempts2(), policy)
    with pytest.raises(LLMConfigurationError):
        PolicyLLMClient(object(), policy)  # 没有 max_attempts 属性也不放行


def test_real_client_exposes_max_attempts() -> None:
    policy = make_policy(FakeClock(), min_interval_seconds=0.0)
    one = OpenAICompatibleLLMClient(
        api_key="k", base_url="http://localhost:1", model="m", max_attempts=1, _client=object()
    )
    assert one.max_attempts == 1
    PolicyLLMClient(one, policy)
    two = OpenAICompatibleLLMClient(
        api_key="k", base_url="http://localhost:1", model="m", max_attempts=2, _client=object()
    )
    assert two.max_attempts == 2
    with pytest.raises(LLMConfigurationError):
        PolicyLLMClient(two, policy)


class _CountingSdk:
    """替代 SDK 对象：每次 chat.completions.create 调用都计数，并抛出真实的 openai 异常。"""

    def __init__(self, error: Exception) -> None:
        self.error = error
        self.http_requests = 0
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.http_requests += 1
        raise self.error


def json_messages() -> list[LLMMessage]:
    """默认模式 json_object 要求 Prompt 含 "JSON"；否则请求在发出前就因配置错误失败。"""

    return [LLMMessage(role="user", content="Return a JSON object.")]


def _sdk_error(cls, status_code: int) -> Exception:
    import httpx

    request = httpx.Request("POST", "https://api.example.invalid/chat/completions")
    response = httpx.Response(status_code, request=request, json={"error": {"message": "x"}})
    return cls("boom", response=response, body=None)


def _real_policy_client(sdk, clock):
    inner = OpenAICompatibleLLMClient(
        api_key="k", base_url="https://api.example.invalid", model="m", max_attempts=1, _client=sdk
    )
    return PolicyLLMClient(inner, make_policy(clock, min_interval_seconds=0.0))


def test_real_sdk_403_sends_one_http_request_and_stops_the_script() -> None:
    import openai

    sdk = _CountingSdk(_sdk_error(openai.PermissionDeniedError, 403))
    client = _real_policy_client(sdk, FakeClock())
    client.begin()

    with pytest.raises(FatalServiceError):
        client.generate_json(json_messages(), schema_name="x", json_schema={})

    assert sdk.http_requests == 1


def test_real_sdk_429_is_retried_by_policy_only_and_classified_as_rate_limit_exhausted() -> None:
    import openai

    clock = FakeClock()
    sdk = _CountingSdk(_sdk_error(openai.RateLimitError, 429))
    client = _real_policy_client(sdk, clock)
    client.begin()

    with pytest.raises(LLMRequestError) as caught:
        client.generate_json(json_messages(), schema_name="x", json_schema={})

    assert sdk.http_requests == 4  # 1 次原始请求 + 策略的 3 次重试；客户端内部不再重试
    assert clock.sleeps == [30, 60, 120]
    assert failure_class(caught.value) == "rate_limit_exhausted"
    assert len(client.question_rate_limit_events) == 4  # 3 次等待 + 最后一次耗尽记录


def test_thinking_ignored_calls_counts_non_empty_reasoning_when_disabled() -> None:
    def call(non_empty):
        return {"reasoning_content_present": True, "reasoning_content_non_empty": non_empty}

    records = [
        {"llm_thinking": "disabled", "calls": [call(False), call(True)]},
        {"llm_thinking": "disabled", "calls": [call(True)]},
        {"llm_thinking": "auto", "calls": [call(True)]},  # 未关闭 thinking 时不算“被忽略”
        {"llm_thinking": "disabled", "error": None, **call(True)},  # 探测记录：本身就是一次调用
    ]

    assert thinking_ignored_calls(records) == 3
    assert summarize_smoke(records[:2])["thinking_ignored_calls"] == 2
