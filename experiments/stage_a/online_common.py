"""A2b 在线脚本的公共层：请求节流、限流重试、403 立即停止、断点续跑与原始回复留存。

这里的逻辑都是纯 Python，可用 Fake client 做离线测试；真实网络调用只发生在
``probe_structured_output.py`` 与 ``run_generation_smoke.py`` 的 ``main`` 中。
"""

from __future__ import annotations

import json
import re
import statistics
import time
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

from evidenceflow.citation import CitationValidationError
from evidenceflow.domain import LLMMessage
from evidenceflow.generation import GenerationContractError
from evidenceflow.llm import LLMConfigurationError, LLMRequestError
from evidenceflow.llm.diagnostics import classify_llm_failure


class FatalServiceError(RuntimeError):
    """403 / 配置类错误：继续请求只会白白消耗次数，整个脚本必须立即停止。"""


class RequestPolicy:
    """统一的请求策略：最小间隔节流 + 429 等待重试 + 配置类错误立即停止。

    - ``min_interval_seconds``：两次请求开始时间之间的最小间隔（含修复调用与 429 重试）。
    - 429：按 ``backoff_seconds`` 依次等待后重试，最多 ``max_rate_limit_retries`` 次；每次都记入
      ``rate_limit_events``（与 LLMRequestError 分开统计）。重试耗尽后抛出原异常，并在异常上标记
      ``rate_limit_exhausted=True``。
    - 403/401 或 ``LLMConfigurationError``：抛出 ``FatalServiceError``，不重试。
    ``sleep`` 与 ``clock`` 可注入，测试无需真实等待。
    """

    def __init__(
        self,
        *,
        min_interval_seconds: float = 6.0,
        max_rate_limit_retries: int = 3,
        backoff_seconds: Sequence[float] = (30.0, 60.0, 120.0),
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.min_interval_seconds = min_interval_seconds
        self.max_rate_limit_retries = max_rate_limit_retries
        self.backoff_seconds = tuple(backoff_seconds)
        self._sleep = sleep
        self._clock = clock
        self._last_started: float | None = None
        self.rate_limit_events: list[dict[str, Any]] = []
        self.request_count = 0

    def _throttle(self) -> None:
        """距上一次请求开始不足最小间隔时补足等待。"""

        if self._last_started is not None:
            wait = self.min_interval_seconds - (self._clock() - self._last_started)
            if wait > 0:
                self._sleep(wait)
        self._last_started = self._clock()

    def call(self, fn: Callable[[], Any], *, label: str = "") -> Any:
        """在策略下执行一次逻辑请求。"""

        attempt = 0
        while True:
            self._throttle()
            self.request_count += 1
            try:
                return fn()
            except Exception as exc:  # 需要按类别分流，未知异常原样上抛
                kind = classify_llm_failure(exc)
                if kind == "configuration" or isinstance(exc, LLMConfigurationError):
                    raise FatalServiceError(
                        f"service rejected the request ({kind}); stopping to avoid wasting requests"
                    ) from exc
                if kind != "rate_limit":
                    raise
                exhausted = attempt >= self.max_rate_limit_retries
                wait = 0.0 if exhausted else self.backoff_seconds[min(attempt, len(self.backoff_seconds) - 1)]
                self.rate_limit_events.append(
                    {"label": label, "attempt": attempt + 1, "wait_seconds": wait, "exhausted": exhausted}
                )
                if exhausted:
                    exc.rate_limit_exhausted = True  # type: ignore[attr-defined]
                    raise
                self._sleep(wait)
                attempt += 1


class PolicyLLMClient:
    """包装真实 LLM Client：套用 RequestPolicy，并留存每次调用的原始回复与诊断。

    ``calls`` 保存当前题目的全部调用（含 Generator 的修复调用）：消息、原始回复、usage、
    reasoning_content 是否存在、耗时；失败的调用记录错误类别。``begin()`` 开始新的一题。
    """

    def __init__(self, inner: Any, policy: RequestPolicy) -> None:
        self._inner = inner
        self.policy = policy
        self.calls: list[dict[str, Any]] = []
        self._last_usage: dict | None = None

    def begin(self) -> None:
        """开始新的一题：清空调用记录，并重置本题的限流事件起点。"""

        self.calls = []
        self._last_usage = None
        self._events_start = len(self.policy.rate_limit_events)

    @property
    def question_rate_limit_events(self) -> list[dict[str, Any]]:
        """本题发生的限流事件（begin 之后新增的部分）。"""

        return list(self.policy.rate_limit_events[getattr(self, "_events_start", 0) :])

    @property
    def last_usage(self) -> dict | None:
        return self._last_usage

    def generate(self, messages: Sequence[LLMMessage]) -> str:
        return self._invoke("generate", messages, lambda: self._inner.generate(messages))

    def generate_json(
        self, messages: Sequence[LLMMessage], *, schema_name: str, json_schema: dict
    ) -> str:
        return self._invoke(
            "generate_json",
            messages,
            lambda: self._inner.generate_json(
                messages, schema_name=schema_name, json_schema=json_schema
            ),
        )

    def _invoke(self, kind: str, messages: Sequence[LLMMessage], fn: Callable[[], str]) -> str:
        started = time.perf_counter()
        record: dict[str, Any] = {
            "kind": kind,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
        }
        self.calls.append(record)
        try:
            raw = self.policy.call(fn, label=kind)
        except FatalServiceError:
            record["error"] = "fatal_service_error"
            raise
        except Exception as exc:
            record["error"] = classify_llm_failure(exc)
            record["error_type"] = type(exc).__name__
            record["rate_limit_exhausted"] = bool(getattr(exc, "rate_limit_exhausted", False))
            self._last_usage = None
            raise
        finally:
            record["elapsed_seconds"] = round(time.perf_counter() - started, 3)
        self._last_usage = getattr(self._inner, "last_usage", None)
        info = getattr(self._inner, "last_response_info", None) or {}
        record.update(
            raw=raw,
            usage=self._last_usage,
            reasoning_content_present=info.get("reasoning_content_present"),
            reasoning_content_non_empty=info.get("reasoning_content_non_empty"),
            finish_reason=info.get("finish_reason"),
        )
        return raw


_LABEL_HEADER = re.compile(r"^\[S(\d+)\]$(?=\nFile: )", re.MULTILINE)


def extract_allowed_labels(messages: Sequence[dict[str, str]]) -> list[str]:
    """从发给模型的 user 消息里提取 Context 的来源标签（与 ContextBuilder 的头部格式一致）。"""

    user = next((m["content"] for m in messages if m["role"] == "user"), "")
    return sorted({f"[S{n}]" for n in _LABEL_HEADER.findall(user)}, key=lambda s: int(s[2:-1]))


# ---- JSONL 断点续跑 -------------------------------------------------------------------------


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    """每完成一项就追加写入一行；即使中途被 403 终止，已完成的结果也不会丢。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """读取全部已完成记录；文件不存在视为空。损坏的最后一行（被中断的写入）被忽略。"""

    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except ValueError:
            continue
    return records


def rewrite_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    """用保留的记录重写 JSONL（仅用于 --retry-llm-errors 清理待重跑项）。"""

    path.write_text(
        "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in records),
        encoding="utf-8",
    )


def pending(items: Iterable[Any], done_keys: set[str], key: Callable[[Any], str]) -> list[Any]:
    """跳过已完成项，保持原顺序。"""

    return [item for item in items if key(item) not in done_keys]


# ---- 冒烟统计（在线冒烟与离线重解析共用，格式一致）------------------------------------------


def failure_class(exc: BaseException | None) -> str | None:
    """把异常归入互不重叠的类别；限流耗尽与 LLMRequestError 分开统计。"""

    if exc is None:
        return None
    if getattr(exc, "rate_limit_exhausted", False) or getattr(
        getattr(exc, "__cause__", None), "rate_limit_exhausted", False
    ):
        return "rate_limit_exhausted"
    if isinstance(exc, GenerationContractError):
        return "generation_contract_error"
    if isinstance(exc, CitationValidationError):
        return "citation_validation_error"
    if isinstance(exc, LLMRequestError):
        return "llm_request_error"
    return "other"


def summarize_smoke(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """汇总冒烟记录：状态分布、格式失败、修复、Citation、服务错误、限流、usage、耗时。"""

    def count(cls: str) -> int:
        return sum(1 for r in records if r.get("failure_class") == cls)

    statuses: dict[str, int] = {}
    for r in records:
        if r.get("status"):
            statuses[r["status"]] = statuses.get(r["status"], 0) + 1
    diag = [r.get("diagnostics") or {} for r in records]
    prompt = completion = 0
    usage_missing = 0
    for d in diag:
        for usage in d.get("usages", []):
            if usage is None:
                usage_missing += 1
                continue
            prompt += usage.get("prompt_tokens") or 0
            completion += usage.get("completion_tokens") or 0
    elapsed = [r["elapsed_seconds"] for r in records if "elapsed_seconds" in r]
    return {
        "question_count": len(records),
        "status_counts": statuses,
        "generation_contract_errors": count("generation_contract_error"),
        "repair_attempted": sum(1 for d in diag if d.get("repair_attempted")),
        "repair_succeeded": sum(1 for d in diag if d.get("repair_succeeded")),
        "citation_validation_errors": count("citation_validation_error"),
        "llm_request_errors": count("llm_request_error"),
        "rate_limit_exhausted": count("rate_limit_exhausted"),
        "rate_limit_events": sum(len(r.get("rate_limit_events") or []) for r in records),
        "other_errors": count("other"),
        "llm_calls": sum(len(d.get("usages", [])) for d in diag),
        "usage": {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "calls_without_usage": usage_missing,
        },
        "latency_seconds": {
            "median": statistics.median(elapsed) if elapsed else None,
            "max": max(elapsed) if elapsed else None,
        },
        "error_questions": {
            r["question_id"]: {"class": r["failure_class"], "type": r.get("exception_type")}
            for r in records
            if r.get("failure_class")
        },
    }
