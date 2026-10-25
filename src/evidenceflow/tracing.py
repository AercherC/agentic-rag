"""RunTrace 记录器：TraceRecorder 协议、空实现、内存实现与安全记录函数（PROJECT_SPEC §12.5）。

本模块只观测，不改变任何回答或异常传播：
- 组件（DenseRerankerRetriever、Generator、RAGService）的 ``trace_recorder`` 默认为 ``None``，
  此时调用方根本不构造 payload，开销与未接入 trace 时一致；
- 组件一律经 ``safe_record`` 记录，recorder 自身的任何异常都被吞掉，只留一条 warning 日志。
- 共享组件服务并发请求时，用 ``trace_scope`` 为当前线程/协程指定本次 run 的 recorder，
  组件经 ``resolve_recorder`` 取得它；构造时显式传入的 recorder 优先。
不得 import ``evidenceflow.eval``（runtime 侧模块不反向依赖评测层）。
"""

from __future__ import annotations

import hashlib
import logging
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any, Protocol

from evidenceflow.domain.trace import RunTrace, TraceEvent

logger = logging.getLogger(__name__)

_COMPLETED_SUFFIX = ".completed"
_STARTED_SUFFIX = ".started"
_ACTIVE_RECORDER: ContextVar["TraceRecorder | None"] = ContextVar(
    "evidenceflow_active_trace_recorder", default=None
)


class TraceRecorder(Protocol):
    """组件记录 trace 事件所需的最小接口。"""

    def record(self, name: str, **payload: Any) -> None:
        """记录一个事件；payload 只能包含 §12.5 白名单字段。"""


class NullTraceRecorder:
    """不做任何事的 recorder，供调用方显式使用（组件默认是 None 而不是它）。"""

    def record(self, name: str, **payload: Any) -> None:
        """丢弃事件。"""


@contextmanager
def trace_scope(recorder: TraceRecorder | None) -> Iterator[TraceRecorder | None]:
    """在当前上下文内把 ``recorder`` 设为本次 run 的 recorder，退出时恢复原值。

    ContextVar 按线程与 asyncio 任务隔离，因此多个请求可以共享同一组 Runtime 组件，
    各自的事件只进入各自的 recorder，而不必在请求之间替换组件上的 ``trace_recorder``。
    """

    token = _ACTIVE_RECORDER.set(recorder)
    try:
        yield recorder
    finally:
        _ACTIVE_RECORDER.reset(token)


def resolve_recorder(explicit: TraceRecorder | None) -> TraceRecorder | None:
    """组件构造时显式传入的 recorder 优先；否则使用 ``trace_scope`` 设定的当前 recorder。"""

    return explicit if explicit is not None else _ACTIVE_RECORDER.get()


def safe_record(recorder: TraceRecorder | None, name: str, **payload: Any) -> None:
    """安全地记录一个事件：recorder 为 None 时直接返回；recorder 抛错只记异常类名后吞掉。

    trace 是旁路观测，recorder 的缺陷绝不能改变回答或原有异常的传播路径；
    日志只写异常类名，不写消息，避免把可能含有正文的内容写进日志。
    """

    if recorder is None:
        return
    try:
        recorder.record(name, **payload)
    except Exception as exc:  # noqa: BLE001 - 观测失败不得影响主流程
        logger.warning(
            "trace recorder failed", extra={"trace_event": name, "error_type": type(exc).__name__}
        )


def _utc_now() -> datetime:
    """当前 UTC 时刻（可在测试中替换为固定时钟）。"""

    return datetime.now(timezone.utc)


class RunTraceRecorder:
    """把一次 run 的事件保存在内存中，并在 ``finish`` 时产出 ``RunTrace``。

    用法：``start_run(query)`` → 组件调用 ``record`` → ``finish()``。同一实例可跨 run 复用，
    每次 run 前必须重新 ``start_run``；之前的事件会被清空。
    """

    def __init__(
        self,
        *,
        clock: Callable[[], float] = time.perf_counter,
        utc_now: Callable[[], datetime] = _utc_now,
    ) -> None:
        """注入单调时钟（秒）与 UTC 时钟，默认使用 ``time.perf_counter`` 与系统时间。"""

        self._clock = clock
        self._utc_now = utc_now
        self._started = False
        self._run_id = ""
        self._query_sha256 = ""
        self._started_at_utc = ""
        self._origin = 0.0
        self._events: list[TraceEvent] = []
        # 记录每个阶段前缀最近一次 started 的 elapsed_ms，用于计算 X.completed 的 duration_ms。
        self._last_started_ms: dict[str, float] = {}

    def start_run(self, query: str, *, run_id: str | None = None) -> None:
        """开始一次新 run：重置事件并记录 query 摘要、开始时刻与单调时钟起点。"""

        self._run_id = run_id if run_id is not None else str(uuid.uuid4())
        # 只保存 UTF-8 query 的 sha256，trace 中不出现 query 原文。
        self._query_sha256 = hashlib.sha256(query.encode("utf-8")).hexdigest()
        self._started_at_utc = self._utc_now().isoformat()
        self._origin = self._clock()
        self._events = []
        self._last_started_ms = {}
        self._started = True

    def record(self, name: str, **payload: Any) -> None:
        """追加一个事件；``start_run`` 之前的调用只记日志并丢弃，不抛异常。"""

        if not self._started:
            logger.warning("trace event recorded before start_run; dropped", extra={"trace_event": name})
            return
        # elapsed_ms 基于单调时钟，不受系统时间调整影响。
        elapsed_ms = (self._clock() - self._origin) * 1000.0
        duration_ms: float | None = None
        if name.endswith(_STARTED_SUFFIX):
            self._last_started_ms[name[: -len(_STARTED_SUFFIX)]] = elapsed_ms
        elif name.endswith(_COMPLETED_SUFFIX):
            # 只有存在配对的 X.started 时才有 duration（citation_validation.completed 没有 started）。
            started_ms = self._last_started_ms.get(name[: -len(_COMPLETED_SUFFIX)])
            if started_ms is not None:
                duration_ms = elapsed_ms - started_ms
        self._events.append(
            TraceEvent(
                seq=len(self._events),
                name=name,
                elapsed_ms=elapsed_ms,
                duration_ms=duration_ms,
                payload=dict(payload),
            )
        )

    @property
    def events(self) -> list[TraceEvent]:
        """当前 run 已记录事件的副本（只读视图）。"""

        return [event.model_copy(deep=True) for event in self._events]

    def finish(self) -> RunTrace:
        """结束当前 run 并返回 RunTrace；最后一个事件必须是 run.completed 或 run.failed。"""

        if not self._events:
            raise ValueError("run trace has no events; cannot finish")
        last = self._events[-1].name
        if last == "run.completed":
            status = "completed"
        elif last == "run.failed":
            status = "failed"
        else:
            raise ValueError(f"run trace must end with run.completed or run.failed, got {last!r}")
        return RunTrace(
            run_id=self._run_id,
            query_sha256=self._query_sha256,
            started_at_utc=self._started_at_utc,
            completed_at_utc=self._utc_now().isoformat(),
            status=status,
            events=[event.model_copy(deep=True) for event in self._events],
        )
