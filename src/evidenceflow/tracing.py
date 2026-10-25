"""RunTrace recorder: the TraceRecorder protocol, a no-op implementation, an in-memory
implementation and a safe recording helper (PROJECT_SPEC §12.5).

This module only observes; it never changes any answer or exception propagation:
- The components (DenseRerankerRetriever, Generator, RAGService) default ``trace_recorder`` to
  ``None``, in which case the caller does not build a payload at all, so the overhead matches an
  untraced deployment;
- Components always record through ``safe_record``, and any exception from the recorder itself is
  swallowed with only a warning log line.
Importing ``evidenceflow.eval`` is not allowed (runtime-side modules must not depend on the
evaluation layer in reverse).
"""

from __future__ import annotations

import hashlib
import logging
import time
import uuid
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any, Protocol

from evidenceflow.domain.trace import RunTrace, TraceEvent

logger = logging.getLogger(__name__)

_COMPLETED_SUFFIX = ".completed"
_STARTED_SUFFIX = ".started"


class TraceRecorder(Protocol):
    """The minimal interface components need to record trace events."""

    def record(self, name: str, **payload: Any) -> None:
        """Record one event; the payload may only contain the §12.5 allowlisted fields."""


class NullTraceRecorder:
    """A recorder that does nothing, for callers to use explicitly (components default to None,
    not to this)."""

    def record(self, name: str, **payload: Any) -> None:
        """Discard the event."""


def safe_record(recorder: TraceRecorder | None, name: str, **payload: Any) -> None:
    """Record one event safely: return immediately when the recorder is None; if the recorder
    raises, log only the exception class name and swallow it.

    Tracing is side-channel observation, so a defect in the recorder must never change the answer
    or the original exception propagation path. The log carries only the exception class name, not
    the message, to avoid writing potentially content-bearing text into the log.
    """

    if recorder is None:
        return
    try:
        recorder.record(name, **payload)
    except Exception as exc:  # noqa: BLE001 - observation failure must not affect the main flow
        logger.warning(
            "trace recorder failed", extra={"trace_event": name, "error_type": type(exc).__name__}
        )


def _utc_now() -> datetime:
    """The current UTC instant (replaceable with a fixed clock in tests)."""

    return datetime.now(timezone.utc)


class RunTraceRecorder:
    """Keep one run's events in memory and produce a ``RunTrace`` on ``finish``.

    Usage: ``start_run(query)`` -> components call ``record`` -> ``finish()``. The same instance
    can be reused across runs, but ``start_run`` must be called again before each run; earlier
    events are cleared.
    """

    def __init__(
        self,
        *,
        clock: Callable[[], float] = time.perf_counter,
        utc_now: Callable[[], datetime] = _utc_now,
    ) -> None:
        """Inject a monotonic clock (seconds) and a UTC clock; the defaults are
        ``time.perf_counter`` and the system time."""

        self._clock = clock
        self._utc_now = utc_now
        self._started = False
        self._run_id = ""
        self._query_sha256 = ""
        self._started_at_utc = ""
        self._origin = 0.0
        self._events: list[TraceEvent] = []
        # Record the elapsed_ms of the latest started event for each stage prefix, used to compute
        # duration_ms for X.completed.
        self._last_started_ms: dict[str, float] = {}

    def start_run(self, query: str, *, run_id: str | None = None) -> None:
        """Start a new run: reset the events and record the query digest, the start instant and the
        monotonic clock origin."""

        self._run_id = run_id if run_id is not None else str(uuid.uuid4())
        # Store only the sha256 of the UTF-8 query; the raw query never appears in the trace.
        self._query_sha256 = hashlib.sha256(query.encode("utf-8")).hexdigest()
        self._started_at_utc = self._utc_now().isoformat()
        self._origin = self._clock()
        self._events = []
        self._last_started_ms = {}
        self._started = True

    def record(self, name: str, **payload: Any) -> None:
        """Append one event; a call before ``start_run`` is only logged and dropped, never
        raised."""

        if not self._started:
            logger.warning("trace event recorded before start_run; dropped", extra={"trace_event": name})
            return
        # elapsed_ms is based on the monotonic clock and is unaffected by system time adjustments.
        elapsed_ms = (self._clock() - self._origin) * 1000.0
        duration_ms: float | None = None
        if name.endswith(_STARTED_SUFFIX):
            self._last_started_ms[name[: -len(_STARTED_SUFFIX)]] = elapsed_ms
        elif name.endswith(_COMPLETED_SUFFIX):
            # A duration exists only when a matching X.started is present
            # (citation_validation.completed has no started event).
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
        """A copy of the events recorded so far in the current run (read-only view)."""

        return [event.model_copy(deep=True) for event in self._events]

    def finish(self) -> RunTrace:
        """Finish the current run and return the RunTrace; the last event must be run.completed or
        run.failed."""

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
