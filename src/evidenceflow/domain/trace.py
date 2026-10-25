"""Read-only observation record of one RAG question (RunTrace, PROJECT_SPEC §12.5).

RunTrace is used for observation only and never enters RAGAnswer or any existing Domain
Model; payloads may only carry the IDs, counters, scores, durations, and token usage on the
§12.5 whitelist, and never any body text or secrets.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

RUN_TRACE_SCHEMA_VERSION = "run-trace-v1"


class TraceEvent(BaseModel):
    """A single event in a RunTrace; the name comes from the event-name list in §12.5."""

    model_config = ConfigDict(extra="forbid")

    seq: int = Field(ge=0, description="事件序号，同一 run 内从 0 起连续递增。")
    name: str = Field(min_length=1, description="事件名，如 run.started、rerank.completed。")
    elapsed_ms: float = Field(ge=0, description="相对 run 开始（start_run）的毫秒数，单调时钟。")
    duration_ms: float | None = Field(
        default=None,
        description="仅 X.completed 事件有值：与最近一个 X.started 的间隔（毫秒）；其余事件为 null。",
    )
    payload: dict[str, Any] = Field(
        default_factory=dict, description="§12.5 白名单内的字段；未列出的事件为空字典。"
    )


class RunTrace(BaseModel):
    """The full event sequence and identity information of one question."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["run-trace-v1"] = Field(
        default=RUN_TRACE_SCHEMA_VERSION, description="RunTrace JSON 格式版本。"
    )
    run_id: str = Field(min_length=1, description="调用方传入的 run 标识，未传时为 uuid4。")
    # Store only the UTF-8 sha256 of the query, never the raw text, so trace files cannot leak user
    # questions.
    query_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$", description="UTF-8 编码 query 的 sha256 十六进制摘要。"
    )
    started_at_utc: str = Field(description="run 开始时刻，ISO 8601 UTC。")
    completed_at_utc: str = Field(description="finish() 时刻，ISO 8601 UTC。")
    status: Literal["completed", "failed"] = Field(
        description="最后一个事件为 run.completed 时为 completed，为 run.failed 时为 failed。"
    )
    events: list[TraceEvent] = Field(description="按 seq 升序排列的事件。")
