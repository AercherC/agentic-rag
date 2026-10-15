"""一次 RAG 提问的只读观测记录（RunTrace，PROJECT_SPEC §12.5）。

RunTrace 只用于观测，不进入 RAGAnswer 或任何现有 Domain Model；payload 只允许
§12.5 白名单内的 ID、计数、分数、耗时与 token 用量，不含任何正文或密钥。
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

RUN_TRACE_SCHEMA_VERSION = "run-trace-v1"


class TraceEvent(BaseModel):
    """RunTrace 中的单个事件；名称取自 §12.5 的事件名列表。"""

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
    """一次提问的完整事件序列与身份信息。"""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["run-trace-v1"] = Field(
        default=RUN_TRACE_SCHEMA_VERSION, description="RunTrace JSON 格式版本。"
    )
    run_id: str = Field(min_length=1, description="调用方传入的 run 标识，未传时为 uuid4。")
    # 只保存 query 的 UTF-8 sha256，不保存原文，避免 trace 文件泄露用户问题。
    query_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$", description="UTF-8 编码 query 的 sha256 十六进制摘要。"
    )
    started_at_utc: str = Field(description="run 开始时刻，ISO 8601 UTC。")
    completed_at_utc: str = Field(description="finish() 时刻，ISO 8601 UTC。")
    status: Literal["completed", "failed"] = Field(
        description="最后一个事件为 run.completed 时为 completed，为 run.failed 时为 failed。"
    )
    events: list[TraceEvent] = Field(description="按 seq 升序排列的事件。")
