"""Phase 15A 最小 BuildJob 状态模型与本地持久化。"""

from __future__ import annotations

import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field

BUILD_JOB_STATUSES = (
    "PENDING",
    "SCANNING",
    "PARSING",
    "CHUNKING",
    "EMBEDDING",
    "INDEXING",
    "VALIDATING",
    "READY",
    "ACTIVATED",
    "FAILED",
    "BLOCKED",
)
BuildJobStatus = Literal[
    "PENDING",
    "SCANNING",
    "PARSING",
    "CHUNKING",
    "EMBEDDING",
    "INDEXING",
    "VALIDATING",
    "READY",
    "ACTIVATED",
    "FAILED",
    "BLOCKED",
]

# 同步构建允许的相邻状态；FAILED/BLOCKED 是终态，避免状态记录掩盖真实失败。
_ALLOWED_TRANSITIONS: dict[BuildJobStatus, frozenset[BuildJobStatus]] = {
    "PENDING": frozenset({"SCANNING", "FAILED", "BLOCKED"}),
    "SCANNING": frozenset({"PARSING", "FAILED", "BLOCKED"}),
    "PARSING": frozenset({"CHUNKING", "FAILED", "BLOCKED"}),
    "CHUNKING": frozenset({"EMBEDDING", "FAILED", "BLOCKED"}),
    "EMBEDDING": frozenset({"INDEXING", "FAILED", "BLOCKED"}),
    "INDEXING": frozenset({"VALIDATING", "FAILED", "BLOCKED"}),
    "VALIDATING": frozenset({"READY", "FAILED", "BLOCKED"}),
    "READY": frozenset({"ACTIVATED", "FAILED", "BLOCKED"}),
    "ACTIVATED": frozenset(),
    "FAILED": frozenset(),
    "BLOCKED": frozenset(),
}


def _utc_now() -> datetime:
    """返回带时区的 UTC 时间，便于不同机器审计同一 Job。"""

    return datetime.now(timezone.utc)


class BuildJob(BaseModel):
    """一次同步 staging build 的可审计状态，不承诺异步 worker 语义。"""

    job_id: str = Field(min_length=1, description="本次构建运行的唯一 Job ID。")
    knowledge_base_id: str = Field(min_length=1, description="所属知识库逻辑身份。")
    source_corpus_signature: str = Field(min_length=1, description="构建输入 corpus signature。")
    target_index_signature: str = Field(min_length=1, description="构建目标 index signature。")
    target_collection_name: str = Field(min_length=1, description="构建目标 STAGING Collection。")
    status: BuildJobStatus = Field(description="当前同步构建状态。")
    created_at: datetime = Field(default_factory=_utc_now, description="Job 创建时间。")
    started_at: datetime | None = Field(default=None, description="开始实际处理的时间。")
    finished_at: datetime | None = Field(default=None, description="进入终态或 READY 的时间。")
    error_code: str | None = Field(default=None, description="FAILED/BLOCKED 的稳定错误分类。")
    error_message: str | None = Field(default=None, description="FAILED/BLOCKED 的诊断信息。")
    document_count: int | None = Field(default=None, ge=0, description="构建输出的文档数。")
    chunk_count: int | None = Field(default=None, ge=0, description="构建输出的 Chunk 数。")

    def transition(
        self,
        status: BuildJobStatus,
        *,
        now: datetime | None = None,
        error_code: str | None = None,
        error_message: str | None = None,
        document_count: int | None = None,
        chunk_count: int | None = None,
    ) -> "BuildJob":
        """生成下一状态 Job，并拒绝跳过中间阶段或复用终态。"""

        if status not in _ALLOWED_TRANSITIONS[self.status]:
            raise ValueError(f"invalid BuildJob transition: {self.status} -> {status}")
        timestamp = now or _utc_now()
        values = self.model_dump()
        values["status"] = status
        if status != "PENDING" and self.started_at is None:
            values["started_at"] = timestamp
        if status in {"READY", "ACTIVATED", "FAILED", "BLOCKED"}:
            values["finished_at"] = timestamp
        if error_code is not None:
            values["error_code"] = error_code
        if error_message is not None:
            values["error_message"] = error_message
        if document_count is not None:
            values["document_count"] = document_count
        if chunk_count is not None:
            values["chunk_count"] = chunk_count
        return BuildJob(**values)


class BuildJobStore:
    """把 BuildJob 以单个 JSON 文件原子保存，供审计和未来 UI 读取。"""

    def __init__(self, path: str | Path) -> None:
        """创建一个指向具体 Job JSON 文件的存储器。"""

        self.path = Path(path)

    def save(self, job: BuildJob) -> None:
        """使用同目录临时文件替换目标，避免留下半截状态 JSON。"""

        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = job.model_dump_json(indent=2, exclude_none=False) + "\n"
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{self.path.name}.",
            suffix=".tmp",
            dir=self.path.parent,
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_name, self.path)
        except Exception:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass
            raise

    def load(self) -> BuildJob:
        """读取并校验一个已持久化的 BuildJob。"""

        return BuildJob.model_validate_json(self.path.read_text(encoding="utf-8"))


class BuildJobTracker:
    """在同步流程中推进并持久化真实 BuildJob 状态。"""

    def __init__(self, job: BuildJob, store: BuildJobStore) -> None:
        """注入当前 Job 和持久化后端；不启动任何后台任务。"""

        self.job = job
        self.store = store
        self.store.save(job)

    @classmethod
    def create(
        cls,
        *,
        knowledge_base_id: str,
        source_corpus_signature: str,
        target_index_signature: str,
        target_collection_name: str,
        store: BuildJobStore,
    ) -> "BuildJobTracker":
        """创建 PENDING Job 并立即落盘，保证构建从可审计状态开始。"""

        job = BuildJob(
            job_id=f"job-{uuid4().hex}",
            knowledge_base_id=knowledge_base_id,
            source_corpus_signature=source_corpus_signature,
            target_index_signature=target_index_signature,
            target_collection_name=target_collection_name,
            status="PENDING",
        )
        return cls(job, store)

    def advance(self, status: BuildJobStatus, **kwargs: object) -> BuildJob:
        """推进一个合法状态并保存最新 Job。"""

        self.job = self.job.transition(status, **kwargs)
        self.store.save(self.job)
        return self.job

    def fail(self, error_code: str, error_message: str) -> BuildJob:
        """将 Job 标记为 FAILED，保留失败原因而不伪造 READY。"""

        return self.advance(
            "FAILED",
            error_code=error_code,
            error_message=error_message,
        )

    def block(self, error_code: str, error_message: str) -> BuildJob:
        """将因外部前提或身份歧义停止的 Job 标记为 BLOCKED。"""

        return self.advance(
            "BLOCKED",
            error_code=error_code,
            error_message=error_message,
        )
