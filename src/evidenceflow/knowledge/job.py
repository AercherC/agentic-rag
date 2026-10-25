"""Minimal Phase 15A BuildJob state model and local persistence."""

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

# Neighbouring statuses allowed by a synchronous build; FAILED/BLOCKED are terminal so the status
# record cannot hide a real failure.
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
    """Return a timezone-aware UTC timestamp so the same Job can be audited from different machines."""

    return datetime.now(timezone.utc)


class BuildJob(BaseModel):
    """Auditable state of one synchronous staging build; no async worker semantics are promised."""

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
        """Produce the next-state Job, refusing to skip intermediate stages or reuse a terminal state."""

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
    """Persist a BuildJob as a single JSON file atomically, for auditing and future UI reads."""

    def __init__(self, path: str | Path) -> None:
        """Create a store pointing at a concrete Job JSON file."""

        self.path = Path(path)

    def save(self, job: BuildJob) -> None:
        """Replace the target using a temporary file in the same directory, so no half-written state JSON is left behind."""

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
        """Read and validate a persisted BuildJob."""

        return BuildJob.model_validate_json(self.path.read_text(encoding="utf-8"))


class BuildJobTracker:
    """Advance and persist real BuildJob state inside a synchronous flow."""

    def __init__(self, job: BuildJob, store: BuildJobStore) -> None:
        """Inject the current Job and the persistence backend; starts no background task."""

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
        """Create a PENDING Job and write it to disk immediately, so the build starts from an auditable state."""

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
        """Advance to a legal status and save the latest Job."""

        self.job = self.job.transition(status, **kwargs)
        self.store.save(self.job)
        return self.job

    def fail(self, error_code: str, error_message: str) -> BuildJob:
        """Mark the Job FAILED, preserving the failure reason instead of faking READY."""

        return self.advance(
            "FAILED",
            error_code=error_code,
            error_message=error_message,
        )

    def block(self, error_code: str, error_message: str) -> BuildJob:
        """Mark a Job BLOCKED when it stopped on an external precondition or ambiguous identity."""

        return self.advance(
            "BLOCKED",
            error_code=error_code,
            error_message=error_message,
        )
