"""Phase 14A 沙盒边界、哈希、产物 IO 与 STOP-ON-ANOMALY 守卫。

本模块是 experiments/phase13c/sandbox.py 的**平行复用版本**：逻辑完全一致，仅把根目录
改为 Phase 14A 的沙盒与产物根。任何写入都必须经过 save_json/save_text，且目标必须落在
允许根目录内（沙盒目录 或 可审计产物目录），否则抛 SandboxViolation。

写操作白名单根：
1. ``<TEMP>/evidenceflow_phase14a`` —— 重资产（venv / 候选 parser 原始输出 / 缓存）。
2. ``<PROJECT>/evals/evaluation_v3/phase14a_multiformat_ingestion_benchmark`` —— 可审计产物。

禁写：src/、tests/、data/、tmp/、PROJECT_SPEC.md、pyproject.toml、AGENTS.md、任何 Chroma 目录。
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

# --------------------------------------------------------------------------------------
# 路径定义（仅此处与 13C 不同）
# --------------------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SANDBOX_ROOT = Path(
    os.environ.get(
        "EF_PHASE14A_SANDBOX",
        Path(os.environ.get("TEMP", "/tmp")) / "evidenceflow_phase14a",
    )
).resolve()

ARTIFACT_ROOT = PROJECT_ROOT / "evals/evaluation_v3/phase14a_multiformat_ingestion_benchmark"

# 只读的项目资产（任何情况下不得写）。
PROJECT_SOURCE_DOCS = PROJECT_ROOT / "data/source_docs"
PROJECT_CHROMA_DIR = PROJECT_ROOT / "data/chroma"
FROZEN_PROJECT_FILES = (
    PROJECT_ROOT / "PROJECT_SPEC.md",
    PROJECT_ROOT / "pyproject.toml",
    PROJECT_ROOT / "AGENTS.md",
)

_WRITE_ALLOWED_ROOTS: tuple[Path, ...] = (SANDBOX_ROOT, ARTIFACT_ROOT)

_WRITE_FORBIDDEN_PREFIXES: tuple[Path, ...] = (
    PROJECT_SOURCE_DOCS,
    PROJECT_CHROMA_DIR,
    PROJECT_ROOT / "src",
    PROJECT_ROOT / "tests",
    PROJECT_ROOT / "data",
    PROJECT_ROOT / "tmp",
)


class SandboxViolation(RuntimeError):
    """表示某次操作越过了 Phase 14A 的沙盒边界。"""


class AnomalyStop(RuntimeError):
    """表示命中 STOP-ON-ANOMALY（方案 §35），当前 arm 必须立即停止。"""

    def __init__(self, arm: str, clause: str, detail: str) -> None:
        super().__init__(f"[{arm}] {clause}: {detail}")
        self.arm = arm
        self.clause = clause
        self.detail = detail


# --------------------------------------------------------------------------------------
# 通用哈希 / 时间
# --------------------------------------------------------------------------------------


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(payload: Any) -> str:
    return sha256_text(json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str))


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# --------------------------------------------------------------------------------------
# 边界校验与 IO
# --------------------------------------------------------------------------------------


def _ensure_writable(path: Path) -> None:
    resolved = Path(path).resolve()
    for forbidden in _WRITE_FORBIDDEN_PREFIXES:
        if resolved == forbidden or forbidden in resolved.parents:
            raise SandboxViolation(f"Write to forbidden project path: {resolved}")
    for allowed in _WRITE_ALLOWED_ROOTS:
        if resolved == allowed or allowed in resolved.parents:
            return
    raise SandboxViolation(
        f"Write outside sandbox roots: {resolved}; allowed roots={_WRITE_ALLOWED_ROOTS}"
    )


def ensure_dir(path: Path) -> Path:
    target = Path(path)
    _ensure_writable(target)
    target.mkdir(parents=True, exist_ok=True)
    return target


def save_json(payload: Any, path: Path) -> Path:
    target = Path(path)
    ensure_dir(target.parent)
    target.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False, default=str) + "\n",
        encoding="utf-8",
    )
    return target


def save_text(text: str, path: Path) -> Path:
    target = Path(path)
    ensure_dir(target.parent)
    target.write_text(text, encoding="utf-8")
    return target


def load_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_anomaly_report(arm: str, clause: str, detail: str) -> Path:
    body = "\n".join(
        [
            f"# Phase 14A Anomaly Report — {arm}",
            "",
            f"- **时间**：{utc_now()}",
            f"- **arm**：`{arm}`",
            f"- **触发条款**：{clause}",
            f"- **细节**：{detail}",
            "- **处理**：立即停止当前 arm（方案 §35）；其余 arm 在公平性未被破坏时继续。",
            "",
        ]
    )
    report = SANDBOX_ROOT / "failures" / arm / "anomaly_report.md"
    save_text(body, report)
    try:
        save_text(body, ARTIFACT_ROOT / "failures" / arm / "anomaly_report.md")
    except SandboxViolation:  # pragma: no cover - 防御性
        pass
    return report


def anomaly_guard(arm: str, condition: bool, clause: str, detail: str) -> None:
    if condition:
        write_anomaly_report(arm, clause, detail)
        raise AnomalyStop(arm, clause, detail)


# --------------------------------------------------------------------------------------
# 输入只读性与工程不变性校验
# --------------------------------------------------------------------------------------


def hash_inputs(paths: Iterable[Path]) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for path in sorted(Path(p) for p in paths):
        stat = path.stat()
        records[path.name] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "size_bytes": stat.st_size,
            "mtime_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
        }
    return records


def verify_inputs_unchanged(recorded: dict[str, dict[str, Any]]) -> dict[str, Any]:
    mismatched: list[str] = []
    missing: list[str] = []
    for name, record in recorded.items():
        path = Path(record["path"])
        if not path.exists():
            missing.append(name)
            continue
        if sha256_file(path) != record["sha256"]:
            mismatched.append(name)
    return {
        "checked": len(recorded),
        "mismatched": mismatched,
        "missing": missing,
        "unchanged": not mismatched and not missing,
    }


def snapshot_frozen_project_files() -> dict[str, dict[str, Any]]:
    return hash_inputs([p for p in FROZEN_PROJECT_FILES if p.exists()])


def snapshot_project_source_tree() -> dict[str, str]:
    files: Sequence[Path] = sorted((PROJECT_ROOT / "src" / "evidenceflow").rglob("*.py"))
    return {str(p.relative_to(PROJECT_ROOT)).replace("\\", "/"): sha256_file(p) for p in files}


def diff_source_tree(before: dict[str, str], after: dict[str, str]) -> dict[str, Any]:
    added = sorted(set(after) - set(before))
    removed = sorted(set(before) - set(after))
    changed = sorted(k for k in set(before) & set(after) if before[k] != after[k])
    return {
        "added": added,
        "removed": removed,
        "changed": changed,
        "identical": not (added or removed or changed),
        "file_count_before": len(before),
        "file_count_after": len(after),
    }


def project_write_guard() -> None:
    for root in _WRITE_ALLOWED_ROOTS:
        for forbidden in _WRITE_FORBIDDEN_PREFIXES:
            if forbidden in root.parents or root == forbidden:
                raise SandboxViolation(f"Sandbox root overlaps forbidden path: {root}")
