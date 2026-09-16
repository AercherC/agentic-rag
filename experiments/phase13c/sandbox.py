"""Sandbox 边界、哈希、产物 IO 与 STOP-ON-ANOMALY 守卫。

本模块是 Phase 13C 唯一的"环境写入口"。所有产物写入都必须经由
:func:`save_json` / :func:`save_text`，它们会校验目标路径落在允许的根目录之内。

允许写入的根目录只有两个：

1. ``<SYSTEM_TEMP>/evidenceflow_phase13c`` —— 重资产（venv、parser 原始输出、
   模型缓存、rasterized 中间文件）；
2. ``<PROJECT>/evals/evaluation_v3/phase13c_scientific_ingestion_benchmark`` ——
   可审计的评测产物（metrics / reports / ground truth / manifest）。

反之，下列路径被显式列为**禁写**：``src/``、``tests/``、``data/``、
``experiments/`` 下的历史阶段脚本、``PROJECT_SPEC.md``、``pyproject.toml``、
``requirements*``、``.env`` 以及任何 Chroma 目录。
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

# --------------------------------------------------------------------------------------
# 路径定义
# --------------------------------------------------------------------------------------

# 本文件位于 <PROJECT>/experiments/phase13c/sandbox.py
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 沙盒根目录：重资产落地处。允许用环境变量覆盖，便于迁移机器时复现。
SANDBOX_ROOT = Path(
    os.environ.get(
        "EF_PHASE13C_SANDBOX",
        Path(os.environ.get("TEMP", "/tmp")) / "evidenceflow_phase13c",
    )
).resolve()

# 可审计产物目录：进入版本库的评测产物。
ARTIFACT_ROOT = PROJECT_ROOT / "evals/evaluation_v3/phase13c_scientific_ingestion_benchmark"

# 只读的项目资产（任何情况下不得写）。
PROJECT_SOURCE_DOCS = PROJECT_ROOT / "data/source_docs"
PROJECT_CHROMA_DIR = PROJECT_ROOT / "data/chroma"
FROZEN_PROJECT_FILES = (
    PROJECT_ROOT / "PROJECT_SPEC.md",
    PROJECT_ROOT / "pyproject.toml",
    PROJECT_ROOT / "AGENTS.md",
)

# 写操作白名单前缀（resolve 后比较）。
_WRITE_ALLOWED_ROOTS: tuple[Path, ...] = (SANDBOX_ROOT, ARTIFACT_ROOT)

# 写操作黑名单前缀：即使落在白名单内也不允许（防御性，正常不会命中）。
_WRITE_FORBIDDEN_PREFIXES: tuple[Path, ...] = (
    PROJECT_SOURCE_DOCS,
    PROJECT_CHROMA_DIR,
    PROJECT_ROOT / "src",
    PROJECT_ROOT / "tests",
    PROJECT_ROOT / "data",
    PROJECT_ROOT / "tmp",
)


class SandboxViolation(RuntimeError):
    """表示某次操作越过了 Phase 13C 的沙盒边界。"""


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
    """返回 bytes 的 SHA-256 十六进制摘要。"""

    return hashlib.sha256(payload).hexdigest()


def sha256_text(text: str) -> str:
    """返回文本 UTF-8 编码后的 SHA-256。"""

    return sha256_bytes(text.encode("utf-8"))


def sha256_file(path: Path) -> str:
    """分块读取文件并返回 SHA-256，避免大 PDF 一次性载入内存。"""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(payload: Any) -> str:
    """对规范化（sort_keys）JSON 取 SHA-256，用于跨产物身份比对。"""

    return sha256_text(json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str))


def utc_now() -> str:
    """返回 UTC ISO-8601 时间戳。"""

    return datetime.now(timezone.utc).isoformat()


def environment_fingerprint() -> dict[str, Any]:
    """采集运行环境身份，用于 parser arm 的可复现性声明。"""

    import torch  # 延迟导入：本模块被测试导入时不必加载 torch。

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }


# --------------------------------------------------------------------------------------
# 边界校验与 IO
# --------------------------------------------------------------------------------------


def _ensure_writable(path: Path) -> None:
    """确认目标路径落在允许写入的根目录内，且不命中禁写前缀。"""

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
    """创建目录（含父级）并返回，写入前先做边界校验。"""

    target = Path(path)
    _ensure_writable(target)
    target.mkdir(parents=True, exist_ok=True)
    return target


def save_json(payload: Any, path: Path) -> Path:
    """以 UTF-8 + 2 空格缩进写出 JSON；拒绝 NaN/Infinity 以保证可复现。"""

    target = Path(path)
    ensure_dir(target.parent)
    target.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False, default=str) + "\n",
        encoding="utf-8",
    )
    return target


def save_text(text: str, path: Path) -> Path:
    """以 UTF-8 写出文本产物。"""

    target = Path(path)
    ensure_dir(target.parent)
    target.write_text(text, encoding="utf-8")
    return target


def load_json(path: Path) -> Any:
    """读取 UTF-8 JSON。"""

    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_anomaly_report(arm: str, clause: str, detail: str) -> Path:
    """把一次异常写入 ``failures/<arm>/anomaly_report.md``（沙盒与产物目录各一份）。"""

    body = "\n".join(
        [
            f"# Phase 13C Anomaly Report — {arm}",
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
    """若 ``condition`` 为真则写入异常报告并抛出 :class:`AnomalyStop`。"""

    if condition:
        write_anomaly_report(arm, clause, detail)
        raise AnomalyStop(arm, clause, detail)


# --------------------------------------------------------------------------------------
# 输入只读性与工程不变性校验
# --------------------------------------------------------------------------------------


def hash_inputs(paths: Iterable[Path]) -> dict[str, dict[str, Any]]:
    """对输入文件记录 SHA-256 / 字节数 / mtime，用于"输入只读"证据。"""

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
    """复核之前记录的输入哈希是否仍然一致（证明只读约束未被破坏）。"""

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
    """对冻结工程文件取样，供结束时证明"未修改项目源码/规范"。"""

    return hash_inputs([p for p in FROZEN_PROJECT_FILES if p.exists()])


def snapshot_project_source_tree() -> dict[str, str]:
    """对 ``src/evidenceflow/**/*.py`` 逐文件取哈希（用于证明 Runtime 未改动）。"""

    files: Sequence[Path] = sorted((PROJECT_ROOT / "src" / "evidenceflow").rglob("*.py"))
    return {str(p.relative_to(PROJECT_ROOT)).replace("\\", "/"): sha256_file(p) for p in files}


def diff_source_tree(before: dict[str, str], after: dict[str, str]) -> dict[str, Any]:
    """比较两次源码树快照，返回新增/删除/修改清单。"""

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
    """断言沙盒根与产物根都不是项目运行目录的子路径（防止误写 Runtime 资产）。"""

    for root in _WRITE_ALLOWED_ROOTS:
        for forbidden in _WRITE_FORBIDDEN_PREFIXES:
            if forbidden in root.parents or root == forbidden:
                raise SandboxViolation(f"Sandbox root overlaps forbidden path: {root}")
