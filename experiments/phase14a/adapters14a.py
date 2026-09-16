"""Phase 14A-A parser arm 适配器（方案 §10 / §29）。

与 13C adapters.py 同一纪律：**主 harness 绝不 import 重库**；所有 parser 通过子进程
在隔离沙盒解释器（PYTHONPATH 指向 experiments + 隔离 site-packages）中运行。

arm 划分：
| arm     | fmt  | kind                | 说明 |
|---------|------|---------------------|------|
| W-A0    | docx | native_subprocess   | python-docx native（主真实性 baseline） |
| W-A1    | docx | blocked             | Docling DOCX（本机未安装 → BLOCKED，非 FAIL） |
| W-A2    | docx | blocked             | Unstructured DOCX（同上） |
| W-A3    | docx | blocked             | Mammoth（同上） |
| X-A0    | xlsx | native_subprocess   | openpyxl native（主真实性 baseline） |
| X-A1    | xlsx | blocked             | Docling XLSX（同上） |
| X-A2    | xlsx | blocked             | Unstructured XLSX（同上） |
"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import sandbox14a

PROJECT_ROOT = sandbox14a.PROJECT_ROOT
SITE_PACKAGES = sandbox14a.SANDBOX_ROOT / "site-packages"


@dataclass(frozen=True)
class ArmSpec:
    arm_id: str
    fmt: str  # docx | xlsx
    label: str
    kind: str  # native_subprocess | blocked
    runner: str | None = None
    blocked_reason: str | None = None


ARMS: tuple[ArmSpec, ...] = (
    ArmSpec("W-A0", "docx", "python-docx native", "native_subprocess",
            runner="phase14a.runners.docx_native_arm"),
    ArmSpec("W-A1", "docx", "Docling DOCX", "blocked",
            blocked_reason="Docling 未安装（用户选择仅安装 python-docx+openpyxl）；本机离线，ML 依赖不可得。"),
    ArmSpec("W-A2", "docx", "Unstructured DOCX", "blocked",
            blocked_reason="Unstructured 未安装（同上）。"),
    ArmSpec("W-A3", "docx", "Mammoth", "blocked",
            blocked_reason="Mammoth 未安装（同上）。"),
    ArmSpec("X-A0", "xlsx", "openpyxl native", "native_subprocess",
            runner="phase14a.runners.xlsx_native_arm"),
    ArmSpec("X-A1", "xlsx", "Docling XLSX", "blocked",
            blocked_reason="Docling 未安装（同上）。"),
    ArmSpec("X-A2", "xlsx", "Unstructured XLSX", "blocked",
            blocked_reason="Unstructured 未安装（同上）。"),
)

ARM_INDEX = {a.arm_id: a for a in ARMS}


def _resolve_python() -> Path:
    """解析用于运行子进程的解释器（隔离，仅通过 PYTHONPATH 载入沙盒 site-packages）。"""

    env_val = os.environ.get("EF_PHASE14A_PYTHON")
    if env_val and Path(env_val).exists():
        return Path(env_val)
    for cand in ("/d/miniconda/python.exe",
                 "/c/Users/chchxh/.workbuddy/binaries/python/versions/3.13.12/python.exe"):
        if Path(cand).exists():
            return Path(cand)
    return Path(sys.executable)


def _subprocess_env() -> dict[str, str]:
    env = dict(os.environ)
    parts = [str(PROJECT_ROOT / "experiments"), str(SITE_PACKAGES)]
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join(parts + ([existing] if existing else []))
    env.pop("SSL_CERT_FILE", None)
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def run_native_arm(arm: ArmSpec, *, file_path: Path, out_path: Path, timeout: int = 600) -> dict[str, Any]:
    """运行一个 native_subprocess arm，返回 {ok, stdout, stderr, error}。"""

    python = _resolve_python()
    module = arm.runner or ""
    cmd = [str(python), "-m", module, f"--{arm.fmt}", str(file_path), "--out", str(out_path)]
    try:
        completed = subprocess.run(
            cmd, cwd=str(PROJECT_ROOT), env=_subprocess_env(),
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        return {"ok": False, "error": f"timeout after {timeout}s",
                "stdout": (exc.stdout or "")[-3000:], "stderr": (exc.stderr or "")[-3000:]}
    return {
        "ok": completed.returncode == 0 and out_path.exists(),
        "returncode": completed.returncode,
        "stdout": completed.stdout[-3000:],
        "stderr": completed.stderr[-4000:],
        "error": None if completed.returncode == 0 else f"exit {completed.returncode}",
    }


def execute_arm(arm: ArmSpec, *, file_path: Path, out_path: Path, timeout: int = 600) -> dict[str, Any]:
    """执行一个 arm：blocked → BLOCKED 状态；native → 子进程运行。"""

    if arm.kind == "blocked":
        return {
            "document": None,
            "availability": {"available": False, "reason": arm.blocked_reason},
            "run": {"status": "BLOCKED", "reason": arm.blocked_reason,
                    "elapsed_seconds": None, "peak_ram_mb": None},
        }
    result = run_native_arm(arm, file_path=file_path, out_path=out_path, timeout=timeout)
    if not result["ok"]:
        return {
            "document": None,
            "availability": {"available": True, "reason": "sandbox subprocess"},
            "run": {"status": "ARM_FAILED", "reason": result.get("error") or "unknown",
                    "elapsed_seconds": None, "peak_ram_mb": None,
                    "stderr_tail": result.get("stderr", "")[-2000:]},
        }
    document = sandbox14a.load_json(out_path)
    return {
        "document": document,
        "availability": {"available": True, "reason": "sandbox subprocess"},
        "run": {"status": "OK", "reason": None, "elapsed_seconds": None, "peak_ram_mb": None},
    }


def arms_for(fmt: str) -> list[ArmSpec]:
    return [a for a in ARMS if a.fmt == fmt]
