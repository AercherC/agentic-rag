"""Phase 14A-B arm 执行（方案 §21）。

四个 arm，全部 python-docx/openpyxl 原生（无新增 parser 依赖）：
| arm      | fmt  | 管线                                    |
|----------|------|-----------------------------------------|
| DOCX-B0  | docx | native → 最小归一（事实保真 baseline）   |
| DOCX-B1  | docx | native → style-aware semantic cleaner   |
| XLSX-B0  | xlsx | native → cell truth                     |
| XLSX-B1  | xlsx | native → cell truth + region structured |

子进程执行（主 harness 不 import 重库），复用 adapters14a 的解释器解析与环境。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import adapters14a, sandbox14a


@dataclass(frozen=True)
class ArmSpecB:
    arm_id: str
    fmt: str
    label: str
    runner: str
    kind: str  # B0 | B1


ARMS_B: tuple[ArmSpecB, ...] = (
    ArmSpecB("DOCX-B0", "docx", "native structured (minimal)", "phase14a.runners.docx_clean14b", "B0"),
    ArmSpecB("DOCX-B1", "docx", "semantic cleaned", "phase14a.runners.docx_clean14b", "B1"),
    ArmSpecB("XLSX-B0", "xlsx", "cell truth baseline", "phase14a.runners.xlsx_clean14b", "B0"),
    ArmSpecB("XLSX-B1", "xlsx", "region structured", "phase14a.runners.xlsx_clean14b", "B1"),
)

ARM_INDEX_B = {a.arm_id: a for a in ARMS_B}


def run_arm(arm: ArmSpecB, *, file_path: Path, out_path: Path, timeout: int = 600) -> dict[str, Any]:
    """运行一个 14B arm 两次由调用方编排；本函数跑单次并返回 {ok, run, document}。"""

    python = adapters14a._resolve_python()
    cmd = [str(python), "-m", arm.runner,
           f"--{arm.fmt}", str(file_path), "--out", str(out_path), "--arm", arm.kind]
    import subprocess
    import os
    try:
        completed = subprocess.run(
            cmd, cwd=str(sandbox14a.PROJECT_ROOT), env=adapters14a._subprocess_env(),
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        return {"ok": False, "run": {"status": "ARM_FAILED", "reason": f"timeout after {timeout}s"},
                "document": None}
    if completed.returncode != 0 or not out_path.exists():
        return {"ok": False,
                "run": {"status": "ARM_FAILED", "reason": f"exit {completed.returncode}",
                        "stderr_tail": completed.stderr[-2000:]},
                "document": None}
    return {"ok": True, "run": {"status": "OK", "reason": None},
            "document": sandbox14a.load_json(out_path)}


def run_arm_twice(arm: ArmSpecB, *, file_path: Path, base_out: Path) -> dict[str, Any]:
    """同输入跑两遍校验确定性（D-G7 / X-G9）；返回 document_a 与 deterministic 标志。"""

    out_a = Path(str(base_out) + "_a.json")
    out_b = Path(str(base_out) + "_b.json")
    ra = run_arm(arm, file_path=file_path, out_path=out_a)
    if not ra["ok"]:
        return {**ra, "deterministic": False}
    rb = run_arm(arm, file_path=file_path, out_path=out_b)
    if not rb["ok"]:
        return {**rb, "deterministic": False}
    from . import schema14b
    deterministic = schema14b.digest_of(ra["document"]) == schema14b.digest_of(rb["document"])
    return {**ra, "deterministic": deterministic}
