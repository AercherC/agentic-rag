"""arm 运行器公共工具：参数、计时、峰值内存、确定性结果输出。

为什么每个 arm 自己报告峰值内存：主 harness 在 Windows 上无法可靠地读取子进程的
PeakWorkingSetSize（需要 psutil 或平台 API），而子进程自己调用 ``GetProcessMemoryInfo``
是零依赖且准确的。方案 §30 明确要求记录 ``peak_RAM``，因此这条信息必须能采到。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from evidenceflow.storage import serialize_chroma_record_metadata

# 保证子进程也能 import 项目内的 phase13c 包与 evidenceflow（只读）。
PROJECT_ROOT = Path(__file__).resolve().parents[3]
for _root in (PROJECT_ROOT, PROJECT_ROOT / "src", PROJECT_ROOT / "experiments"):
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))


def chroma_metadatas_for_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """为 13C/C、13E、13F 的原生 Chroma 写入生成统一 metadata 列表。"""

    # 13C/C 与 13E 的历史评测记录不含生产 Chunk 的 file_name/content_hash，
    # 因此在边界层按已有字段序列化，不伪造缺失的来源信息。
    return [serialize_chroma_record_metadata(record) for record in records]


class Timer:
    """秒级计时器。"""

    def __init__(self) -> None:
        self._start = time.perf_counter()

    @property
    def elapsed(self) -> float:
        """已用秒数。"""

        return time.perf_counter() - self._start


def peak_rss_mb() -> float | None:
    """返回当前进程的峰值工作集（MB）；不支持的平台返回 ``None``。"""

    if os.name != "nt":  # pragma: no cover - Windows 为主，非 Windows 退回 ru_maxrss
        try:
            import resource

            return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1)
        except Exception:
            return None

    import ctypes
    from ctypes import wintypes

    class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    counters = PROCESS_MEMORY_COUNTERS()
    counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
    try:
        handle = ctypes.windll.kernel32.GetCurrentProcess()
        ok = ctypes.windll.psapi.GetProcessMemoryInfo(
            handle, ctypes.byref(counters), counters.cb
        )
        if not ok:
            return None
        return round(counters.PeakWorkingSetSize / (1024.0 * 1024.0), 1)
    except Exception:
        return None


def build_parser(arm: str) -> argparse.ArgumentParser:
    """构造 arm 运行器共用的命令行参数。"""

    parser = argparse.ArgumentParser(description=f"Phase 13C parser arm: {arm}")
    parser.add_argument("--pdf", required=True, help="输入 PDF 路径（只读）")
    parser.add_argument("--out", required=True, help="结果 JSON 输出路径")
    parser.add_argument("--document-id", default=None, help="覆盖 document_id（默认由文件路径派生）")
    parser.add_argument(
        "--ocr-mode",
        default="auto",
        choices=["none", "auto", "force"],
        help="OCR 策略：none=仅原生文本层；auto=条件 OCR；force=强制 OCR（方案 §12 O0/O1/O2）",
    )
    parser.add_argument("--device", default="cpu", help="推理设备")
    parser.add_argument("--layout-class", default=None, help="语料类别标签，仅用于报告分组")
    return parser


def document_id_for(pdf_path: Path, override: str | None) -> str:
    """派生稳定 document_id：文件名去除扩展名 + 内容摘要前 8 位。"""

    if override:
        return override
    from phase13c import sandbox

    digest = sandbox.sha256_file(pdf_path)
    return f"{pdf_path.stem}-{digest[:8]}"


def emit_result(
    *,
    out_path: Path,
    payload: dict[str, Any],
    timer: Timer,
    extra: dict[str, Any] | None = None,
) -> None:
    """写出 arm 结果 JSON，并在 stdout 打印一行摘要供主 harness 解析。"""

    payload = dict(payload)
    payload.setdefault("arm_meta", {})
    payload["arm_meta"].update(
        {
            "elapsed_seconds": round(timer.elapsed, 3),
            "peak_ram_mb": peak_rss_mb(),
            "python": sys.version.split()[0],
            "executable": sys.executable,
            "hf_endpoint": os.environ.get("HF_ENDPOINT"),
        }
    )
    if extra:
        payload["arm_meta"].update(extra)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False, default=str) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": payload.get("status", "OK"),
                "out": str(out_path),
                "element_count": len(payload.get("elements", [])),
                "elapsed_seconds": payload["arm_meta"]["elapsed_seconds"],
                "peak_ram_mb": payload["arm_meta"]["peak_ram_mb"],
            }
        )
    )


def fail_result(*, out_path: Path, arm: str, error: str, timer: Timer) -> None:
    """把一次 arm 失败写成结构化结果（而不是让子进程直接崩溃丢掉证据）。"""

    emit_result(
        out_path=out_path,
        payload={
            "status": "ARM_FAILED",
            "arm": arm,
            "error": error,
            "elements": [],
        },
        timer=timer,
    )
