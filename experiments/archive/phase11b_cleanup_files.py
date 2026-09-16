#!/usr/bin/env python
"""已冻结的 Phase 11B 文件清理历史脚本。

删除判定（全部满足才删）：
  1. 无任何 src / tests / experiments 代码引用（已用 grep 核实）；
  2. 不属于 Gold / qrels / Mapping / Chroma / Phase 6–11A 关键 report；
  3. 可由运行确定性重建（缓存），或属于一次性 debug probe / 重复日志；
  4. 目录删除仅限**空目录**（历史 eval 版本遗留的空骨架）。

每个被删文件记录 path / sha256 / size / reason。
"""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(".")
OUT = Path("evals/evaluation_v3/phase11b_cleanup_v1")

# 1) 字节码缓存：由解释器确定性重建
PYCACHE_DIRS = sorted(p for p in ROOT.rglob("__pycache__") if p.is_dir())
PYC_FILES = sorted(p for p in ROOT.rglob("*.pyc") if p.is_file())
CACHE_DIRS = [p for p in (ROOT / ".pytest_cache",) if p.exists()]

# 2) 一次性 debug probe / 重复实验日志（无 import、无 test、无 artifact 依赖）
TMP_FILES = [
    "tmp/inspect_artifacts.py",
    "tmp/inspect_deep.py",
    "tmp/inspect_depths.py",
    "tmp/inspect_p7.py",
    "tmp/recompute_probe.py",
    "tmp/hybrid_retrieval_run.log",
    "tmp/query_rewrite_run.log",
]

# 3) 历史 eval 版本遗留的**空**目录（find -empty 核实，无任何文件）
EMPTY_DIRS = [
    "evals/diagnostics",
    "evals/experiments",
    "evals/experiments/multilingual_fusion",
    "evals/experiments/query_translation",
    "evals/experiments/weighted_rrf",
    "evals/fine_grained",
    "evals/fine_grained_holdout/query_variants",
    "evals/fine_grained_holdout/results",
    "evals/holdout/query_variants",
    "evals/holdout/results",
    "evals/query_variants",
    "evals/rag_unanswerable",
    "evals/results",
]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    deleted: list[dict] = []
    skipped: list[dict] = []

    for path in PYC_FILES:
        deleted.append(
            {
                "path": str(path).replace("\\", "/"),
                "kind": "bytecode",
                "size": path.stat().st_size,
                "sha256": _sha256(path),
                "reason": "Python 字节码缓存，解释器可确定性重建",
            }
        )
        path.unlink()

    for directory in PYCACHE_DIRS:
        deleted.append(
            {
                "path": str(directory).replace("\\", "/"),
                "kind": "directory",
                "size": 0,
                "sha256": None,
                "reason": "__pycache__ 目录，字节码缓存容器",
            }
        )
        shutil.rmtree(directory, ignore_errors=True)

    for directory in CACHE_DIRS:
        deleted.append(
            {
                "path": str(directory).replace("\\", "/"),
                "kind": "directory",
                "size": 0,
                "sha256": None,
                "reason": "pytest 缓存，可由下次运行确定性重建",
            }
        )
        shutil.rmtree(directory, ignore_errors=True)

    for rel in TMP_FILES:
        path = ROOT / rel
        if not path.exists():
            skipped.append({"path": rel, "reason": "not found"})
            continue
        deleted.append(
            {
                "path": rel,
                "kind": "temp_file",
                "size": path.stat().st_size,
                "sha256": _sha256(path),
                "reason": "一次性 debug probe / 实验日志副本，无代码或 artifact 引用",
            }
        )
        path.unlink()

    for rel in EMPTY_DIRS:
        path = ROOT / rel
        if not path.exists():
            skipped.append({"path": rel, "reason": "not found"})
            continue
        if any(path.iterdir()):
            skipped.append({"path": rel, "reason": "not empty — refused"})
            continue
        deleted.append(
            {
                "path": rel,
                "kind": "empty_directory",
                "size": 0,
                "sha256": None,
                "reason": "历史 eval 版本遗留空骨架目录，无任何文件且无代码引用",
            }
        )
        path.rmdir()

    payload = {
        "artifact_type": "phase11b_deleted_files",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "deleted_count": len(deleted),
        "deleted": deleted,
        "skipped": skipped,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "deleted_files.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"deleted entries: {len(deleted)}, skipped: {len(skipped)}")
    for item in deleted:
        if item["kind"] != "bytecode":
            print(f"  - {item['path']}  ({item['kind']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
