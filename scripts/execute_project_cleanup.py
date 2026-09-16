# -*- coding: utf-8 -*-
"""按二次确认计划逐文件清理 EvidenceFlow 的可再生缓存。

默认是 dry-run；必须显式传 ``--execute`` 才会删除。执行前重新校验计划中的
exact allowlist、路径边界与 snapshot，绝不删除源码、Gold/qrels、PDF、Chroma
或 artifacts。被删内容均为可由源码/测试/评测输入重建的缓存。
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from audit_project_cleanup import ROOT, safe_candidate_reason

OUTPUT_DIR = ROOT / "artifacts" / "cleanup"


def sha256_of(path: Path) -> str:
    """删除前计算摘要，保留在 snapshot 以便验证候选集合。"""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_plan(plan: dict) -> list[tuple[Path, dict]]:
    """校验计划中的每个文件仍属于 exact DELETE_SAFE 范围。"""

    invalid = []
    targets = []
    for item in plan.get("delete_safe", []):
        rel = item.get("path", "")
        reason = safe_candidate_reason(rel)
        target = (ROOT / rel).resolve()
        try:
            target.relative_to(ROOT.resolve())
        except ValueError:
            invalid.append((rel, "outside project root"))
            continue
        if reason is None or reason != item.get("reason"):
            invalid.append((rel, "exact allowlist mismatch"))
            continue
        targets.append((target, item))
    if invalid:
        raise RuntimeError(f"cleanup plan validation failed: {invalid[:5]}")
    return targets


def execute(plan_path: Path, *, perform: bool) -> dict:
    """生成 snapshot；perform=True 时逐项删除并记录结果。"""

    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if not plan.get("pass"):
        raise RuntimeError("cleanup plan did not pass anomaly gate")
    targets = validate_plan(plan)
    snapshot = []
    for path, item in targets:
        if path.is_file():
            stat = path.stat()
            snapshot.append(
                {
                    "path": item["path"],
                    "size_bytes": stat.st_size,
                    "mtime": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
                    "sha256": sha256_of(path),
                    "reason": item["reason"],
                }
            )
    snapshot_path = OUTPUT_DIR / "project_cleanup_snapshot.json"
    snapshot_path.write_text(
        json.dumps(
            {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "plan": str(plan_path.relative_to(ROOT)).replace("\\", "/"),
                "recovery_note": "候选均为可再生缓存；snapshot 保存删除前身份，恢复方式为重新运行对应生成流程。",
                "files": snapshot,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    removed = []
    removed_directories = []
    missing = []
    errors = []
    if perform:
        for entry in snapshot:
            path = ROOT / entry["path"]
            try:
                # 删除前再次确认当前文件未被替换，避免误删新生成文件。
                if not path.is_file():
                    missing.append(entry["path"])
                    continue
                if path.stat().st_size != entry["size_bytes"] or sha256_of(path) != entry["sha256"]:
                    errors.append({"path": entry["path"], "reason": "file changed after snapshot"})
                    continue
                path.unlink()
                removed.append(entry)
            except OSError as exc:
                errors.append({"path": entry["path"], "reason": str(exc)})
        # 文件逐项校验删除后，只移除已经为空的标准缓存目录；源码目录和普通空目录不处理。
        cache_names = {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
        scan_roots = [ROOT / name for name in ("src", "tests", "experiments", "scripts")]
        cache_roots = {
            path
            for base in scan_roots
            for path in ([base] if base.name in cache_names else base.rglob("*"))
            if path.is_dir()
            and path.name in cache_names
            and ".cache" not in path.relative_to(ROOT).parts
            and "artifacts" not in path.relative_to(ROOT).parts
        }
        root_pytest_cache = ROOT / ".pytest_cache"
        if root_pytest_cache.is_dir():
            cache_roots.add(root_pytest_cache)
        # 先删除缓存根内部的空子目录，再删除缓存根本身，避免 pytest 的 v/cache 壳目录残留。
        candidates = {
            child
            for cache_root in cache_roots
            for child in [cache_root, *cache_root.rglob("*")]
            if child.is_dir()
        }
        for directory in sorted(candidates, key=lambda path: len(path.parts), reverse=True):
            try:
                if not any(directory.iterdir()):
                    directory.rmdir()
                    removed_directories.append(
                        str(directory.relative_to(ROOT)).replace("\\", "/") + "/"
                    )
            except OSError as exc:
                errors.append(
                    {
                        "path": str(directory.relative_to(ROOT)).replace("\\", "/"),
                        "reason": str(exc),
                    }
                )
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dry_run": not perform,
        "planned_count": len(targets),
        "snapshot_count": len(snapshot),
        "removed": removed,
        "removed_directories": removed_directories,
        "missing": missing,
        "errors": errors,
        "bytes_reclaimed": sum(item["size_bytes"] for item in removed),
        "pass": not errors,
    }


def main() -> int:
    """命令行入口；无 --execute 时只生成删除前 snapshot。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plan",
        type=Path,
        default=OUTPUT_DIR / "project_cleanup_plan.json",
    )
    parser.add_argument("--execute", action="store_true", help="执行已验证的逐文件删除")
    args = parser.parse_args()
    result = execute(args.plan.resolve(), perform=args.execute)
    (OUTPUT_DIR / "project_cleanup_execution.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
