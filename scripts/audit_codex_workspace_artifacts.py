# -*- coding: utf-8 -*-
"""只读检查 EvidenceFlow workspace 中可能由 Codex 生成的文件。

分类结果仅用于人工审查：tracked project file、codex-generated、unknown。未知项
永远标记 REVIEW，不提供删除动作。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "artifacts" / "cleanup" / "codex"
CODEX_PREFIXES = (
    ".codex-log/",
    ".codex/",
    ".agents/",
    ".tmp/",
    "coverage/",
)


def _iter_workspace_files(root: Path):
    """遍历工作区但排除本审计产物与模型/插件缓存。"""

    excluded = {".cache", ".workbuddy", ".vscode", ".git", "artifacts"}
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames[:] = [
            name
            for name in dirnames
            if name not in excluded and not (Path(dirpath) / name).is_symlink()
        ]
        for name in filenames:
            yield Path(dirpath) / name


def load_known_project_paths(root: Path) -> set[str]:
    """读取既有清单作为无 Git 工作区的 tracked-like 基线。"""

    manifest = root / "artifacts" / "cleanup" / "post13f" / "pre_cleanup_manifest.json"
    if not manifest.is_file():
        return set()
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return set()
    paths = data.get("files", {})
    return set(paths) if isinstance(paths, dict) else set()


def classify_workspace_path(relative_path: str, known_paths: set[str]) -> tuple[str, str]:
    """给工作区文件分类并给出审查动作。"""

    rel = relative_path.replace("\\", "/")
    if rel.startswith(CODEX_PREFIXES):
        return "codex-generated", "REVIEW"
    if rel in known_paths:
        return "tracked project file", "KEEP"
    # 本轮不把 tmp/ 中既有研究文件误判成 Codex 产物；新增项进入 unknown。
    return "unknown", "REVIEW"


def audit_workspace(root: Path) -> dict[str, Any]:
    """返回 workspace 分类清单，不修改文件。"""

    known_paths = load_known_project_paths(root)
    records = []
    for path in _iter_workspace_files(root):
        rel = path.relative_to(root).as_posix()
        classification, action = classify_workspace_path(rel, known_paths)
        stat = path.stat()
        records.append(
            {
                "path": rel,
                "size_bytes": stat.st_size,
                "mtime": stat.st_mtime,
                "classification": classification,
                "recommended_action": action,
            }
        )
    records.sort(key=lambda item: item["path"])
    return {
        "workspace": str(root.resolve()),
        "known_baseline": "artifacts/cleanup/post13f/pre_cleanup_manifest.json",
        "file_count": len(records),
        "files": records,
    }


def main() -> int:
    """执行 workspace artifact 审计并写入 report。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=ROOT)
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_DIR / "codex_workspace_inventory.json",
    )
    args = parser.parse_args()
    result = audit_workspace(args.workspace.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    counts: dict[str, int] = {}
    for item in result["files"]:
        counts[item["classification"]] = counts.get(item["classification"], 0) + 1
    summary = [
        "# Codex Workspace Artifact Audit",
        "",
        f"- workspace: `{result['workspace']}`",
        f"- files: {result['file_count']}",
        "",
        "| classification | count |",
        "|---|---:|",
    ]
    summary.extend(f"| {key} | {counts[key]} |" for key in sorted(counts))
    summary += ["", "All `unknown` items remain REVIEW; this audit performs no cleanup."]
    (args.output.parent / "codex_workspace_summary.md").write_text(
        "\n".join(summary) + "\n", encoding="utf-8"
    )
    print(json.dumps(counts, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
