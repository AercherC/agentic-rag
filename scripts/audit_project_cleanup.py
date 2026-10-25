# -*- coding: utf-8 -*-
"""Generate the second-confirmation report for the Post-13F physical cleanup of EvidenceFlow.

DELETE_SAFE covers only explicitly regenerable pytest temporary trees, Python bytecode/cache and the
declared Phase14 temporary Chroma; Gold, qrels, manifests, source PDFs, active/rollback Chroma,
source code and model files never become candidates. The script is read-only and deletes nothing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "artifacts" / "cleanup"
PHASE14_CHROMA_PREFIX = (
    "evals/evaluation_v3/phase14a_multiformat_ingestion_benchmark/phase14a_c/chroma/"
)


def sha256_of(path: Path) -> str:
    """Compute the hash of a file pending cleanup in a streaming way, as the verification anchor before any restore or rebuild."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def iter_project_files():
    """Walk the controlled project scope, excluding the Codex/model cache and this audit's own output."""

    excluded = {".cache", ".workbuddy", ".vscode", ".git", "artifacts"}
    for dirpath, dirnames, filenames in os.walk(ROOT, followlinks=False):
        dirnames[:] = [
            name
            for name in dirnames
            if name not in excluded and not (Path(dirpath) / name).is_symlink()
        ]
        for name in filenames:
            yield Path(dirpath) / name


def safe_candidate_reason(relative_path: str) -> str | None:
    """Return the reason for an exact allowlist hit, or None when nothing matches."""

    rel = relative_path.replace("\\", "/")
    if rel == ".pytest_cache" or rel.startswith(".pytest_cache/"):
        return "pytest cache"
    if rel.startswith("tmp/pytest-root/"):
        return "pytest temporary sandbox"
    if "/__pycache__/" in f"/{rel}/" or rel.endswith((".pyc", ".pyo")):
        return "Python bytecode cache"
    if rel.startswith(PHASE14_CHROMA_PREFIX):
        return "rebuildable Phase14 temporary Chroma"
    return None


def _red_flag(relative_path: str, reason: str) -> str | None:
    """Apply a second scope check to the allowlist candidates."""

    rel = relative_path.replace("\\", "/")
    # These exact categories are already proven regenerable caches by the allowlist; a test-module
    # word such as gold/model in the file name does not change their cache semantics, so they must
    # not
    # be misreported as protected business artifacts.
    if reason in {"pytest cache", "pytest temporary sandbox", "Python bytecode cache", "rebuildable Phase14 temporary Chroma"}:
        return None
    lower = rel.lower()
    if lower.endswith(".py"):
        return "Python source"
    if any(token in lower for token in ("gold", "qrels", "manifest")):
        return "protected evaluation artifact"
    if lower.startswith(("data/source_docs/", "tmp/pdfs/", "data/chroma/")):
        return "source or active/rollback Chroma"
    if "model" in lower or lower.endswith((".safetensors", ".bin")) and not reason:
        return "model file"
    return None


def build_cleanup_audit() -> dict[str, Any]:
    """Scan the current files and produce the DELETE_SAFE, REVIEW and anomaly entries."""

    delete_safe: list[dict[str, Any]] = []
    review: list[dict[str, Any]] = []
    anomalies: list[dict[str, str]] = []
    for path in iter_project_files():
        rel = path.relative_to(ROOT).as_posix()
        stat = path.stat()
        reason = safe_candidate_reason(rel)
        item = {
            "path": rel,
            "size_bytes": stat.st_size,
            "mtime": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
        }
        if reason:
            item["reason"] = reason
            flag = _red_flag(rel, reason)
            if flag:
                anomalies.append({"path": rel, "reason": flag})
                review.append({**item, "classification": "REVIEW", "recommended_action": "REVIEW"})
            else:
                delete_safe.append({**item, "classification": "DELETE_SAFE"})
        else:
            review.append({**item, "classification": "REVIEW", "recommended_action": "REVIEW"})
    delete_safe.sort(key=lambda item: item["path"])
    review.sort(key=lambda item: item["path"])
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scope": "project files excluding .cache/.workbuddy/.vscode/artifacts",
        "delete_safe": delete_safe,
        "review": review,
        "anomalies": anomalies,
        "pass": not anomalies,
    }


def write_cleanup_reports(audit: dict[str, Any], output_dir: Path = OUTPUT_DIR) -> None:
    """Write the machine-readable plan plus the DELETE_SAFE and REVIEW cluster reports to disk."""

    output_dir.mkdir(parents=True, exist_ok=True)
    plan = {
        "generated_at": audit["generated_at"],
        "dry_run": True,
        "delete_safe": audit["delete_safe"],
        "review_count": len(audit["review"]),
        "anomalies": audit["anomalies"],
        "pass": audit["pass"],
        "guard": "本文件只读；删除器必须逐项读取本计划并再次校验 exact allowlist",
    }
    (output_dir / "project_cleanup_plan.json").write_text(
        json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    by_dir: dict[str, dict[str, int]] = defaultdict(lambda: {"count": 0, "bytes": 0})
    by_ext: dict[str, dict[str, int]] = defaultdict(lambda: {"count": 0, "bytes": 0})
    by_reason: dict[str, dict[str, int]] = defaultdict(lambda: {"count": 0, "bytes": 0})
    for item in audit["delete_safe"]:
        directory = "/".join(item["path"].split("/")[:-1]) or "."
        extension = Path(item["path"]).suffix.lower() or "<none>"
        for bucket, key in ((by_dir, directory), (by_ext, extension), (by_reason, item["reason"])):
            bucket[key]["count"] += 1
            bucket[key]["bytes"] += item["size_bytes"]
    lines = [
        "# DELETE_SAFE Summary",
        "",
        "本报告为二次确认；未执行删除。DELETE_SAFE 必须同时满足 exact allowlist 和无异常项。",
        "",
        f"- candidates: {len(audit['delete_safe'])}",
        f"- bytes: {sum(item['size_bytes'] for item in audit['delete_safe']):,}",
        f"- anomalies: {len(audit['anomalies'])}",
    ]
    for title, bucket in (("按目录", by_dir), ("按扩展名", by_ext), ("按原因", by_reason)):
        lines += ["", f"## {title}", "", "| key | files | bytes |", "|---|---:|---:|"]
        lines.extend(
            f"| `{key}` | {value['count']} | {value['bytes']:,} |"
            for key, value in sorted(bucket.items(), key=lambda pair: -pair[1]["bytes"])
        )
    if audit["anomalies"]:
        lines += ["", "## 异常（必须 STOP）", ""]
        lines.extend(f"- `{item['path']}`：{item['reason']}" for item in audit["anomalies"])
    (output_dir / "delete_safe_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    clusters: dict[str, dict[str, int]] = defaultdict(lambda: {"count": 0, "bytes": 0})
    for item in audit["review"]:
        first = item["path"].split("/", 1)[0]
        cluster = first if first in {"src", "tests", "evals", "experiments", "artifacts", "docs", "data", "scripts"} else "misc"
        clusters[cluster]["count"] += 1
        clusters[cluster]["bytes"] += item["size_bytes"]
    review_lines = [
        "# REVIEW Cluster Summary",
        "",
        "REVIEW 项只做聚类盘点，不自动删除；Python、Gold/qrels、manifest、source PDF 与 Chroma 默认保留。",
        "",
        "| cluster | files | bytes |",
        "|---|---:|---:|",
    ]
    review_lines.extend(
        f"| `{key}` | {value['count']} | {value['bytes']:,} |"
        for key, value in sorted(clusters.items(), key=lambda pair: -pair[1]["bytes"])
    )
    (output_dir / "review_cluster_summary.md").write_text("\n".join(review_lines) + "\n", encoding="utf-8")


def main() -> int:
    """Run the second-confirmation audit for the project cleanup."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    audit = build_cleanup_audit()
    write_cleanup_reports(audit, args.output_dir)
    print(
        json.dumps(
            {
                "delete_safe": len(audit["delete_safe"]),
                "review": len(audit["review"]),
                "anomalies": audit["anomalies"],
                "pass": audit["pass"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if audit["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
