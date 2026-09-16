# -*- coding: utf-8 -*-
"""Phase 13F 晋升后指纹重基线与一致性验证（只读盘点 + 证据落盘，不修改 src/evals）。

职责：
  1. 对比清理阶段冻结的 src_hash_baseline.json，断言本轮 src 变更**恰好等于**
     PLAN §3 声明的变更集（防未授权扩散）；
  2. 重算 runtime_source_fingerprint（src/evidenceflow/**/*.py 的 sha256 全集），
     作为晋升后的新基线记录；
  3. 复验冻结评测证据（standard_ir / phase12 / phase13c_b / 13C-C / 13E）未被晋升触碰；
  4. 复验归档门（src/tests 不得 import experiments/phase14a）仍然成立。
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "evidenceflow"
OLD_BASELINE = ROOT / "artifacts" / "cleanup" / "src_hash_baseline.json"
OUT = ROOT / "artifacts" / "cleanup" / "post_promotion_rebaseline.json"

# PLAN §3（evals/evaluation_v3/phase13f_hybrid_runtime_promotion/PLAN.md）声明的 src 变更集。
DECLARED_MODIFIED = {
    "src/evidenceflow/config.py",
    "src/evidenceflow/ingestion/chunker.py",
    "src/evidenceflow/ingestion/service.py",
    "src/evidenceflow/ingestion/__init__.py",
    "src/evidenceflow/runtime.py",
}
DECLARED_ADDED = {"src/evidenceflow/ingestion/structure.py"}
DECLARED_DELETED: set[str] = set()

# 冻结评测证据：cleanup Gate 冻结的三个 manifest + 13C-C/13E 产物 manifest。
FROZEN_MANIFESTS = [
    ROOT / "evals" / "evaluation_v3" / "standard_ir_v1" / "manifest.json",
    ROOT / "evals" / "evaluation_v3" / "phase12_rag_quality_v1" / "manifest.json",
    ROOT / "evals" / "evaluation_v3" / "phase13c_scientific_ingestion_benchmark"
    / "phase13c_c" / "manifest.json",
    ROOT / "evals" / "evaluation_v3" / "phase13d_candidate_augmentation_benchmark" / "manifest.json",
    ROOT / "evals" / "evaluation_v3" / "phase13e_hybrid_chunk_benchmark" / "manifest.json",
    ROOT / "evals" / "evaluation_v3" / "phase13f_hybrid_runtime_promotion" / "manifest.json",
]


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def current_py_hashes() -> dict[str, str]:
    """src/evidenceflow 全部 .py 的相对路径 → sha256（不含 __pycache__）。"""

    return {
        str(path.relative_to(ROOT)).replace("\\", "/"): sha256_of(path)
        for path in sorted(SRC.rglob("*.py"))
    }


def verify_manifest_integrity(manifest_path: Path) -> dict:
    """重算 manifest 记录的每个文件 hash，返回一致性结果。"""

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = manifest.get("files", {})
    mismatched = []
    missing = []
    for rel, expected in files.items():
        path = manifest_path.parent / rel.replace("/", os.sep)
        if not path.exists():
            missing.append(rel)
        elif sha256_of(path) != expected:
            mismatched.append(rel)
    return {
        "manifest": str(manifest_path.relative_to(ROOT)).replace("\\", "/"),
        "file_count": len(files),
        "missing": missing[:5],
        "mismatched": mismatched[:5],
        "intact": not missing and not mismatched,
    }


def main() -> int:
    now = datetime.now(timezone.utc).isoformat()

    # ---- 1. src 变更集 vs PLAN §3 声明 ----
    old_baseline = json.loads(OLD_BASELINE.read_text(encoding="utf-8"))["files"]
    old_py = {rel: h for rel, h in old_baseline.items() if rel.endswith(".py")}
    current = current_py_hashes()
    added = sorted(set(current) - set(old_py))
    deleted = sorted(set(old_py) - set(current))
    modified = sorted(
        rel for rel in set(current) & set(old_py) if current[rel] != old_py[rel]
    )
    change_set_ok = (
        set(added) == DECLARED_ADDED
        and set(deleted) == DECLARED_DELETED
        and set(modified) == DECLARED_MODIFIED
    )

    # ---- 2. 新指纹基线 ----
    fingerprint_payload = "".join(f"{rel}:{current[rel]}\n" for rel in sorted(current))
    runtime_source_fingerprint = hashlib.sha256(fingerprint_payload.encode("utf-8")).hexdigest()

    # ---- 3. 冻结评测证据完整性 ----
    frozen_results = [verify_manifest_integrity(path) for path in FROZEN_MANIFESTS if path.exists()]

    # ---- 4. 归档门：src/tests 不得 import experiments/phase14a（清理阶段冻结口径）----
    violations = []
    for scope in ("src", "tests"):
        for path in sorted((ROOT / scope).rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                modules = []
                if isinstance(node, ast.Import):
                    modules = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    modules = [node.module]
                for module in modules:
                    # 仅禁止已归档的 phase14a（experiments.phase13c 为 13 线活跃实验代码，
                    # 根 pytest 级 13C 专项测试合法引用）。
                    if module == "experiments.phase14a" or module.startswith("experiments.phase14a."):
                        violations.append(f"{path.relative_to(ROOT)} -> {module}")
    archive_gate_ok = not violations

    result = {
        "artifact_type": "phase13f_post_promotion_rebaseline",
        "generated_at": now,
        "change_set": {
            "added": added,
            "modified": modified,
            "deleted": deleted,
            "declared_in_plan": {
                "added": sorted(DECLARED_ADDED),
                "modified": sorted(DECLARED_MODIFIED),
                "deleted": sorted(DECLARED_DELETED),
            },
            "matches_plan": change_set_ok,
        },
        "runtime_source_fingerprint": runtime_source_fingerprint,
        "py_file_count": len(current),
        "baseline_source": "artifacts/cleanup/src_hash_baseline.json（清理阶段冻结，含 .pyc，此处仅比对 .py）",
        "frozen_evidence": frozen_results,
        "archive_gate": {"violations": violations[:10], "ok": archive_gate_ok},
    }
    result["pass"] = (
        change_set_ok and archive_gate_ok and all(item["intact"] for item in frozen_results)
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"change_set matches PLAN: {change_set_ok}")
    print(f"  added:    {added}")
    print(f"  modified: {modified}")
    print(f"  deleted:  {deleted}")
    print(f"runtime_source_fingerprint: {runtime_source_fingerprint}")
    print(f"frozen evidence intact: {all(item['intact'] for item in frozen_results)}")
    print(f"archive gate ok: {archive_gate_ok}")
    print(f"OVERALL: {'PASS' if result['pass'] else 'FAIL'}")
    return 0 if result["pass"] else 2


if __name__ == "__main__":
    sys.exit(main())
