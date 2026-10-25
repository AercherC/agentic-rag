# -*- coding: utf-8 -*-
"""Phase 13F post-promotion fingerprint re-baseline and consistency verification (read-only inventory + evidence on disk; modifies neither src/ nor evals/).

Responsibilities:
  1. Compare against src_hash_baseline.json frozen during the cleanup phase and assert this round's src changes are **exactly**
     the change set declared in PLAN §3 (guarding against unauthorised spread);
  2. Recompute runtime_source_fingerprint (the full set of sha256 over src/evidenceflow/**/*.py)
     and record it as the new post-promotion baseline;
  3. Re-verify that the frozen evaluation evidence (standard_ir / phase12 / phase13c_b / 13C-C / 13E) was not touched by the promotion;
  4. Re-verify that the archive gate still holds (src/tests must not import experiments/phase14a).
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

# The src change set declared in PLAN §3
# (evals/evaluation_v3/phase13f_hybrid_runtime_promotion/PLAN.md).
DECLARED_MODIFIED = {
    "src/evidenceflow/config.py",
    "src/evidenceflow/ingestion/chunker.py",
    "src/evidenceflow/ingestion/service.py",
    "src/evidenceflow/ingestion/__init__.py",
    "src/evidenceflow/runtime.py",
}
DECLARED_ADDED = {"src/evidenceflow/ingestion/structure.py"}
DECLARED_DELETED: set[str] = set()

# Frozen evaluation evidence: the three manifests frozen by the cleanup Gate plus the 13C-C/13E
# artifact manifests.
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
    """Relative path -> sha256 for every .py under src/evidenceflow (excluding __pycache__)."""

    return {
        str(path.relative_to(ROOT)).replace("\\", "/"): sha256_of(path)
        for path in sorted(SRC.rglob("*.py"))
    }


def verify_manifest_integrity(manifest_path: Path) -> dict:
    """Recompute the hash of every file recorded in the manifest and return the consistency result."""

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

    # ---- 1. src change set vs. the PLAN §3 declaration ----
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

    # ---- 2. New fingerprint baseline ----
    fingerprint_payload = "".join(f"{rel}:{current[rel]}\n" for rel in sorted(current))
    runtime_source_fingerprint = hashlib.sha256(fingerprint_payload.encode("utf-8")).hexdigest()

    # ---- 3. Integrity of the frozen evaluation evidence ----
    frozen_results = [verify_manifest_integrity(path) for path in FROZEN_MANIFESTS if path.exists()]

    # ---- 4. Archive gate: src/tests must not import experiments/phase14a (definition frozen during the cleanup phase) ----
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
                    # Only the already-archived phase14a is forbidden (experiments.phase13c is live
                    # 13-line experiment code,
                    # and the root pytest-level 13C-specific tests may legitimately reference it).
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
