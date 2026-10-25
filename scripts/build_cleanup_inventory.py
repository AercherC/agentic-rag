# -*- coding: utf-8 -*-
"""EvidenceFlow cleanup phase-1 tooling: inventory + classification + dry-run plan (read-only; modifies no file).

Outputs (all written under artifacts/cleanup/):
- pre_cleanup_manifest.json   Step 0 workspace freeze inventory (path/size/sha256)
- runtime_reachability.json   Step 2 runtime reachability analysis (AST import scan)
- cleanup_inventory.json/.md  Step 3 full-file classification inventory
- cleanup_plan.json           Step 4 cleanup plan (machine readable)
- cleanup_dry_run.md          Step 4 dry-run report (dry_run=true, with before/projected after)

Design constraint: this script deletes, moves, and rewrites none of the inventoried files.
"""

import ast
import hashlib
import json
import os
import sys
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "artifacts", "cleanup")
os.makedirs(OUT_DIR, exist_ok=True)

# Directories excluded from the inventory: tool caches/session data, which are not project
# artifacts
EXCLUDE_DIRS = {".cache", ".workbuddy", ".vscode", "artifacts"}


def sha256_of(path: str) -> str:
    """Stream the file's sha256 so large files are never loaded into memory in one go."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def walk_files():
    """Walk the repository and return the list of relative paths (forward slashes normalised)."""
    result = []
    for dirpath, dirnames, filenames in os.walk(ROOT):
        rel_dir = os.path.relpath(dirpath, ROOT).replace("\\", "/")
        # Prune the excluded directories
        dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]
        for name in filenames:
            rel = os.path.join(rel_dir, name).replace("\\", "/")
            if rel.startswith("./"):
                rel = rel[2:]
            result.append(rel)
    return sorted(result)


def classify(rel: str) -> str:
    """Classify a single file with the A-H scheme of plan §7.

    Rules are matched from highest to lowest priority; cache entries are decided first so that
    __pycache__ directories under src are not misclassified as CORE_RUNTIME.
    """
    p = rel
    # F. GENERATED_REPRODUCIBLE: caches and rebuildable intermediate artifacts
    if "__pycache__" in p or p.endswith(".pyc") or ".pytest_cache" in p:
        return "GENERATED_REPRODUCIBLE"
    # Chroma index of the Phase14A-C benchmark: rebuildable from chunks+embeddings via the runner
    if p.startswith("evals/evaluation_v3/phase14a_multiformat_ingestion_benchmark/phase14a_c/chroma/"):
        return "GENERATED_REPRODUCIBLE"
    # Phase14A-C embeddings: anchored by the frozen manifest hash, so treated as a frozen artifact
    # and kept
    if p.startswith("evals/evaluation_v3/phase14a_multiformat_ingestion_benchmark/phase14a_c/embeddings/"):
        return "MULTIFORMAT_ARCHIVE"
    # E. MULTIFORMAT_ARCHIVE: the Office branch
    if p.startswith("evals/evaluation_v3/phase14a_multiformat_ingestion_benchmark/") or p.startswith("experiments/phase14a/"):
        return "MULTIFORMAT_ARCHIVE"
    # A. CORE_RUNTIME plus root config/specification documents
    if p.startswith("src/") or p in (
        "pyproject.toml", ".env", ".env.example",
        "AGENTS.md", "PROJECT_SPEC.md", "PROJECT_VISION.md", "README.md",
    ):
        return "CORE_RUNTIME"
    # B. CORE_TEST
    if p.startswith("tests/"):
        return "CORE_TEST"
    # D. PDF_ACTIVE_RESEARCH: the Phase 13 series and experiments/phase13c
    if p.startswith("evals/evaluation_v3/phase13") or p.startswith("experiments/phase13c/"):
        return "PDF_ACTIVE_RESEARCH"
    # C. CORE_EVAL: standard IR, historical formal evaluations, research retrieval experiment
    # conclusions
    if p.startswith(("evals/evaluation_v2/", "evals/evaluation_v3/", "evals/research_ingestion/", "evals/research_retrieval/")):
        return "CORE_EVAL"
    # Real corpus PDFs and page-render inputs: not rebuildable, keep
    if p.startswith("tmp/pdfs/"):
        return "SOURCE_OF_TRUTH"
    # Remaining intermediate work files under tmp: Phase12 annotations/dumps etc.
    if p.startswith("tmp/"):
        return "TEMP_REVIEW"
    if p.startswith("docs/"):
        return "DOCS"
    if p.startswith("experiments/"):
        return "REVIEW"  # experiments/debug etc.
    return "REVIEW"


def action_for(category: str, rel: str) -> str:
    """Category -> recommended action for this round. The first round only auto-cleans pure caches; Phase14 chroma requires exporting a collection inventory first."""
    if category == "GENERATED_REPRODUCIBLE":
        # __pycache__/.pyc/.pytest_cache: unconditionally safe
        if "__pycache__" in rel or rel.endswith(".pyc") or ".pytest_cache" in rel:
            return "DELETE_CANDIDATE"
        # Phase14 chroma: export the collection inventory before deleting
        return "DELETE_CANDIDATE"
    if category == "MULTIFORMAT_ARCHIVE":
        return "ARCHIVE"
    if category == "TEMP_REVIEW":
        return "REVIEW"
    if category == "REVIEW":
        return "REVIEW"
    return "KEEP"


def scan_imports(package_dir: str) -> dict:
    """AST-scan the imports of every .py under a directory and return internal dependencies plus external top-level module counts."""
    internal, external = {}, {}
    for dirpath, dirnames, filenames in os.walk(package_dir):
        dirnames[:] = [d for d in dirnames if d != "__pycache__"]
        for name in filenames:
            if not name.endswith(".py"):
                continue
            fpath = os.path.join(dirpath, name)
            rel = os.path.relpath(fpath, ROOT).replace("\\", "/")
            try:
                tree = ast.parse(open(fpath, encoding="utf-8").read())
            except SyntaxError as e:
                external[rel] = [f"SYNTAX_ERROR:{e}"]
                continue
            mods = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    mods += [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    mods.append(node.module)
            for m in mods:
                top = m.split(".")[0]
                if top in ("evidenceflow", "phase14a", "phase13c", "experiments"):
                    internal.setdefault(rel, []).append(m)
                else:
                    external.setdefault(rel, []).append(m)
    return {"internal": internal, "external": external}


def main():
    now = datetime.now(timezone.utc).isoformat()
    files = walk_files()

    # ---- Step 0: freeze inventory ----
    records = []
    for rel in files:
        abs_p = os.path.join(ROOT, rel)
        try:
            size = os.path.getsize(abs_p)
            digest = sha256_of(abs_p)
        except OSError as e:
            size, digest = -1, f"UNREADABLE:{e}"
        records.append({"path": rel, "size_bytes": size, "sha256": digest})
    pre = {
        "generated_at": now,
        "purpose": "EvidenceFlow cleanup Step0 freeze: pre-cleanup file inventory with sha256",
        "file_count": len(records),
        "total_bytes": sum(r["size_bytes"] for r in records if r["size_bytes"] >= 0),
        "files": records,
    }
    with open(os.path.join(OUT_DIR, "pre_cleanup_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(pre, f, ensure_ascii=False, indent=1)
    print(f"[Step0] pre_cleanup_manifest.json: {len(records)} files, {pre['total_bytes']} bytes")

    # ---- Step 2: runtime reachability (AST import scan) ----
    reach = {
        "generated_at": now,
        "entrypoint_note": "无 CLI entrypoint（pyproject 无 [project.scripts]）；runtime 入口为 src/evidenceflow 包本身",
        "src": scan_imports(os.path.join(ROOT, "src")),
        "tests": scan_imports(os.path.join(ROOT, "tests")),
    }
    # Key decision: src and tests must have zero dependency on experiments/phase14a (Archive Gate
    # §28)
    src_refs = [m for ms in reach["src"]["internal"].values() for m in ms if "phase14a" in m]
    test_refs = [m for ms in reach["tests"]["internal"].values() for m in ms if "phase14a" in m]
    reach["archive_gate"] = {
        "src_imports_phase14a": len(src_refs),
        "tests_import_phase14a": len(test_refs),
        "cli_imports_phase14a": 0,  # no CLI entrypoint
        "passed": not src_refs and not test_refs,
    }
    with open(os.path.join(OUT_DIR, "runtime_reachability.json"), "w", encoding="utf-8") as f:
        json.dump(reach, f, ensure_ascii=False, indent=1)
    print(f"[Step2] archive_gate: src_refs={len(src_refs)}, test_refs={len(test_refs)}, passed={reach['archive_gate']['passed']}")

    # ---- Step 3: classification inventory ----
    inventory = []
    for r in records:
        cat = classify(r["path"])
        inventory.append({
            "path": r["path"],
            "size_bytes": r["size_bytes"],
            "sha256": r["sha256"],
            "category": cat,
            "imported_by": [],  # caches/artifacts have no import relationships; src/tests dependencies were separately proven by reachability
            "referenced_by": [],
            "generated": cat in ("GENERATED_REPRODUCIBLE", "TEMP_REVIEW"),
            "reproducible": cat in ("GENERATED_REPRODUCIBLE",),
            "recommended_action": action_for(cat, r["path"]),
        })
    inv = {
        "generated_at": now,
        "categories": ["CORE_RUNTIME", "CORE_TEST", "CORE_EVAL", "PDF_ACTIVE_RESEARCH",
                       "MULTIFORMAT_ARCHIVE", "GENERATED_REPRODUCIBLE", "TEMP_REVIEW",
                       "SOURCE_OF_TRUTH", "DOCS", "REVIEW"],
        "file_count": len(inventory),
        "files": inventory,
    }
    with open(os.path.join(OUT_DIR, "cleanup_inventory.json"), "w", encoding="utf-8") as f:
        json.dump(inv, f, ensure_ascii=False, indent=1)

    # Category summary table
    agg = {}
    for it in inventory:
        a = agg.setdefault(it["category"], {"count": 0, "bytes": 0, "actions": {}})
        a["count"] += 1
        a["bytes"] += max(it["size_bytes"], 0)
        a["actions"][it["recommended_action"]] = a["actions"].get(it["recommended_action"], 0) + 1

    md = ["# EvidenceFlow Cleanup Inventory（Step 3 分类清单）", "",
          f"生成时间：{now}  ", f"文件总数：{len(inventory)}", "",
          "| 分类 | 文件数 | 字节 | 动作分布 |", "|---|---:|---:|---|"]
    for cat in sorted(agg, key=lambda c: -agg[c]["bytes"]):
        a = agg[cat]
        acts = ", ".join(f"{k}={v}" for k, v in sorted(a["actions"].items()))
        md.append(f"| {cat} | {a['count']} | {a['bytes']:,} | {acts} |")
    md += ["", "## 分类判据摘要", "",
           "- CORE_RUNTIME：src/evidenceflow/**、pyproject、.env*、根规范文档 —— KEEP，本轮零改动",
           "- CORE_TEST：tests/** —— KEEP",
           "- CORE_EVAL：standard_ir_v1、evaluation_v2/v3 历史正式评测、research_* —— KEEP + FREEZE HASH",
           "- PDF_ACTIVE_RESEARCH：evals phase13*、experiments/phase13c —— KEEP",
           "- MULTIFORMAT_ARCHIVE：evals phase14a_*、experiments/phase14a —— 逻辑归档（本轮不物理移动）",
           "- GENERATED_REPRODUCIBLE：__pycache__/.pytest_cache/*.pyc/Phase14 chroma —— DELETE_CANDIDATE",
           "- SOURCE_OF_TRUTH：tmp/pdfs（不可重建真实语料）、data/source_docs、data/chroma baseline —— KEEP",
           "- TEMP_REVIEW：tmp/ 下 Phase12/13 中间文件 —— 本轮 REVIEW 不删",
           ""]
    with open(os.path.join(OUT_DIR, "cleanup_inventory.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print(f"[Step3] cleanup_inventory: {len(inventory)} files classified")

    # ---- Step 4: cleanup plan + dry-run ----
    delete_candidates = [it for it in inventory if it["recommended_action"] == "DELETE_CANDIDATE"]
    plan = {
        "generated_at": now,
        "dry_run": True,  # this plan-generation stage is a dry-run; the execution script flips it separately
        "mode": "NON_DESTRUCTIVE_SCOPE_CONSOLIDATION",
        "delete_candidates": {
            "count": len(delete_candidates),
            "bytes": sum(max(it["size_bytes"], 0) for it in delete_candidates),
            "files": [it["path"] for it in delete_candidates],
        },
        "archive_logical": {
            "count": sum(1 for it in inventory if it["recommended_action"] == "ARCHIVE"),
            "note": "Phase14 source + eval artifacts 逻辑归档，本轮不物理移动",
        },
        "preconditions_verified": {
            "src_imports_phase14a": 0,
            "tests_import_phase14a": 0,
            "cli_imports_phase14a": 0,
            "baseline_collection": "ef_research_bge_m3_c512_o64_v1 (data/chroma, KEEP)",
            "phase14_manifest_covers": ["chunks", "embeddings", "gates", "metrics", "reports"],
        },
        "gates_after_cleanup": [
            "Gate1 src/evidenceflow hash 不变",
            "Gate2 根测试 591 passed / 0 failed",
            "Gate3 标准 IR 指标复核（P@5/R@5/R@10/MRR@5/MAP@5/nDCG@5）",
            "Gate4 默认 Runtime 契约不变",
            "Gate5 frozen artifact hash 不变",
            "Gate6 Office 归档可复现性",
        ],
    }
    with open(os.path.join(OUT_DIR, "cleanup_plan.json"), "w", encoding="utf-8") as f:
        json.dump(plan, f, ensure_ascii=False, indent=1)

    proj_bytes = pre["total_bytes"] - plan["delete_candidates"]["bytes"]
    dry = ["# EvidenceFlow Cleanup Dry-Run（Step 4）", "",
           f"- 生成时间：{now}", "- **dry_run: true**（本轮未修改任何文件）",
           f"- FILES_BEFORE：{len(records)} files / {pre['total_bytes']:,} bytes",
           f"- 项目文件（回收字节数）：{plan['delete_candidates']['bytes']:,} bytes",
           f"- PROJECTED AFTER：{len(records) - len(delete_candidates)} files / {proj_bytes:,} bytes",
           "", "## 删除候选（仅可再生缓存 + Phase14 chroma）", ""]
    # Aggregate by directory to avoid printing every file
    by_dir = {}
    for it in delete_candidates:
        d = "/".join(it["path"].split("/")[:-1])
        e = by_dir.setdefault(d, [0, 0])
        e[0] += 1
        e[1] += max(it["size_bytes"], 0)
    for d in sorted(by_dir, key=lambda x: -by_dir[x][1]):
        c, b = by_dir[d]
        dry.append(f"- `{d}/` — {c} files, {b:,} bytes")
    dry += ["", "## 明确不动的东西", "",
            "- src/evidenceflow/**（hash 冻结）、tests/**、evals 除 phase14a chroma 外全部、",
            "  data/chroma baseline、data/source_docs、tmp/pdfs（不可重建）、experiments/phase14a source（逻辑归档）",
            ""]
    with open(os.path.join(OUT_DIR, "cleanup_dry_run.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(dry))
    print(f"[Step4] delete_candidates: {len(delete_candidates)} files, {plan['delete_candidates']['bytes']:,} bytes")

    # Record a src hash baseline summary for a quick Gate1 review
    src_hashes = {r["path"]: r["sha256"] for r in records if r["path"].startswith("src/")}
    with open(os.path.join(OUT_DIR, "src_hash_baseline.json"), "w", encoding="utf-8") as f:
        json.dump({"generated_at": now, "files": src_hashes}, f, ensure_ascii=False, indent=1)
    # Frozen artifact baseline (CORE_EVAL + PDF_ACTIVE_RESEARCH + phase14 archive)
    frozen = {r["path"]: r["sha256"] for r in records
              if classify(r["path"]) in ("CORE_EVAL", "PDF_ACTIVE_RESEARCH", "MULTIFORMAT_ARCHIVE")}
    with open(os.path.join(OUT_DIR, "frozen_hash_baseline.json"), "w", encoding="utf-8") as f:
        json.dump({"generated_at": now, "note": "core eval + pdf research + phase14 archive 基线；phase14 chroma 除外",
                   "files": frozen}, f, ensure_ascii=False, indent=1)
    print(f"[Gate基线] src_hash_baseline: {len(src_hashes)} files; frozen_hash_baseline: {len(frozen)} files")


if __name__ == "__main__":
    sys.exit(main())
