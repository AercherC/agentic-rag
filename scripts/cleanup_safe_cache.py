# -*- coding: utf-8 -*-
"""EvidenceFlow cleanup Step 5: export the Phase14 Chroma collection inventory, then clean safe caches.

Deletion scope (strictly limited, plan §14/§15):
1. every __pycache__/ directory, .pytest_cache/, *.pyc - purely regenerable caches
2. evals/.../phase14a_multiformat_ingestion_benchmark/phase14a_c/chroma/
   - the 8-arm benchmark vector index, rebuildable from chunks+embeddings via runners/embed14c.py;
      before deleting, export the collection name/record count/config/model to
      chroma_collection_inventory.json

Never deleted: any .py source, embeddings (anchored by the frozen manifest hash), the baseline
data/chroma.
"""

import json
import os
import shutil
import sqlite3
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "artifacts", "cleanup")
CHROMA_DIR = os.path.join(
    ROOT, "evals", "evaluation_v3", "phase14a_multiformat_ingestion_benchmark",
    "phase14a_c", "chroma")


def export_chroma_inventory():
    """Export the collection inventory before deletion: name, record count, dimension, embedding
    model identifier.

    Plan §15 requires that, before deletion, the collection manifest, record count, config hash,
    embedding model and source hash be saved; the source hash is already anchored by
    phase14a_c/manifests/manifest.json.
    """
    db_path = os.path.join(CHROMA_DIR, "chroma.sqlite3")
    inv = {"exported_at": datetime.now(timezone.utc).isoformat(),
           "chroma_dir": os.path.relpath(CHROMA_DIR, ROOT).replace("\\", "/"),
           "collections": []}
    con = sqlite3.connect(db_path)
    # collections table: id/name; dimension information is taken from the segment metadata
    for cid, name in con.execute("SELECT id, name FROM collections").fetchall():
        # Number of embedding records: the embeddings table joins on the collection
        try:
            n = con.execute(
                "SELECT COUNT(*) FROM embeddings WHERE collection_id = ?", (cid,)).fetchone()[0]
        except sqlite3.Error:
            n = -1
        # Write-queue dimension (surrogate evidence for the config hash: record the segment
        # metadata JSON)
        seg_meta = con.execute(
            "SELECT key, str_value, int_value, float_value FROM collection_metadata WHERE collection_id = ?",
            (cid,)).fetchall() if True else []
        inv["collections"].append({
            "collection_name": name,
            "collection_id": cid,
            "record_count": n,
            "metadata": [{"key": k, "str": s, "int": i, "float": fl} for k, s, i, fl in seg_meta],
        })
    con.close()
    # Record the size and file count of the whole chroma directory as evidence of the reclaimed
    # space
    total, files_n = 0, 0
    for dirpath, _, filenames in os.walk(CHROMA_DIR):
        for fn in filenames:
            total += os.path.getsize(os.path.join(dirpath, fn))
            files_n += 1
    inv["dir_bytes"] = total
    inv["dir_files"] = files_n
    # Embedding model and rebuild entry point (fixed facts read from the 14A-C runner contract)
    inv["rebuild_hint"] = "chunks/*.jsonl + embeddings/*.npz 经 experiments/phase14a/runners/embed14c.py 可重建"
    inv["embedding_model_note"] = "见 phase14a_c/metrics/metrics_by_arm.json 与 manifest.json（bge-m3 离线本地模型）"
    out = os.path.join(OUT_DIR, "chroma_collection_inventory.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(inv, f, ensure_ascii=False, indent=1)
    return inv


def main():
    now = datetime.now(timezone.utc).isoformat()
    log = {"executed_at": now, "dry_run": False, "removed": [], "errors": []}

    # ---- Export the chroma inventory first (a precondition for deletion) ----
    inv = export_chroma_inventory()
    print(f"[chroma inventory] {len(inv['collections'])} collections, "
          f"{inv['dir_files']} files, {inv['dir_bytes']:,} bytes -> chroma_collection_inventory.json")

    # ---- 1. Clean __pycache__ / .pytest_cache / *.pyc ----
    removed_files, removed_bytes = 0, 0
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if d not in (".cache", ".workbuddy")]
        for d in list(dirnames):
            if d == "__pycache__" or d == ".pytest_cache":
                target = os.path.join(dirpath, d)
                # Count first, then delete the whole directory
                for dp, _, fns in os.walk(target):
                    for fn in fns:
                        removed_bytes += os.path.getsize(os.path.join(dp, fn))
                        removed_files += 1
                shutil.rmtree(target, ignore_errors=False)
                log["removed"].append(os.path.relpath(target, ROOT).replace("\\", "/") + "/")
                dirnames.remove(d)
        for fn in filenames:
            if fn.endswith(".pyc"):
                p = os.path.join(dirpath, fn)
                removed_bytes += os.path.getsize(p)
                os.remove(p)
                removed_files += 1
    print(f"[cache] removed {removed_files} cache files, {removed_bytes:,} bytes")

    # ---- 2. Delete the Phase14 chroma directory ----
    if os.path.isdir(CHROMA_DIR):
        shutil.rmtree(CHROMA_DIR)
        log["removed"].append(os.path.relpath(CHROMA_DIR, ROOT).replace("\\", "/") + "/")
        print(f"[phase14 chroma] removed {inv['dir_files']} files, {inv['dir_bytes']:,} bytes")

    log["summary"] = {
        "cache_files_removed": removed_files,
        "cache_bytes_removed": removed_bytes,
        "chroma_bytes_removed": inv["dir_bytes"],
        "chroma_files_removed": inv["dir_files"],
        "total_bytes_reclaimed": removed_bytes + inv["dir_bytes"],
    }
    with open(os.path.join(OUT_DIR, "cleanup_execution_log.json"), "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=1)
    print(f"[total] reclaimed {removed_bytes + inv['dir_bytes']:,} bytes")


if __name__ == "__main__":
    main()
