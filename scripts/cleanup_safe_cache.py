# -*- coding: utf-8 -*-
"""EvidenceFlow 清理 Step 5：导出 Phase14 Chroma collection inventory 后清理安全缓存。

删除范围（严格限定，方案 §14/§15）：
1. 所有 __pycache__/ 目录、.pytest_cache/、*.pyc —— 纯可再生缓存
2. evals/.../phase14a_multiformat_ingestion_benchmark/phase14a_c/chroma/
   —— 8-arm benchmark 向量索引，可由 chunks+embeddings 经 runners/embed14c.py 重建；
      删除前导出 collection 名称/记录数/config/model 到 chroma_collection_inventory.json

不删除：任何 .py 源码、embeddings（被冻结 manifest hash 锚定）、baseline data/chroma。
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
    """删除前导出 collection inventory：名称、记录数、维度、embedding 模型标识。

    方案 §15 要求删除前必须保存 collection manifest、record count、config hash、
    embedding model、source hash；source hash 已由 phase14a_c/manifests/manifest.json 锚定。
    """
    db_path = os.path.join(CHROMA_DIR, "chroma.sqlite3")
    inv = {"exported_at": datetime.now(timezone.utc).isoformat(),
           "chroma_dir": os.path.relpath(CHROMA_DIR, ROOT).replace("\\", "/"),
           "collections": []}
    con = sqlite3.connect(db_path)
    # collections 表：id/name；segment 元数据中取维度信息
    for cid, name in con.execute("SELECT id, name FROM collections").fetchall():
        # 嵌入记录数：embeddings 表关联 collection
        try:
            n = con.execute(
                "SELECT COUNT(*) FROM embeddings WHERE collection_id = ?", (cid,)).fetchone()[0]
        except sqlite3.Error:
            n = -1
        # 写入队列维度（config hash 的替代证据：记录 segment 元数据 JSON）
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
    # 记录整个 chroma 目录的大小与文件数，作为删除收益证据
    total, files_n = 0, 0
    for dirpath, _, filenames in os.walk(CHROMA_DIR):
        for fn in filenames:
            total += os.path.getsize(os.path.join(dirpath, fn))
            files_n += 1
    inv["dir_bytes"] = total
    inv["dir_files"] = files_n
    # embedding 模型与重建入口说明（从 14A-C runner 契约读取到的固定事实）
    inv["rebuild_hint"] = "chunks/*.jsonl + embeddings/*.npz 经 experiments/phase14a/runners/embed14c.py 可重建"
    inv["embedding_model_note"] = "见 phase14a_c/metrics/metrics_by_arm.json 与 manifest.json（bge-m3 离线本地模型）"
    out = os.path.join(OUT_DIR, "chroma_collection_inventory.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(inv, f, ensure_ascii=False, indent=1)
    return inv


def main():
    now = datetime.now(timezone.utc).isoformat()
    log = {"executed_at": now, "dry_run": False, "removed": [], "errors": []}

    # ---- 先导出 chroma inventory（删除前置条件）----
    inv = export_chroma_inventory()
    print(f"[chroma inventory] {len(inv['collections'])} collections, "
          f"{inv['dir_files']} files, {inv['dir_bytes']:,} bytes -> chroma_collection_inventory.json")

    # ---- 1. 清理 __pycache__ / .pytest_cache / *.pyc ----
    removed_files, removed_bytes = 0, 0
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if d not in (".cache", ".workbuddy")]
        for d in list(dirnames):
            if d == "__pycache__" or d == ".pytest_cache":
                target = os.path.join(dirpath, d)
                # 统计后整目录删除
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

    # ---- 2. 删除 Phase14 chroma 目录 ----
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
