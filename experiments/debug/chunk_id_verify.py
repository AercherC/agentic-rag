"""临时验证：对 Chroma 中已持久化的 chunk 复算 chunk_id，验证确定性 ID 合同。

同时检验一个设计观察：Chroma metadata 里没有落盘 chunk_size / chunk_overlap，
但 chunk_id 依赖它们。如果全部复算命中，说明当前 collection 是同构的（全是 C512/O64）。
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# 数据和模型缓存相对项目根目录，而非 experiments 目录。
ROOT = Path(__file__).resolve().parents[2]
os.environ["HF_HOME"] = str((ROOT / ".cache" / "huggingface").resolve())
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
sys.path.insert(0, str(ROOT / "src"))

from evidenceflow.config import Settings  # noqa: E402
from evidenceflow.domain.ids import build_chunk_id  # noqa: E402
from evidenceflow.storage.chroma import ChromaVectorStore  # noqa: E402


def main() -> None:
    settings = Settings()
    store = ChromaVectorStore(settings.vector_db_path, settings.collection_name)
    raw = store._collection.get(include=["documents", "metadatas"])

    ids = raw["ids"]
    docs = raw["documents"]
    metas = raw["metadatas"]
    print(f"collection = {settings.collection_name}")
    print(f"chunk 总数 = {len(ids)}")

    # 用当前 Settings 的切分配置复算；若 collection 混入了其它配置，这里就会失配。
    cs, co = settings.chunk_size, settings.chunk_overlap
    print(f"复算配置 chunk_size={cs} chunk_overlap={co}\n")

    mismatch = []
    for cid, content, meta in zip(ids, docs, metas):
        recomputed = build_chunk_id(
            document_id=meta["document_id"],
            chunk_size=cs,
            chunk_overlap=co,
            chunk_index=meta["chunk_index"],
            content=content,
        )
        if recomputed != cid:
            mismatch.append({"stored": cid, "recomputed": recomputed, "file": meta["file_name"]})

    print(f"chunk_id 复算命中: {len(ids) - len(mismatch)}/{len(ids)}")
    if mismatch:
        print(f"失配 {len(mismatch)} 条（说明 collection 内混入了不同切分配置）")
        for item in mismatch[:5]:
            print("  ", item)
    else:
        print("全部命中 → 确定性 ID 合同在真实持久化数据上成立，且 collection 切分配置同构。")

    # 附加检查：metadata 里到底有哪些键
    print(f"\nmetadata 键: {sorted(metas[0].keys())}")
    print(f"是否落盘 chunk_size/overlap: "
          f"{'chunk_size' in metas[0] or 'chunk_overlap' in metas[0]}")

    # 每文档 chunk 数与页码覆盖
    per_file: dict[str, int] = {}
    no_page = 0
    for meta in metas:
        per_file[meta["file_name"]] = per_file.get(meta["file_name"], 0) + 1
        if "page_start" not in meta:
            no_page += 1
    print(f"\n文档数 {len(per_file)}，无页码 chunk {no_page}")
    for name, count in sorted(per_file.items()):
        print(f"  {count:>4}  {name}")


if __name__ == "__main__":
    main()
