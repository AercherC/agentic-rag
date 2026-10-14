"""A3：只读记录 active 索引与 Chroma collection 列表的身份，用于真实验证前后对比。

只通过 chromadb 只读接口读取（``get_collection`` / ``list_collections`` / ``get``），
不会创建、修改或删除任何 collection；输出写入调用方指定的 JSON。
身份包含：计数、chunk_id_hash、内容哈希（id + 文本 + metadata + 向量字节）、
collection metadata，以及 index_state.json 与 active manifest 的 sha256。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import chromadb  # noqa: E402
import numpy as np  # noqa: E402

from evidenceflow.config import Settings  # noqa: E402
from evidenceflow.knowledge import LifecycleState  # noqa: E402


def sha256_file(path: Path) -> str:
    """文件字节级 sha256；用于判断 JSON 是否被改写（含格式变化）。"""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def chunk_id_hash(chunk_ids: list[str]) -> str:
    """与 Phase 15A 相同的口径：排序后的紧凑 JSON 数组的 sha256。"""

    serialized = json.dumps(sorted(chunk_ids), ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def collection_identity(client: Any, name: str) -> dict[str, Any]:
    """读取一个 collection 的计数、metadata 与内容哈希（向量按 float32 字节参与）。"""

    collection = client.get_collection(name=name)
    records = collection.get(include=["documents", "metadatas", "embeddings"])
    order = sorted(range(len(records["ids"])), key=lambda i: records["ids"][i])
    digest = hashlib.sha256()
    for index in order:
        digest.update(records["ids"][index].encode("utf-8"))
        digest.update((records["documents"][index] or "").encode("utf-8"))
        digest.update(
            json.dumps(records["metadatas"][index], sort_keys=True, ensure_ascii=False).encode("utf-8")
        )
        digest.update(np.asarray(records["embeddings"][index], dtype=np.float32).tobytes())
    return {
        "name": name,
        "count": collection.count(),
        "chunk_id_hash": chunk_id_hash(list(records["ids"])),
        "content_hash": digest.hexdigest(),
        "metadata": dict(collection.metadata or {}),
    }


def snapshot(settings: Settings) -> dict[str, Any]:
    """生成完整身份快照：全部 collection 列表 + active 详情 + 状态文件 sha256。"""

    state_path = Path(settings.index_state_path)
    state = LifecycleState.model_validate_json(state_path.read_text(encoding="utf-8"))
    manifest_path = ROOT / state.active.manifest_path
    client = chromadb.PersistentClient(path=str(settings.vector_db_path))
    listed = sorted(client.list_collections(), key=lambda item: item.name)
    collections = {
        item.name: {"count": client.get_collection(item.name).count(), "metadata": dict(item.metadata or {})}
        for item in listed
    }
    return {
        "collections": collections,
        "collection_names": sorted(collections),
        "active": {
            **collection_identity(client, state.active.collection_name),
            "index_signature": state.active.index_signature,
        },
        "state_file": {
            "path": str(state_path),
            "sha256": sha256_file(state_path),
            "active": state.active.model_dump(mode="json"),
            "previous": state.previous.model_dump(mode="json") if state.previous else None,
            "staging": state.staging.model_dump(mode="json") if state.staging else None,
        },
        "active_manifest": {"path": state.active.manifest_path, "sha256": sha256_file(manifest_path)},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = snapshot(Settings())
    # 与项目 write_json 一致：UTF-8、不转义中文、键排序。
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(args.output), "collections": len(result["collection_names"]),
                      "active_count": result["active"]["count"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
