# -*- coding: utf-8 -*-
"""Production Chroma Index Integrity Smoke。

该脚本只读检查当前配置指向的 Collection：存在性、记录数、metadata 覆盖率、
Domain Chunk 全量恢复、样本查询以及 Collection 身份。它不会修复、删除或切换
任何 Collection，适合在 ingestion、runtime promotion 和 cleanup 后重复执行。

默认只把 JSON 结果打印到 stdout，不写任何文件；只有显式传入 ``--output PATH`` 时才额外
写入该文件（避免误改被 git 跟踪的历史产物）。退出码：0 = 全部检查通过；2 = 有检查失败。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import chromadb

from evidenceflow.config import Settings
from evidenceflow.storage import ChromaVectorStore


def _raw_records(collection: Any) -> list[dict[str, Any]]:
    """分批读取 Collection 的 id/document/metadata，不读取无关向量。"""

    records: list[dict[str, Any]] = []
    offset = 0
    page_size = 1024
    while True:
        result = collection.get(
            limit=page_size,
            offset=offset,
            include=["documents", "metadatas"],
        )
        ids = result.get("ids") or []
        documents = result.get("documents") or []
        metadatas = result.get("metadatas") or []
        if not ids:
            break
        if not (len(ids) == len(documents) == len(metadatas)):
            raise ValueError("Chroma returned misaligned ids/documents/metadatas")
        records.extend(
            {
                "chunk_id": chunk_id,
                "document": document,
                "metadata": metadata,
            }
            for chunk_id, document, metadata in zip(ids, documents, metadatas)
        )
        if len(ids) < page_size:
            break
        offset += page_size
    return records


def inspect_active_index(
    persist_path: str | Path,
    collection_name: str,
    *,
    expected_count: int | None = None,
    expected_strategy: str = "hybrid",
) -> dict[str, Any]:
    """返回完整性 smoke 结果；失败项保留诊断而不是吞掉异常。"""

    result: dict[str, Any] = {
        "persist_path": str(Path(persist_path).resolve()),
        "collection_name": collection_name,
        "expected_count": expected_count,
        "checks": {},
    }
    try:
        client = chromadb.PersistentClient(path=str(persist_path))
        collection_names = {item.name for item in client.list_collections()}
        exists = collection_name in collection_names
        result["checks"]["collection_exists"] = exists
        if not exists:
            result["error"] = f"Collection does not exist: {collection_name}"
            result["pass"] = False
            return result

        collection = client.get_collection(name=collection_name)
        count = collection.count()
        result["count"] = count
        result["checks"]["count_positive"] = count > 0
        result["checks"]["count_expected"] = (
            expected_count is None or count == expected_count
        )

        raw = _raw_records(collection)
        metadata_count = sum(isinstance(item["metadata"], dict) for item in raw)
        result["metadata_count"] = metadata_count
        result["metadata_coverage"] = metadata_count / len(raw) if raw else 0.0
        result["checks"]["metadata_coverage_100"] = bool(raw) and metadata_count == len(raw)

        metadata = collection.metadata or {}
        configuration = collection.configuration or {}
        hnsw = configuration.get("hnsw") or {}
        result["collection_metadata"] = metadata
        result["collection_configuration"] = {
            "space": hnsw.get("space"),
            "embedding_function": type(configuration.get("embedding_function")).__name__
            if configuration.get("embedding_function") is not None
            else None,
        }
        result["checks"]["collection_identity"] = (
            metadata.get("chunk_strategy") == expected_strategy
            and hnsw.get("space") == "cosine"
        )

        store = ChromaVectorStore(persist_path, collection_name)
        try:
            restored = list(store.iter_all_chunks())
            result["restored_count"] = len(restored)
            result["checks"]["iter_all_chunks"] = len(restored) == count
        except Exception as exc:  # 记录失败原因，供部署门直接诊断
            result["checks"]["iter_all_chunks"] = False
            result["iter_all_chunks_error"] = f"{type(exc).__name__}: {exc}"

        embedding_result = collection.get(limit=1, include=["embeddings"])
        raw_embeddings = embedding_result.get("embeddings")
        embeddings = list(raw_embeddings) if raw_embeddings is not None else []
        if embeddings:
            try:
                # Chroma 1.x 可能返回 numpy.ndarray；Adapter 合同接收 Python 序列。
                sample_vector = embeddings[0].tolist() if hasattr(embeddings[0], "tolist") else embeddings[0]
                sample = store.search(sample_vector, top_k=1)
                result["sample_query_count"] = len(sample)
                result["checks"]["sample_query"] = len(sample) == 1
            except Exception as exc:
                result["checks"]["sample_query"] = False
                result["sample_query_error"] = f"{type(exc).__name__}: {exc}"
        else:
            result["checks"]["sample_query"] = False
            result["sample_query_error"] = "Collection has no stored embedding"
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["checks"]["collection_exists"] = False

    result["pass"] = all(result["checks"].values()) if result["checks"] else False
    return result


def main(argv: list[str] | None = None) -> int:
    """执行只读 smoke，结果打印到 stdout；仅在显式给出 ``--output`` 时写文件。"""

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--path", type=Path, default=None, help="Chroma 持久化目录")
    parser.add_argument("--collection", default=None, help="待检查 Collection 名称")
    parser.add_argument("--expected-count", type=int, default=None, help="可选的精确记录数")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="可选：同时把 JSON 结果写入该路径（默认不写文件，只输出到 stdout）",
    )
    args = parser.parse_args(argv)
    settings = Settings()
    path = args.path or settings.vector_db_path
    collection = args.collection or settings.collection_name
    result = inspect_active_index(
        path,
        collection,
        expected_count=args.expected_count,
    )
    if args.output is not None:
        # 只有用户显式指定路径才落盘；默认保持完全只读。
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
