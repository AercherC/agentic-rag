# -*- coding: utf-8 -*-
"""Production Chroma Index Integrity Smoke.

This script read-only checks the Collection the current config points at: existence, record count, metadata coverage,
full recovery of Domain Chunks, a sample query, and the Collection identity. It never repairs, deletes or switches any
Collection, so it can be run repeatedly after ingestion, runtime promotion and cleanup.

Genuinely read-only: a missing persist directory fails immediately instead of constructing ``PersistentClient`` (which
would create the directory); ``ChromaVectorStore`` is opened with ``create_if_missing=False``, so even if the Collection
is deleted between the existence check and the open, no empty Collection is left behind.

By default it only prints the JSON result to stdout and writes no file; only an explicit ``--output PATH`` additionally
writes that file (so git-tracked historical artifacts are not modified by accident). Exit codes: 0 = all checks passed;
2 = some check failed.
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
    """Read the Collection's id/document/metadata in batches, without fetching the unrelated vectors."""

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
    """Return the integrity smoke result; failed checks keep their diagnostics instead of swallowing the exception."""

    result: dict[str, Any] = {
        "persist_path": str(Path(persist_path).resolve()),
        "collection_name": collection_name,
        "expected_count": expected_count,
        "checks": {},
    }
    # Check the directory first: chromadb.PersistentClient creates it when missing, and a read-only
    # script must not have that side effect.
    if not Path(persist_path).is_dir():
        result["checks"]["collection_exists"] = False
        result["error"] = f"Persist path does not exist: {result['persist_path']}"
        result["pass"] = False
        return result
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

        # create_if_missing=False: if the Collection disappears between the existence check and the
        # open, this must raise
        # rather than create an empty one.
        store = ChromaVectorStore(persist_path, collection_name, create_if_missing=False)
        try:
            restored = list(store.iter_all_chunks())
            result["restored_count"] = len(restored)
            result["checks"]["iter_all_chunks"] = len(restored) == count
        except Exception as exc:  # record the failure reason for direct diagnosis by the deployment gate
            result["checks"]["iter_all_chunks"] = False
            result["iter_all_chunks_error"] = f"{type(exc).__name__}: {exc}"

        embedding_result = collection.get(limit=1, include=["embeddings"])
        raw_embeddings = embedding_result.get("embeddings")
        embeddings = list(raw_embeddings) if raw_embeddings is not None else []
        if embeddings:
            try:
                # Chroma 1.x may return numpy.ndarray; the Adapter contract accepts a Python
                # sequence.
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
    """Run the read-only smoke and print the result to stdout; write a file only when ``--output`` is given explicitly."""

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
        # Persist only when the user explicitly gives a path; by default the run stays completely
        # read-only.
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
