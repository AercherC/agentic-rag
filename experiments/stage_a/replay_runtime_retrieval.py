"""A1：通过线上装配路径 ``runtime.build_retriever`` 回放冻结的 40 题检索。

输出 schema 与 ``production_reranked_baseline.json`` 一致，便于复用 ``compare_replay``。
身份常量与工具函数全部复用 ``run_production_reranked_baseline_replay``，不重复定义 hash。
数值 Gate 必须在 CPU 下运行：解析设备不是 cpu 且未传 ``--allow-gpu`` 时直接报错退出。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
for extra in (ROOT, ROOT / "src"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from evidenceflow.config import Settings  # noqa: E402
from evidenceflow.eval.ir_metrics import evaluate_ir_run  # noqa: E402
from evidenceflow.model_device import resolve_model_device  # noqa: E402
from evidenceflow.runtime import build_retriever, resolve_active_collection  # noqa: E402
from evidenceflow.storage.chroma import ChromaVectorStore  # noqa: E402
from experiments.phase15b_active_relevance import project_active_relevance  # noqa: E402
from experiments.run_production_reranked_baseline_replay import (  # noqa: E402
    CHUNK_ID_HASH,
    COLLECTION_NAME,
    CORPUS_SIGNATURE,
    INDEX_SIGNATURE,
    QUESTIONS_V1_PATH,
    QUESTIONS_V2_PATH,
    SNAPSHOT_PATH,
    configure_offline_cache,
    metric_summary,
    read_json,
    sha256_file,
    sorted_chunk_id_hash,
    validate_identity,
    write_json,
)

PIPELINE = "runtime Exact Dense Top20 -> BAAI/bge-reranker-base -> Top5"


class DeviceError(RuntimeError):
    """数值 Gate 要求 CPU，但解析出的设备不是 CPU。"""


def run(output: Path, *, candidates_only: bool, allow_gpu: bool) -> dict[str, Any]:
    """执行一次线上路径回放并写出 JSON。"""

    configure_offline_cache()
    started = time.perf_counter()
    settings = Settings()
    # 在加载任何模型之前先解析设备：冻结基线都是 CPU 结果，GPU 浮点差异可能改变排序。
    device = resolve_model_device(settings.model_device)
    if device != "cpu" and not allow_gpu:
        raise DeviceError(
            f"resolved device is {device!r}; set MODEL_DEVICE=cpu or pass --allow-gpu"
        )

    snapshot = read_json(SNAPSHOT_PATH)
    questions_v1 = read_json(QUESTIONS_V1_PATH)
    questions_v2 = read_json(QUESTIONS_V2_PATH)
    question_ids = [q["question_id"] for q in questions_v1["questions"]]
    if question_ids != [q["question_id"] for q in questions_v2["questions"]] or len(question_ids) != 40:
        raise RuntimeError("replay requires the frozen identical 40Q question order")
    by_id = {q["question_id"]: q for q in questions_v1["questions"]}

    # A3：collection 由 lifecycle active 指针决定（与 build_retriever 同一个解析入口）。
    active_collection = resolve_active_collection(settings)
    # 只读校验 active collection 身份，并取得 chunk 用于构建 active relevance 投影。
    # validate_identity 属于已冻结的历史脚本，读的是 settings.collection_name，这里传入解析后的副本。
    identity_store = ChromaVectorStore(settings.vector_db_path, active_collection)
    chunks = validate_identity(
        settings.model_copy(update={"collection_name": active_collection}), snapshot, identity_store
    )
    projection = project_active_relevance(chunks, questions_v2)

    # 线上装配路径：与 build_rag_service 使用同一个 build_retriever。
    build_started = time.perf_counter()
    retriever = build_retriever(settings)
    build_seconds = time.perf_counter() - build_started
    dense = retriever.dense_retriever
    index = dense.vector_store
    actual_device = dense.embedding_provider.device
    actual_chunk_hash = sorted_chunk_id_hash(index.chunk_ids)
    if actual_chunk_hash != CHUNK_ID_HASH:
        raise RuntimeError("Exact index chunk IDs do not match the frozen chunk_id_hash")

    # 单独计时 Exact 索引的纯构建耗时（读 Chroma 向量 + 构建矩阵），不含模型加载。
    index_started = time.perf_counter()
    pairs = list(identity_store.iter_all_chunks_with_embeddings())
    type(index)([c for c, _ in pairs], [e for _, e in pairs])
    index_build_seconds = time.perf_counter() - index_started

    candidate_run: dict[str, list[str]] = {}
    candidate_scores: dict[str, list[float]] = {}
    final_run: dict[str, list[str]] = {}
    per_query: dict[str, Any] = {}
    search_ms: list[float] = []
    for qid in question_ids:
        query = by_id[qid]["english_query"]
        # 先单独跑一次 Dense Top20（取分数与耗时）；Embedding 在 CPU 上是确定性的。
        embedding = dense.embedding_provider.embed_query(query)
        t0 = time.perf_counter()
        results = index.search(embedding, top_k=settings.reranker_input_k)
        search_ms.append((time.perf_counter() - t0) * 1000)
        candidate_run[qid] = [r.chunk.chunk_id for r in results]
        candidate_scores[qid] = [r.similarity_score for r in results]
        entry: dict[str, Any] = {
            "question_id": qid,
            "query": query,
            "candidate_chunk_ids": candidate_run[qid],
            "candidate_scores": candidate_scores[qid],
        }
        if not candidates_only:
            top = retriever.retrieve(query, top_k=settings.reranker_final_k)
            final_run[qid] = [r.chunk.chunk_id for r in top]
            entry["top5"] = [
                {
                    "rank": r.rank,
                    "chunk_id": r.chunk.chunk_id,
                    "document_id": r.chunk.document_id,
                    "candidate_rank": r.candidate_rank,
                    "reranker_score": r.reranker_score,
                }
                for r in top
            ]
        per_query[qid] = entry

    payload: dict[str, Any] = {
        "schema_version": "stage-a-runtime-replay-v1",
        "status": "PASS",
        "mode": "candidates_only" if candidates_only else "full",
        "pipeline": PIPELINE,
        # 身份字段与 compare_replay 对齐；常量来自冻结脚本，chunk_id_hash 另由实际索引复核。
        "collection_name": active_collection,
        "corpus_signature": CORPUS_SIGNATURE,
        "index_signature": INDEX_SIGNATURE,
        "chunk_id_hash": actual_chunk_hash,
        "expected_collection_name": COLLECTION_NAME,
        "document_count": 10,
        "chunk_count": index.size,
        "query_count": len(question_ids),
        "query_source": "evals/research_retrieval/research_questions_v1.json:english_query",
        "device": actual_device,
        "backend": index.backend,
        "index_size": index.size,
        "dimension": index.dimension,
        "active_relevance_projection_hash": projection["active_relevance_projection_hash"],
        "candidate_run": candidate_run,
        "candidate_scores": candidate_scores,
        "per_query": per_query,
        "timing": {
            "build_retriever_seconds": build_seconds,
            "exact_index_build_seconds": index_build_seconds,
            "dense_search_ms_mean": sum(search_ms) / len(search_ms),
            "dense_search_ms_max": max(search_ms),
        },
        "input_hashes": {
            "research_questions_v1": sha256_file(QUESTIONS_V1_PATH),
            "research_questions_v2": sha256_file(QUESTIONS_V2_PATH),
            "phase15a_snapshot": sha256_file(SNAPSHOT_PATH),
        },
    }
    if not candidates_only:
        ir = evaluate_ir_run(projection["qrels"], final_run, depths=(1, 3, 5, 10, 20))
        payload["final_run"] = final_run
        payload["standard_ir"] = ir
        payload["summary_at_5"] = metric_summary(ir)
    payload["duration_seconds"] = time.perf_counter() - started
    write_json(output, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidates-only", action="store_true", help="只输出 Dense Top20，不做 rerank")
    parser.add_argument("--allow-gpu", action="store_true", help="允许非 CPU 设备（结果不作为 Gate）")
    args = parser.parse_args()
    try:
        payload = run(args.output, candidates_only=args.candidates_only, allow_gpu=args.allow_gpu)
    except DeviceError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": payload["status"],
                "output": str(args.output),
                "device": payload["device"],
                "backend": payload["backend"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
