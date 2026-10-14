"""独立回放当前 Production Dense→Learned Reranker→Top5 基线。

该脚本不导入 Phase 15B benchmark runner；它直接装配生产检索 primitive 与
``DenseRerankerRetriever``，将结果冻结为后续 Phase 15B v2 的 B0 parity oracle。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evidenceflow.config import Settings
from evidenceflow.embedding.bge_m3 import BgeM3EmbeddingProvider
from evidenceflow.eval.ir_metrics import evaluate_ir_run
from evidenceflow.retrieval.reranker import (
    DenseRerankerRetriever,
    LearnedReranker,
    TransformersCrossEncoderScorer,
)
from evidenceflow.retrieval.retriever import DenseRetriever
from evidenceflow.storage.chroma import ChromaVectorStore
from experiments.phase15b_active_relevance import project_active_relevance


SNAPSHOT_PATH = ROOT / "artifacts" / "phase15a" / "baseline_snapshot.json"
QUESTIONS_V1_PATH = ROOT / "evals" / "research_retrieval" / "research_questions_v1.json"
QUESTIONS_V2_PATH = (
    ROOT
    / "evals"
    / "evaluation_v2"
    / "phase35_gold_stabilization_v1"
    / "research_questions_v2_gold_stable.json"
)
OUTPUT_PATH = (
    ROOT
    / "evals"
    / "evaluation_v3"
    / "phase15b_hybrid_candidate_v2"
    / "production_reranked_baseline.json"
)

COLLECTION_NAME = "ef_research_bge_m3_hybrid_c512_v2"
CHUNK_ID_HASH = "e87a613ebe65b10262996e83974b55839e0b6c27fbe22053ee7b8826dd1f153d"
CORPUS_SIGNATURE = "bb184726b7f98756bd519d70e1a2a1f1224b5108e689680b2c328023ba1f4332"
INDEX_SIGNATURE = "b4e9432c7170615ade2c53803a887fa86d0f793949934a57db421f0da57739e5"


def read_json(path: Path) -> Any:
    """读取冻结输入 JSON。"""

    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    """计算冻结输入文件的字节级 SHA-256。"""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    """以确定性缩进写出 replay oracle。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def configure_offline_cache() -> None:
    """只允许使用本地模型缓存，避免 replay 依赖隐式网络状态。"""

    os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def resolve_local_model(settings: Settings) -> str:
    """在模型身份不变的前提下选择仓库内已有 checkpoint 路径。"""

    local = ROOT / ".cache" / "models" / "bge-reranker-base"
    if (local / "config.json").is_file() and (local / "model.safetensors").is_file():
        return str(local)
    return settings.reranker_model


def sorted_chunk_id_hash(chunk_ids: Sequence[str]) -> str:
    """按 Phase 15A 的紧凑 JSON 数组规则计算 active chunk ID Hash。"""

    serialized = json.dumps(
        sorted(chunk_ids), ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def validate_identity(settings: Settings, snapshot: Mapping[str, Any], store: ChromaVectorStore):
    """只读验证 Production replay 使用的 Collection 身份。"""

    if settings.collection_name != COLLECTION_NAME:
        raise RuntimeError("Settings collection does not match frozen active collection")
    if store.metadata != snapshot["authoritative_identity"]["chroma_active_collection_metadata"]:
        raise RuntimeError("active Collection metadata differs from Phase 15A snapshot")
    chunks = list(store.iter_all_chunks())
    ids = [chunk.chunk_id for chunk in chunks]
    if len(chunks) != 817 or len(set(ids)) != 817 or sorted_chunk_id_hash(ids) != CHUNK_ID_HASH:
        raise RuntimeError("active Collection chunks do not match frozen identity")
    if len({chunk.document_id for chunk in chunks}) != 10:
        raise RuntimeError("active Collection document count is not 10")
    return chunks


def metric_summary(ir_result: Mapping[str, Any]) -> dict[str, Any]:
    """提取 post-reranker B0 的标准 @5 指标。"""

    aggregate = ir_result["aggregate"]
    return {
        "P@5": aggregate["precision_at_k"][5],
        "Recall@5": aggregate["recall_at_k"][5],
        "MRR@5": aggregate["mrr_at_k"][5],
        "MAP@5": aggregate["map_at_k"][5],
        "nDCG@5": aggregate["ndcg_at_k"][5],
        "eligible_count": aggregate["eligible_count"],
        "excluded_count": aggregate["excluded_count"],
    }


def run_replay(output_path: Path = OUTPUT_PATH) -> dict[str, Any]:
    """执行一次独立 production retrieval replay 并冻结逐题 Top5。"""

    configure_offline_cache()
    started = time.perf_counter()
    snapshot = read_json(SNAPSHOT_PATH)
    questions_v1 = read_json(QUESTIONS_V1_PATH)
    questions_v2 = read_json(QUESTIONS_V2_PATH)
    question_ids = [question["question_id"] for question in questions_v1["questions"]]
    v2_ids = [question["question_id"] for question in questions_v2["questions"]]
    if question_ids != v2_ids or len(question_ids) != 40:
        raise RuntimeError("replay requires the frozen identical 40Q question order")

    settings = Settings()
    store = ChromaVectorStore(settings.vector_db_path, settings.collection_name)
    chunks = validate_identity(settings, snapshot, store)
    projection = project_active_relevance(chunks, questions_v2)

    # 这里按现有生产 wrapper 装配单一路径；查询文本使用 benchmark 冻结的
    # english_query，与 Phase 13F active hybrid retrieval 的 40Q 口径相同。
    embedding_provider = BgeM3EmbeddingProvider(
        settings.embedding_model, device=settings.model_device
    )
    dense = DenseRetriever(embedding_provider, store, top_k=settings.reranker_input_k)
    scorer = TransformersCrossEncoderScorer(
        resolve_local_model(settings),
        device=settings.model_device,
        dtype=settings.reranker_dtype,
        max_length=settings.reranker_max_length,
        inference_mode=False,
    )
    reranker = LearnedReranker(
        scorer,
        batch_size=settings.reranker_batch_size,
        max_candidates=settings.reranker_input_k,
    )
    production_retriever = DenseRerankerRetriever(
        dense,
        reranker,
        candidate_pool_k=settings.reranker_input_k,
        reranker_input_k=settings.reranker_input_k,
        final_top_k=settings.reranker_final_k,
    )

    questions_by_id = {question["question_id"]: question for question in questions_v1["questions"]}
    candidate_run: dict[str, list[str]] = {}
    final_run: dict[str, list[str]] = {}
    per_query: dict[str, dict[str, Any]] = {}
    for question_id in question_ids:
        query = questions_by_id[question_id]["english_query"]
        # wrapper 内部先生成 Production Top20，再用同一 checkpoint 输出 Top5。
        results = production_retriever.retrieve(query, top_k=settings.reranker_final_k)
        final_ids = [result.chunk.chunk_id for result in results]
        # candidate_rank 由 wrapper 输出保留；通过结果无法恢复完整候选，因此单独 replay
        # 一次 Dense Top20 只取 ID，不改变 production final ranking 或模型输入。
        dense_results = dense.retrieve(query, top_k=settings.reranker_input_k)
        candidate_ids = [result.chunk.chunk_id for result in dense_results]
        candidate_run[question_id] = candidate_ids
        final_run[question_id] = final_ids
        per_query[question_id] = {
            "question_id": question_id,
            "query": query,
            "candidate_count": len(candidate_ids),
            "candidate_chunk_ids": candidate_ids,
            "top5": [
                {
                    "rank": result.rank,
                    "chunk_id": result.chunk.chunk_id,
                    "document_id": result.chunk.document_id,
                    "candidate_rank": result.candidate_rank,
                    "reranker_score": result.reranker_score,
                }
                for result in results
            ],
        }

    ir = evaluate_ir_run(projection["qrels"], final_run, depths=(1, 3, 5, 10, 20))
    result = {
        "schema_version": "phase15b-production-reranked-baseline-v1",
        "status": "PASS",
        "pipeline": "production Dense Top20 -> BAAI/bge-reranker-base -> Top5",
        "collection_name": COLLECTION_NAME,
        "corpus_signature": CORPUS_SIGNATURE,
        "index_signature": INDEX_SIGNATURE,
        "chunk_id_hash": CHUNK_ID_HASH,
        "document_count": 10,
        "chunk_count": 817,
        "query_count": len(question_ids),
        "query_source": "evals/research_retrieval/research_questions_v1.json:english_query",
        "reranker": {
            "model": settings.reranker_model,
            "resolved_model_path": str(Path(resolve_local_model(settings)).relative_to(ROOT))
            if Path(resolve_local_model(settings)).is_absolute()
            else resolve_local_model(settings),
            "max_length": settings.reranker_max_length,
            "batch_size": settings.reranker_batch_size,
            "device": scorer.device,
            "dtype": settings.reranker_dtype,
            "final_top_k": settings.reranker_final_k,
        },
        "active_relevance_projection_hash": projection["active_relevance_projection_hash"],
        "active_relevance_projection": {
            "qrels": projection["qrels"],
            "groups": projection["groups_by_question"],
            "hash_serialization": projection["hash_serialization"],
        },
        "candidate_run": candidate_run,
        "final_run": final_run,
        "per_query": per_query,
        "standard_ir": ir,
        "summary_at_5": metric_summary(ir),
        "input_hashes": {
            "research_questions_v1": sha256_file(QUESTIONS_V1_PATH),
            "research_questions_v2": sha256_file(QUESTIONS_V2_PATH),
            "phase15a_snapshot": sha256_file(SNAPSHOT_PATH),
        },
        "duration_seconds": time.perf_counter() - started,
    }
    write_json(output_path, result)
    return result


def parse_args() -> argparse.Namespace:
    """解析 replay 输出路径。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    payload = run_replay(args.output)
    print(json.dumps({"status": payload["status"], "output": str(args.output)}, ensure_ascii=False))
