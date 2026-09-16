"""已冻结的 Phase 12 fallback 历史脚本。

设计边界（严守计划 §3「禁止修改 Runtime」）：
- 不修改 src/ 任何文件；本脚本是独立 experiments 脚本。
- 复用**冻结**的 DenseRetriever + BgeReranker + ContextBuilder（与正式 Runtime 完全一致）。
- 仅把 Generator 替换为本地 Qwen1.5-0.5B（transformers 直连），符合异常报告列的「换 LLM 供应商」恢复路径。
- 0.5B 不遵守 rag-generator-v2 的 STATUS:/ANSWER: 控制头，故本脚本**不解析控制头**，
  直接把模型自由文本作为 answer_text，并复用项目 CitationParser 抽取 [S1] 类引用（src 不改）。

所有产物写入 evals/evaluation_v3/phase12_rag_quality_v1/*_fallback.json，并打 FALLBACK_MODEL 标记。
指标为启发式（rule-based），**不代表生产 Runtime 答案质量**，仅验证评测管线可端到端执行。
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

# 允许以 experiments/ 为工作目录运行（PYTHONPATH 已含 src）
sys.path.insert(0, "src")

from evidenceflow.citation.parser import parse_citations  # noqa: E402
from evidenceflow.config import Settings  # noqa: E402
from evidenceflow.context import ContextBuilder  # noqa: E402
from evidenceflow.embedding import BgeM3EmbeddingProvider  # noqa: E402
from evidenceflow.ingestion import BgeM3Tokenizer  # noqa: E402
from evidenceflow.retrieval import (  # noqa: E402
    DenseRerankerRetriever,
    DenseRetriever,
    LearnedReranker,
    TransformersCrossEncoderScorer,
)
from evidenceflow.storage import ChromaVectorStore  # noqa: E402

LOCAL_MODEL = "D:/Agent_learn/local_qwen_model_demo/Qwen1.5-0.5B-Chat"
BASE = Path("evals/evaluation_v3/phase12_rag_quality_v1")

_FALLBACK_SYSTEM = (
    "You are a research assistant. Answer the question using ONLY the supplied "
    "knowledge-base context. Every factual claim must cite the relevant source label, "
    "e.g. [S1]. You may cite only source labels that appear in the context. Never invent "
    "a source label. If the context does not contain enough information to answer, say "
    "you cannot answer from the provided context. Answer in the same language as the question."
)


def build_frozen_retriever_context(settings: Settings):
    """复刻 runtime.py 的检索+Context 装配（去掉 Generator）。"""
    embedding_provider = BgeM3EmbeddingProvider(settings.embedding_model)
    vector_store = ChromaVectorStore(settings.vector_db_path, settings.collection_name)
    dense_retriever = DenseRetriever(
        embedding_provider,
        vector_store,
        top_k=settings.reranker_input_k,
    )
    scorer = TransformersCrossEncoderScorer(
        settings.reranker_model,
        device=settings.reranker_device,
        dtype=settings.reranker_dtype,
        max_length=settings.reranker_max_length,
    )
    retriever = DenseRerankerRetriever(
        dense_retriever,
        LearnedReranker(
            scorer,
            batch_size=settings.reranker_batch_size,
            max_candidates=settings.reranker_input_k,
        ),
        candidate_pool_k=settings.reranker_input_k,
        reranker_input_k=settings.reranker_input_k,
        final_top_k=settings.reranker_final_k,
    )
    context_builder = ContextBuilder(
        BgeM3Tokenizer(settings.embedding_model),
        settings.max_context_tokens,
    )
    return retriever, context_builder


def build_local_generator():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(LOCAL_MODEL, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        LOCAL_MODEL, local_files_only=True, torch_dtype="auto"
    )
    return tok, model


def generate_one(tok, model, query: str, context_text: str) -> str:
    user = (
        "Knowledge-base context:\n\n<SOURCE_CONTEXT>\n"
        f"{context_text}\n</SOURCE_CONTEXT>\n\n"
        "Question:\n\n<USER_QUERY>\n" + query + "\n</USER_QUERY>"
    )
    messages = [
        {"role": "system", "content": _FALLBACK_SYSTEM},
        {"role": "user", "content": user},
    ]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inp = tok(text, return_tensors="pt").input_ids
    out = model.generate(
        inp, max_new_tokens=256, do_sample=False, temperature=1.0
    )
    return tok.decode(out[0][inp.shape[1]:], skip_special_tokens=True).strip()


_INSUFFICIENT_HINTS = (
    "cannot answer", "cannot provide", "do not have", "not contain", "not enough",
    "insufficient", "无法回答", "没有足够", "不包含", "知识库中没有", "不能回答",
    "无法提供", "不足",
)


def predict_status(answer_text: str) -> str:
    low = answer_text.lower()
    if any(h in low for h in _INSUFFICIENT_HINTS) and len(answer_text) < 200:
        return "INSUFFICIENT_CONTEXT"
    return "ANSWERED"


def cmd_gen() -> None:
    import os

    os.environ.setdefault("RERANKER_MODEL", ".cache/models/bge-reranker-base")
    settings = Settings()
    settings.reranker_model = os.environ.get("RERANKER_MODEL", settings.reranker_model)

    dataset = json.load(open(BASE / "generation_eval_dataset.json", encoding="utf-8"))
    questions = dataset["questions"]

    retriever, context_builder = build_frozen_retriever_context(settings)
    tok, model = build_local_generator()

    outputs = []
    t0 = time.time()
    for i, q in enumerate(questions, 1):
        query = q["query"]
        retrieved = retriever.retrieve(query)
        built = context_builder.build(retrieved)
        t1 = time.time()
        answer_text = generate_one(tok, model, query, built.context_text)
        gen_seconds = time.time() - t1

        parsed = parse_citations(answer_text, built)
        record = {
            "question_id": q["question_id"],
            "answerability_gold": q["answerability"],
            "query": query,
            "query_language": q.get("query_language"),
            "retrieved_chunk_ids": [c.chunk.chunk_id for c in retrieved],
            "retrieved_ranks": [c.rank for c in retrieved],
            "context_chunk_ids": [e.chunk_id for e in built.entries],
            "context_source_labels": [e.source_label for e in built.entries],
            "context_text_sha256": _sha(built.context_text),
            "context_total_tokens": built.total_tokens,
            "context_included_count": built.included_count,
            "context_skipped_count": built.skipped_count,
            "answer_text": answer_text,
            "valid_citations": list(parsed.valid_labels),
            "invalid_citations": list(parsed.invalid_labels),
            "predicted_status": predict_status(answer_text),
            "generation_seconds": round(gen_seconds, 3),
        }
        outputs.append(record)
        print(f"[{i}/{len(questions)}] {q['question_id']} "
              f"ctx={built.included_count} cit={len(parsed.valid_labels)} "
              f"status={record['predicted_status']} {round(gen_seconds,1)}s")

    payload = {
        "fallback_model": LOCAL_MODEL,
        "note": "FALLBACK_MODEL / 非生产质量；0.5B 不遵守 STATUS:/ANSWER: 控制头，answer_text 为自由文本",
        "question_count": len(outputs),
        "total_seconds": round(time.time() - t0, 1),
        "outputs": outputs,
    }
    json.dump(payload, open(BASE / "runtime_outputs_fallback.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("wrote runtime_outputs_fallback.json")


def _sha(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["gen", "metrics"])
    args = ap.parse_args()
    if args.cmd == "gen":
        cmd_gen()
    else:
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "phase12_fallback_metrics",
            str(Path(__file__).with_name("phase12_fallback_metrics.py")),
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mod.cmd_metrics()
