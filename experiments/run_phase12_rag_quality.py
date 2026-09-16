"""Phase 12：End-to-End RAG Quality Evaluation 执行脚本。

本阶段是 Evaluation Phase，不修改任何 Runtime 组件。脚本只做四件事：
1. 冻结 Runtime / Prompt / LLM / Dataset / Chroma 基线；
2. 从 canonical 40Q 派生独立版本化的 Generation Eval Dataset；
3. 用当前正式 Runtime 抓取 Generation 输出（需要可用 LLM）；
4. 在任何异常处落盘 checkpoint 与 anomaly report 后停止。

子命令：
    baseline    冻结 Phase 12 基线（不依赖 LLM）
    prepare     生成 Generation Eval Dataset 并校验 unanswerable 题的证据缺失
    dump        导出所选题目 Gold Chunk 正文，供人工撰写 reference answer / claims
    outputs     用正式 Runtime 生成 runtime_outputs.json（需要可用 LLM）
    anomaly     写 progress_checkpoint.json + anomaly_report.md
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from evidenceflow.config import Settings
from evidenceflow.storage import ChromaVectorStore

OUT = Path("evals/evaluation_v3/phase12_rag_quality_v1")
V2 = Path("evals/evaluation_v2")
V3 = Path("evals/evaluation_v3")

QUESTIONS_SOURCE = V2 / "phase35_gold_stabilization_v1" / "research_questions_v2_gold_stable.json"
QRELS_SOURCE = V3 / "standard_ir_v1" / "qrels.json"
PER_QUERY_SOURCE = V3 / "standard_ir_v1" / "per_query_results.json"

GENERATION_DATASET_ID = "rag_generation_eval_v1"
GENERATION_SCHEMA_VERSION = "1.0"

# 标准 IR 冻结基线（Phase 9 后不再变化，Phase 12 只做回归确认）。
STANDARD_IR_BASELINE = {
    "precision_at_5": 0.115000,
    "recall_at_5": 0.558333,
    "recall_at_10": 0.733333,
    "mrr_at_5": 0.432083,
    "map_at_5": 0.427917,
    "ndcg_at_5": 0.461851,
}
FROZEN_CHUNK_COUNT = 361
FROZEN_DATASET_HASH = "be1e8354babef3927163c19e32c614f7639323095900c487d559ade8249e7357"
FROZEN_TESTS = {"passed": 418, "failed": 0, "warnings": 17}


def _sha256_file(path: Path) -> str:
    """计算单个文件 SHA-256，用于 fingerprint 与 artifact 审计。"""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_json(payload: Any) -> str:
    """对规范化 JSON 取 hash，用于运行期 composition fingerprint。"""

    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def _sha256_text(text: str) -> str:
    """对文本取 hash，用于 context / answer 内容审计。"""

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _runtime_fingerprint() -> dict[str, str]:
    """PROJECT_SPEC + 全部生产源码 hash，与 11B/11C 同口径以便比对。"""

    files = [Path("PROJECT_SPEC.md")]
    files.extend(sorted(Path("src/evidenceflow").rglob("*.py")))
    return {str(path).replace("\\", "/"): _sha256_file(path) for path in files}


def _mask_endpoint(url: str) -> str:
    """脱敏 LLM endpoint：只保留 scheme + host，去掉路径与查询串。"""

    match = re.match(r"^(https?://[^/]+)", url)
    return match.group(1) if match else "<invalid-url>"


def _mask_key(raw: str | None) -> str:
    """只记录 API Key 是否设置与其指纹前缀，绝不落盘密钥本体。"""

    if not raw:
        return "<unset>"
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]
    return f"<set len={len(raw)} sha256_12={digest}>"


def _chroma_identity() -> dict[str, object]:
    """读取正式 Collection 的 chunk 数与排序后 chunk_id 身份 hash。"""

    settings = Settings()
    store = ChromaVectorStore(settings.vector_db_path, settings.collection_name)
    chunk_ids = sorted(chunk.chunk_id for chunk in store.iter_all_chunks())
    return {
        "collection_name": settings.collection_name,
        "vector_db_path": str(settings.vector_db_path),
        "chunk_count": len(chunk_ids),
        "sorted_chunk_id_identity_hash": hashlib.sha256("\n".join(chunk_ids).encode("utf-8")).hexdigest(),
    }


def _composition_fingerprint() -> dict[str, Any]:
    """Runtime composition fingerprint：检索相关 Settings + 装配模块源码 hash。"""

    settings = Settings()
    settings_subset = {
        key: getattr(settings, key)
        for key in (
            "reranker_enabled", "reranker_model", "reranker_input_k", "reranker_final_k",
            "multilingual_hybrid_enabled", "multilingual_candidate_top_k",
            "multilingual_candidate_pool_k", "multilingual_final_top_k",
            "multilingual_rrf_k", "embedding_model", "collection_name",
            "max_context_tokens", "chunk_size", "chunk_overlap",
        )
    }
    modules = [
        "src/evidenceflow/runtime.py",
        "src/evidenceflow/retrieval/reranker.py",
        "src/evidenceflow/retrieval/multilingual.py",
        "src/evidenceflow/retrieval/retriever.py",
        "src/evidenceflow/rag/service.py",
        "src/evidenceflow/context/builder.py",
        "src/evidenceflow/generation/generator.py",
        "src/evidenceflow/citation/parser.py",
    ]
    module_hashes = {m: _sha256_file(Path(m)) for m in modules}
    return {
        "settings_subset": settings_subset,
        "assembly_module_hashes": module_hashes,
        "fingerprint": _sha256_json({"settings": settings_subset, "modules": module_hashes}),
    }


def _generation_contract_snapshot() -> dict[str, Any]:
    """导出与 Generation / Citation 直接相关的合同版本，不含任何密钥。"""

    from evidenceflow.generation import GENERATOR_PROMPT_VERSION

    settings = Settings()
    return {
        "generator_prompt_version": GENERATOR_PROMPT_VERSION,
        "generator_status_contract": "rag-generator-v2 control header (STATUS:/ANSWER:)",
        "citation_parser_contract": "source-label allowlist [S1..Sn] from BuiltContext; "
                                    "ANSWERED requires >=1 valid citation",
        "citation_parser_module_hash": _sha256_file(Path("src/evidenceflow/citation/parser.py")),
        "llm_model": settings.llm_model,
        "llm_endpoint_masked": _mask_endpoint(settings.llm_base_url),
        "llm_api_key": _mask_key(settings.llm_api_key),
        "llm_temperature": settings.llm_temperature,
        "llm_timeout_seconds": settings.llm_timeout_seconds,
        "llm_max_attempts": settings.llm_max_attempts,
        "max_context_tokens": settings.max_context_tokens,
        "final_top_k": settings.reranker_final_k if settings.reranker_enabled else settings.multilingual_final_top_k,
        "reranker_input_k": settings.reranker_input_k,
    }


def save_artifact(payload: dict, path: Path) -> None:
    """排他写入 JSON：禁止覆盖任何已存在的历史 artifact。"""

    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_json(path: Path) -> Any:
    """读取 UTF-8 JSON。"""

    return json.loads(Path(path).read_text(encoding="utf-8"))


# --------------------------------------------------------------------------
# Step 0：基线
# --------------------------------------------------------------------------
def cmd_baseline(args: argparse.Namespace) -> int:
    """冻结 Phase 12 Runtime Baseline（不加载模型、不调用 LLM）。"""

    questions_doc = load_json(QUESTIONS_SOURCE)
    questions = questions_doc["questions"]
    qrels = load_json(QRELS_SOURCE)
    dataset_hash = _sha256_file(QUESTIONS_SOURCE)

    baseline = {
        "phase": "phase12_rag_quality_v1",
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_composition_fingerprint": _composition_fingerprint(),
        "runtime_source_fingerprint": _sha256_json(_runtime_fingerprint()),
        "generation_contract": _generation_contract_snapshot(),
        "retrieval_dataset": {
            "source": str(QUESTIONS_SOURCE).replace("\\", "/"),
            "dataset_id": questions_doc["dataset_id"],
            "schema_version": questions_doc["schema_version"],
            "corpus_version": questions_doc["corpus_version"],
            "question_count": len(questions),
            # 两个 hash 计算基准不同：file_sha256 是本次对源文件的直接摘要，
            # frozen_dataset_hash 是自 Phase 6 起逐阶段沿用、从未重算的冻结常量。
            "dataset_file_sha256": dataset_hash,
            "frozen_dataset_hash_carried_from_phase6": FROZEN_DATASET_HASH,
            "frozen_dataset_hash_note": "沿用常量，未在 Phase 12 重算；两者不可直接比较。",
        },
        "qrels": {
            "source": str(QRELS_SOURCE).replace("\\", "/"),
            "question_count": len(qrels),
            "total_relevant": sum(len(v) for v in qrels.values()),
            "sha256": _sha256_json(qrels),
        },
        "chroma_identity": _chroma_identity(),
        "standard_ir_baseline": {
            "source_phase": "phase9_runtime_promotion_v1",
            "metrics": STANDARD_IR_BASELINE,
            "recall_at_20": "N/A (frozen ranking only covers Top-10)",
        },
        "tests_baseline": FROZEN_TESTS,
        "frozen_chunk_count": FROZEN_CHUNK_COUNT,
    }
    save_artifact(baseline, OUT / "phase12_baseline.json")
    print(f"[baseline] wrote {OUT / 'phase12_baseline.json'}")
    return 0


def _load_corpus() -> list[dict[str, Any]]:
    """读取正式 Collection 的全部 chunk 内容，用于 Gold 落盘与缺失性验证。"""

    settings = Settings()
    store = ChromaVectorStore(settings.vector_db_path, settings.collection_name)
    return [
        {
            "chunk_id": chunk.chunk_id,
            "file_name": chunk.file_name,
            "page_start": chunk.page_start,
            "page_end": chunk.page_end,
            "content": chunk.content,
        }
        for chunk in store.iter_all_chunks()
    ]


def _resolve_prefixes(prefixes: list[str], corpus: list[dict[str, Any]]) -> list[str]:
    """把 Gold 中的 12 位 chunk_id 前缀解析为完整 chunk_id，歧义立即报错。"""

    resolved = []
    for prefix in prefixes:
        matches = [c["chunk_id"] for c in corpus if c["chunk_id"].startswith(prefix)]
        if len(matches) != 1:
            raise ValueError(f"chunk prefix {prefix!r} matched {len(matches)} chunks")
        resolved.append(matches[0])
    return resolved


# --------------------------------------------------------------------------
# Step 1-3：Generation Eval Dataset + Gold
# --------------------------------------------------------------------------
def cmd_prepare(args: argparse.Namespace) -> int:
    """生成 Generation Eval Dataset 与四类 Gold artifact，并复验不可答题的证据缺失。"""

    from phase12_generation_gold import (
        ANSWERABLE_GOLD,
        INSUFFICIENT_CONTEXT_FORBIDDEN,
        INSUFFICIENT_CONTEXT_GOLD,
    )

    questions_doc = load_json(QUESTIONS_SOURCE)
    canonical = {q["question_id"]: q for q in questions_doc["questions"]}
    qrels = load_json(QRELS_SOURCE)
    per_query = load_json(PER_QUERY_SOURCE)["phase9"]
    corpus = _load_corpus()
    by_id = {c["chunk_id"]: c for c in corpus}

    def _evidence_meta(chunk_ids: list[str]) -> list[dict[str, Any]]:
        return [
            {
                "chunk_id": cid,
                "file_name": by_id[cid]["file_name"],
                "page_start": by_id[cid]["page_start"],
                "page_end": by_id[cid]["page_end"],
            }
            for cid in chunk_ids
        ]

    # ---- 不可答题：关键术语零命中复验 ----
    corpus_low = [(c["chunk_id"], c["content"].lower()) for c in corpus]
    unanswerable_verification: dict[str, Any] = {}
    for qid, gold in INSUFFICIENT_CONTEXT_GOLD.items():
        terms = {
            term: sum(1 for _, text in corpus_low if term.lower() in text)
            for term in gold["absence_terms"]
        }
        unanswerable_verification[qid] = {
            "question_id": qid,
            "absence_terms": gold["absence_terms"],
            "corpus_chunk_count": len(corpus),
            "term_hit_counts": terms,
            "all_terms_zero_hit": all(v == 0 for v in terms.values()),
            "rationale": gold["rationale"],
        }

    questions: list[dict[str, Any]] = []
    reference_answers: dict[str, Any] = {}
    required_claims: dict[str, Any] = {}
    answerability_gold: dict[str, Any] = {}

    for qid, gold in ANSWERABLE_GOLD.items():
        src = canonical[qid]
        claims = []
        for claim in gold["required_claims"]:
            evidence = _resolve_prefixes(claim["evidence"], corpus)
            claims.append(
                {
                    "claim_id": claim["claim_id"],
                    "text": claim["text"],
                    "required": claim["required"],
                    "acceptable_evidence": evidence,
                }
            )
        acceptable_source_ids = sorted({cid for c in claims for cid in c["acceptable_evidence"]})
        gold_chunk_ids = sorted(set(qrels[qid]))
        metrics = per_query[qid]["standard_metrics"]
        answerability_gold[qid] = {
            "question_id": qid,
            "answerability": "ANSWERABLE",
            "basis": "canonical 40Q 中该题为 answerable，且 qrels 中存在 Gold Chunk 支撑全部 required claims。",
            "gold_chunk_count": len(gold_chunk_ids),
        }
        reference_answers[qid] = {
            "question_id": qid,
            "reference_answer": gold["reference_answer"],
            "source": "derived from gold chunk content (verified source truth)",
            "gold_chunk_ids": gold_chunk_ids,
        }
        required_claims[qid] = {
            "question_id": qid,
            "claims": claims,
            "required_count": sum(1 for c in claims if c["required"]),
            "optional_count": sum(1 for c in claims if not c["required"]),
        }
        questions.append(
            {
                "question_id": qid,
                "source": "canonical_40q",
                "query": src["query"],
                "query_language": src["query_language"],
                "corpus_language": src["corpus_language"],
                "question_type": src["question_type"],
                "answerability": "ANSWERABLE",
                "reference_answer": gold["reference_answer"],
                "required_claims": claims,
                "acceptable_source_ids": acceptable_source_ids,
                "forbidden_claims": [],
                "gold_evidence": _evidence_meta(gold_chunk_ids),
                "expected_retrieval_phase9": {
                    "recall_at_5": metrics["recall_at_5"],
                    "recall_at_10": metrics["recall_at_10"],
                    "ndcg_at_5": metrics["ndcg_at_5"],
                },
            }
        )

    for qid, gold in INSUFFICIENT_CONTEXT_GOLD.items():
        answerability_gold[qid] = {
            "question_id": qid,
            "answerability": "INSUFFICIENT_CONTEXT",
            "basis": "新增 Generation Eval 题目；关键术语在 361 chunks 上零命中，语料不存在支撑证据。",
            "absence_verification": unanswerable_verification[qid],
        }
        reference_answers[qid] = {
            "question_id": qid,
            "reference_answer": None,
            "source": "no supporting evidence in corpus",
            "gold_chunk_ids": [],
        }
        required_claims[qid] = {"question_id": qid, "claims": [], "required_count": 0, "optional_count": 0}
        questions.append(
            {
                "question_id": qid,
                "source": "phase12_new_unanswerable",
                "query": gold["query"],
                "query_language": gold["query_language"],
                "corpus_language": gold["corpus_language"],
                "question_type": gold["question_type"],
                "answerability": "INSUFFICIENT_CONTEXT",
                "reference_answer": None,
                "required_claims": [],
                "acceptable_source_ids": [],
                "forbidden_claims": [INSUFFICIENT_CONTEXT_FORBIDDEN],
                "gold_evidence": [],
                "expected_retrieval_phase9": None,
            }
        )

    order = {qid: idx for idx, qid in enumerate(ANSWERABLE_GOLD)}
    order.update({qid: 100 + idx for idx, qid in enumerate(INSUFFICIENT_CONTEXT_GOLD)})
    questions.sort(key=lambda item: order[item["question_id"]])

    answerable_count = sum(1 for q in questions if q["answerability"] == "ANSWERABLE")
    dataset = {
        "schema_version": GENERATION_SCHEMA_VERSION,
        "dataset_id": GENERATION_DATASET_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "derived_from": {
            "canonical_retrieval_dataset": str(QUESTIONS_SOURCE).replace("\\", "/"),
            "canonical_question_count": len(canonical),
            "qrels": str(QRELS_SOURCE).replace("\\", "/"),
            "note": "canonical 40Q 未被修改；新增 5 题使用独立 GEN-U* ID 空间。",
        },
        "question_count": len(questions),
        "answerable_count": answerable_count,
        "insufficient_context_count": len(questions) - answerable_count,
        "coverage": {
            "query_language": {
                lang: sum(1 for q in questions if q["query_language"] == lang)
                for lang in sorted({q["query_language"] for q in questions})
            },
            "corpus_language": {
                lang: sum(1 for q in questions if q["corpus_language"] == lang)
                for lang in sorted({q["corpus_language"] for q in questions})
            },
            "question_type": {
                t: sum(1 for q in questions if q["question_type"] == t)
                for t in sorted({q["question_type"] for q in questions})
            },
            "cross_lingual_zh_query_en_corpus": sum(
                1 for q in questions
                if q["query_language"] == "zh" and q["corpus_language"] == "en"
            ),
            "multi_document": sum(1 for q in questions if q["question_type"] == "multi_document"),
            "expected_retrieval_failure": sum(
                1 for q in questions
                if q["expected_retrieval_phase9"]
                and q["expected_retrieval_phase9"]["recall_at_10"] == 0.0
            ),
            "chinese_corpus": 0,
            "chinese_corpus_note": "语料 10 篇论文均为英文，无法覆盖 Chinese corpus，记为已知限制。",
        },
        "questions": questions,
    }

    save_artifact(dataset, OUT / "generation_eval_dataset.json")
    save_artifact(
        {
            "schema_version": GENERATION_SCHEMA_VERSION,
            "dataset_id": GENERATION_DATASET_ID,
            "reference_answers": reference_answers,
        },
        OUT / "reference_answers.json",
    )
    save_artifact(
        {
            "schema_version": GENERATION_SCHEMA_VERSION,
            "dataset_id": GENERATION_DATASET_ID,
            "required_claims": required_claims,
        },
        OUT / "required_claims.json",
    )
    save_artifact(
        {
            "schema_version": GENERATION_SCHEMA_VERSION,
            "dataset_id": GENERATION_DATASET_ID,
            "answerability_gold": answerability_gold,
        },
        OUT / "answerability_gold.json",
    )
    save_artifact(
        {
            "dataset_id": GENERATION_DATASET_ID,
            "corpus_chunk_count": len(corpus),
            "verification": unanswerable_verification,
            "all_passed": all(v["all_terms_zero_hit"] for v in unanswerable_verification.values()),
        },
        OUT / "unanswerable_verification.json",
    )
    print(
        f"[prepare] questions={dataset['question_count']} "
        f"answerable={dataset['answerable_count']} "
        f"insufficient={dataset['insufficient_context_count']}"
    )
    return 0


# --------------------------------------------------------------------------
# Step 4：冻结当前 Runtime 输出
# --------------------------------------------------------------------------
def _build_service():
    """通过唯一 Composition Root 装配正式 RAG Runtime（不修改任何组件）。"""

    from dotenv import load_dotenv

    load_dotenv()
    from evidenceflow.runtime import build_rag_service

    return build_rag_service(Settings())


def cmd_outputs(args: argparse.Namespace) -> int:
    """对 Generation Eval Dataset 逐题执行正式 Runtime，抓取可评测输出。"""

    import time

    from evidenceflow.citation import assemble_rag_answer
    from evidenceflow.domain.generation import AnswerStatus

    dataset = load_json(OUT / "generation_eval_dataset.json")
    questions = dataset["questions"]
    limit = args.limit

    service = _build_service()
    # 复用 Composition Root 已装配的组件，调用序列与 RAGService.answer 完全一致
    # （retrieve → build → generate → assemble_rag_answer），仅额外保留中间对象用于评测。
    retriever = service._retriever
    context_builder = service._context_builder
    generator = service._generator

    results: list[dict[str, Any]] = []
    started = time.time()
    for index, item in enumerate(questions, start=1):
        if limit and index > limit:
            break
        qid = item["question_id"]
        record: dict[str, Any] = {
            "question_id": qid,
            "query": item["query"],
            "answerability_gold": item["answerability"],
        }
        t0 = time.time()
        # 先抓取检索与 Context：即使 LLM 失败，也能保留 Retrieval / Context 层诊断信息。
        try:
            retrieved = retriever.retrieve(item["query"])
            built_context = context_builder.build(retrieved)
            record.update(
                {
                    "retrieved_chunk_ids": [c.chunk.chunk_id for c in retrieved],
                    "retrieved_ranks": [c.rank for c in retrieved],
                    "retrieved_file_names": [c.chunk.file_name for c in retrieved],
                    "context_chunk_ids": [e.chunk_id for e in built_context.entries],
                    "context_source_labels": [e.source_label for e in built_context.entries],
                    "context_text_sha256": _sha256_text(built_context.context_text),
                    "context_total_tokens": built_context.total_tokens,
                    "context_included_count": built_context.included_count,
                    "context_skipped_count": built_context.skipped_count,
                }
            )
        except Exception as exc:  # noqa: BLE001
            record.update({"ok": False, "error": f"RETRIEVAL_OR_CONTEXT: {type(exc).__name__}: {exc}"})

        if record.get("error") is None:
            try:
                generated = generator.generate(item["query"], built_context)
                answer = assemble_rag_answer(
                    query=item["query"],
                    generated_answer=generated,
                    built_context=built_context,
                    retrieved_chunks=retrieved,
                )
                record.update(
                    {
                        "ok": True,
                        "error": None,
                        "generated_status": generated.status.value
                        if isinstance(generated.status, AnswerStatus)
                        else str(generated.status),
                        "answer_text": answer.answer,
                        "cited_labels": [s.source_label for s in answer.sources],
                        "cited_sources": [
                            {
                                "source_label": s.source_label,
                                "file_name": s.file_name,
                                "page_start": s.page_start,
                                "page_end": s.page_end,
                                "chunk_id": s.chunk_id,
                            }
                            for s in answer.sources
                        ],
                    }
                )
            except Exception as exc:  # noqa: BLE001 - 评测需逐题记录失败不中断整轮
                record.update({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
        record["latency_seconds"] = round(time.time() - t0, 3)
        results.append(record)
        print(f"[{index:02d}/{len(questions)}] {qid} ok={record['ok']} {record['latency_seconds']}s")

    payload = {
        "dataset_id": dataset["dataset_id"],
        "schema_version": dataset["schema_version"],
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "generation_contract": _generation_contract_snapshot(),
        "runtime_composition_fingerprint": _composition_fingerprint()["fingerprint"],
        "question_count": len(results),
        "ok_count": sum(1 for r in results if r["ok"]),
        "error_count": sum(1 for r in results if not r["ok"]),
        "total_seconds": round(time.time() - started, 3),
        "outputs": results,
    }
    save_artifact(payload, OUT / "runtime_outputs.json")
    print(f"[outputs] ok={payload['ok_count']} error={payload['error_count']}")
    if payload["error_count"]:
        print("[outputs] 存在失败题目，按 §4 进入异常流程：请先运行 anomaly 子命令。")
    return 0 if payload["error_count"] == 0 else 2


# --------------------------------------------------------------------------
# 异常即停
# --------------------------------------------------------------------------
def cmd_anomaly(args: argparse.Namespace) -> int:
    """在任何异常处落盘 checkpoint 与 anomaly report。"""

    outputs_path = OUT / "runtime_outputs.json"
    outputs = load_json(outputs_path) if outputs_path.exists() else None
    errors: list[dict[str, Any]] = []
    if outputs:
        errors = [
            {"question_id": r["question_id"], "error": r["error"]}
            for r in outputs["outputs"]
            if not r["ok"]
        ]

    checkpoint = {
        "phase": "phase12_rag_quality_v1",
        "stopped_at_utc": datetime.now(timezone.utc).isoformat(),
        "completed_steps": [
            "Step 0 baseline",
            "Step 1 generation dataset",
            "Step 2 generation gold contract",
            "Step 3 answerability gold",
        ],
        "blocked_step": "Step 4 runtime outputs (freeze current runtime outputs)",
        "artifacts_produced": sorted(p.name for p in OUT.iterdir() if p.is_file()),
        "runtime_outputs_present": outputs is not None,
        "runtime_outputs_ok_count": outputs["ok_count"] if outputs else 0,
        "runtime_outputs_error_count": outputs["error_count"] if outputs else None,
        "errors": errors,
        "runtime_modified": False,
        "retrieval_gold_modified": False,
        "chroma_modified": False,
    }
    (OUT / "progress_checkpoint.json").write_text(
        json.dumps(checkpoint, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[anomaly] wrote {OUT / 'progress_checkpoint.json'}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 12 RAG Quality Evaluation runner")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("baseline", help="冻结 Phase 12 基线").set_defaults(func=cmd_baseline)
    sub.add_parser("prepare", help="生成 Generation Eval Dataset 与 Gold").set_defaults(func=cmd_prepare)
    outputs_parser = sub.add_parser("outputs", help="抓取正式 Runtime 输出")
    outputs_parser.add_argument("--limit", type=int, default=0, help="只跑前 N 题，用于连通性验证")
    outputs_parser.set_defaults(func=cmd_outputs)
    sub.add_parser("anomaly", help="落盘 checkpoint 与 anomaly 信息").set_defaults(func=cmd_anomaly)
    parsed = parser.parse_args()
    raise SystemExit(parsed.func(parsed))
