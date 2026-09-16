"""已冻结的 Hybrid Runtime 一致性历史验证。

§17.3.1 §4 要求：Gate 之后必须用冻结 40Q 在新 Runtime 上重跑，确认
``hit_at_5`` 与 ``candidate_retrieval_failure`` 与离线
``run_research_hybrid_retrieval_experiment`` 的 treatment 一致；不一致即回滚。

实现要点：

- 复用 ``run_research_dense_baseline`` 的共享计算（``_load_and_verify`` /
  ``_all_chunks`` / ``_gold_groups`` / ``_metrics`` / ``_source_ndcg`` /
  ``_breakdowns`` / ``FrozenTranslationProvider``），保证行同构。
- 直接实例化正式 ``MultilingualRetriever(bm25_index=BM25Index(...))``，让
  §17.3.1 Runtime 集成路径自然执行末轮 Equal RRF。
- 与 ``evals/research_retrieval/hybrid_retrieval_experiment_v1.json`` 中
  treatment 分桶逐项对比；通过条件（详见 ``_gate``）：

    * 硬指标：``hit_at_5`` 与 ``candidate_retrieval_failure`` 须与离线完全一致；
    * 非退化校验：其余 Overall 指标 Runtime 不得低于 Offline，但允许正向漂移。

注意：脚本写入 ``runtime_validation_output.json``，并把对照结果以 Markdown
表格打印到 stderr，便于在 Runtime 升级后第一时间肉眼复核。
"""

from __future__ import annotations

import json
import math
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence


# 让脚本既能从 ``experiments/`` 直接运行，也能作为模块被 ``experiments.run_...``
# 调用：自动注入 ``experiments/`` 与 ``src/`` 到 sys.path；不依赖 PYTHONPATH 跨平台写法。
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _candidate in (str(_PROJECT_ROOT / "src"), str(_PROJECT_ROOT)):
    if _candidate not in sys.path:
        sys.path.insert(0, _candidate)

from evidenceflow.config import Settings
from evidenceflow.embedding import BgeM3EmbeddingProvider
from evidenceflow.ingestion import BgeM3Tokenizer
from evidenceflow.retrieval import DenseRetriever, MultilingualRetriever
from evidenceflow.storage import ChromaVectorStore

from experiments.diagnose_research_candidate_failures import (  # noqa: E402
    BM25_B,
    BM25_K1,
    EXPECTED_COLLECTION,
    EXPECTED_DATASET_HASH,
)
from experiments.run_research_dense_baseline import (  # noqa: E402
    DEPTHS,
    FrozenTranslationProvider,
    _all_chunks,
    _breakdowns,
    _gold_groups,
    _load_and_verify,
    _metrics,
    _source_ndcg,
)


ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "evals" / "research_retrieval" / "research_questions_v1.json"
OFFLINE_TREATMENT_PATH = (
    ROOT / "evals" / "research_retrieval" / "hybrid_retrieval_experiment_v1.json"
)
RUNTIME_OUTPUT_JSON = (
    ROOT / "evals" / "research_retrieval" / "hybrid_runtime_validation_v1.json"
)

CANDIDATE_BUDGET = 20


def _evaluate(
    question: Mapping[str, Any],
    chunks: Sequence[Any],
    results: Sequence[Any],
) -> dict[str, Any]:
    """构造单题结果行，保持与离线 hybrid 实验完全相同的字段集。"""

    from evidenceflow.eval.gold import gold_group_rank

    groups = _gold_groups(question, chunks)
    extraction_failure = False  # 构建阶段已逐页验证全部原文锚点。
    chunk_failure = any(not group for group in groups)
    ranked_ids = [result.chunk.chunk_id for result in results]
    rank = None if chunk_failure else gold_group_rank(ranked_ids, required_gold_groups=groups)
    if extraction_failure:
        primary_failure = "extraction_failure"
    elif chunk_failure:
        primary_failure = "chunk_failure"
    elif rank is None:
        primary_failure = "candidate_retrieval_failure"
    elif rank > 5:
        primary_failure = "ranking_failure"
    else:
        primary_failure = "success"
    row = {
        "question_id": question["question_id"],
        "query": question["query"],
        "gold_group_rank": rank,
        "primary_failure": primary_failure,
        # 与离线 hybrid 实验同构：补齐 _breakdowns 需要的分组维度字段。
        "language_group": question["language_group"],
        "question_type": question["question_type"],
        "evidence_type": question["evidence_type"],
        "gold_files": [source["file_name"] for source in question["gold_sources"]],
    }
    for depth in DEPTHS:
        row[f"ndcg_at_{depth}"] = _source_ndcg(results, groups, depth)
    return row


def _runtime_rows(
    questions: Sequence[Mapping[str, Any]],
    chunks: Sequence[Any],
    retriever: MultilingualRetriever,
) -> list[dict[str, Any]]:
    """逐题调用正式 MultilingualRetriever.retrieve()，记录全题结果。"""

    rows: list[dict[str, Any]] = []
    for index, question in enumerate(questions, start=1):
        results = retriever.retrieve(question["query"], top_k=CANDIDATE_BUDGET)
        row = _evaluate(question, chunks, results)
        rows.append(row)
        print(
            f"[runtime {index:02d}/{len(questions)}] {question['question_id']} "
            f"rank={row['gold_group_rank']} {row['primary_failure']}",
            flush=True,
        )
    return rows


def _compare_overall(
    runtime_overall: Mapping[str, float], offline_overall: Mapping[str, float]
) -> dict[str, dict[str, float]]:
    """逐深度对比 runtime 与 offline hybrid treatment 的 Overall 指标。"""

    comparison: dict[str, dict[str, float]] = {}
    for depth in DEPTHS:
        key = f"hit_at_{depth}"
        runtime_value = runtime_overall[key]
        offline_value = offline_overall[key]
        comparison[key] = {
            "runtime": round(runtime_value, 6),
            "offline_treatment": round(offline_value, 6),
            "abs_diff": round(abs(runtime_value - offline_value), 6),
        }
    for key in ("mrr", "ndcg_at_20"):
        runtime_value = runtime_overall[key]
        offline_value = offline_overall[key]
        comparison[key] = {
            "runtime": round(runtime_value, 6),
            "offline_treatment": round(offline_value, 6),
            "abs_diff": round(abs(runtime_value - offline_value), 6),
        }
    return comparison


def _compare_failures(
    runtime_rows: Sequence[Mapping[str, Any]],
    offline_rows: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, int]]:
    """按 primary_failure 类型对比 Runtime 与 Offline 的归因计数。"""

    runtime_counts = dict(Counter(r["primary_failure"] for r in runtime_rows))
    offline_counts = dict(Counter(r["primary_failure"] for r in offline_rows))
    keys = set(runtime_counts) | set(offline_counts)
    return {
        key: {
            "runtime": runtime_counts.get(key, 0),
            "offline_treatment": offline_counts.get(key, 0),
            "abs_diff": abs(runtime_counts.get(key, 0) - offline_counts.get(key, 0)),
        }
        for key in sorted(keys)
    }


def _gate(
    comparison: Mapping[str, Mapping[str, float]],
    failure_comparison: Mapping[str, Mapping[str, int]],
) -> dict[str, Any]:
    """判定 Runtime 与 Offline 是否一致。

    §17.3.1 §4 字面要求：

        用冻结 40Q Fine-grained Holdout 在新 Runtime 上重跑一次，确认
        ``hit_at_5`` 与 ``candidate_retrieval_failure`` 与离线实验一致；
        不一致即回滚。

    因此本 Gate 拆为两层：

    1. **硬指标**（must_pass）：``hit_at_5`` 与 ``candidate_retrieval_failure`` 须与离线
       完全一致（计数级一致即整数差为 0，浮点容忍 1e-9）。任一硬指标不达标 → 回滚。
    2. **非退化校验**（non_regression）：其余 Overall 指标（``hit_at_1`` /
       ``hit_at_3`` / ``hit_at_10`` / ``hit_at_20`` / ``mrr`` / ``ndcg_at_20``）允许小幅
       正向漂移，但 Runtime 不得低于 Offline（容忍 1e-9）。一旦退化即视为潜在异常，
       报告显眼告警，但不触发自动回滚（避免单次浮点噪声导致误回滚）；维护人员应人工
       复核。
    """

    MUST_PASS_METRICS = ("hit_at_5",)
    MUST_PASS_FAILURES = ("candidate_retrieval_failure",)
    NON_REGRESSION_METRICS = (
        "hit_at_1",
        "hit_at_3",
        "hit_at_10",
        "hit_at_20",
        "mrr",
        "ndcg_at_20",
    )
    HARD_TOLERANCE = 1e-9  # 浮点容差，避免数值噪声误判硬指标失败。

    failed_hard_metrics = [
        key
        for key in MUST_PASS_METRICS
        if key in comparison and comparison[key]["abs_diff"] > HARD_TOLERANCE
    ]
    failed_hard_failures = [
        key
        for key in MUST_PASS_FAILURES
        if key in failure_comparison
        and failure_comparison[key]["abs_diff"] > 0  # 计数维度要求整数级一致
    ]
    # 非退化校验：Runtime 不得低于 Offline；若退化则列出供人工复核。
    regressed_metrics = [
        key
        for key in NON_REGRESSION_METRICS
        if key in comparison
        and comparison[key]["runtime"] + HARD_TOLERANCE < comparison[key]["offline_treatment"]
    ]

    passed = not failed_hard_metrics and not failed_hard_failures
    if not passed:
        decision = (
            "NOT PASSED → 必须回滚 multilingual_hybrid_enabled=False 并回退 "
            "runtime.py/multilingual.py/chroma.py"
        )
    elif regressed_metrics:
        decision = (
            "PASSED（硬指标）→ 存在非退化漂移 "
            f"{regressed_metrics}，请人工复核后决定是否保留"
        )
    else:
        decision = (
            "PASSED → Runtime 集成与离线 hybrid 实验一致，保持开启 "
            "multilingual_hybrid_enabled"
        )

    return {
        "must_pass_metrics": list(MUST_PASS_METRICS),
        "must_pass_failures": list(MUST_PASS_FAILURES),
        "non_regression_metrics": list(NON_REGRESSION_METRICS),
        "failed_hard_metrics": failed_hard_metrics,
        "failed_hard_failure_buckets": failed_hard_failures,
        "regressed_metrics": regressed_metrics,
        "passed": passed,
        "decision": decision,
    }


def _runtime_fingerprint() -> dict[str, str]:
    """记录新 Runtime 关键源文件 Hash，便于审计。"""

    import hashlib

    files = (
        "PROJECT_SPEC.md",
        "src/evidenceflow/runtime.py",
        "src/evidenceflow/config.py",
        "src/evidenceflow/retrieval/retriever.py",
        "src/evidenceflow/retrieval/multilingual.py",
        "src/evidenceflow/retrieval/hybrid_fusion.py",
        "src/evidenceflow/retrieval/bm25.py",
        "src/evidenceflow/retrieval/fusion.py",
        "src/evidenceflow/domain/chunk.py",
        "src/evidenceflow/storage/base.py",
        "src/evidenceflow/storage/chroma.py",
    )
    return {
        relative: hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        for relative in files
    }


def run() -> dict[str, Any]:
    """运行 Runtime 一致性验证，返回对照结果与 Gate 判定。"""

    dataset, translations_by_id, audit = _load_and_verify()
    if dataset.get("dataset_hash") != EXPECTED_DATASET_HASH:
        raise ValueError("dataset hash mismatch against frozen research baseline")
    os.environ["HF_HOME"] = str((ROOT / ".cache" / "huggingface").resolve())
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    settings = Settings(
        multilingual_hybrid_enabled=True,
        multilingual_hybrid_bm25_k1=BM25_K1,
        multilingual_hybrid_bm25_b=BM25_B,
        multilingual_hybrid_rrf_k=60,
    )
    if (
        settings.collection_name,
        settings.chunk_size,
        settings.chunk_overlap,
    ) != (EXPECTED_COLLECTION, 512, 64):
        raise ValueError("active Settings do not match the frozen research baseline")
    store = ChromaVectorStore(settings.vector_db_path, settings.collection_name)
    if store._collection.count() != 361:
        raise ValueError("research collection must contain exactly 361 frozen chunks")
    chunks = _all_chunks(store)

    translations_by_query = {
        question["query"]: translations_by_id[question["question_id"]]
        for question in dataset["questions"]
    }
    dense = DenseRetriever(
        BgeM3EmbeddingProvider(settings.embedding_model),
        store,
        top_k=settings.multilingual_candidate_top_k,
    )
    frozen_provider = FrozenTranslationProvider(translations_by_query)

    # Runtime 路径：先复用 iter_all_chunks 把正式 Chunk 灌入 BM25Index；
    # 然后构造带 bm25_index 的 MultilingualRetriever，让 §17.3.1 末轮 Equal RRF
    # 自然生效。这与 runtime._build_bm25_index() 等价的实验验证路径。
    from evidenceflow.retrieval import BM25Index

    bm25_index = BM25Index(
        list(store.iter_all_chunks()),
        BgeM3Tokenizer(settings.embedding_model),
        k1=settings.multilingual_hybrid_bm25_k1,
        b=settings.multilingual_hybrid_bm25_b,
    )
    runtime_retriever = MultilingualRetriever(
        dense,
        frozen_provider,
        candidate_top_k=settings.multilingual_candidate_top_k,
        final_top_k=settings.multilingual_final_top_k,
        rrf_k=settings.multilingual_rrf_k,
        original_weight=settings.multilingual_original_weight,
        translation_weight=settings.multilingual_translation_weight,
        bm25_index=bm25_index,
        hybrid_rrf_k=settings.multilingual_hybrid_rrf_k,
        hybrid_candidate_budget=CANDIDATE_BUDGET,
    )

    print("=== runtime (multilingual_dense + bm25, §17.3.1 path) ===", flush=True)
    runtime_rows = _runtime_rows(dataset["questions"], chunks, runtime_retriever)
    runtime_breakdowns = _breakdowns(runtime_rows)

    if not OFFLINE_TREATMENT_PATH.exists():
        raise FileNotFoundError(
            f"Missing offline treatment artifact: {OFFLINE_TREATMENT_PATH}"
        )
    offline_payload = json.loads(OFFLINE_TREATMENT_PATH.read_text(encoding="utf-8"))
    offline_overall = offline_payload["treatment"]["breakdowns"]["overall"]
    offline_rows = offline_payload["treatment"]["question_results"]
    overall_comparison = _compare_overall(runtime_breakdowns["overall"], offline_overall)
    failure_comparison = _compare_failures(runtime_rows, offline_rows)
    gate = _gate(overall_comparison, failure_comparison)

    return {
        "evaluation_version": "research-hybrid-runtime-validation-v1",
        "dataset_hash": dataset["dataset_hash"],
        "runtime_supersession_marker": {
            "applied": True,
            "by": "§17.3.1 Hybrid Runtime Integration",
            "note": (
                "本报告本身在 §17.3.1 Runtime 集成完成后运行；Runtime 路径已包含"
                " multilingual_hybrid_enabled=True 与 BM25→Equal RRF 末轮融合。"
            ),
        },
        "configuration": {
            "runtime_path": "MultilingualRetriever(bm25_index=BM25Index(iter_all_chunks()))",
            "settings": {
                "multilingual_hybrid_enabled": settings.multilingual_hybrid_enabled,
                "multilingual_hybrid_bm25_k1": settings.multilingual_hybrid_bm25_k1,
                "multilingual_hybrid_bm25_b": settings.multilingual_hybrid_bm25_b,
                "multilingual_hybrid_rrf_k": settings.multilingual_hybrid_rrf_k,
                "candidate_top_k_per_branch": settings.multilingual_candidate_top_k,
                "final_runtime_top_k": settings.multilingual_final_top_k,
                "hybrid_candidate_budget": CANDIDATE_BUDGET,
            },
            "offline_treatment_artifact": str(
                OFFLINE_TREATMENT_PATH.relative_to(ROOT)
            ),
        },
        "runtime": {
            "breakdowns": runtime_breakdowns,
            "failure_counts": dict(Counter(r["primary_failure"] for r in runtime_rows)),
            "question_results": runtime_rows,
        },
        "comparison": {
            "overall": overall_comparison,
            "failures": failure_comparison,
        },
        "gate": gate,
        "runtime_fingerprint": _runtime_fingerprint(),
    }


def _render_markdown(payload: Mapping[str, Any]) -> str:
    """渲染 Runtime 对照报告，便于人工复核。"""

    runtime_overall = payload["runtime"]["breakdowns"]["overall"]
    comparison = payload["comparison"]["overall"]
    failures = payload["comparison"]["failures"]
    gate = payload["gate"]
    rows: list[str] = []
    rows.append("# Hybrid Runtime Validation v1")
    rows.append("")
    rows.append(f"- Dataset hash: `{payload['dataset_hash']}`")
    rows.append(
        "- Runtime: `MultilingualRetriever(bm25_index=BM25Index(iter_all_chunks()))` "
        "(§17.3.1 集成路径)"
    )
    rows.append("- Offline 对照：hybrid_retrieval_experiment_v1.json treatment")
    rows.append("")
    rows.append("## Overall 一致性")
    rows.append("")
    rows.append("| Metric | Runtime | Offline | abs_diff |")
    rows.append("|---|---:|---:|---:|")
    for key in (
        "hit_at_1",
        "hit_at_3",
        "hit_at_5",
        "hit_at_10",
        "hit_at_20",
        "mrr",
        "ndcg_at_20",
    ):
        values = comparison[key]
        rows.append(
            f"| {key} | {values['runtime']:.6f} | "
            f"{values['offline_treatment']:.6f} | {values['abs_diff']:.6f} |"
        )
    rows.append("")
    rows.append("## 失败归因对比")
    rows.append("")
    rows.append("| failure | Runtime | Offline | abs_diff |")
    rows.append("|---|---:|---:|---:|")
    for key, values in failures.items():
        rows.append(
            f"| {key} | {values['runtime']} | {values['offline_treatment']} | "
            f"{values['abs_diff']} |"
        )
    rows.append("")
    rows.append("## Gate（§17.3.1 §4）")
    rows.append("")
    rows.append(
        f"- 硬指标（必须完全一致）：{gate['must_pass_metrics']} + "
        f"{gate['must_pass_failures']}"
    )
    rows.append(f"- 硬指标失败：{gate['failed_hard_metrics']} {gate['failed_hard_failure_buckets']}")
    rows.append(
        f"- 非退化校验（其余 Overall 指标 Runtime 不得低于 Offline）："
        f"{gate['non_regression_metrics']}"
    )
    rows.append(f"- 退化指标：{gate['regressed_metrics']}")
    rows.append(f"- **判定：{gate['decision']}**")
    return "\n".join(rows)


def main() -> None:
    """执行 Runtime 一致性验证，落盘 JSON + Markdown。"""

    payload = run()
    RUNTIME_OUTPUT_JSON.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    markdown = _render_markdown(payload)
    print(markdown)
    overall = payload["runtime"]["breakdowns"]["overall"]
    print(
        "=== summary ===\n"
        f"hit_at_5={overall['hit_at_5']:.4f} hit_at_10={overall['hit_at_10']:.4f} "
        f"mrr={overall['mrr']:.4f} candidate_retrieval_failure="
        f"{payload['runtime']['failure_counts'].get('candidate_retrieval_failure', 0)} "
        f"passed={payload['gate']['passed']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
