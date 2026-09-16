"""已冻结的 Dense + BM25 Hybrid 历史实验。

背景
----
失败归因显示真正瓶颈在 recall：12 个 `candidate_retrieval_failure`（Gold 从未进入
Top-20）。Query Rewrite 实验已证明查询表层改写只是弱杠杆（Gate 未通过，仅救回 2/12）。
另一条候选路径是**在检索时融合 lexical（BM25）分支**：语义 Dense 召回靠向量近邻，
而 BM25 能在词汇/术语精确匹配上补 Dense 的盲区（尤其是缩写、公式、专有名词），
有望把部分 Gold 拉进候选集合。

本实验直接复用离线评测专用工具 `evidenceflow.eval.lexical.BM25Index` 与
`evidenceflow.eval.hybrid.equal_ranked_rrf_fusion`，**不改 Runtime、不引入 LLM**，
仅对冻结 40Q Gold 做受控 A/B。

实验设计（受控 A/B，不改 Runtime）
--------------------------------
- 控制组 `control`：当前冻结基线 = 多语 Dense 检索器（中文原始分支 + 冻结人工英文
  翻译分支，内部 RRF）。与 `run_research_dense_baseline` 完全一致。
- 处理组 `treatment`：在上述多语 Dense 结果之上，**再融合一路 BM25 lexical 分支**
  （Equal-Rank RRF, k=60），即检索时 Dense+BM25 混合。
- 复用 `diagnose_research_candidate_failures` 的共享契约（`_load_and_verify` /
  `EXPECTED_COLLECTION` / `EXPECTED_DATASET_HASH` / `BM25_K1` / `BM25_B`）与
  `run_research_dense_baseline` 的共享计算（`_all_chunks` / `_gold_groups` /
  `_metrics` / `_source_ndcg` / `FrozenTranslationProvider` / `_breakdowns`），
  只读冻结 40Q Gold，不修改任何生产代码。

判定（与 Query Rewrite 实验同构，便于跨实验比较）
------------------------------------------------
- Phase A（定向）：在控制组判定的 12 个候选失败题上，统计处理组把 Gold 拉进
  Top-20 / Top-5 的题数。
- Phase B（全量）：对比控制组 / 处理组的 Hit@5、MRR、nDCG@20 与失败分布。
- Gate：`recovered_to_top20 >= PHASE_A_GATE`（且全量 Hit@5 不劣化超过容差）
  才建议进入 Runtime 集成候选，否则停止。
"""

from __future__ import annotations

import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from evidenceflow.config import Settings
from evidenceflow.domain.generation import LLMMessage
from evidenceflow.embedding import BgeM3EmbeddingProvider
from evidenceflow.eval.gold import gold_group_rank
from evidenceflow.ingestion import BgeM3Tokenizer
from evidenceflow.eval.hybrid import RankedChunkCandidate, equal_ranked_rrf_fusion
from evidenceflow.eval.lexical import BM25Index
from evidenceflow.retrieval import DenseRetriever, MultilingualRetriever
from evidenceflow.storage import ChromaVectorStore
from evidenceflow.translation.base import TranslationError, TranslationResult

# 复用诊断模块共享 BM25 超参与冻结集合/数据集哈希常量，与 augmentation / reranking 实验一致。
from experiments.diagnose_research_candidate_failures import (  # noqa: E402
    BM25_B,
    BM25_K1,
    EXPECTED_COLLECTION,
    EXPECTED_DATASET_HASH,
)
# 复用冻结基线的共享契约（_load_and_verify 在基线中返回 3 元组）。
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
OUTPUT_JSON = ROOT / "evals" / "research_retrieval" / "hybrid_retrieval_experiment_v1.json"
OUTPUT_MD = ROOT / "evals" / "research_retrieval" / "hybrid_retrieval_experiment_v1_report.md"

PHASE_A_GATE = 4                    # 12 个定向失败题中至少救回 4 个才放行。
HIT5_REGRESS_TOLERANCE = 0.025      # 全量 Hit@5 允许的最大劣化幅度。
RRF_K = 60                          # Equal-Rank RRF 的 k（与 hybrid.py 默认一致）。
CANDIDATE_BUDGET = 20               # 融合后候选预算（与基线候选池一致）。


# --------------------------------------------------------------------------- #
# 单题评估（与冻结基线 / Query Rewrite 实验同构，便于 _breakdowns 聚合）
# --------------------------------------------------------------------------- #
def _evaluate(
    question: Mapping[str, Any],
    chunks: Sequence[Any],
    results: Sequence[Any],
) -> dict[str, Any]:
    """对一组检索结果计算 Gold 来源排名、失败归因与 nDCG 指标。"""

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
        # 与冻结基线 `_breakdowns` 同构：补齐分组维度字段，否则聚合阶段 KeyError。
        "language_group": question["language_group"],
        "question_type": question["question_type"],
        "evidence_type": question["evidence_type"],
        "gold_files": [source["file_name"] for source in question["gold_sources"]],
    }
    for depth in DEPTHS:
        row[f"ndcg_at_{depth}"] = _source_ndcg(results, groups, depth)
    return row


def _control_rows(
    questions: Sequence[Mapping[str, Any]],
    chunks: Sequence[Any],
    retriever: MultilingualRetriever,
) -> list[dict[str, Any]]:
    """控制组：多语 Dense 检索器直接评估。"""

    rows: list[dict[str, Any]] = []
    for index, question in enumerate(questions, start=1):
        results = retriever.retrieve(question["query"], top_k=CANDIDATE_BUDGET)
        row = _evaluate(question, chunks, results)
        rows.append(row)
        print(
            f"[control {index:02d}/{len(questions)}] {question['question_id']} "
            f"rank={row['gold_group_rank']} {row['primary_failure']}",
            flush=True,
        )
    return rows


def _treatment_rows(
    questions: Sequence[Mapping[str, Any]],
    chunks: Sequence[Any],
    retriever: MultilingualRetriever,
    bm25: BM25Index,
) -> list[dict[str, Any]]:
    """处理组：多语 Dense 结果再融合 BM25 lexical 分支（Equal-Rank RRF）。"""

    rows: list[dict[str, Any]] = []
    for index, question in enumerate(questions, start=1):
        # Dense 分支：复用与基线完全一致的检索结果，转为 RankedChunkCandidate。
        dense_results = retriever.retrieve(question["query"], top_k=CANDIDATE_BUDGET)
        dense_ranked = [
            RankedChunkCandidate(chunk=r.chunk, rank=i + 1)
            for i, r in enumerate(dense_results)
        ]
        # Lexical 分支：BM25 在冻结 361 Chunk 上检索。
        bm25_hits = bm25.search(question["query"], top_k=CANDIDATE_BUDGET)
        bm25_ranked = [
            RankedChunkCandidate(chunk=h.chunk, rank=h.rank) for h in bm25_hits
        ]
        # 检索时融合：Dense + BM25（rank-only Equal RRF，不污染分数语义）。
        fused = equal_ranked_rrf_fusion(
            dense_ranked, bm25_ranked, rrf_k=RRF_K, candidate_budget=CANDIDATE_BUDGET
        )
        row = _evaluate(question, chunks, fused)
        rows.append(row)
        print(
            f"[treatment {index:02d}/{len(questions)}] {question['question_id']} "
            f"rank={row['gold_group_rank']} {row['primary_failure']}",
            flush=True,
        )
    return rows


# --------------------------------------------------------------------------- #
# Phase A / Gate
# --------------------------------------------------------------------------- #
def _phase_a_table(
    control_rows: list[dict[str, Any]],
    treatment_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """构造 Phase A 定向失败题的处理组效果明细。"""

    by_id = {r["question_id"]: r for r in treatment_rows}
    table: list[dict[str, Any]] = []
    for ctl in control_rows:
        if ctl["primary_failure"] != "candidate_retrieval_failure":
            continue
        trt = by_id[ctl["question_id"]]
        trt_rank = trt["gold_group_rank"]
        if trt_rank is None:
            effect = "still_missed"
        elif trt_rank <= 5:
            effect = "RECOVERED→Top5"
        else:
            effect = "RECOVERED→Top20"
        table.append(
            {
                "question_id": ctl["question_id"],
                "control_rank": ctl["gold_group_rank"],
                "treatment_rank": trt_rank,
                "effect": effect,
            }
        )
    return table


def _gate(
    phase_a: list[dict[str, Any]],
    control_overall: dict[str, Any],
    treatment_overall: dict[str, Any],
) -> dict[str, Any]:
    """按 Phase A 定向召回 + 全量 Hit@5 不劣化判定 Gate。"""

    recovered_to_top20 = sum(
        1 for r in phase_a if r["treatment_rank"] is not None
    )
    recovered_to_top5 = sum(
        1 for r in phase_a if r["treatment_rank"] is not None and r["treatment_rank"] <= 5
    )
    control_hit5 = control_overall["hit_at_5"]
    treatment_hit5 = treatment_overall["hit_at_5"]
    hit5_regression = round(control_hit5 - treatment_hit5, 4)
    passed = (recovered_to_top20 >= PHASE_A_GATE) and (
        treatment_hit5 >= control_hit5 - HIT5_REGRESS_TOLERANCE
    )
    decision = (
        "PASSED → 建议进入 Runtime 集成候选（Dense+BM25 检索时融合）"
        if passed
        else "NOT PASSED → 停止，不回写 Runtime"
    )
    return {
        "phase_a_targets": len(phase_a),
        "recovered_to_top20": recovered_to_top20,
        "recovered_to_top5": recovered_to_top5,
        "phase_a_gate_threshold": PHASE_A_GATE,
        "control_hit_at_5": control_hit5,
        "treatment_hit_at_5": treatment_hit5,
        "hit5_regression": hit5_regression,
        "hit5_regression_tolerance": HIT5_REGRESS_TOLERANCE,
        "passed": passed,
        "decision": decision,
    }


# --------------------------------------------------------------------------- #
# 报告渲染
# --------------------------------------------------------------------------- #
def _overall_row(label: str, overall: dict[str, Any]) -> str:
    return (
        f"| {label} | {overall['N']} | {overall['hit_at_1']:.3f} | "
        f"{overall['hit_at_3']:.3f} | {overall['hit_at_5']:.3f} | "
        f"{overall['hit_at_10']:.3f} | {overall['hit_at_20']:.3f} | "
        f"{overall['mrr']:.3f} | {overall['ndcg_at_20']:.3f} |"
    )


def render_markdown(payload: Mapping[str, Any]) -> str:
    """把完整 payload 渲染为可读 Markdown 报告。"""

    control = payload["control"]["breakdowns"]["overall"]
    treatment = payload["treatment"]["breakdowns"]["overall"]
    gate = payload["gate"]
    phase_a = payload["phase_a"]
    lines: list[str] = []
    lines.append("# Academic Research Hybrid Retrieval (Dense + BM25) Experiment v1")
    lines.append("")
    lines.append(f"- Dataset hash: `{payload['dataset_hash']}`")
    lines.append("- 控制组：多语 Dense（中文原始分支 + 冻结人工英文翻译分支，内部 RRF）。")
    lines.append("- 处理组：多语 Dense 结果再融合 BM25 lexical 分支（Equal-Rank RRF）。")
    lines.append(
        f"- BM25 配置：k1={payload['configuration']['bm25_k1']}, "
        f"b={payload['configuration']['bm25_b']}；融合 rrf_k={payload['configuration']['rrf_k']}。"
    )
    lines.append("")
    lines.append("## Overall（控制组 vs 处理组）")
    lines.append("")
    lines.append(
        "| Group | N | Hit@1 | Hit@3 | Hit@5 | Hit@10 | Hit@20 | MRR | nDCG@20 |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    lines.append(_overall_row("control", control))
    lines.append(_overall_row("treatment", treatment))
    lines.append("")
    lines.append("## Failure distribution")
    lines.append("")
    lines.append(f"- control: {payload['control']['failure_counts']}")
    lines.append(f"- treatment: {payload['treatment']['failure_counts']}")
    lines.append("")
    lines.append("## Phase A：定向候选失败题的处理组效果")
    lines.append("")
    lines.append("| question_id | control_rank | treatment_rank | effect |")
    lines.append("|---|---:|---:|---|")
    for r in phase_a:
        lines.append(
            f"| {r['question_id']} | {r['control_rank']} | {r['treatment_rank']} | "
            f"{r['effect']} |"
        )
    lines.append("")
    lines.append("## Gate")
    lines.append("")
    lines.append(f"- Phase A 定向题数：{gate['phase_a_targets']}")
    lines.append(f"- 救回至 Top-20：{gate['recovered_to_top20']}（阈值 {gate['phase_a_gate_threshold']}）")
    lines.append(f"- 救回至 Top-5：{gate['recovered_to_top5']}")
    lines.append(
        f"- 控制组 Hit@5：{gate['control_hit_at_5']:.3f}；"
        f"处理组 Hit@5：{gate['treatment_hit_at_5']:.3f}"
    )
    lines.append(
        f"- Hit@5 劣化：{gate['hit5_regression']}（容差 {gate['hit5_regression_tolerance']}）"
    )
    lines.append(f"- **判定：{gate['decision']}**")
    lines.append("")
    lines.append(
        "结论：若 Gate 通过，说明在检索时融合 BM25 lexical 分支能稳定把部分 Gold 拉进"
        "候选集合，是比查询改写更可靠的 recall 杠杆；否则瓶颈仍在语料覆盖 / chunk 切分 / "
        "嵌入表示。"
    )
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def run() -> dict[str, Any]:
    """执行 Dense + BM25 混合检索受控实验，返回带 Gate 判定的完整产物。"""

    dataset, translations_by_id, audit = _load_and_verify()
    if dataset.get("dataset_hash") != EXPECTED_DATASET_HASH:
        raise ValueError("dataset hash mismatch against frozen research baseline")
    os.environ["HF_HOME"] = str((ROOT / ".cache" / "huggingface").resolve())
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    settings = Settings()
    if (settings.collection_name, settings.chunk_size, settings.chunk_overlap) != (
        EXPECTED_COLLECTION,
        512,
        64,
    ):
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
    control_retriever = MultilingualRetriever(
        dense,
        frozen_provider,
        candidate_top_k=settings.multilingual_candidate_top_k,
        final_top_k=settings.multilingual_final_top_k,
        rrf_k=settings.multilingual_rrf_k,
        original_weight=settings.multilingual_original_weight,
        translation_weight=settings.multilingual_translation_weight,
    )
    # 离线 lexical 索引：在冻结 361 Chunk 上用 BGE Tokenizer 建 BM25。
    bm25 = BM25Index(
        chunks, BgeM3Tokenizer(settings.embedding_model), k1=BM25_K1, b=BM25_B
    )

    print("=== control (multilingual dense) ===", flush=True)
    control_rows = _control_rows(dataset["questions"], chunks, control_retriever)
    print("=== treatment (dense + BM25 fusion) ===", flush=True)
    treatment_rows = _treatment_rows(
        dataset["questions"], chunks, control_retriever, bm25
    )

    control_breakdowns = _breakdowns(control_rows)
    treatment_breakdowns = _breakdowns(treatment_rows)
    phase_a = _phase_a_table(control_rows, treatment_rows)
    gate = _gate(phase_a, control_breakdowns["overall"], treatment_breakdowns["overall"])

    return {
        "evaluation_version": "research-hybrid-retrieval-v1",
        "dataset_hash": dataset["dataset_hash"],
        "configuration": {
            "control": "multilingual dense (chinese original + frozen english translation, internal RRF)",
            "treatment": "multilingual dense fused with BM25 lexical branch (equal-rank RRF)",
            "embedding_model": settings.embedding_model,
            "collection_name": settings.collection_name,
            "candidate_top_k_per_branch": settings.multilingual_candidate_top_k,
            "final_runtime_top_k": settings.multilingual_final_top_k,
            "rrf_k": RRF_K,
            "bm25_k1": BM25_K1,
            "bm25_b": BM25_B,
            "phase_a_gate": PHASE_A_GATE,
            "hit5_regression_tolerance": HIT5_REGRESS_TOLERANCE,
        },
        "control": {
            "breakdowns": control_breakdowns,
            "failure_counts": dict(Counter(r["primary_failure"] for r in control_rows)),
            "question_results": control_rows,
        },
        "treatment": {
            "breakdowns": treatment_breakdowns,
            "failure_counts": dict(Counter(r["primary_failure"] for r in treatment_rows)),
            "question_results": treatment_rows,
        },
        "phase_a": phase_a,
        "gate": gate,
        "runtime_fingerprint": _runtime_fingerprint(),
    }


def _runtime_fingerprint() -> dict[str, Any]:
    """记录运行时指纹，保证实验产物可复现、可审计。"""

    import hashlib

    files = (
        "PROJECT_SPEC.md",
        "src/evidenceflow/runtime.py",
        "src/evidenceflow/retrieval/retriever.py",
        "src/evidenceflow/retrieval/multilingual.py",
        "src/evidenceflow/retrieval/fusion.py",
        "src/evidenceflow/eval/lexical.py",
        "src/evidenceflow/eval/hybrid.py",
    )
    return {
        relative: hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        for relative in files
    }


def main() -> None:
    """运行实验并将完整 JSON 与精简 Markdown 同步落盘。"""

    payload = run()
    OUTPUT_JSON.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    OUTPUT_MD.write_text(render_markdown(payload), encoding="utf-8")
    print(
        json.dumps(
            {
                "control_overall": payload["control"]["breakdowns"]["overall"],
                "treatment_overall": payload["treatment"]["breakdowns"]["overall"],
                "gate": payload["gate"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
