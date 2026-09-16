"""运行 References-aware metadata/filtering 单变量受控实验。"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
for import_root in (ROOT, ROOT / "src"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from experiments.run_research_dense_baseline import (
    DEPTHS,
    FrozenTranslationProvider,
    _all_chunks,
    _breakdowns,
    _canonical_hash,
    _gold_groups,
    _source_ndcg,
)


DATASET_PATH = ROOT / "evals" / "research_retrieval" / "research_questions_v1.json"
TRANSLATIONS_PATH = ROOT / "evals" / "research_retrieval" / "research_translations_v1.json"
BASELINE_PATH = ROOT / "evals" / "research_retrieval" / "research_dense_baseline_v1.json"
METADATA_AUDIT_PATH = ROOT / "evals" / "research_retrieval" / "reference_metadata_collection_v1.json"
OUTPUT_JSON = ROOT / "evals" / "research_retrieval" / "references_aware_experiment_v1.json"
OUTPUT_MD = ROOT / "evals" / "research_retrieval" / "references_aware_report_v1.md"
EXPECTED_DATASET_HASH = "be1e8354babef3927163c19e32c614f7639323095900c487d559ade8249e7357"
BASELINE_COLLECTION = "ef_research_bge_m3_c512_o64_v1"
VARIANT_COLLECTION = "ef_research_bge_m3_c512_o64_refmeta_v1"
OVERSAMPLE_TOP_K = 20
BRANCH_TOP_K = 10
FINAL_TOP_K = 5
RRF_K = 60


class ReferencesFilteredDenseRetriever:
    """每路固定 Top-20 oversampling，过滤 References 后按原顺序回填 Top-10。"""

    def __init__(self, dense_retriever: Any) -> None:
        self.dense_retriever = dense_retriever

    def retrieve(self, query: str, top_k: int | None = None) -> list[Any]:
        """保持非 References 候选的 cosine 次序，并重建连续的分支排名。"""

        effective_top_k = BRANCH_TOP_K if top_k is None else top_k
        if effective_top_k != BRANCH_TOP_K:
            raise ValueError("reference-aware branch budget is frozen at Top-10")
        oversampled = self.dense_retriever.retrieve(query, top_k=OVERSAMPLE_TOP_K)
        body_candidates = [
            item for item in oversampled
            if item.chunk.metadata.get("section_type") != "references"
        ][:BRANCH_TOP_K]
        return [
            item.model_copy(update={"rank": rank}, deep=True)
            for rank, item in enumerate(body_candidates, start=1)
        ]


def _load_frozen_inputs() -> tuple[dict[str, Any], dict[str, str], dict[str, Any], dict[str, Any]]:
    """在初始化 Embedding 前验证冻结 Dataset、Baseline 和 metadata Collection 审计。"""

    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    claimed_hash = dataset.pop("dataset_hash")
    if claimed_hash != EXPECTED_DATASET_HASH or _canonical_hash(dataset) != claimed_hash:
        raise ValueError("frozen research dataset hash mismatch")
    dataset["dataset_hash"] = claimed_hash
    translations = json.loads(TRANSLATIONS_PATH.read_text(encoding="utf-8"))
    if translations.get("dataset_hash") != claimed_hash:
        raise ValueError("frozen translations hash mismatch")
    baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    if baseline.get("dataset_hash") != claimed_hash:
        raise ValueError("baseline uses a different dataset")
    metadata_audit = json.loads(METADATA_AUDIT_PATH.read_text(encoding="utf-8"))
    if not (
        metadata_audit.get("chunk_count") == 361
        and metadata_audit.get("chunk_ids_identical")
        and metadata_audit.get("chunk_text_identical")
    ):
        raise ValueError("reference metadata collection failed identity audit")
    return dataset, translations["translations"], baseline, metadata_audit


def audit_reference_targeted_questions(questions: Sequence[Mapping[str, Any]]) -> list[str]:
    """识别明确索要引用或 bibliography 信息的题；普通“reference point”不触发。"""

    patterns = (
        r"\bbibliograph(?:y|ies|ic)\b",
        r"\bcitation list\b",
        r"\bwhich (?:paper|work|method)s? (?:is|are|were) cited\b",
        r"\bwhat (?:paper|work|method)s? (?:is|are|were) cited\b",
        r"引用了(?:谁|哪些)",
        r"参考文献(?:有哪些|列表|信息)",
    )
    return [
        question["question_id"]
        for question in questions
        if any(re.search(pattern, question["query"], re.IGNORECASE) for pattern in patterns)
    ]


def _rank(results: Sequence[Any], groups: Sequence[Sequence[str]]) -> int | None:
    """复用共享 Gold group 语义计算全部来源同时覆盖的最早深度。"""

    from evidenceflow.eval.gold import gold_group_rank

    if any(not group for group in groups):
        return None
    return gold_group_rank(
        [result.chunk.chunk_id for result in results],
        required_gold_groups=groups,
    )


def _primary_failure(rank: int | None, groups: Sequence[Sequence[str]]) -> str:
    """按冻结 Extraction→Chunk→Candidate→Ranking 分层返回主要结果。"""

    if any(not group for group in groups):
        return "chunk_failure"
    if rank is None:
        return "candidate_retrieval_failure"
    if rank > FINAL_TOP_K:
        return "ranking_failure"
    return "success"


def _evaluate_question(question: Mapping[str, Any], retriever: Any, chunks: Sequence[Any]) -> dict[str, Any]:
    """运行一道 Variant Query，并保留 Gold、污染槽位及多文档覆盖诊断。"""

    groups = _gold_groups(question, chunks)
    results = retriever.retrieve(question["query"], top_k=20)
    rank = _rank(results, groups)
    covered_groups = sum(
        any(result.chunk.chunk_id in set(group) for result in results)
        for group in groups
    )
    row = {
        "question_id": question["question_id"],
        "query": question["query"],
        "language": question["language"],
        "language_group": question["language_group"],
        "question_type": question["question_type"],
        "evidence_type": question["evidence_type"],
        "gold_files": [source["file_name"] for source in question["gold_sources"]],
        "gold_chunk_group_sizes": [len(group) for group in groups],
        "gold_group_rank": rank,
        "primary_failure": _primary_failure(rank, groups),
        "covered_gold_source_count": covered_groups,
        "gold_source_count": len(groups),
        "reference_candidates_in_top5": sum(
            result.chunk.metadata.get("section_type") == "references"
            for result in results[:FINAL_TOP_K]
        ),
        "candidates": [
            {
                "rank": result.rank,
                "chunk_id": result.chunk.chunk_id,
                "file_name": result.chunk.file_name,
                "page_start": result.chunk.page_start,
                "page_end": result.chunk.page_end,
                "section_type": result.chunk.metadata.get("section_type"),
                "original_rank": result.original_rank,
                "translated_rank": result.translated_rank,
                "fusion_score": result.fusion_score,
            }
            for result in results
        ],
    }
    for depth in DEPTHS:
        row[f"ndcg_at_{depth}"] = _source_ndcg(results, groups, depth)
    return row


def _baseline_multi_coverage(row: Mapping[str, Any], groups: Sequence[Sequence[str]]) -> int:
    """从已冻结 Baseline 候选计算多来源部分覆盖数。"""

    candidate_ids = {candidate["chunk_id"] for candidate in row["candidates"]}
    return sum(bool(candidate_ids & set(group)) for group in groups)


def _runtime_fingerprint() -> dict[str, str]:
    """记录未修改正式路径的 SHA-256，供实验结束后复核。"""

    files = (
        "src/evidenceflow/runtime.py",
        "src/evidenceflow/retrieval/retriever.py",
        "src/evidenceflow/retrieval/multilingual.py",
        "src/evidenceflow/retrieval/fusion.py",
        "src/evidenceflow/ingestion/chunker.py",
        "src/evidenceflow/ingestion/cleaner.py",
        "PROJECT_SPEC.md",
    )
    return {
        relative: hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        for relative in files
    }


def run_experiment() -> dict[str, Any]:
    """先执行 22 题机制检查；有 Gold promotion 后才完成正式 40 题评测。"""

    dataset, translations_by_id, baseline, metadata_audit = _load_frozen_inputs()
    targeted_ids = audit_reference_targeted_questions(dataset["questions"])
    if targeted_ids:
        raise RuntimeError(
            "reference-targeted questions exist; global filter contract must be reconsidered"
        )

    from evidenceflow.config import Settings
    from evidenceflow.embedding import BgeM3EmbeddingProvider
    from evidenceflow.retrieval import DenseRetriever, MultilingualRetriever
    from evidenceflow.storage import ChromaVectorStore

    settings = Settings()
    variant_store = ChromaVectorStore(settings.vector_db_path, VARIANT_COLLECTION)
    if variant_store._collection.count() != 361:
        raise ValueError("variant collection must contain 361 chunks")
    chunks = _all_chunks(variant_store)
    translations_by_query = {
        question["query"]: translations_by_id[question["question_id"]]
        for question in dataset["questions"]
    }
    raw_dense = DenseRetriever(
        BgeM3EmbeddingProvider(settings.embedding_model),
        variant_store,
        top_k=OVERSAMPLE_TOP_K,
    )
    filtered_dense = ReferencesFilteredDenseRetriever(raw_dense)
    retriever = MultilingualRetriever(
        filtered_dense,
        FrozenTranslationProvider(translations_by_query),
        candidate_top_k=BRANCH_TOP_K,
        final_top_k=FINAL_TOP_K,
        rrf_k=RRF_K,
        original_weight=1.0,
        translation_weight=1.0,
    )

    baseline_by_id = {row["question_id"]: row for row in baseline["question_results"]}
    questions_by_id = {question["question_id"]: question for question in dataset["questions"]}
    mechanism_ids = [
        row["question_id"]
        for row in baseline["question_results"]
        if row["reference_candidates_in_top5"] > 0
    ]
    variant_by_id: dict[str, dict[str, Any]] = {}
    mechanism_rows = []
    for question_id in mechanism_ids:
        variant_row = _evaluate_question(questions_by_id[question_id], retriever, chunks)
        variant_by_id[question_id] = variant_row
        baseline_row = baseline_by_id[question_id]
        baseline_rank = baseline_row["gold_group_rank"]
        variant_rank = variant_row["gold_group_rank"]
        improved = variant_rank is not None and (
            baseline_rank is None or variant_rank < baseline_rank
        )
        top5_promoted = (
            baseline_rank is None or baseline_rank > FINAL_TOP_K
        ) and variant_rank is not None and variant_rank <= FINAL_TOP_K
        regressed = (
            baseline_rank is not None
            and (variant_rank is None or variant_rank > baseline_rank)
        )
        mechanism_rows.append(
            {
                "question_id": question_id,
                "baseline_rank": baseline_rank,
                "variant_rank": variant_rank,
                "reference_slots_removed": baseline_row["reference_candidates_in_top5"],
                "gold_rank_improved": improved,
                "gold_promoted_into_top5": top5_promoted,
                "unexpected_rank_regression": regressed,
            }
        )
        print(f"[mechanism] {question_id}: {baseline_rank} -> {variant_rank}", flush=True)

    mechanism = {
        "question_count": len(mechanism_rows),
        "removed_reference_slots": sum(row["reference_slots_removed"] for row in mechanism_rows),
        "gold_rank_improved_count": sum(row["gold_rank_improved"] for row in mechanism_rows),
        "gold_promoted_into_top5_count": sum(row["gold_promoted_into_top5"] for row in mechanism_rows),
        "no_change_count": sum(
            row["baseline_rank"] == row["variant_rank"] for row in mechanism_rows
        ),
        "unexpected_rank_regression_count": sum(
            row["unexpected_rank_regression"] for row in mechanism_rows
        ),
        "rows": mechanism_rows,
    }
    if mechanism["gold_rank_improved_count"] == 0:
        return {
            "experiment_version": "references-aware-filtering-v1",
            "status": "mechanism_failed_stop",
            "dataset_hash": dataset["dataset_hash"],
            "reference_targeted_question_count": 0,
            "metadata_audit": metadata_audit,
            "mechanism_check": mechanism,
            "runtime_modified": False,
            "runtime_fingerprint": _runtime_fingerprint(),
            "project_spec_version": "0.14",
        }

    # 机制成立后只补跑剩余问题；前 22 题复用机制检查结果，避免重复 Query。
    for question in dataset["questions"]:
        if question["question_id"] not in variant_by_id:
            variant_by_id[question["question_id"]] = _evaluate_question(
                question, retriever, chunks
            )
    variant_rows = [variant_by_id[question["question_id"]] for question in dataset["questions"]]
    variant_breakdowns = _breakdowns(variant_rows)
    failure_counts = Counter(row["primary_failure"] for row in variant_rows)
    for label in (
        "success",
        "chunk_failure",
        "candidate_retrieval_failure",
        "ranking_failure",
    ):
        failure_counts.setdefault(label, 0)

    transitions = []
    for row in variant_rows:
        baseline_row = baseline_by_id[row["question_id"]]
        baseline_hit = baseline_row["gold_group_rank"] is not None and baseline_row["gold_group_rank"] <= 5
        variant_hit = row["gold_group_rank"] is not None and row["gold_group_rank"] <= 5
        transition = "recovered" if not baseline_hit and variant_hit else (
            "regressed" if baseline_hit and not variant_hit else "preserved"
        )
        transitions.append({"question_id": row["question_id"], "transition": transition})
    transition_counts = Counter(item["transition"] for item in transitions)

    candidate_recovered = sum(
        baseline_by_id[row["question_id"]]["primary_failure"] == "candidate_retrieval_failure"
        and row["primary_failure"] != "candidate_retrieval_failure"
        for row in variant_rows
    )
    ranking_recovered = sum(
        baseline_by_id[row["question_id"]]["primary_failure"] == "ranking_failure"
        and row["primary_failure"] == "success"
        for row in variant_rows
    )

    multi_questions = [q for q in dataset["questions"] if q["question_type"] == "multi_document"]
    multi_rows = []
    for question in multi_questions:
        groups = _gold_groups(question, chunks)
        baseline_row = baseline_by_id[question["question_id"]]
        variant_row = variant_by_id[question["question_id"]]
        multi_rows.append(
            {
                "question_id": question["question_id"],
                "source_count": len(groups),
                "baseline_covered_sources": _baseline_multi_coverage(baseline_row, groups),
                "variant_covered_sources": variant_row["covered_gold_source_count"],
                "baseline_complete": baseline_row["gold_group_rank"] is not None,
                "variant_complete": variant_row["gold_group_rank"] is not None,
            }
        )

    subgroup_regressions = []
    for dimension in ("language_group", "question_type", "evidence_type"):
        for name, metric in variant_breakdowns[dimension].items():
            baseline_metric = baseline["breakdowns"][dimension][name]
            if metric["hit_at_5"] < baseline_metric["hit_at_5"]:
                subgroup_regressions.append(
                    {
                        "dimension": dimension,
                        "group": name,
                        "baseline_hit_at_5": baseline_metric["hit_at_5"],
                        "variant_hit_at_5": metric["hit_at_5"],
                    }
                )

    overall = variant_breakdowns["overall"]
    decision = "pass" if (
        overall["hit_at_5"] > baseline["breakdowns"]["overall"]["hit_at_5"]
        and transition_counts["regressed"] <= 1
        and (candidate_recovered > 0 or ranking_recovered > 0)
        and not subgroup_regressions
    ) else (
        "partial" if (
            sum(row["reference_candidates_in_top5"] for row in variant_rows) == 0
            and transition_counts["regressed"] <= 1
            and overall["hit_at_5"] >= baseline["breakdowns"]["overall"]["hit_at_5"]
        ) else "fail"
    )
    return {
        "experiment_version": "references-aware-filtering-v1",
        "status": "formal_evaluation_complete",
        "dataset_hash": dataset["dataset_hash"],
        "hypothesis": (
            "References consume limited Dense slots; deterministic metadata filtering and "
            "backfill can promote answer-bearing non-reference evidence."
        ),
        "reference_targeted_question_count": 0,
        "configuration": {
            "baseline_collection": BASELINE_COLLECTION,
            "variant_collection": VARIANT_COLLECTION,
            "oversample_top_k_per_branch": OVERSAMPLE_TOP_K,
            "candidate_top_k_per_branch": BRANCH_TOP_K,
            "final_top_k": FINAL_TOP_K,
            "rrf_k": RRF_K,
            "weights": [1.0, 1.0],
            "embedding_model": settings.embedding_model,
            "generator_or_llm_called": False,
            "sweeps_run": [],
        },
        "metadata_audit": metadata_audit,
        "mechanism_check": mechanism,
        "baseline_breakdowns": baseline["breakdowns"],
        "variant_breakdowns": variant_breakdowns,
        "transitions": {
            "counts": dict(transition_counts),
            "rows": transitions,
            "candidate_failure_recovered_count": candidate_recovered,
            "ranking_failure_recovered_count": ranking_recovered,
        },
        "reference_pollution": {
            "baseline_questions_with_reference_in_top5": baseline["retrieval_artifacts"]["questions_with_reference_chunks_in_top5"],
            "variant_questions_with_reference_in_top5": sum(
                row["reference_candidates_in_top5"] > 0 for row in variant_rows
            ),
            "baseline_reference_slots_in_top5": baseline["retrieval_artifacts"]["reference_chunks_in_all_top5_results"],
            "variant_reference_slots_in_top5": sum(
                row["reference_candidates_in_top5"] for row in variant_rows
            ),
        },
        "baseline_failure_counts": baseline["failure_counts"],
        "variant_failure_counts": dict(failure_counts),
        "multi_document": {
            "baseline_complete_count": sum(row["baseline_complete"] for row in multi_rows),
            "variant_complete_count": sum(row["variant_complete"] for row in multi_rows),
            "rows": multi_rows,
        },
        "subgroup_regressions": subgroup_regressions,
        "decision": {
            "gate": decision,
            "references_pollution_proven_causal": mechanism["gold_rank_improved_count"] > 0,
            "recommend_runtime_integration": decision == "pass",
        },
        "variant_question_results": variant_rows,
        "runtime_modified": False,
        "runtime_fingerprint": _runtime_fingerprint(),
        "project_spec_version": "0.14",
    }


def render_markdown(payload: Mapping[str, Any]) -> str:
    """把机制证据、正式指标和预冻结决策门渲染为审计报告。"""

    metadata = payload["metadata_audit"]
    mechanism = payload["mechanism_check"]
    lines = [
        "# References-Aware Metadata / Filtering Controlled Experiment v1",
        "",
        f"- Status: `{payload['status']}`",
        f"- Dataset hash: `{payload['dataset_hash']}`",
        f"- Reference-targeted questions: {payload['reference_targeted_question_count']}/40.",
        f"- Chunks: {metadata['chunk_count']}（body {metadata['section_counts']['body']}；references {metadata['section_counts']['references']}；supplementary {metadata['section_counts']['supplementary']}）。",
        f"- Detection: References {metadata['detection_accuracy']['references_correct']}/10；Supplementary {metadata['detection_accuracy']['supplementary_correct']}/10；FP/FN {metadata['detection_accuracy']['false_positive_count']}/{metadata['detection_accuracy']['false_negative_count']}。",
        "- Chunk ID、文本和 embedding 不变；只增加实验 metadata。",
        "",
        "## Mechanism check",
        "",
        f"- Questions: {mechanism['question_count']}.",
        f"- Reference slots removed: {mechanism['removed_reference_slots']}.",
        f"- Gold rank improved: {mechanism['gold_rank_improved_count']}.",
        f"- Gold promoted into Top-5: {mechanism['gold_promoted_into_top5_count']}.",
        f"- No change: {mechanism['no_change_count']}.",
        f"- Unexpected rank regression: {mechanism['unexpected_rank_regression_count']}.",
    ]
    if payload["status"] != "formal_evaluation_complete":
        lines += ["", "机制未成立，按预冻结合同停止；未运行正式 40 题评测。", ""]
        return "\n".join(lines)

    baseline = payload["baseline_breakdowns"]["overall"]
    variant = payload["variant_breakdowns"]["overall"]
    lines += [
        "",
        "## Formal frozen 40Q evaluation",
        "",
        "| Metric | Baseline | Variant |",
        "|---|---:|---:|",
    ]
    for key, label in (
        ("hit_at_1", "Hit@1"), ("hit_at_3", "Hit@3"), ("hit_at_5", "Hit@5"),
        ("hit_at_10", "Hit@10"), ("hit_at_20", "Hit@20"), ("mrr", "MRR"),
        ("ndcg_at_20", "nDCG@20"),
    ):
        lines.append(f"| {label} | {baseline[key]:.3f} | {variant[key]:.3f} |")
    transitions = payload["transitions"]
    pollution = payload["reference_pollution"]
    lines += [
        "",
        "## Outcomes",
        "",
        f"- Recovered / preserved / regressed: {transitions['counts'].get('recovered', 0)} / {transitions['counts'].get('preserved', 0)} / {transitions['counts'].get('regressed', 0)}.",
        f"- Candidate failure recovered: {transitions['candidate_failure_recovered_count']}；ranking failure recovered: {transitions['ranking_failure_recovered_count']}.",
        f"- Reference Top-5 questions: {pollution['baseline_questions_with_reference_in_top5']} → {pollution['variant_questions_with_reference_in_top5']}.",
        f"- Reference Top-5 slots: {pollution['baseline_reference_slots_in_top5']} → {pollution['variant_reference_slots_in_top5']}.",
        f"- Candidate / ranking / chunk failures: {payload['baseline_failure_counts'].get('candidate_retrieval_failure', 0)}/{payload['baseline_failure_counts'].get('ranking_failure', 0)}/{payload['baseline_failure_counts'].get('chunk_failure', 0)} → {payload['variant_failure_counts'].get('candidate_retrieval_failure', 0)}/{payload['variant_failure_counts'].get('ranking_failure', 0)}/{payload['variant_failure_counts'].get('chunk_failure', 0)}.",
        f"- Multi-document complete hit: {payload['multi_document']['baseline_complete_count']}/4 → {payload['multi_document']['variant_complete_count']}/4.",
        f"- Subgroup regressions: {len(payload['subgroup_regressions'])}.",
        "",
        "## Decision",
        "",
        f"- Gate: **{payload['decision']['gate'].upper()}**.",
        f"- References pollution proven causal for at least one failure: {payload['decision']['references_pollution_proven_causal']}.",
        f"- Recommend Runtime integration: {payload['decision']['recommend_runtime_integration']}.",
        "- Runtime 未修改；PROJECT_SPEC 保持 v0.14；未调用 Generator/LLM。",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    """执行受控实验并同步写出 JSON/Markdown。"""

    payload = run_experiment()
    OUTPUT_JSON.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    OUTPUT_MD.write_text(render_markdown(payload), encoding="utf-8")
    print(json.dumps({
        "status": payload["status"],
        "mechanism_check": {k: v for k, v in payload["mechanism_check"].items() if k != "rows"},
        "decision": payload.get("decision"),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
