"""已冻结的 BM25 Candidate Augmentation 历史实验。"""

from __future__ import annotations

import hashlib
import json
import math
import os
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
for import_root in (ROOT, ROOT / "src"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from experiments.diagnose_research_candidate_failures import (
    BM25_B,
    BM25_K1,
    EXPECTED_COLLECTION,
    EXPECTED_DATASET_HASH,
    _load_and_verify,
)
from experiments.run_research_dense_baseline import _all_chunks, _gold_groups


OUTPUT_JSON = ROOT / "evals" / "research_retrieval" / "academic_candidate_augmentation_v1.json"
OUTPUT_MD = ROOT / "evals" / "research_retrieval" / "academic_candidate_augmentation_report_v1.md"
DENSE_BRANCH_TOP_K = 10
BM25_TOP_K = 10
MAX_AUGMENTED_POOL_SIZE = 30
PHASE_A_MIN_RECOVERY = 4
GENERALIZATION_MIN_RECOVERY = 3


@dataclass
class AttributedCandidate:
    """保存同一 Chunk 的实验级来源归因，不改变正式 Domain Model。"""

    chunk: Any
    retrieval_sources: list[str] = field(default_factory=list)
    source_ranks: dict[str, int] = field(default_factory=dict)


def build_candidate_pool(
    original_dense: Sequence[Any],
    translated_dense: Sequence[Any],
    bm25_results: Sequence[Any] = (),
) -> list[AttributedCandidate]:
    """按当前 chunk_id 去重并合并来源；BM25 只追加，绝不驱逐 Dense 候选。"""

    by_id: dict[str, AttributedCandidate] = {}
    for source_name, results in (
        ("dense_original", original_dense),
        ("dense_translation", translated_dense),
        ("bm25", bm25_results),
    ):
        for result in results:
            chunk_id = result.chunk.chunk_id
            candidate = by_id.setdefault(
                chunk_id,
                AttributedCandidate(chunk=result.chunk.model_copy(deep=True)),
            )
            if source_name not in candidate.retrieval_sources:
                candidate.retrieval_sources.append(source_name)
            candidate.source_ranks[source_name] = result.rank
    return list(by_id.values())


def candidate_coverage(
    pool: Sequence[AttributedCandidate],
    required_gold_groups: Sequence[Sequence[str]],
) -> tuple[bool, int]:
    """按共享 Gold group 语义计算完整覆盖及已覆盖 Required Source 数。"""

    pool_ids = {candidate.chunk.chunk_id for candidate in pool}
    covered = sum(bool(pool_ids & set(group)) for group in required_gold_groups)
    return covered == len(required_gold_groups), covered


def _pool_stats(values: Sequence[int]) -> dict[str, float | int]:
    """返回 Candidate 成本的 min/median/mean/P95/max；P95 使用 nearest-rank。"""

    ordered = sorted(values)
    if not ordered:
        raise ValueError("candidate pool statistics require non-empty values")
    p95_index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return {
        "min": min(ordered),
        "median": statistics.median(ordered),
        "mean": statistics.mean(ordered),
        "p95": ordered[p95_index],
        "max": max(ordered),
    }


def _serialize_pool(pool: Sequence[AttributedCandidate]) -> list[dict[str, Any]]:
    """序列化候选身份、来源和来源内 Rank，不发明跨来源最终排名。"""

    return [
        {
            "chunk_id": candidate.chunk.chunk_id,
            "file_name": candidate.chunk.file_name,
            "page_start": candidate.chunk.page_start,
            "page_end": candidate.chunk.page_end,
            "retrieval_sources": candidate.retrieval_sources,
            "source_ranks": candidate.source_ranks,
        }
        for candidate in pool
    ]


def _runtime_fingerprint() -> dict[str, str]:
    """记录正式文件 Hash，证明 add-only 实验未接入 Runtime。"""

    files = (
        "PROJECT_SPEC.md",
        "src/evidenceflow/runtime.py",
        "src/evidenceflow/retrieval/retriever.py",
        "src/evidenceflow/retrieval/multilingual.py",
        "src/evidenceflow/retrieval/fusion.py",
        "src/evidenceflow/eval/lexical.py",
    )
    return {
        relative: hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        for relative in files
    }


def _evaluate_question(
    question: Mapping[str, Any],
    *,
    dense: Any,
    bm25: Any,
    translations_by_id: Mapping[str, str],
    chunks: Sequence[Any],
) -> dict[str, Any]:
    """构建 Baseline/Variant Pool，验证子集 invariant 并计算 Gold 来源归因。"""

    original = dense.retrieve(question["query"], top_k=DENSE_BRANCH_TOP_K)
    translated = []
    if question["language"] == "zh":
        translated = dense.retrieve(
            translations_by_id[question["question_id"]],
            top_k=DENSE_BRANCH_TOP_K,
        )
    lexical = bm25.search(question["query"], top_k=BM25_TOP_K)
    baseline_pool = build_candidate_pool(original, translated)
    augmented_pool = build_candidate_pool(original, translated, lexical)
    baseline_ids = {candidate.chunk.chunk_id for candidate in baseline_pool}
    augmented_ids = {candidate.chunk.chunk_id for candidate in augmented_pool}
    if not baseline_ids <= augmented_ids:
        raise RuntimeError("add-only invariant violated: Dense candidate was evicted")
    if len(augmented_pool) > MAX_AUGMENTED_POOL_SIZE:
        raise RuntimeError("augmented candidate pool exceeds theoretical maximum")

    groups = _gold_groups(question, chunks)
    if any(not group for group in groups):
        # Chunk failures remain uncovered under the unchanged Gold/Chunk contract.
        baseline_complete = augmented_complete = False
        baseline_sources = augmented_sources = 0
    else:
        baseline_complete, baseline_sources = candidate_coverage(baseline_pool, groups)
        augmented_complete, augmented_sources = candidate_coverage(augmented_pool, groups)
    if baseline_complete and not augmented_complete:
        raise RuntimeError("add-only augmentation regressed a Baseline Gold hit")

    augmented_by_id = {candidate.chunk.chunk_id: candidate for candidate in augmented_pool}
    gold_attributions = []
    for group in groups:
        matched = [augmented_by_id[chunk_id] for chunk_id in group if chunk_id in augmented_by_id]
        sources = sorted({source for candidate in matched for source in candidate.retrieval_sources})
        gold_attributions.append(
            {
                "covered": bool(matched),
                "matched_chunk_ids": [candidate.chunk.chunk_id for candidate in matched],
                "retrieval_sources": sources,
                "bm25_only": bool(matched) and all(
                    candidate.retrieval_sources == ["bm25"] for candidate in matched
                ),
                "dense_and_bm25_shared": any(
                    "bm25" in candidate.retrieval_sources
                    and any(source.startswith("dense_") for source in candidate.retrieval_sources)
                    for candidate in matched
                ),
            }
        )

    return {
        "question_id": question["question_id"],
        "question_type": question["question_type"],
        "evidence_type": question["evidence_type"],
        "language_group": question["language_group"],
        "baseline_candidate_count": len(baseline_pool),
        "augmented_candidate_count": len(augmented_pool),
        "unique_bm25_additions": len(augmented_ids - baseline_ids),
        "bm25_duplicate_count": BM25_TOP_K - len(augmented_ids - baseline_ids),
        "baseline_complete_coverage": baseline_complete,
        "augmented_complete_coverage": augmented_complete,
        "baseline_covered_source_count": baseline_sources,
        "augmented_covered_source_count": augmented_sources,
        "required_source_count": len(groups),
        "transition": (
            "recovered" if not baseline_complete and augmented_complete else "preserved"
        ),
        "gold_attributions": gold_attributions,
        "baseline_pool": _serialize_pool(baseline_pool),
        "augmented_pool": _serialize_pool(augmented_pool),
    }


def _breakdown(rows: Sequence[Mapping[str, Any]], field_name: str) -> dict[str, Any]:
    """按题型、证据或语言聚合 Baseline/Variant coverage 和增量恢复。"""

    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row[field_name])].append(row)
    return {
        name: {
            "N": len(selected),
            "baseline_covered": sum(row["baseline_complete_coverage"] for row in selected),
            "variant_covered": sum(row["augmented_complete_coverage"] for row in selected),
            "recovered": sum(row["transition"] == "recovered" for row in selected),
        }
        for name, selected in sorted(groups.items())
    }


def run_experiment() -> dict[str, Any]:
    """先跑固定 12 题机制门，通过后一次性验证 Frozen 40Q Candidate Coverage。"""

    dataset, translations_by_id, _baseline, failure_ids = _load_and_verify()
    if dataset["dataset_hash"] != EXPECTED_DATASET_HASH:
        raise ValueError("dataset hash guard failed")
    os.environ["HF_HOME"] = str((ROOT / ".cache" / "huggingface").resolve())
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    from evidenceflow.config import Settings
    from evidenceflow.embedding import BgeM3EmbeddingProvider
    from evidenceflow.eval.lexical import BM25Index
    from evidenceflow.ingestion import BgeM3Tokenizer
    from evidenceflow.retrieval import DenseRetriever
    from evidenceflow.storage import ChromaVectorStore

    settings = Settings()
    if (
        settings.collection_name != EXPECTED_COLLECTION
        or settings.chunk_size != 512
        or settings.chunk_overlap != 64
        or settings.embedding_model != "BAAI/bge-m3"
    ):
        raise ValueError("Academic Baseline runtime configuration drifted")
    store = ChromaVectorStore(settings.vector_db_path, settings.collection_name)
    if store._collection.count() != 361:
        raise ValueError("Academic Baseline collection must contain 361 chunks")
    chunks = _all_chunks(store)
    dense = DenseRetriever(
        BgeM3EmbeddingProvider(settings.embedding_model),
        store,
        top_k=DENSE_BRANCH_TOP_K,
    )
    bm25 = BM25Index(
        chunks,
        BgeM3Tokenizer(settings.embedding_model),
        k1=BM25_K1,
        b=BM25_B,
    )
    questions_by_id = {question["question_id"]: question for question in dataset["questions"]}

    cached_rows: dict[str, dict[str, Any]] = {}
    phase_a_rows = []
    for question_id in failure_ids:
        row = _evaluate_question(
            questions_by_id[question_id],
            dense=dense,
            bm25=bm25,
            translations_by_id=translations_by_id,
            chunks=chunks,
        )
        cached_rows[question_id] = row
        phase_a_rows.append(row)
        print(f"[phase-a] {question_id}: {row['transition']}", flush=True)
    phase_a_recovered = [row["question_id"] for row in phase_a_rows if row["transition"] == "recovered"]
    phase_a = {
        "candidate_failure_count": len(phase_a_rows),
        "recovered_count": len(phase_a_recovered),
        "recovered_ids": phase_a_recovered,
        "still_missed_ids": [
            row["question_id"] for row in phase_a_rows
            if not row["augmented_complete_coverage"]
        ],
        "success_gate_minimum": PHASE_A_MIN_RECOVERY,
        "passed": len(phase_a_recovered) >= PHASE_A_MIN_RECOVERY,
    }
    base_payload = {
        "experiment_id": "academic_candidate_augmentation_v1",
        "dataset_hash": dataset["dataset_hash"],
        "configuration": {
            "collection_name": settings.collection_name,
            "dense_original_top_k": DENSE_BRANCH_TOP_K,
            "dense_translation_top_k": DENSE_BRANCH_TOP_K,
            "bm25_top_k": BM25_TOP_K,
            "bm25_query": "original_only",
            "bm25_k1": BM25_K1,
            "bm25_b": BM25_B,
            "augmentation": "add_only",
            "dedup_identity": "chunk_id",
            "ranking": "none",
            "generator_context_citation_llm_called": False,
            "reranker_used": False,
            "parameter_sweeps": [],
        },
        "phase_a": phase_a,
    }
    if not phase_a["passed"]:
        return base_payload | {
            "status": "phase_a_failed_stop",
            "decision": {
                "gate": "fail",
                "candidate_augmentation_generalizes": False,
                "next_experiment": "dense_only_reranker",
            },
            "runtime_modified": False,
            "project_spec_version": "0.14",
            "runtime_fingerprint": _runtime_fingerprint(),
        }

    # Phase A 通过后只补跑其余 28 题，避免重复检索固定失败题。
    for question in dataset["questions"]:
        if question["question_id"] not in cached_rows:
            cached_rows[question["question_id"]] = _evaluate_question(
                question,
                dense=dense,
                bm25=bm25,
                translations_by_id=translations_by_id,
                chunks=chunks,
            )
    rows = [cached_rows[question["question_id"]] for question in dataset["questions"]]
    baseline_covered = sum(row["baseline_complete_coverage"] for row in rows)
    variant_covered = sum(row["augmented_complete_coverage"] for row in rows)
    recovered_rows = [row for row in rows if row["transition"] == "recovered"]
    regressed = sum(
        row["baseline_complete_coverage"] and not row["augmented_complete_coverage"]
        for row in rows
    )
    bm25_only_recovery = sum(
        row["transition"] == "recovered"
        and any(attribution["bm25_only"] for attribution in row["gold_attributions"])
        for row in rows
    )
    shared_gold = sum(
        any(attribution["dense_and_bm25_shared"] for attribution in row["gold_attributions"])
        for row in rows
    )
    total_bm25 = len(rows) * BM25_TOP_K
    total_duplicates = sum(row["bm25_duplicate_count"] for row in rows)
    cost = {
        "baseline_pool": _pool_stats([row["baseline_candidate_count"] for row in rows]),
        "augmented_pool": _pool_stats([row["augmented_candidate_count"] for row in rows]),
        "unique_bm25_additions": _pool_stats([row["unique_bm25_additions"] for row in rows]),
        "bm25_duplicate_count": total_duplicates,
        "bm25_candidate_count": total_bm25,
        "bm25_duplicate_rate": total_duplicates / total_bm25,
    }
    multi_rows = [row for row in rows if row["question_type"] == "multi_document"]
    multi = {
        "question_count": len(multi_rows),
        "baseline_complete": sum(row["baseline_complete_coverage"] for row in multi_rows),
        "variant_complete": sum(row["augmented_complete_coverage"] for row in multi_rows),
        "baseline_partial": sum(
            0 < row["baseline_covered_source_count"] < row["required_source_count"]
            for row in multi_rows
        ),
        "variant_partial": sum(
            0 < row["augmented_covered_source_count"] < row["required_source_count"]
            for row in multi_rows
        ),
        "rows": [
            {
                "question_id": row["question_id"],
                "baseline_covered_sources": row["baseline_covered_source_count"],
                "variant_covered_sources": row["augmented_covered_source_count"],
                "required_sources": row["required_source_count"],
            }
            for row in multi_rows
        ],
    }
    attribution_valid = all(
        any(attribution["covered"] and "bm25" in attribution["retrieval_sources"] for attribution in row["gold_attributions"])
        for row in recovered_rows
    )
    gates = {
        "coverage_recovery_minimum": GENERALIZATION_MIN_RECOVERY,
        "coverage_gate_passed": len(recovered_rows) >= GENERALIZATION_MIN_RECOVERY,
        "preservation_gate_passed": regressed == 0,
        "attribution_gate_passed": attribution_valid,
        "cost_reported": True,
    }
    passed = all(
        gates[key]
        for key in (
            "coverage_gate_passed",
            "preservation_gate_passed",
            "attribution_gate_passed",
            "cost_reported",
        )
    )
    return base_payload | {
        "status": "formal_40q_complete",
        "candidate_coverage": {
            "question_count": len(rows),
            "baseline_covered": baseline_covered,
            "baseline_rate": baseline_covered / len(rows),
            "variant_covered": variant_covered,
            "variant_rate": variant_covered / len(rows),
            "net_recovery": len(recovered_rows),
            "recovered_ids": [row["question_id"] for row in recovered_rows],
            # Preserved 严格表示 Baseline hit → Variant hit，不包含两路都 miss 的题。
            "preserved_count": baseline_covered - regressed,
            "regressed_count": regressed,
            "still_missed_count": len(rows) - variant_covered,
        },
        "candidate_cost": cost,
        "attribution": {
            "bm25_only_gold_recovery_count": bm25_only_recovery,
            "dense_and_bm25_shared_gold_question_count": shared_gold,
            "all_recoveries_attributable_to_bm25": attribution_valid,
        },
        "breakdowns": {
            "question_type": _breakdown(rows, "question_type"),
            "evidence_type": _breakdown(rows, "evidence_type"),
            "language_group": _breakdown(rows, "language_group"),
        },
        "multi_document": multi,
        "generalization_gates": gates,
        "decision": {
            "gate": "pass" if passed else "fail",
            "candidate_augmentation_generalizes": passed,
            "benefit_vs_complexity": (
                f"Recovered {len(recovered_rows)}/40 complete Gold sets with "
                f"{cost['unique_bm25_additions']['mean']:.2f} average unique candidates added."
            ),
            "next_experiment": (
                "augmented_candidate_reranking" if passed else "dense_only_reranker"
            ),
            "recommend_runtime_integration": False,
        },
        "questions": rows,
        "runtime_modified": False,
        "project_spec_version": "0.14",
        "runtime_fingerprint": _runtime_fingerprint(),
    }


def render_markdown(payload: Mapping[str, Any]) -> str:
    """渲染 Candidate Coverage、成本、归因和预冻结 Gate。"""

    phase_a = payload["phase_a"]
    lines = [
        "# Academic Candidate Augmentation Controlled Experiment v1",
        "",
        f"- Dataset hash: `{payload['dataset_hash']}`",
        "- Variant: Original Dense Top-10 + Translation Dense Top-10 + Original-query BM25 Top-10。",
        "- Add-only；dedup=`chunk_id`；ranking=none；不报告 Variant Hit@5。",
        "- 未调用 Generator/Context/Citation/LLM，未使用 Reranker，未运行参数 sweep。",
        "",
        "## Phase A — fixed 12 failures",
        "",
        f"- Recovered: {phase_a['recovered_count']}/12。",
        f"- IDs: {phase_a['recovered_ids']}。",
        f"- Gate ≥{phase_a['success_gate_minimum']}/12: {'PASS' if phase_a['passed'] else 'FAIL'}。",
    ]
    if payload["status"] != "formal_40q_complete":
        lines += ["", "Phase A 未通过，按冻结合同停止。", ""]
        return "\n".join(lines)
    coverage = payload["candidate_coverage"]
    cost = payload["candidate_cost"]
    question_types = payload["breakdowns"]["question_type"]
    evidence_types = payload["breakdowns"]["evidence_type"]
    languages = payload["breakdowns"]["language_group"]
    multi_document = payload["multi_document"]
    still_missed_ids = [
        row["question_id"]
        for row in payload["questions"]
        if row["transition"] == "still_missed"
    ]
    lines += [
        "",
        "## Phase B — Frozen 40Q Candidate Coverage",
        "",
        "| Metric | Baseline | Variant |",
        "|---|---:|---:|",
        f"| Complete Candidate Coverage | {coverage['baseline_covered']}/40 ({coverage['baseline_rate']:.3f}) | {coverage['variant_covered']}/40 ({coverage['variant_rate']:.3f}) |",
        f"| Regressed | 0 | {coverage['regressed_count']} |",
        "",
        f"- Net recovery: {coverage['net_recovery']}；IDs: {coverage['recovered_ids']}。",
        f"- Preserved / regressed / still missed: {coverage['preserved_count']} / {coverage['regressed_count']} / {coverage['still_missed_count']}。",
        f"- BM25-only Gold recovery: {payload['attribution']['bm25_only_gold_recovery_count']}；Dense+BM25 shared Gold questions: {payload['attribution']['dense_and_bm25_shared_gold_question_count']}。",
        "",
        "## Candidate cost",
        "",
        "| Pool | Min | Median | Mean | P95 | Max |",
        "|---|---:|---:|---:|---:|---:|",
        f"| Baseline | {cost['baseline_pool']['min']} | {cost['baseline_pool']['median']:.1f} | {cost['baseline_pool']['mean']:.2f} | {cost['baseline_pool']['p95']} | {cost['baseline_pool']['max']} |",
        f"| Augmented | {cost['augmented_pool']['min']} | {cost['augmented_pool']['median']:.1f} | {cost['augmented_pool']['mean']:.2f} | {cost['augmented_pool']['p95']} | {cost['augmented_pool']['max']} |",
        f"| Unique BM25 additions | {cost['unique_bm25_additions']['min']} | {cost['unique_bm25_additions']['median']:.1f} | {cost['unique_bm25_additions']['mean']:.2f} | {cost['unique_bm25_additions']['p95']} | {cost['unique_bm25_additions']['max']} |",
        "",
        f"- BM25 duplicate rate: {cost['bm25_duplicate_count']}/{cost['bm25_candidate_count']} ({cost['bm25_duplicate_rate']:.3f})。",
        "",
        "## Recovery breakdown",
        "",
        "| Slice | N | Baseline | Variant | Recovered |",
        "|---|---:|---:|---:|---:|",
        f"| Method | {question_types['method']['N']} | {question_types['method']['baseline_covered']} | {question_types['method']['variant_covered']} | {question_types['method']['recovered']} |",
        f"| Experiment setting | {question_types['experiment_setup']['N']} | {question_types['experiment_setup']['baseline_covered']} | {question_types['experiment_setup']['variant_covered']} | {question_types['experiment_setup']['recovered']} |",
        f"| Quantitative results / dataset-metric related | {question_types['quantitative_results']['N']} | {question_types['quantitative_results']['baseline_covered']} | {question_types['quantitative_results']['variant_covered']} | {question_types['quantitative_results']['recovered']} |",
        f"| Table evidence | {evidence_types['table']['N']} | {evidence_types['table']['baseline_covered']} | {evidence_types['table']['variant_covered']} | {evidence_types['table']['recovered']} |",
        f"| Ablation | {question_types['ablation']['N']} | {question_types['ablation']['baseline_covered']} | {question_types['ablation']['variant_covered']} | {question_types['ablation']['recovered']} |",
        f"| Multi-document | {question_types['multi_document']['N']} | {question_types['multi_document']['baseline_covered']} | {question_types['multi_document']['variant_covered']} | {question_types['multi_document']['recovered']} |",
        "",
        "| Language group | N | Baseline | Variant | Recovered |",
        "|---|---:|---:|---:|---:|",
        f"| English → English | {languages['english_to_english']['N']} | {languages['english_to_english']['baseline_covered']} | {languages['english_to_english']['variant_covered']} | {languages['english_to_english']['recovered']} |",
        f"| Chinese → English paper | {languages['chinese_to_english']['N']} | {languages['chinese_to_english']['baseline_covered']} | {languages['chinese_to_english']['variant_covered']} | {languages['chinese_to_english']['recovered']} |",
        f"| Mixed / multi-document | {languages['mixed_multiple_sources']['N']} | {languages['mixed_multiple_sources']['baseline_covered']} | {languages['mixed_multiple_sources']['variant_covered']} | {languages['mixed_multiple_sources']['recovered']} |",
        "",
        "## Multi-document",
        "",
        f"- Complete coverage: {multi_document['baseline_complete']}/4 → {multi_document['variant_complete']}/4。",
        f"- Partial coverage: {multi_document['baseline_partial']}/4 → {multi_document['variant_partial']}/4。",
        f"- Miss: {multi_document['question_count'] - multi_document['baseline_complete'] - multi_document['baseline_partial']}/4 → {multi_document['question_count'] - multi_document['variant_complete'] - multi_document['variant_partial']}/4。",
        "",
        "## Remaining misses",
        "",
        f"- Dense + BM25 still miss complete Gold sets: {coverage['still_missed_count']}/40。",
        f"- IDs: {still_missed_ids}。",
        "",
        "## Decision",
        "",
        f"- Generalization gate: **{payload['decision']['gate'].upper()}**。",
        f"- Benefit vs complexity: {payload['decision']['benefit_vs_complexity']}",
        f"- Next controlled experiment: **{payload['decision']['next_experiment']}**。",
        f"- Recommend Runtime integration now: {payload['decision']['recommend_runtime_integration']}。",
        "- Runtime 未修改；PROJECT_SPEC 保持 v0.14。",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    """运行两阶段实验并同步写出 JSON/Markdown。"""

    payload = run_experiment()
    OUTPUT_JSON.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    OUTPUT_MD.write_text(render_markdown(payload), encoding="utf-8")
    print(json.dumps({
        "phase_a": payload["phase_a"],
        "coverage": payload.get("candidate_coverage"),
        "decision": payload["decision"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
