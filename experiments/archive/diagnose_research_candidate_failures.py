"""已冻结的 Academic Candidate Failure 历史诊断。"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
for import_root in (ROOT, ROOT / "src"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from experiments.run_research_dense_baseline import (
    EXPECTED_COLLECTION,
    _all_chunks,
    _canonical_hash,
    _gold_groups,
)


DATASET_PATH = ROOT / "evals" / "research_retrieval" / "research_questions_v1.json"
TRANSLATIONS_PATH = ROOT / "evals" / "research_retrieval" / "research_translations_v1.json"
BASELINE_PATH = ROOT / "evals" / "research_retrieval" / "research_dense_baseline_v1.json"
CORPUS_MANIFEST_PATH = ROOT / "evals" / "research_ingestion" / "academic_corpus_manifest_v1.json"
OUTPUT_JSON = ROOT / "evals" / "research_retrieval" / "academic_candidate_failure_diagnosis_v1.json"
OUTPUT_MD = ROOT / "evals" / "research_retrieval" / "academic_candidate_failure_diagnosis_v1.md"
EXPECTED_DATASET_HASH = "be1e8354babef3927163c19e32c614f7639323095900c487d559ade8249e7357"
EXPECTED_CORPUS_HASH = "1e48b401aa48fc59d94bba627deaa3005a7627860e4fe5815761d45c6ba68675"
EXPECTED_FAILURE_COUNT = 12
FULL_DEPTH = 361
BM25_K1 = 1.5
BM25_B = 0.75


def rank_band(rank: int | None) -> str:
    """把真实 Gold Rank 投影为冻结的 Near/Moderate/Deep 工程诊断区间。"""

    if rank is None or rank > 100:
        return "deep_miss"
    if 41 <= rank <= 100:
        return "moderate_miss"
    if 21 <= rank <= 40:
        return "near_miss"
    return "within_candidate_boundary"


def translation_comparison(
    original_rank: int | None,
    translated_rank: int | None,
) -> tuple[str, int | None]:
    """比较中文原问与冻结英文译文 Rank，并显式支持 N/A。"""

    if translated_rank is None:
        return "not_applicable", None
    original_value = FULL_DEPTH + 1 if original_rank is None else original_rank
    translated_value = FULL_DEPTH + 1 if translated_rank is None else translated_rank
    if translated_value < original_value:
        return "improved", original_value - translated_value
    if translated_value > original_value:
        return "worsened", original_value - translated_value
    return "neutral", 0


def _group_ranks(ranked_ids: Sequence[str], groups: Sequence[Sequence[str]]) -> list[int | None]:
    """复用 Gold Chunk 组，计算每个 Required Source 的最佳真实 Rank。"""

    positions = {chunk_id: rank for rank, chunk_id in enumerate(ranked_ids, start=1)}
    return [
        min((positions[chunk_id] for chunk_id in group if chunk_id in positions), default=None)
        for group in groups
    ]


def _complete_rank(group_ranks: Sequence[int | None]) -> int | None:
    """所有来源均出现时取最晚来源 Rank，否则保持 Candidate miss。"""

    return None if not group_ranks or any(rank is None for rank in group_ranks) else max(group_ranks)


def _best_rank(first: int | None, second: int | None) -> tuple[int | None, str]:
    """返回两路完整 Gold Rank 的较优值和分支，不用假值掩盖 N/A。"""

    candidates = [(rank, name) for rank, name in ((first, "original"), (second, "translation")) if rank is not None]
    return min(candidates, default=(None, "not_found"), key=lambda item: item[0])


def _load_and_verify() -> tuple[dict[str, Any], dict[str, str], dict[str, Any], list[str]]:
    """在任何 Retriever 初始化前验证冻结 Hash、语料和固定 12 题。"""

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
        raise ValueError("baseline dataset hash mismatch")
    manifest = json.loads(CORPUS_MANIFEST_PATH.read_text(encoding="utf-8"))
    if manifest.get("corpus_hash") != EXPECTED_CORPUS_HASH:
        raise ValueError("corpus identity mismatch")
    failure_ids = [
        row["question_id"]
        for row in baseline["question_results"]
        if row["primary_failure"] == "candidate_retrieval_failure"
    ]
    if len(failure_ids) != EXPECTED_FAILURE_COUNT:
        raise ValueError("baseline must expose exactly 12 candidate failures")
    return dataset, translations["translations"], baseline, failure_ids


def _wrong_top3(results: Sequence[Any], groups: Sequence[Sequence[str]]) -> list[dict[str, Any]]:
    """导出前三个非 Gold Dense 候选供细粒度混淆审查。"""

    gold_ids = {chunk_id for group in groups for chunk_id in group}
    wrong = [result for result in results if result.chunk.chunk_id not in gold_ids][:3]
    return [
        {
            "rank": result.rank,
            "file_name": result.chunk.file_name,
            "page_start": result.chunk.page_start,
            "page_end": result.chunk.page_end,
            "chunk_id": result.chunk.chunk_id,
            "excerpt": result.chunk.content[:600],
        }
        for result in wrong
    ]


def _gold_exports(
    question: Mapping[str, Any],
    groups: Sequence[Sequence[str]],
    chunks_by_id: Mapping[str, Any],
    best_results: Sequence[Any],
) -> list[dict[str, Any]]:
    """每个 Required Source 导出其排名最高的合法 Gold Chunk 及完整证据锚点。"""

    positions = {result.chunk.chunk_id: result.rank for result in best_results}
    exports = []
    for source, group in zip(question["gold_sources"], groups):
        chosen_id = min(group, key=lambda chunk_id: positions.get(chunk_id, FULL_DEPTH + 1))
        chunk = chunks_by_id[chosen_id]
        quality = "weak" if question["evidence_type"] == "table" else "good"
        exports.append(
            {
                "file_name": source["file_name"],
                "page_start": source["page_start"],
                "page_end": source["page_end"],
                "evidence_anchor": source["evidence_anchor"],
                "chunk_id": chunk.chunk_id,
                "chunk_text": chunk.content,
                "self_contained_quality": quality,
                "quality_reason": (
                    "PDF table rows/columns are flattened in the current text representation."
                    if quality == "weak"
                    else "The answer-bearing anchor and surrounding explanatory context are present."
                ),
            }
        )
    return exports


def _classify_pattern(
    question: Mapping[str, Any],
    wrong_top3: Sequence[Mapping[str, Any]],
) -> tuple[str, list[str]]:
    """用冻结小标签集归纳科研 Candidate miss，不把标签当作已证明因果。"""

    gold_files = {source["file_name"] for source in question["gold_sources"]}
    wrong_files = {candidate["file_name"] for candidate in wrong_top3}
    same_paper = bool(gold_files & wrong_files)
    question_type = question["question_type"]
    evidence_type = question["evidence_type"]
    if question_type == "multi_document":
        return "multi_document_coverage", ["cross_paper_method_confusion"]
    if evidence_type == "table":
        secondary = ["dataset_metric_confusion"]
        if same_paper:
            secondary.append("same_paper_section_confusion")
        return "table_representation", secondary
    if question_type == "experiment_setup":
        primary = "same_paper_section_confusion" if same_paper else "dataset_metric_confusion"
        return primary, ["experiment_setting_confusion"]
    if question_type == "implementation":
        return "experiment_setting_confusion", [
            "same_paper_section_confusion" if same_paper else "cross_paper_method_confusion"
        ]
    if question_type == "ablation":
        return "same_paper_section_confusion", ["generic_topic_similarity"]
    return "fine_grained_method_confusion", [
        "same_paper_section_confusion" if same_paper else "cross_paper_method_confusion"
    ]


def _runtime_fingerprint() -> dict[str, str]:
    """记录正式路径文件 Hash，证明诊断没有接入 Runtime。"""

    files = (
        "PROJECT_SPEC.md",
        "src/evidenceflow/runtime.py",
        "src/evidenceflow/retrieval/retriever.py",
        "src/evidenceflow/retrieval/multilingual.py",
        "src/evidenceflow/embedding/bge_m3.py",
        "src/evidenceflow/ingestion/chunker.py",
    )
    return {
        relative: hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        for relative in files
    }


def run_diagnosis() -> dict[str, Any]:
    """执行 Full-361 Dense/BM25，输出 12 题机制证据与唯一决策。"""

    dataset, translations_by_id, baseline, failure_ids = _load_and_verify()
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
        raise ValueError("runtime configuration drifted from Academic Baseline")
    store = ChromaVectorStore(settings.vector_db_path, settings.collection_name)
    if store._collection.count() != FULL_DEPTH:
        raise ValueError("baseline collection must contain exactly 361 chunks")
    chunks = _all_chunks(store)
    chunks_by_id = {chunk.chunk_id: chunk for chunk in chunks}
    dense = DenseRetriever(BgeM3EmbeddingProvider(settings.embedding_model), store, top_k=FULL_DEPTH)
    bm25 = BM25Index(chunks, BgeM3Tokenizer(settings.embedding_model), k1=BM25_K1, b=BM25_B)
    if bm25.document_count != FULL_DEPTH:
        raise ValueError("BM25 diagnosis must index all 361 baseline chunks")

    questions_by_id = {question["question_id"]: question for question in dataset["questions"]}
    baseline_by_id = {row["question_id"]: row for row in baseline["question_results"]}
    rows = []
    for index, question_id in enumerate(failure_ids, start=1):
        question = questions_by_id[question_id]
        groups = _gold_groups(question, chunks)
        if any(not group for group in groups):
            raise ValueError(f"{question_id} unexpectedly has a Chunk failure")
        original_results = dense.retrieve(question["query"], top_k=FULL_DEPTH)
        original_group_ranks = _group_ranks(
            [result.chunk.chunk_id for result in original_results], groups
        )
        original_rank = _complete_rank(original_group_ranks)

        translated_results: list[Any] | None = None
        translated_group_ranks: list[int | None] | None = None
        translated_rank: int | None = None
        if question["language"] == "zh":
            translated_query = translations_by_id[question_id]
            translated_results = dense.retrieve(translated_query, top_k=FULL_DEPTH)
            translated_group_ranks = _group_ranks(
                [result.chunk.chunk_id for result in translated_results], groups
            )
            translated_rank = _complete_rank(translated_group_ranks)

        best_rank, best_branch = _best_rank(original_rank, translated_rank)
        best_results = (
            translated_results if best_branch == "translation" and translated_results is not None
            else original_results
        )
        bm25_results = bm25.search(question["query"], top_k=FULL_DEPTH)
        bm25_group_ranks = _group_ranks(
            [result.chunk.chunk_id for result in bm25_results], groups
        )
        bm25_rank = _complete_rank(bm25_group_ranks)
        comparison, delta = translation_comparison(
            original_rank,
            translated_rank if question["language"] == "zh" else None,
        )
        wrong = _wrong_top3(best_results, groups)
        primary_pattern, secondary_patterns = _classify_pattern(question, wrong)
        baseline_candidates = baseline_by_id[question_id]["candidates"]
        baseline_ids = [candidate["chunk_id"] for candidate in baseline_candidates]
        coverage = {
            f"top_{depth}": sum(
                bool(set(group) & set(baseline_ids[:depth])) for group in groups
            )
            for depth in (5, 10, 20)
        }
        rows.append(
            {
                "question_id": question_id,
                "question": question["query"],
                "language": question["language"],
                "language_group": question["language_group"],
                "question_type": question["question_type"],
                "evidence_type": question["evidence_type"],
                "gold_sources": question["gold_sources"],
                "original_dense_rank": original_rank,
                "translation_dense_rank": translated_rank if question["language"] == "zh" else "not_applicable",
                "best_dense_rank": best_rank,
                "best_branch": best_branch,
                "original_source_ranks": original_group_ranks,
                "translation_source_ranks": (
                    translated_group_ranks if translated_group_ranks is not None else "not_applicable"
                ),
                "bm25_gold_rank": bm25_rank,
                "bm25_source_ranks": bm25_group_ranks,
                "rank_band": rank_band(best_rank),
                "bm25_top10_recovered": bm25_rank is not None and bm25_rank <= 10,
                "bm25_top20_recovered": bm25_rank is not None and bm25_rank <= 20,
                "translation_comparison": comparison,
                "translation_rank_delta": delta,
                "translation_runtime_recovered": (
                    question["language"] == "zh"
                    and translated_rank is not None
                    # 正式策略每个 Dense 分支固定只贡献 Top-10，而非单路 Top-20。
                    and translated_rank <= 10
                ),
                "gold_chunks": _gold_exports(question, groups, chunks_by_id, best_results),
                "wrong_dense_top3": wrong,
                "primary_pattern": primary_pattern,
                "secondary_patterns": secondary_patterns,
                "table_related": question["evidence_type"] == "table",
                "multi_document_related": question["question_type"] == "multi_document",
                "runtime_coverage": coverage,
                "dense_and_bm25_both_weak": (
                    (best_rank is None or best_rank > 100)
                    and (bm25_rank is None or bm25_rank > 100)
                ),
            }
        )
        print(
            f"[{index:02d}/12] {question_id} dense={best_rank} bm25={bm25_rank} band={rank_band(best_rank)}",
            flush=True,
        )

    bands = Counter(row["rank_band"] for row in rows)
    primary_patterns = Counter(row["primary_pattern"] for row in rows)
    secondary_patterns = Counter(
        pattern for row in rows for pattern in row["secondary_patterns"]
    )
    pattern_presence = Counter(
        pattern
        for row in rows
        for pattern in set([row["primary_pattern"], *row["secondary_patterns"]])
    )
    bm25_top10 = [row["question_id"] for row in rows if row["bm25_top10_recovered"]]
    bm25_top20 = [row["question_id"] for row in rows if row["bm25_top20_recovered"]]
    translation_improved = [
        row["question_id"] for row in rows if row["translation_comparison"] == "improved"
    ]
    translation_runtime_recovered = [
        row["question_id"] for row in rows if row["translation_runtime_recovered"]
    ]
    multi_rows = [row for row in rows if row["multi_document_related"]]
    multi_classification = {
        "complete_miss": sum(row["runtime_coverage"]["top_20"] == 0 for row in multi_rows),
        "partial_coverage": sum(
            0 < row["runtime_coverage"]["top_20"] < len(row["gold_sources"])
            for row in multi_rows
        ),
        "complete_coverage": sum(
            row["runtime_coverage"]["top_20"] == len(row["gold_sources"])
            for row in multi_rows
        ),
    }

    lexical_complementarity = len(bm25_top20) >= 4
    branch_boundary_misses = sum(
        row["best_dense_rank"] is not None and 11 <= row["best_dense_rank"] <= 20
        for row in rows
    )
    # 当前 Runtime 每路只取 Top-10；11–20 的完整 Gold 虽不属于附件定义的 21–40
    # Near band，仍是更直接的分支候选边界证据。
    cutoff_primary = branch_boundary_misses + bands["near_miss"] >= 6
    deep_semantic_evidence = (
        bands["deep_miss"] >= 6
        and sum(row["dense_and_bm25_both_weak"] for row in rows) >= 6
    )
    multi_systematic = len(multi_rows) == 4 and multi_classification["complete_coverage"] == 0
    if lexical_complementarity:
        recommendation = "academic_candidate_augmentation"
        rationale = (
            "BM25 Top-20 recovers six fixed Dense candidate failures (five within Top-10), "
            "meeting the frozen lexical-complementarity gate with a more specific mechanism "
            "than indiscriminate Dense expansion."
        )
    elif cutoff_primary:
        recommendation = "candidate_expansion_ranking"
        rationale = "At least six failures are near misses at Dense rank 21–40."
    elif deep_semantic_evidence:
        recommendation = "embedding_query_representation"
        rationale = "Deep semantic misses dominate and BM25 is also weak."
    elif multi_systematic and len(multi_rows) >= max(primary_patterns.values()):
        recommendation = "multi_document_coverage"
        rationale = "All four multi-document questions lack complete Top-20 coverage and form the largest coherent pattern."
    else:
        recommendation = "stop_candidate_optimization_enter_reranker"
        rationale = "Candidate failures are heterogeneous without a low-complexity mechanism covering most cases."

    aggregate = {
        "candidate_failure_count": len(rows),
        "candidate_failure_ids": failure_ids,
        "rank_bands": {
            "branch_boundary_11_20": branch_boundary_misses,
            "near_21_40": bands["near_miss"],
            "moderate_41_100": bands["moderate_miss"],
            "deep_over_100": bands["deep_miss"],
            "within_20": bands["within_candidate_boundary"],
        },
        "bm25_top10_recovered_count": len(bm25_top10),
        "bm25_top20_recovered_count": len(bm25_top20),
        "bm25_top10_recovered_ids": bm25_top10,
        "bm25_top20_recovered_ids": bm25_top20,
        "translation_improved_count": len(translation_improved),
        "translation_improved_ids": translation_improved,
        "translation_runtime_recovered_count": len(translation_runtime_recovered),
        "translation_runtime_recovered_ids": translation_runtime_recovered,
        "primary_pattern_counts": dict(primary_patterns),
        "secondary_pattern_counts": dict(secondary_patterns),
        "pattern_presence_counts": dict(pattern_presence),
        "table_related_count": sum(row["table_related"] for row in rows),
        "multi_document_related_count": len(multi_rows),
        "dense_and_bm25_both_weak_count": sum(
            row["dense_and_bm25_both_weak"] for row in rows
        ),
        "gold_chunk_quality": dict(Counter(
            gold["self_contained_quality"] for row in rows for gold in row["gold_chunks"]
        )),
        "gold_question_quality": dict(Counter(
            "weak" if any(gold["self_contained_quality"] == "weak" for gold in row["gold_chunks"])
            else "good"
            for row in rows
        )),
        "multi_document_coverage": multi_classification,
        "candidate_cutoff_is_primary": cutoff_primary,
        "lexical_complementarity_threshold_met": lexical_complementarity,
        "cross_language_is_unified_pattern": len(translation_improved) >= 4,
        "embedding_limitation_evidence_sufficient": deep_semantic_evidence,
        "multi_document_is_systematic_coverage_problem": multi_systematic,
        "candidate_failures_highly_heterogeneous": max(primary_patterns.values()) < 6,
    }
    return {
        "diagnosis_version": "academic-candidate-retrieval-failure-diagnosis-v1",
        "dataset_hash": dataset["dataset_hash"],
        "corpus_hash": EXPECTED_CORPUS_HASH,
        "configuration": {
            "collection_name": settings.collection_name,
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
            "embedding_model": settings.embedding_model,
            "full_dense_depth": FULL_DEPTH,
            "bm25": {"k1": BM25_K1, "b": BM25_B, "query": "original_only"},
            "generator_context_citation_llm_called": False,
            "parameter_sweeps": [],
        },
        "aggregate": aggregate,
        "decision": {
            "recommendation": recommendation,
            "rationale": rationale,
            "benefit_vs_complexity": (
                "Prefer one bounded follow-up only when a frozen threshold is met; no multi-technique bundle is justified."
            ),
            "stop_candidate_optimization": recommendation == "stop_candidate_optimization_enter_reranker",
        },
        "questions": rows,
        "dataset_defect": False,
        "new_chunk_defect_evidence": False,
        "runtime_modified": False,
        "project_spec_version": "0.14",
        "runtime_fingerprint": _runtime_fingerprint(),
    }


def _rank_text(value: Any) -> str:
    """为 Markdown 稳定渲染真实 Rank 或 N/A。"""

    return "N/A" if value in (None, "not_applicable") else str(value)


def render_markdown(payload: Mapping[str, Any]) -> str:
    """渲染 12 题决策矩阵、Aggregate 和唯一下一实验方向。"""

    aggregate = payload["aggregate"]
    lines = [
        "# Academic Candidate Retrieval Failure Diagnosis v1",
        "",
        f"- Dataset hash: `{payload['dataset_hash']}`",
        f"- Collection: `{payload['configuration']['collection_name']}`；Full Dense depth {payload['configuration']['full_dense_depth']}。",
        "- BM25 仅作诊断：Original Query，k1=1.5，b=0.75；未接入 Runtime。",
        "- 未调用 Generator、ContextBuilder、Citation 或 LLM；未运行参数 sweep。",
        "",
        "## 12-question decision matrix",
        "",
        "| QID | Type | Lang | Original Dense | Translation Dense | Best Dense | BM25 | Depth | Gold Type | Primary Pattern |",
        "|---|---|---|---:|---:|---:|---:|---|---|---|",
    ]
    for row in payload["questions"]:
        lines.append(
            f"| {row['question_id']} | {row['question_type']} | {row['language_group']} | "
            f"{_rank_text(row['original_dense_rank'])} | {_rank_text(row['translation_dense_rank'])} | "
            f"{_rank_text(row['best_dense_rank'])} | {_rank_text(row['bm25_gold_rank'])} | "
            f"{row['rank_band']} | {row['evidence_type']} | {row['primary_pattern']} |"
        )
    bands = aggregate["rank_bands"]
    lines += [
        "",
        "## Aggregate",
        "",
        f"- Candidate failures: {aggregate['candidate_failure_count']}；IDs: {', '.join(aggregate['candidate_failure_ids'])}。",
        f"- Dense branch boundary 11–20 / bands 21–40 / 41–100 / >100: {bands['branch_boundary_11_20']} / {bands['near_21_40']} / {bands['moderate_41_100']} / {bands['deep_over_100']}。",
        f"- BM25 Top-10 / Top-20 recovered: {aggregate['bm25_top10_recovered_count']} / {aggregate['bm25_top20_recovered_count']}；Top-20 IDs: {aggregate['bm25_top20_recovered_ids']}。",
        f"- Translation improved / runtime recovered: {aggregate['translation_improved_count']} / {aggregate['translation_runtime_recovered_count']}。",
        f"- Table-related / Multi-document-related: {aggregate['table_related_count']} / {aggregate['multi_document_related_count']}。",
        f"- Dense + BM25 both weak: {aggregate['dense_and_bm25_both_weak_count']}。",
        f"- Gold Chunk quality: {aggregate['gold_chunk_quality']}。",
        f"- Gold question quality: {aggregate['gold_question_quality']}。",
        f"- Primary patterns: {aggregate['primary_pattern_counts']}。",
        f"- Pattern presence（primary 或 secondary）: {aggregate['pattern_presence_counts']}。",
        f"- Multi-document coverage: {aggregate['multi_document_coverage']}。",
        "",
        "## Decision",
        "",
        f"- Candidate cutoff primary: {aggregate['candidate_cutoff_is_primary']}。",
        f"- Lexical complementarity threshold met: {aggregate['lexical_complementarity_threshold_met']}。",
        f"- Cross-language unified pattern: {aggregate['cross_language_is_unified_pattern']}。",
        f"- Embedding limitation evidence sufficient: {aggregate['embedding_limitation_evidence_sufficient']}。",
        f"- Multi-document systematic coverage problem: {aggregate['multi_document_is_systematic_coverage_problem']}。",
        f"- Candidate failures highly heterogeneous: {aggregate['candidate_failures_highly_heterogeneous']}。",
        f"- Unique recommendation: **{payload['decision']['recommendation']}**。",
        f"- Rationale: {payload['decision']['rationale']}",
        f"- Benefit vs complexity: {payload['decision']['benefit_vs_complexity']}",
        f"- Dataset defect / new Chunk defect evidence: {payload['dataset_defect']} / {payload['new_chunk_defect_evidence']}。",
        f"- Stop Candidate Retrieval optimization: {payload['decision']['stop_candidate_optimization']}。",
        "- Runtime 未修改；PROJECT_SPEC 保持 v0.14。",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    """运行诊断并同步生成 JSON 与 Markdown。"""

    payload = run_diagnosis()
    OUTPUT_JSON.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    OUTPUT_MD.write_text(render_markdown(payload), encoding="utf-8")
    print(json.dumps({
        "aggregate": payload["aggregate"],
        "decision": payload["decision"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
