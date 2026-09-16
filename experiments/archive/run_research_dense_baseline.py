"""已冻结的 Academic Research v1 Dense 基线历史 runner。"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "evals" / "research_retrieval" / "research_questions_v1.json"
TRANSLATIONS_PATH = ROOT / "evals" / "research_retrieval" / "research_translations_v1.json"
INGESTION_AUDIT_PATH = ROOT / "evals" / "research_ingestion" / "academic_ingestion_audit_v1.json"
OUTPUT_JSON = ROOT / "evals" / "research_retrieval" / "research_dense_baseline_v1.json"
OUTPUT_MD = ROOT / "evals" / "research_retrieval" / "research_dense_baseline_v1.md"
DEPTHS = (1, 3, 5, 10, 20)
EXPECTED_COLLECTION = "ef_research_bge_m3_c512_o64_v1"


def _normalize(text: str) -> str:
    """对证据与 Chunk 做同一确定性规范化，并兼容学术 PDF 断词。"""

    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"(?<=\w)-\s*\n\s*(?=\w)", "", text)
    return re.sub(r"\s+", " ", text).strip().casefold()


def _canonical_hash(payload: Any) -> str:
    """复算冻结 JSON 的稳定 SHA-256，禁止评测时数据漂移。"""

    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class FrozenTranslationProvider:
    """仅以检索前冻结的人工英文 Query 驱动当前多语 Dense 编排。"""

    def __init__(self, translations: Mapping[str, str]) -> None:
        self._translations = dict(translations)

    def translate_to_english(self, text: str):
        """精确返回冻结译文；缺失时明确失败而不调用外部模型。"""

        from evidenceflow.translation import TranslationResult

        translated = self._translations[text]
        return TranslationResult(
            original_text=text,
            translated_text=translated,
            was_translated=translated.strip() != text.strip(),
        )


def _load_and_verify() -> tuple[dict[str, Any], dict[str, str], dict[str, Any]]:
    """在模型初始化和首次检索前验证 Dataset、译文及摄取审计状态。"""

    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    claimed_hash = dataset.pop("dataset_hash")
    if _canonical_hash(dataset) != claimed_hash:
        raise ValueError("research dataset hash mismatch")
    dataset["dataset_hash"] = claimed_hash
    translations = json.loads(TRANSLATIONS_PATH.read_text(encoding="utf-8"))
    if translations.get("dataset_hash") != claimed_hash:
        raise ValueError("frozen translations target a different dataset")
    by_id = translations.get("translations")
    if set(by_id) != {question["question_id"] for question in dataset["questions"]}:
        raise ValueError("frozen translations do not cover the 40 questions exactly")
    audit = json.loads(INGESTION_AUDIT_PATH.read_text(encoding="utf-8"))
    if audit["corpus_hash"] != dataset["corpus_hash"]:
        raise ValueError("ingestion audit and retrieval dataset use different corpora")
    return dataset, by_id, audit


def _all_chunks(store: Any) -> list[Any]:
    """从研究 Collection 恢复全部 Domain Chunk，用于独立 Gold 映射和诊断。"""

    from evidenceflow.storage.chroma import _restore_chunk

    raw = store._collection.get(include=["documents", "metadatas"])
    return [
        _restore_chunk(chunk_id, document, metadata)
        for chunk_id, document, metadata in zip(
            raw["ids"], raw["documents"], raw["metadatas"]
        )
    ]


def _gold_groups(question: Mapping[str, Any], chunks: Sequence[Any]) -> list[list[str]]:
    """将页码与原文锚点解析为每个来源可互换的合法 Chunk 组。"""

    groups: list[list[str]] = []
    for source in question["gold_sources"]:
        anchor = _normalize(source["evidence_anchor"])
        groups.append(
            [
                chunk.chunk_id
                for chunk in chunks
                if chunk.document_id == source["document_id"]
                and chunk.page_start is not None
                and chunk.page_start <= source["page_end"]
                and chunk.page_end >= source["page_start"]
                and anchor in _normalize(chunk.content)
            ]
        )
    return groups


def _source_ndcg(results: Sequence[Any], groups: Sequence[Sequence[str]], depth: int) -> float:
    """复用统一核心，保留历史 CUSTOM nDCG 语义。"""
    from evidenceflow.eval.metrics import custom_evidence_group_ndcg
    return custom_evidence_group_ndcg([r.chunk.chunk_id for r in results], groups, depth)


def _metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """保留旧报告字段，通过共享核心聚合历史完整组覆盖指标。"""
    from evidenceflow.eval.metrics import legacy_research_metrics
    return legacy_research_metrics(rows, DEPTHS)


def _breakdowns(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """按语言、题型、证据类型和论文分别生成基线指标。"""

    dimensions = {
        "language_group": "language_group",
        "question_type": "question_type",
        "evidence_type": "evidence_type",
    }
    output = {"overall": _metrics(rows)}
    for dimension, field in dimensions.items():
        values = sorted({row[field] for row in rows})
        output[dimension] = {
            value: _metrics([row for row in rows if row[field] == value]) for value in values
        }
    paper_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        for file_name in row["gold_files"]:
            paper_rows[file_name].append(row)
    output["paper"] = {name: _metrics(selected) for name, selected in sorted(paper_rows.items())}
    return output


def run() -> dict[str, Any]:
    """执行一次冻结 Current Dense 基线，输出逐题证据链及自动失败归因。"""

    dataset, translations_by_id, audit = _load_and_verify()
    os.environ["HF_HOME"] = str((ROOT / ".cache" / "huggingface").resolve())
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    from evidenceflow.config import Settings
    from evidenceflow.embedding import BgeM3EmbeddingProvider
    from evidenceflow.eval.gold import gold_group_rank
    from evidenceflow.retrieval import DenseRetriever, MultilingualRetriever
    from evidenceflow.storage import ChromaVectorStore

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
    retriever = MultilingualRetriever(
        dense,
        FrozenTranslationProvider(translations_by_query),
        candidate_top_k=settings.multilingual_candidate_top_k,
        final_top_k=settings.multilingual_final_top_k,
        rrf_k=settings.multilingual_rrf_k,
        original_weight=settings.multilingual_original_weight,
        translation_weight=settings.multilingual_translation_weight,
    )
    audit_by_file = {paper["file_name"]: paper for paper in audit["papers"]}
    rows: list[dict[str, Any]] = []
    for index, question in enumerate(dataset["questions"], start=1):
        groups = _gold_groups(question, chunks)
        extraction_failure = False  # 构建阶段已逐页验证全部原文锚点。
        chunk_failure = any(not group for group in groups)
        results = retriever.retrieve(question["query"], top_k=20)
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

        academic_labels: set[str] = set()
        if question["evidence_type"] == "table" and primary_failure != "success":
            academic_labels.add("table_structure_loss")
        if question["evidence_type"] == "figure_caption" and primary_failure != "success":
            academic_labels.add("figure_semantics_missing")
        if question["evidence_type"] == "supplementary" and primary_failure != "success":
            academic_labels.add("supplement_boundary")
        if chunk_failure:
            academic_labels.add("section_boundary")
        reference_candidates = 0
        for result in results[:5]:
            paper = audit_by_file[result.chunk.file_name]
            if result.chunk.page_end >= paper["references_page"]:
                reference_candidates += 1
        if reference_candidates and primary_failure != "success":
            academic_labels.add("reference_pollution")

        row = {
            "question_id": question["question_id"],
            "query": question["query"],
            "english_query": question["english_query"],
            "language": question["language"],
            "language_group": question["language_group"],
            "question_type": question["question_type"],
            "evidence_type": question["evidence_type"],
            "gold_files": [source["file_name"] for source in question["gold_sources"]],
            "gold_chunk_group_sizes": [len(group) for group in groups],
            "gold_group_rank": rank,
            "primary_failure": primary_failure,
            "academic_labels": sorted(academic_labels),
            "reference_candidates_in_top5": reference_candidates,
            "candidates": [
                {
                    "rank": result.rank,
                    "chunk_id": result.chunk.chunk_id,
                    "file_name": result.chunk.file_name,
                    "page_start": result.chunk.page_start,
                    "page_end": result.chunk.page_end,
                    "original_rank": result.original_rank,
                    "translated_rank": result.translated_rank,
                    "fusion_score": result.fusion_score,
                }
                for result in results
            ],
        }
        for depth in DEPTHS:
            row[f"ndcg_at_{depth}"] = _source_ndcg(results, groups, depth)
        rows.append(row)
        print(f"[{index:02d}/40] {question['question_id']} rank={rank} {primary_failure}", flush=True)

    failure_counts = Counter(row["primary_failure"] for row in rows)
    for label in (
        "success",
        "extraction_failure",
        "chunk_failure",
        "candidate_retrieval_failure",
        "ranking_failure",
    ):
        failure_counts.setdefault(label, 0)
    academic_counts = Counter(label for row in rows for label in row["academic_labels"])
    for label in (
        "reference_pollution",
        "section_boundary",
        "table_structure_loss",
        "figure_semantics_missing",
        "formula_representation",
        "supplement_boundary",
    ):
        academic_counts.setdefault(label, 0)
    return {
        "evaluation_version": "research-dense-baseline-v1",
        "dataset_hash": dataset["dataset_hash"],
        "configuration": {
            "strategy": "current multilingual Dense: Original/Translation branch Top-10 + Equal RRF",
            "embedding_model": settings.embedding_model,
            "collection_name": settings.collection_name,
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
            "candidate_top_k_per_branch": settings.multilingual_candidate_top_k,
            "final_runtime_top_k": settings.multilingual_final_top_k,
            "evaluation_depth": 20,
            "rrf_k": settings.multilingual_rrf_k,
            "weights": [settings.multilingual_original_weight, settings.multilingual_translation_weight],
            "forbidden_experiments_run": [],
        },
        "breakdowns": _breakdowns(rows),
        "failure_counts": dict(failure_counts),
        "academic_failure_distribution": dict(sorted(academic_counts.items())),
        "retrieval_artifacts": {
            "questions_with_reference_chunks_in_top5": sum(
                row["reference_candidates_in_top5"] > 0 for row in rows
            ),
            "reference_chunks_in_all_top5_results": sum(
                row["reference_candidates_in_top5"] for row in rows
            ),
        },
        "question_results": rows,
    }


def _metric_line(name: str, metric: Mapping[str, Any]) -> str:
    """把一组核心指标渲染为 Markdown 表格行。"""

    return (
        f"| {name} | {metric['N']} | {metric['hit_at_1']:.3f} | {metric['hit_at_3']:.3f} | "
        f"{metric['hit_at_5']:.3f} | {metric['hit_at_10']:.3f} | {metric['hit_at_20']:.3f} | "
        f"{metric['mrr']:.3f} | {metric['ndcg_at_20']:.3f} |"
    )


def render_markdown(payload: Mapping[str, Any]) -> str:
    """生成面向阶段决策的 Dense 基线与失败诊断报告。"""

    header = "| Group | N | Hit@1 | Hit@3 | Hit@5 | Hit@10 | Hit@20 | MRR | nDCG@20 |"
    divider = "|---|---:|---:|---:|---:|---:|---:|---:|---:|"
    lines = [
        "# Academic Research Current Dense Baseline v1",
        "",
        f"- Dataset hash: `{payload['dataset_hash']}`",
        "- Strategy: current Original/Translation Dense Top-10 per branch + Equal RRF；evaluation union depth 20。",
        "- 未运行 BM25、Hybrid、Reranker、Query Rewrite 或参数搜索。",
        "",
        "## Overall",
        "",
        header,
        divider,
        _metric_line("overall", payload["breakdowns"]["overall"]),
    ]
    for dimension in ("language_group", "question_type", "evidence_type", "paper"):
        lines += ["", f"## {dimension}", "", header, divider]
        lines.extend(
            _metric_line(name, metric)
            for name, metric in payload["breakdowns"][dimension].items()
        )
    lines += ["", "## Failure diagnosis", ""]
    lines.extend(f"- {name}: {count}" for name, count in sorted(payload["failure_counts"].items()))
    lines += ["", "## Academic-specific diagnostic labels", ""]
    lines.extend(
        f"- {name}: {count}" for name, count in payload["academic_failure_distribution"].items()
    )
    artifacts = payload["retrieval_artifacts"]
    lines += [
        "",
        "## Retrieval artifacts",
        "",
        f"- Top-5 中出现 References Chunk 的问题：{artifacts['questions_with_reference_chunks_in_top5']}/40。",
        f"- 40×Top-5 中的 References Chunk 总数：{artifacts['reference_chunks_in_all_top5_results']}。",
    ]
    lines += [
        "",
        "诊断标签是相关性线索，不把版面问题自动宣称为唯一因果；逐题候选、页码和分支排名保存在 JSON。",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    """运行基线并将完整 JSON 与精简 Markdown 同步落盘。"""

    payload = run()
    OUTPUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    OUTPUT_MD.write_text(render_markdown(payload), encoding="utf-8")
    print(json.dumps({"overall": payload["breakdowns"]["overall"], "failure_counts": payload["failure_counts"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
