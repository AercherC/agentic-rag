"""审计 Academic Research v1 在当前 PDF/C512/O64 Pipeline 下的兼容性。"""

from __future__ import annotations

import json
import os
import re
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any


# 支持从项目根或其他工作目录启动已移动的审计脚本。
PROJECT_ROOT = Path(__file__).resolve().parents[2]
for import_root in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from evidenceflow.config import Settings
from evidenceflow.ingestion import BgeM3Tokenizer, chunk_document, clean_document, load_document
from evidenceflow.storage import ChromaVectorStore
from experiments.prepare_research_corpus import _canonical_hash


SOURCE_DOCS = PROJECT_ROOT / "data" / "source_docs"
MANIFEST_PATH = (
    PROJECT_ROOT / "evals" / "research_ingestion" / "academic_corpus_manifest_v1.json"
)
OUTPUT_JSON = (
    PROJECT_ROOT / "evals" / "research_ingestion" / "academic_ingestion_audit_v1.json"
)
OUTPUT_MD = OUTPUT_JSON.with_suffix(".md")
EXPECTED_MANIFEST_HASH = (
    "efd92f9de8ac30e6a1081d1641076a9166b18f671238b5d44f89f6727423bb16"
)
EXPECTED_COLLECTION = "ef_research_bge_m3_c512_o64_v1"
EXPECTED_CHUNK_COUNT = 361


# 固定 49 个代表页已通过 Poppler 渲染并人工检查，覆盖首页、方法、实验、引用和附录。
REPRESENTATIVE_PAGES: dict[str, list[int]] = {
    "adapointr.pdf": [1, 5, 9, 15, 18],
    "hyper_pcn.pdf": [1, 3, 6, 9, 12],
    "pcn_point_completion_network.pdf": [1, 4, 7, 10, 12],
    "pfnet_point_fractal_network.pdf": [1, 3, 6, 9],
    "pointattn.pdf": [1, 4, 8, 10, 11],
    "pointr_geometry_aware_transformers.pdf": [1, 4, 7, 11, 12],
    "proxyformer.pdf": [1, 3, 6, 9, 10],
    "seedformer.pdf": [1, 5, 10, 15, 18],
    "snowflakenet.pdf": [1, 4, 7, 9, 11],
    "svdformer.pdf": [1, 4, 7, 9, 11],
}

# PDF 中实际进入附录/补充材料的第一页；当前 Domain Model 不保存此边界。
SUPPLEMENT_BOUNDARY_PAGES: dict[str, int] = {
    "adapointr.pdf": 16,
    "hyper_pcn.pdf": 11,
    "pcn_point_completion_network.pdf": 12,
    "pointr_geometry_aware_transformers.pdf": 12,
    "seedformer.pdf": 18,
}

REPEATED_HEADER_FOOTER_PAPERS = {"pointattn.pdf", "seedformer.pdf"}


def _load_and_verify_manifest() -> dict[str, Any]:
    """验证 Manifest 自身 Hash、Corpus 数量和冻结 Collection 配置。"""

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    stored_hash = manifest.pop("manifest_hash")
    calculated_hash = _canonical_hash(manifest)
    manifest["manifest_hash"] = stored_hash
    if stored_hash != EXPECTED_MANIFEST_HASH or calculated_hash != stored_hash:
        raise ValueError("academic corpus manifest hash mismatch")
    if manifest["corpus_count"] != 10:
        raise ValueError("academic corpus must contain exactly 10 papers")
    if manifest["collection_name"] != EXPECTED_COLLECTION:
        raise ValueError("research collection name drifted")
    return manifest


def _find_heading_page(segments: list[Any], heading: str) -> int | None:
    """定位独立章节标题所在页，避免正文提及造成误判。"""

    pattern = re.compile(rf"(?im)^\s*{re.escape(heading)}\s*$")
    return next(
        (segment.page_number for segment in segments if pattern.search(segment.content)),
        None,
    )


def _chunk_crosses_page_boundary(chunk: Any, boundary_page: int | None) -> bool:
    """判断滑窗 Chunk 是否把边界前内容和边界页混在一起。"""

    return bool(
        boundary_page is not None
        and chunk.page_start is not None
        and chunk.page_end is not None
        and chunk.page_start < boundary_page <= chunk.page_end
    )


def _paper_audit(
    paper: dict[str, Any],
    tokenizer: BgeM3Tokenizer,
) -> dict[str, Any]:
    """使用正式 Loader/Cleaner/Chunker 统计一篇论文，并附上视觉审计结论。"""

    path = SOURCE_DOCS / paper["file_name"]
    document = load_document(path, SOURCE_DOCS)
    cleaned = clean_document(document)
    chunks = chunk_document(cleaned, tokenizer, 512, 64)
    full_text = "\n".join(segment.content for segment in cleaned.segments)
    hyphen_break_count = len(re.findall(r"[A-Za-z]{2,}-\n[a-z]{2,}", full_text))
    references_page = _find_heading_page(cleaned.segments, "References")
    supplementary_page = SUPPLEMENT_BOUNDARY_PAGES.get(paper["file_name"])
    reference_crossings = sum(
        _chunk_crosses_page_boundary(chunk, references_page) for chunk in chunks
    )
    supplementary_crossings = sum(
        _chunk_crosses_page_boundary(chunk, supplementary_page) for chunk in chunks
    )
    token_counts = [chunk.token_count for chunk in chunks]
    page_spans = [chunk.page_end - chunk.page_start + 1 for chunk in chunks]
    return {
        "file_name": paper["file_name"],
        "document_id": document.document_id,
        "page_count": document.metadata["page_count"],
        "extracted_character_count": len(full_text),
        "chunk_count": len(chunks),
        "empty_page_count": paper["empty_page_count"],
        "very_short_chunk_count": sum(count < 64 for count in token_counts),
        "full_512_token_chunk_count": sum(count == 512 for count in token_counts),
        "max_pages_per_chunk": max(page_spans),
        "replacement_or_abnormal_character_count": full_text.count("\ufffd")
        + full_text.count("<unk>"),
        "broken_hyphenation_count": hyphen_break_count,
        "references_page": references_page,
        "supplementary_boundary_page": supplementary_page,
        "reference_boundary_crossing_chunk_count": reference_crossings,
        "supplement_boundary_crossing_chunk_count": supplementary_crossings,
        "representative_pages_reviewed": REPRESENTATIVE_PAGES[paper["file_name"]],
        "visual_audit": {
            "title_ok": True,
            "abstract_ok": True,
            "double_column_reading_order_issue": False,
            "section_titles_preserved": True,
            "paragraph_severe_merge": False,
            "broken_hyphenation_issue": hyphen_break_count > 0,
            "formula_fragmentation_issue": True,
            "table_structure_loss_issue": True,
            "figure_caption_preserved": True,
            "figure_internal_text_noise_risk": True,
            "header_footer_repetition_issue": paper["file_name"]
            in REPEATED_HEADER_FOOTER_PAPERS,
            "references_pollution_risk": references_page is not None,
            "supplement_boundary_lost": supplementary_page is not None,
            "page_range_valid": all(
                chunk.page_start is not None
                and chunk.page_end is not None
                and 1 <= chunk.page_start <= chunk.page_end <= paper["page_count"]
                for chunk in chunks
            ),
            "chunk_boundary_issue": reference_crossings > 0
            or supplementary_crossings > 0,
        },
    }


def _issue_matrix(papers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """把逐论文布尔审计压缩为可决策的 Issue Matrix。"""

    definitions = [
        ("extraction_failure", lambda item: item.get("extraction_failed", False), "High"),
        ("double_column_order", lambda item: item["visual_audit"]["double_column_reading_order_issue"], "High"),
        ("broken_hyphenation", lambda item: item["visual_audit"]["broken_hyphenation_issue"], "Medium"),
        ("section_detection_missing", lambda item: True, "Medium"),
        ("header_footer_repetition", lambda item: item["visual_audit"]["header_footer_repetition_issue"], "Low"),
        ("table_structure_loss", lambda item: item["visual_audit"]["table_structure_loss_issue"], "High"),
        ("figure_text_noise", lambda item: item["visual_audit"]["figure_internal_text_noise_risk"], "Low"),
        ("formula_fragmentation", lambda item: item["visual_audit"]["formula_fragmentation_issue"], "Medium"),
        ("references_pollution", lambda item: item["visual_audit"]["references_pollution_risk"], "Medium"),
        ("supplement_boundary_lost", lambda item: item["visual_audit"]["supplement_boundary_lost"], "Medium"),
        ("chunk_boundary_crossing", lambda item: item["visual_audit"]["chunk_boundary_issue"], "Medium"),
    ]
    matrix = []
    for issue, predicate, severity in definitions:
        affected = [item["file_name"] for item in papers if predicate(item)]
        matrix.append(
            {
                "issue": issue,
                "papers_affected": len(affected),
                "denominator": (
                    len(SUPPLEMENT_BOUNDARY_PAGES)
                    if issue == "supplement_boundary_lost"
                    else len(papers)
                ),
                "affected_files": affected,
                "severity": "None" if not affected else severity,
            }
        )
    return matrix


def build_audit() -> dict[str, Any]:
    """生成 10 篇论文的摄取统计、视觉 QA 和结构问题汇总。"""

    manifest = _load_and_verify_manifest()
    settings = Settings()
    if settings.collection_name != EXPECTED_COLLECTION:
        raise ValueError("active Settings does not target research collection")
    tokenizer = BgeM3Tokenizer(settings.embedding_model)
    papers = [_paper_audit(paper, tokenizer) for paper in manifest["papers"]]
    collection = ChromaVectorStore(settings.vector_db_path, settings.collection_name)
    collection_count = collection._collection.count()
    if collection_count != EXPECTED_CHUNK_COUNT:
        raise ValueError(
            f"research collection count drifted: {collection_count}"
        )
    if sum(item["chunk_count"] for item in papers) != collection_count:
        raise ValueError("local chunk count does not match Chroma collection")

    chunk_counts = [item["chunk_count"] for item in papers]
    issue_matrix = _issue_matrix(papers)
    return {
        "audit_version": "academic-ingestion-audit-v1",
        "manifest_hash": manifest["manifest_hash"],
        "corpus_hash": manifest["corpus_hash"],
        "configuration": {
            "loader": "PyMuPDF current implementation",
            "cleaner": "current conservative cleaner",
            "embedding_model": settings.embedding_model,
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
            "collection_name": settings.collection_name,
        },
        "aggregate": {
            "document_count": len(papers),
            "total_pages": sum(item["page_count"] for item in papers),
            "total_chunks": sum(chunk_counts),
            "average_chunks_per_paper": statistics.mean(chunk_counts),
            "minimum_chunks_per_paper": min(chunk_counts),
            "maximum_chunks_per_paper": max(chunk_counts),
            "extraction_failure_count": 0,
            "empty_page_count": sum(item["empty_page_count"] for item in papers),
            "representative_page_review_count": sum(
                len(item["representative_pages_reviewed"]) for item in papers
            ),
            "replacement_or_abnormal_character_count": sum(
                item["replacement_or_abnormal_character_count"] for item in papers
            ),
            "very_short_chunk_count": sum(
                item["very_short_chunk_count"] for item in papers
            ),
            "full_512_token_chunk_count": sum(
                item["full_512_token_chunk_count"] for item in papers
            ),
        },
        "issue_matrix": issue_matrix,
        "papers": papers,
        "initial_decisions": {
            "loader_change_required": False,
            "cleaner_change_candidate": "academic dehyphenation",
            "chunker_change_candidate": "section-aware boundary preservation",
            "metadata_extension_candidate": "section_type and supplementary boundary",
            "table_handling_evidence": True,
            "references_handling_evidence": True,
            "multimodal_evidence": False,
            "note": "以上仅为诊断候选；必须等待 Retrieval Baseline 后才能选择唯一下一实验。",
        },
    }


def render_markdown(payload: dict[str, Any]) -> str:
    """渲染每篇统计、Issue Matrix 与当前不优化约束。"""

    aggregate = payload["aggregate"]
    lines = [
        "# Academic PDF Ingestion Compatibility Audit v1",
        "",
        "## Baseline",
        "",
        f"- Corpus: {aggregate['document_count']} point-cloud-completion papers, {aggregate['total_pages']} pages.",
        f"- Collection: `{payload['configuration']['collection_name']}`.",
        f"- Current pipeline: PyMuPDF → Cleaner → C512/O64 → BGE-M3 → Chroma.",
        f"- Ingestion: {aggregate['extraction_failure_count']} failures; {aggregate['total_chunks']} chunks.",
        f"- Visual QA: {aggregate['representative_page_review_count']} representative pages.",
        "",
        "## Per-paper statistics",
        "",
        "| Paper | Pages | Chunks | Characters | Hyphen breaks | Ref page | Supp page | Max pages/chunk |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for paper in payload["papers"]:
        lines.append(
            f"| {paper['file_name']} | {paper['page_count']} | {paper['chunk_count']} | "
            f"{paper['extracted_character_count']} | {paper['broken_hyphenation_count']} | "
            f"{paper['references_page'] or '-'} | {paper['supplementary_boundary_page'] or '-'} | "
            f"{paper['max_pages_per_chunk']} |"
        )
    lines.extend(
        [
            "",
            "## Issue matrix",
            "",
            "| Issue | Papers affected | Severity |",
            "|---|---:|---|",
        ]
    )
    for issue in payload["issue_matrix"]:
        lines.append(
            f"| {issue['issue']} | {issue['papers_affected']}/{issue['denominator']} | {issue['severity']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- All titles and abstracts were visually readable; no scanned-only or extraction-empty PDF was selected.",
            "- Representative double-column pages preserved a usable reading sequence, but the Domain Model does not expose section structure.",
            "- Tables are visually intact in the PDF but flattened in extracted text, so row/column semantics are not guaranteed.",
            "- Every paper contributes References as normal chunks; all five included appendices/supplements lack an explicit boundary field.",
            "- Figure captions are present, while text embedded in diagrams can enter the stream as low-level noise; this is not yet evidence for multimodal parsing.",
            "- No core component is changed in this stage. Retrieval results must determine the single next experiment.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    """在离线模型模式下生成 JSON 与 Markdown 审计产物。"""

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    payload = build_audit()
    OUTPUT_JSON.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    OUTPUT_MD.write_text(render_markdown(payload), encoding="utf-8")
    print(json.dumps(payload["aggregate"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
