"""构建仅增加 section_type metadata 的 References-aware 实验 Collection。"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
for import_root in (ROOT, ROOT / "src"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from evidenceflow.config import Settings
from evidenceflow.ingestion import clean_document, load_document
from evidenceflow.storage import ChromaVectorStore
from evidenceflow.storage.chroma import _restore_chunk


SOURCE_DIR = ROOT / "data" / "source_docs"
BASELINE_COLLECTION = "ef_research_bge_m3_c512_o64_v1"
VARIANT_COLLECTION = "ef_research_bge_m3_c512_o64_refmeta_v1"
EXPECTED_CHUNK_COUNT = 361
OUTPUT_PATH = ROOT / "evals" / "research_retrieval" / "reference_metadata_collection_v1.json"

# 代表页人工核对得到的实际结构边界。PoinTr 的第 12 页仍为 References，
# 因而纠正早期 ingestion audit 中把它记为 supplementary 的启发式误判。
EXPECTED_REFERENCE_PAGES = {
    "adapointr.pdf": 15,
    "hyper_pcn.pdf": 9,
    "pcn_point_completion_network.pdf": 10,
    "pfnet_point_fractal_network.pdf": 9,
    "pointattn.pdf": 10,
    "pointr_geometry_aware_transformers.pdf": 11,
    "proxyformer.pdf": 9,
    "seedformer.pdf": 15,
    "snowflakenet.pdf": 9,
    "svdformer.pdf": 9,
}
EXPECTED_SUPPLEMENTARY_PAGES = {
    "adapointr.pdf": 16,
    "hyper_pcn.pdf": 11,
    "pcn_point_completion_network.pdf": 12,
    "seedformer.pdf": 18,
}


def _normalized_lines(text: str) -> list[str]:
    """折叠行内空白但保留行边界，避免普通正文单词触发章节状态。"""

    return [re.sub(r"\s+", " ", line).strip() for line in text.splitlines()]


def is_references_heading(text: str) -> bool:
    """仅识别独立的 References 标题行，正文提及 references 不触发。"""

    return any(line.casefold() == "references" for line in _normalized_lines(text))


def is_supplementary_heading(text: str) -> bool:
    """识别 References 后的 Supplementary/Appendix 重置边界。"""

    for line in _normalized_lines(text):
        folded = line.casefold()
        if folded == "supplementary material":
            return True
        if folded.startswith("in this supplementary material"):
            return True
        if re.fullmatch(r"appendix(?:\s+[a-z])?", folded):
            return True
        # PCN 的补充部分没有总标题，以 References 后独立的 A. Overview 开始。
        if folded == "a. overview":
            return True
    return False


def detect_section_boundaries(segments: Sequence[Any]) -> tuple[int | None, int | None]:
    """按页面顺序确定 References 起点及其后的 Supplementary 重置页。"""

    references_page: int | None = None
    supplementary_page: int | None = None
    for segment in segments:
        if references_page is None and is_references_heading(segment.content):
            references_page = segment.page_number
            continue
        if references_page is not None and is_supplementary_heading(segment.content):
            supplementary_page = segment.page_number
            break
    return references_page, supplementary_page


def classify_chunk_section(
    chunk: Any,
    *,
    references_page: int,
    supplementary_page: int | None,
) -> str:
    """用页码跨度给既有 Chunk 标注单一 section_type，不改变内容或身份。

    跨边界 Chunk 按其覆盖的后一个区域归类，使 References 污染候选能够被过滤；
    Supplementary 优先级更高，确保 References 状态可以确定性重置。
    """

    if chunk.page_end is None:
        return "body"
    if supplementary_page is not None and chunk.page_end >= supplementary_page:
        return "supplementary"
    if chunk.page_end >= references_page:
        return "references"
    return "body"


def _raw_collection(store: ChromaVectorStore) -> dict[str, Any]:
    """读取完整 Chunk、metadata 和 embedding，保证 Variant 不重新编码文本。"""

    return store._collection.get(include=["documents", "metadatas", "embeddings"])


def build_collection() -> dict[str, Any]:
    """复制 Baseline embedding，只为 Chunk 增加实验性 section_type metadata。"""

    settings = Settings()
    baseline = ChromaVectorStore(settings.vector_db_path, BASELINE_COLLECTION)
    if baseline._collection.count() != EXPECTED_CHUNK_COUNT:
        raise ValueError("baseline research collection must contain 361 chunks")
    raw = _raw_collection(baseline)
    baseline_chunks = [
        _restore_chunk(chunk_id, document, metadata)
        for chunk_id, document, metadata in zip(
            raw["ids"], raw["documents"], raw["metadatas"]
        )
    ]
    stored_embeddings = raw["embeddings"]
    if stored_embeddings is None or len(stored_embeddings) != EXPECTED_CHUNK_COUNT:
        raise ValueError("baseline embeddings are incomplete")
    # Chroma 返回 NumPy array；转为项目 VectorStore Protocol 接受的普通数值序列。
    embeddings = [vector.tolist() for vector in stored_embeddings]

    boundaries: dict[str, tuple[int, int | None]] = {}
    detection_rows = []
    for pdf_path in sorted(SOURCE_DIR.glob("*.pdf")):
        cleaned = clean_document(load_document(pdf_path, SOURCE_DIR))
        references_page, supplementary_page = detect_section_boundaries(cleaned.segments)
        if references_page is None:
            raise ValueError(f"References heading was not detected: {pdf_path.name}")
        boundaries[pdf_path.name] = (references_page, supplementary_page)
        expected_ref = EXPECTED_REFERENCE_PAGES[pdf_path.name]
        expected_supp = EXPECTED_SUPPLEMENTARY_PAGES.get(pdf_path.name)
        detection_rows.append(
            {
                "file_name": pdf_path.name,
                "references_start_page": references_page,
                "supplementary_exists": supplementary_page is not None,
                "supplementary_start_page": supplementary_page,
                "reference_heading_correct": references_page == expected_ref,
                "supplementary_boundary_correct": supplementary_page == expected_supp,
                "false_positive": references_page != expected_ref
                or (supplementary_page is not None and expected_supp is None),
                "false_negative": references_page is None
                or (supplementary_page is None and expected_supp is not None),
            }
        )

    variant_chunks = []
    for chunk in baseline_chunks:
        references_page, supplementary_page = boundaries[chunk.file_name]
        section_type = classify_chunk_section(
            chunk,
            references_page=references_page,
            supplementary_page=supplementary_page,
        )
        metadata = dict(chunk.metadata)
        metadata["section_type"] = section_type
        variant_chunks.append(chunk.model_copy(update={"metadata": metadata}, deep=True))

    variant = ChromaVectorStore(settings.vector_db_path, VARIANT_COLLECTION)
    # 只重建明确命名的实验 Collection；Baseline Collection 永不覆盖。
    if variant._collection.count():
        variant.clear()
    variant.add_chunks(variant_chunks, embeddings)
    if variant._collection.count() != EXPECTED_CHUNK_COUNT:
        raise ValueError("variant collection count differs from baseline")

    # 所有非 metadata 字段及向量必须逐条相同，排除 Chunking/Embedding 混杂变量。
    for baseline_chunk, variant_chunk in zip(baseline_chunks, variant_chunks):
        baseline_core = baseline_chunk.model_dump(exclude={"metadata"})
        variant_core = variant_chunk.model_dump(exclude={"metadata"})
        if baseline_core != variant_core:
            raise ValueError("variant unexpectedly changed chunk identity or text")

    section_counts = Counter(chunk.metadata["section_type"] for chunk in variant_chunks)
    per_paper_counts: dict[str, Counter[str]] = {}
    for file_name in sorted(boundaries):
        per_paper_counts[file_name] = Counter(
            chunk.metadata["section_type"]
            for chunk in variant_chunks
            if chunk.file_name == file_name
        )
    payload = {
        "build_version": "reference-metadata-collection-v1",
        "baseline_collection": BASELINE_COLLECTION,
        "variant_collection": VARIANT_COLLECTION,
        "chunk_count": len(variant_chunks),
        "chunk_ids_identical": True,
        "chunk_text_identical": True,
        "embeddings_reused_without_reencoding": True,
        "section_counts": dict(section_counts),
        "detection_accuracy": {
            "references_correct": sum(row["reference_heading_correct"] for row in detection_rows),
            "references_total": len(detection_rows),
            "supplementary_correct": sum(row["supplementary_boundary_correct"] for row in detection_rows),
            "supplementary_total": len(detection_rows),
            "false_positive_count": sum(row["false_positive"] for row in detection_rows),
            "false_negative_count": sum(row["false_negative"] for row in detection_rows),
        },
        "papers": [
            row
            | {
                "body_chunk_count": per_paper_counts[row["file_name"]]["body"],
                "references_chunk_count": per_paper_counts[row["file_name"]]["references"],
                "supplementary_chunk_count": per_paper_counts[row["file_name"]]["supplementary"],
            }
            for row in detection_rows
        ],
        "known_boundary_rule": (
            "A chunk spanning a boundary is assigned to the later section; this preserves text "
            "identity but mixed boundary chunks remain an acknowledged metadata approximation."
        ),
    }
    OUTPUT_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return payload


def main() -> None:
    """构建独立 Collection 并输出检测与内容一致性审计。"""

    payload = build_collection()
    print(json.dumps({
        "section_counts": payload["section_counts"],
        "detection_accuracy": payload["detection_accuracy"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
