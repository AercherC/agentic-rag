"""冻结 10 篇点云补全论文的 Academic Research v1 Corpus Manifest。"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import pymupdf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
for import_root in (PROJECT_ROOT, SOURCE_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from evidenceflow.domain.ids import build_document_id


SOURCE_DOCS = PROJECT_ROOT / "data" / "source_docs"
OUTPUT_PATH = (
    PROJECT_ROOT
    / "evals"
    / "research_ingestion"
    / "academic_corpus_manifest_v1.json"
)
COLLECTION_NAME = "ef_research_bge_m3_c512_o64_v1"
SCANNED_PDF_COUNT = 107


# 元数据来自论文首页；选择原因用于冻结 Corpus 的覆盖逻辑，不参与 Runtime。
PAPER_METADATA: dict[str, dict[str, Any]] = {
    "pcn_point_completion_network.pdf": {
        "source_path": r"点云补全\2019\PCN_Point_Completion_Network.pdf",
        "title": "PCN: Point Completion Network",
        "authors": ["Wentao Yuan", "Tejas Khot", "David Held", "Christoph Mertz", "Martial Hebert"],
        "year": 2018,
        "venue": "3DV",
        "has_supplementary": True,
        "selection_reason": "经典 coarse-to-fine baseline；长 PDF 含补充材料，可检查正文到附录边界。",
    },
    "pfnet_point_fractal_network.pdf": {
        "source_path": r"点云补全\2020\Huang_PF-Net_Point_Fractal_Network_for_3D_Point_Cloud_Completion_CVPR_2020_paper.pdf",
        "title": "PF-Net: Point Fractal Network for 3D Point Cloud Completion",
        "authors": ["Zitian Huang", "Yikuan Yu", "Jiawen Xu", "Feng Ni", "Xinyi Le"],
        "year": 2020,
        "venue": "CVPR",
        "has_supplementary": False,
        "selection_reason": "紧凑双栏会议论文，包含多阶段损失、公式、表格和定性图。",
    },
    "pointr_geometry_aware_transformers.pdf": {
        "source_path": r"点云补全\2021\ICCV2021_PoinTr.pdf",
        "title": "PoinTr: Diverse Point Cloud Completion with Geometry-Aware Transformers",
        "authors": ["Xumin Yu", "Yongming Rao", "Ziyi Wang", "Zuyan Liu", "Jiwen Lu", "Jie Zhou"],
        "year": 2021,
        "venue": "ICCV",
        "has_supplementary": True,
        "selection_reason": "Transformer 代表方法；正文、详细结果表与追加可视化并存。",
    },
    "snowflakenet.pdf": {
        "source_path": r"点云补全\2021\ICCV2021_SnowflakeNet.pdf",
        "title": "SnowflakeNet: Point Cloud Completion by Snowflake Point Deconvolution with Skip-Transformer",
        "authors": ["Peng Xiang", "Xin Wen", "Yu-Shen Liu", "Yan-Pei Cao", "Pengfei Wan", "Wen Zheng", "Zhizhong Han"],
        "year": 2021,
        "venue": "ICCV",
        "has_supplementary": False,
        "selection_reason": "层级点反卷积方法；适合检查复杂模块名、公式和消融表。",
    },
    "seedformer.pdf": {
        "source_path": r"点云补全\2022\SeedFormer_ECCV2022.pdf",
        "title": "SeedFormer: Patch Seeds based Point Cloud Completion with Upsample Transformer",
        "authors": ["Haoran Zhou", "Yun Cao", "Wenqing Chu", "Junwei Zhu", "Tong Lu", "Ying Tai", "Chengjie Wang"],
        "year": 2022,
        "venue": "ECCV",
        "has_supplementary": True,
        "selection_reason": "22 页长论文且明确包含 Supplementary，可审计主文与补充材料边界。",
    },
    "adapointr.pdf": {
        "source_path": r"点云补全\2023\ICCV2023_AdaPoinTr Diverse Point Cloud Completion with.pdf",
        "title": "AdaPoinTr: Diverse Point Cloud Completion with Adaptive Geometry-Aware Transformers",
        "authors": ["Xumin Yu", "Yongming Rao", "Ziyi Wang", "Jiwen Lu", "Jie Zhou"],
        "year": 2023,
        "venue": "IEEE TPAMI",
        "has_supplementary": True,
        "selection_reason": "长篇期刊版，包含大量详细结果表和 Appendix，检验长文档兼容性。",
    },
    "proxyformer.pdf": {
        "source_path": r"点云补全\2023\ProxyFormer_CVPR_2023_paper.pdf",
        "title": "ProxyFormer: Proxy Alignment Assisted Point Cloud Completion with Missing Part Sensitive Transformer",
        "authors": ["Shanshan Li", "Pan Gao", "Xiaoyang Tan", "Mingqiang Wei"],
        "year": 2023,
        "venue": "CVPR",
        "has_supplementary": False,
        "selection_reason": "标准十页双栏论文，包含代理对齐、Transformer、表格与参考文献。",
    },
    "svdformer.pdf": {
        "source_path": r"点云补全\2023\Zhu_SVDFormer_Complementing_Point_Cloud_via_Self-view_Augmentation_and_Self-structure_Dual-generator_ICCV_2023_paper.pdf",
        "title": "SVDFormer: Complementing Point Cloud via Self-view Augmentation and Self-structure Dual-generator",
        "authors": ["Zhe Zhu", "Honghua Chen", "Xing He", "Weiming Wang", "Jing Qin", "Mingqiang Wei"],
        "year": 2023,
        "venue": "ICCV",
        "has_supplementary": False,
        "selection_reason": "多视图和双生成器结构，提供图像说明、复杂表格及跨模态术语。",
    },
    "pointattn.pdf": {
        "source_path": r"点云补全\2024\POINTATTN YOU ONLY NEED ATTENTION FOR POINT CLOUD completion.pdf",
        "title": "PointAttN: You Only Need Attention for Point Cloud Completion",
        "authors": ["Jun Wang", "Ying Cui", "Dongyan Guo", "Junxia Li", "Qingshan Liu", "Chunhua Shen"],
        "year": 2024,
        "venue": "AAAI",
        "has_supplementary": False,
        "selection_reason": "较新 Attention 架构，包含多数据集实验、消融和密集引用页。",
    },
    "hyper_pcn.pdf": {
        "source_path": r"点云补全\2026\CVPR2026_hyper-pcn.pdf",
        "title": "Hyper-PCN: Hypergraph-Based Point Cloud Completion via High-Order Correlation Modeling",
        "authors": ["Linfei Li", "Pei Tan", "Siqi Li", "Changqing Zou", "Yue Gao"],
        "year": 2026,
        "venue": "CVPR",
        "has_supplementary": True,
        "selection_reason": "最新 Hypergraph 方法，含正文、附录和丰富可视化，是迁移目标代表样本。",
    },
}


def _sha256_bytes(value: bytes) -> str:
    """返回 PDF、Corpus 或 Manifest 的确定性 SHA-256。"""

    return hashlib.sha256(value).hexdigest()


def _canonical_hash(value: Any) -> str:
    """对结构化数据执行 UTF-8 确定性序列化后计算 Hash。"""

    serialized = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return _sha256_bytes(serialized)


def build_manifest() -> dict[str, Any]:
    """验证选定 PDF 并构造冻结的 10 篇论文 Corpus Manifest。"""

    actual_names = {path.name for path in SOURCE_DOCS.glob("*.pdf")}
    if actual_names != set(PAPER_METADATA):
        raise ValueError(
            f"research corpus file set drifted: {sorted(actual_names)}"
        )

    papers: list[dict[str, Any]] = []
    for file_name in sorted(PAPER_METADATA):
        path = SOURCE_DOCS / file_name
        metadata = PAPER_METADATA[file_name]
        file_bytes = path.read_bytes()
        with pymupdf.open(path) as pdf:
            pages = [page.get_text() for page in pdf]
            page_count = pdf.page_count
        if any(not page.strip() for page in pages):
            raise ValueError(f"selected PDF contains an empty text page: {file_name}")
        papers.append(
            {
                "file_name": file_name,
                "selected_from": metadata["source_path"],
                "document_id": build_document_id(file_name),
                "sha256": _sha256_bytes(file_bytes),
                "title": metadata["title"],
                "authors": metadata["authors"],
                "year": metadata["year"],
                "venue": metadata["venue"],
                "page_count": page_count,
                "extracted_character_count": sum(len(page) for page in pages),
                "empty_page_count": 0,
                "has_two_column": True,
                "has_tables": any("table" in page.casefold() for page in pages),
                "has_figures": any("figure" in page.casefold() for page in pages),
                "has_equations": True,
                "has_references": any("references" in page.casefold() for page in pages),
                "has_supplementary": metadata["has_supplementary"],
                "selection_reason": metadata["selection_reason"],
            }
        )

    corpus_identity = [
        {
            "file_name": item["file_name"],
            "document_id": item["document_id"],
            "sha256": item["sha256"],
        }
        for item in papers
    ]
    payload: dict[str, Any] = {
        "manifest_version": "academic-corpus-manifest-v1",
        "domain": "point_cloud_completion",
        "candidate_inventory_pdf_count": SCANNED_PDF_COUNT,
        "corpus_count": len(papers),
        "collection_name": COLLECTION_NAME,
        "embedding_model": "BAAI/bge-m3",
        "chunk_size": 512,
        "chunk_overlap": 64,
        "corpus_hash": _canonical_hash(corpus_identity),
        "papers": papers,
    }
    payload["manifest_hash"] = _canonical_hash(payload)
    return payload


def main() -> None:
    """写出 Corpus Manifest，并打印冻结摘要。"""

    payload = build_manifest()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "corpus_count": payload["corpus_count"],
                "total_pages": sum(item["page_count"] for item in payload["papers"]),
                "corpus_hash": payload["corpus_hash"],
                "manifest_hash": payload["manifest_hash"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
