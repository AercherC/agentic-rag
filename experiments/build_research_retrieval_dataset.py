"""构建并冻结 Academic Research v1 的 40 题检索评测集。"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

import fitz


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "data" / "source_docs"
INGESTION_MANIFEST = ROOT / "evals" / "research_ingestion" / "academic_corpus_manifest_v1.json"
OUTPUT_DIR = ROOT / "evals" / "research_retrieval"


def _q(
    question_id: str,
    language: str,
    question_type: str,
    evidence_type: str,
    query: str,
    english_query: str,
    sources: list[tuple[str, str]],
) -> dict[str, Any]:
    """声明一道题；证据页码由原始 PDF 中的冻结锚点自动定位。"""

    return {
        "question_id": question_id,
        "language": language,
        "language_group": (
            "mixed_multiple_sources"
            if len(sources) > 1
            else ("english_to_english" if language == "en" else "chinese_to_english")
        ),
        "question_type": question_type,
        "evidence_type": evidence_type,
        "query": query,
        "english_query": english_query,
        "sources": sources,
    }


# 问题在检索前一次性人工设计；锚点仅依赖原文，不依赖任何检索结果或 Chunk ID。
QUESTION_SPECS = [
    _q("ARR-001", "en", "motivation_concept", "body_text", "Why is point cloud completion needed according to PCN?", "Why is point cloud completion needed according to PCN?", [("pcn_point_completion_network.pdf", "Real-world 3D data are often incomplete")]),
    _q("ARR-002", "zh", "motivation_concept", "body_text", "PoinTr 认为真实点云不完整的三个常见原因是什么？", "What three common causes of incomplete real-world point clouds does PoinTr identify?", [("pointr_geometry_aware_transformers.pdf", "limited sensor resolution, single viewpoint")]),
    _q("ARR-003", "en", "motivation_concept", "body_text", "What acquisition problems motivate ProxyFormer?", "What acquisition problems motivate ProxyFormer?", [("proxyformer.pdf", "equipment defects or limited viewpoints")]),
    _q("ARR-004", "zh", "motivation_concept", "body_text", "SVDFormer 试图缓解真实扫描中的哪类稀疏性问题？", "What sparsity problem in real-world scans does SVDFormer seek to mitigate?", [("svdformer.pdf", "highly sparse real-world")]),
    _q("ARR-005", "en", "motivation_concept", "body_text", "Why does PointAttN question using a fixed neighborhood parameter?", "Why does PointAttN question using a fixed neighborhood parameter?", [("pointattn.pdf", "fixed parameter to process point clouds is unreasonable")]),
    _q("ARR-006", "zh", "motivation_concept", "body_text", "Hyper-PCN 认为只建模成对关系会遗漏什么？", "What does Hyper-PCN argue is missed by modeling only pairwise relations?", [("hyper_pcn.pdf", "high-order correlations")]),
    _q("ARR-007", "en", "method", "figure_caption", "What two outputs are shown in the PCN architecture?", "What two outputs are shown in the PCN architecture?", [("pcn_point_completion_network.pdf", "Figure 2: PCN architecture")]),
    _q("ARR-008", "zh", "method", "figure_caption", "PF-Net 的总体架构图由哪三个主要模块组成？", "Which three main modules form the overall PF-Net architecture?", [("pfnet_point_fractal_network.pdf", "The architecture of PF-Net")]),
    _q("ARR-009", "en", "method", "body_text", "How does PoinTr represent a point cloud for transformer processing?", "How does PoinTr represent a point cloud for transformer processing?", [("pointr_geometry_aware_transformers.pdf", "sequence of point proxies")]),
    _q("ARR-010", "zh", "method", "figure_caption", "SnowflakeNet 的架构图包含哪种点反卷积模块？", "What point deconvolution module is shown in the SnowflakeNet architecture?", [("snowflakenet.pdf", "architecture of SnowflakeNet")]),
    _q("ARR-011", "en", "method", "body_text", "What new shape representation does SeedFormer introduce?", "What new shape representation does SeedFormer introduce?", [("seedformer.pdf", "shape representation, namely Patch Seeds")]),
    _q("ARR-012", "zh", "method", "body_text", "ProxyFormer 如何利用 proxy alignment 改善缺失区域？", "How does ProxyFormer use proxy alignment to improve missing regions?", [("proxyformer.pdf", "proxy alignment")]),
    _q("ARR-013", "en", "method", "body_text", "What two generators make up SVDFormer's dual-generator design?", "What two generators make up SVDFormer's dual-generator design?", [("svdformer.pdf", "self-structure dual-generator")]),
    _q("ARR-014", "zh", "method", "figure_caption", "Hyper-PCN 总体框架中的 HyperRS 与 A-HGNN 分别承担什么角色？", "What roles do HyperRS and A-HGNN play in the overall Hyper-PCN framework?", [("hyper_pcn.pdf", "overall Hyper-PCN framework")]),
    _q("ARR-015", "en", "implementation", "body_text", "How many ShapeNet models and categories are used by PCN?", "How many ShapeNet models and categories are used by PCN?", [("pcn_point_completion_network.pdf", "30974 models from 8 categories")]),
    _q("ARR-016", "zh", "implementation", "body_text", "SnowflakeNet 训练时采用什么主要距离损失？", "What primary distance loss is used to train SnowflakeNet?", [("snowflakenet.pdf", "Chamfer distance")]),
    _q("ARR-017", "en", "implementation", "body_text", "What values are used for the number of patches and seed feature dimensions in SeedFormer?", "What values are used for the number of patches and seed feature dimensions in SeedFormer?", [("seedformer.pdf", "Np = 128")]),
    _q("ARR-018", "zh", "implementation", "body_text", "ProxyFormer 在 ShapeNet-55/34 中输入点云和完整点云各有多少点？", "How many points are in the input and complete point clouds for ProxyFormer on ShapeNet-55/34?", [("proxyformer.pdf", "2048 points")]),
    _q("ARR-019", "en", "implementation", "body_text", "How many ground-truth points are used in PointAttN's PCN benchmark?", "How many ground-truth points are used in PointAttN's PCN benchmark?", [("pointattn.pdf", "16384 points")]),
    _q("ARR-020", "zh", "experiment_setup", "body_text", "PCN 的合成训练数据来自哪个 CAD 数据集？", "Which CAD dataset supplies PCN's synthetic training data?", [("pcn_point_completion_network.pdf", "synthetic CAD models from ShapeNet")]),
    _q("ARR-021", "en", "experiment_setup", "body_text", "How many ShapeNet categories are used in PoinTr's ShapeNet-55 benchmark?", "How many ShapeNet categories are used in PoinTr's ShapeNet-55 benchmark?", [("pointr_geometry_aware_transformers.pdf", "55 categories")]),
    _q("ARR-022", "zh", "experiment_setup", "body_text", "SnowflakeNet 的主要实验使用哪两个公开基准？", "Which two public benchmarks are used in SnowflakeNet's main experiments?", [("snowflakenet.pdf", "PCN [57] and Completion3D")]),
    _q("ARR-023", "en", "experiment_setup", "body_text", "Which projected ShapeNet datasets does AdaPoinTr introduce?", "Which projected ShapeNet datasets does AdaPoinTr introduce?", [("adapointr.pdf", "Projected-ShapeNet-55")]),
    _q("ARR-024", "zh", "experiment_setup", "body_text", "Hyper-PCN 在哪些主要数据集上进行评估？", "On which main datasets is Hyper-PCN evaluated?", [("hyper_pcn.pdf", "PCN dataset")]),
    _q("ARR-025", "en", "quantitative_results", "table", "Which methods are compared with AdaPoinTr in its ShapeNet-55 results table?", "Which methods are compared with AdaPoinTr in its ShapeNet-55 results table?", [("adapointr.pdf", "Results on ShapeNet-55")]),
    _q("ARR-026", "zh", "quantitative_results", "table", "PoinTr 的 PCN 数据集结果表使用什么指标比较各方法？", "What metric is used to compare methods in PoinTr's PCN dataset results table?", [("pointr_geometry_aware_transformers.pdf", "Results on the PCN dataset")]),
    _q("ARR-027", "en", "quantitative_results", "table", "What variants are reported in SeedFormer's ShapeNet-55 completion table?", "What variants are reported in SeedFormer's ShapeNet-55 completion table?", [("seedformer.pdf", "Completion results on ShapeNet-55")]),
    _q("ARR-028", "zh", "quantitative_results", "table", "ProxyFormer 的 PCN 定量对比表包含哪些物体类别？", "Which object categories appear in ProxyFormer's quantitative PCN comparison table?", [("proxyformer.pdf", "Quantitative comparison")]),
    _q("ARR-029", "en", "quantitative_results", "table", "How does SVDFormer report results on seen and unseen ShapeNet-34 categories?", "How does SVDFormer report results on seen and unseen ShapeNet-34 categories?", [("svdformer.pdf", "Quantitative results on ShapeNet-34")]),
    _q("ARR-030", "zh", "quantitative_results", "table", "PointAttN 的 PCN 结果表采用什么 Chamfer Distance 缩放？", "What Chamfer Distance scaling is used in PointAttN's PCN results table?", [("pointattn.pdf", "Point cloud completion results on the PCN dataset")]),
    _q("ARR-031", "en", "ablation", "supplementary", "What point-proxy formulation is ablated in AdaPoinTr's appendix?", "What point-proxy formulation is ablated in AdaPoinTr's appendix?", [("adapointr.pdf", "Ablation on formulation of point proxies")]),
    _q("ARR-032", "zh", "ablation", "supplementary", "Hyper-PCN 补充材料如何通过替换方式消融核心模块？", "How does the Hyper-PCN supplement ablate core modules by replacement?", [("hyper_pcn.pdf", "Ablation of core modules by replacement")]),
    _q("ARR-033", "en", "ablation", "supplementary", "What additional evidence does the PCN supplementary document provide?", "What additional evidence does the PCN supplementary document provide?", [("pcn_point_completion_network.pdf", "technical details and additional quantitative")]),
    _q("ARR-034", "zh", "ablation", "supplementary", "SeedFormer 补充材料列出了哪些额外实现与实验内容？", "What additional implementation and experimental content is listed in the SeedFormer supplement?", [("seedformer.pdf", "supplementary material")]),
    _q("ARR-035", "en", "conclusion_limitation", "body_text", "What core approach does SVDFormer summarize in its conclusion?", "What core approach does SVDFormer summarize in its conclusion?", [("svdformer.pdf", "We propose SVDFormer")]),
    _q("ARR-036", "zh", "conclusion_limitation", "body_text", "PointAttN 在结论中如何概括其核心网络？", "How does PointAttN summarize its core network in the conclusion?", [("pointattn.pdf", "we present a novel deep neural network")]),
    _q("ARR-037", "en", "multi_document", "multi_document", "Compare how PoinTr and AdaPoinTr construct or adapt transformer queries.", "Compare how PoinTr and AdaPoinTr construct or adapt transformer queries.", [("pointr_geometry_aware_transformers.pdf", "dynamic queries"), ("adapointr.pdf", "dynamic query")]),
    _q("ARR-038", "zh", "multi_document", "multi_document", "比较 PF-Net 与 SeedFormer 的层级式点云生成表示。", "Compare the hierarchical point-cloud generation representations of PF-Net and SeedFormer.", [("pfnet_point_fractal_network.pdf", "point pyramid decoder"), ("seedformer.pdf", "Patch Seeds")]),
    _q("ARR-039", "en", "multi_document", "multi_document", "Compare the attention mechanisms used by SnowflakeNet and PointAttN.", "Compare the attention mechanisms used by SnowflakeNet and PointAttN.", [("snowflakenet.pdf", "Skip-Transformer"), ("pointattn.pdf", "self-attention and cross-attention")]),
    _q("ARR-040", "zh", "multi_document", "multi_document", "比较 SVDFormer 与 Hyper-PCN 如何建模点云结构关系。", "Compare how SVDFormer and Hyper-PCN model structural relationships in point clouds.", [("svdformer.pdf", "self-structure"), ("hyper_pcn.pdf", "hypergraph")]),
]


def _normalize(text: str) -> str:
    """统一 Unicode、断行和 PDF 连字符，供锚点确定性匹配。"""

    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"(?<=\w)-\s*\n\s*(?=\w)", "", text)
    return re.sub(r"\s+", " ", text).strip().casefold()


def _canonical_hash(payload: Any) -> str:
    """以稳定 JSON 序列化计算 SHA-256，保证数据集可追踪。"""

    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def main() -> None:
    """定位 Gold、校验分布并写出不可依赖 Chunk ID 的冻结数据集。"""

    corpus = json.loads(INGESTION_MANIFEST.read_text(encoding="utf-8"))
    papers = {paper["file_name"]: paper for paper in corpus["papers"]}
    page_cache: dict[str, list[str]] = {}
    questions: list[dict[str, Any]] = []

    for spec in QUESTION_SPECS:
        gold_sources = []
        for file_name, anchor in spec["sources"]:
            if file_name not in papers:
                raise ValueError(f"unknown corpus file: {file_name}")
            if file_name not in page_cache:
                with fitz.open(SOURCE_DIR / file_name) as pdf:
                    page_cache[file_name] = [page.get_text("text") for page in pdf]
            normalized_anchor = _normalize(anchor)
            matches = [
                page_number
                for page_number, page_text in enumerate(page_cache[file_name], start=1)
                if normalized_anchor in _normalize(page_text)
            ]
            if not matches:
                raise ValueError(f"anchor not found: {file_name}: {anchor!r}")
            # 短术语可能跨页重复；冻结首次出现页，并保留原文锚点供独立复核。
            gold_sources.append(
                {
                    "file_name": file_name,
                    "document_id": papers[file_name]["document_id"],
                    "page_start": matches[0],
                    "page_end": matches[0],
                    "evidence_anchor": anchor,
                }
            )
        questions.append(
            {
                key: value for key, value in spec.items() if key != "sources"
            }
            | {"gold_sources": gold_sources}
        )

    if len(questions) != 40 or len({q["question_id"] for q in questions}) != 40:
        raise ValueError("research dataset must contain 40 unique questions")
    expected_types = {
        "motivation_concept": 6,
        "method": 8,
        "implementation": 5,
        "experiment_setup": 5,
        "quantitative_results": 6,
        "ablation": 4,
        "conclusion_limitation": 2,
        "multi_document": 4,
    }
    if Counter(q["question_type"] for q in questions) != expected_types:
        raise ValueError("question type distribution changed")
    if Counter(q["evidence_type"] for q in questions) != {
        "body_text": 22,
        "figure_caption": 4,
        "table": 6,
        "supplementary": 4,
        "multi_document": 4,
    }:
        raise ValueError("evidence type distribution changed")
    if Counter(q["language"] for q in questions) != {"en": 20, "zh": 20}:
        raise ValueError("language distribution changed")

    dataset_core = {
        "dataset_version": "research-retrieval-v1",
        "corpus_hash": corpus["corpus_hash"],
        "corpus_manifest_hash": corpus["manifest_hash"],
        "gold_contract": "document_id + 1-based page_range + evidence_anchor; no chunk_id",
        "questions": questions,
    }
    dataset_hash = _canonical_hash(dataset_core)
    dataset = {**dataset_core, "dataset_hash": dataset_hash}
    translations = {
        "translation_version": "research-frozen-translations-v1",
        "dataset_hash": dataset_hash,
        "translations": {
            question["question_id"]: question["english_query"] for question in questions
        },
    }
    manifest = {
        "dataset_version": dataset["dataset_version"],
        "dataset_hash": dataset_hash,
        "question_count": len(questions),
        "language_distribution": dict(Counter(q["language"] for q in questions)),
        "language_group_distribution": dict(Counter(q["language_group"] for q in questions)),
        "question_type_distribution": dict(Counter(q["question_type"] for q in questions)),
        "evidence_type_distribution": dict(Counter(q["evidence_type"] for q in questions)),
        "paper_coverage": dict(Counter(source["file_name"] for q in questions for source in q["gold_sources"])),
        "frozen_before_retrieval": True,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "research_questions_v1.json").write_text(
        json.dumps(dataset, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (OUTPUT_DIR / "research_translations_v1.json").write_text(
        json.dumps(translations, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (OUTPUT_DIR / "research_manifest_v1.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
