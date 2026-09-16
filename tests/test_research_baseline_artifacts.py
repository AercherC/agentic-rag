"""验证 Academic Research v1 冻结语料、评测集和基线报告的一致性。"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load(relative_path: str) -> dict:
    """读取项目内 JSON 评测产物。"""

    return json.loads((ROOT / relative_path).read_text(encoding="utf-8"))


def _canonical_hash(payload: dict) -> str:
    """复用冻结数据集声明的确定性 JSON Hash 规则。"""

    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def test_research_corpus_manifest_matches_active_sources() -> None:
    """研究语料必须恰好包含冻结的 10 篇 PDF，避免旧领域文档重新混入。"""

    manifest = _load("evals/research_ingestion/academic_corpus_manifest_v1.json")
    actual_files = sorted(path.name for path in (ROOT / "data/source_docs").glob("*.pdf"))
    expected_files = sorted(paper["file_name"] for paper in manifest["papers"])

    assert manifest["corpus_count"] == 10
    assert actual_files == expected_files
    assert manifest["collection_name"] == "ef_research_bge_m3_c512_o64_v1"


def test_research_dataset_hash_and_contract_are_frozen() -> None:
    """40 题必须保持 Gold 无 Chunk ID、题型及证据类型分布不漂移。"""

    dataset = _load("evals/research_retrieval/research_questions_v1.json")
    claimed_hash = dataset.pop("dataset_hash")

    assert _canonical_hash(dataset) == claimed_hash
    assert len(dataset["questions"]) == 40
    assert all("chunk_id" not in source for q in dataset["questions"] for source in q["gold_sources"])
    assert Counter(q["question_type"] for q in dataset["questions"]) == {
        "motivation_concept": 6,
        "method": 8,
        "implementation": 5,
        "experiment_setup": 5,
        "quantitative_results": 6,
        "ablation": 4,
        "conclusion_limitation": 2,
        "multi_document": 4,
    }
    assert Counter(q["evidence_type"] for q in dataset["questions"]) == {
        "body_text": 22,
        "figure_caption": 4,
        "table": 6,
        "supplementary": 4,
        "multi_document": 4,
    }


def test_research_dense_report_targets_same_dataset() -> None:
    """Current Dense 报告必须覆盖全部冻结题且不声明运行禁用实验。"""

    dataset = _load("evals/research_retrieval/research_questions_v1.json")
    result = _load("evals/research_retrieval/research_dense_baseline_v1.json")

    assert result["dataset_hash"] == dataset["dataset_hash"]
    assert result["breakdowns"]["overall"]["N"] == 40
    assert len(result["question_results"]) == 40
    assert result["configuration"]["forbidden_experiments_run"] == []
    assert sum(result["failure_counts"].values()) == 40
