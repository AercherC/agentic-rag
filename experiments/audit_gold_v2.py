"""只读正式索引和来源，全量生成版本化 v2 数据、映射及历史排名回放。"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sqlite3
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evidenceflow.config import Settings
from evidenceflow.eval.dataset import adapt_research_v1
from evidenceflow.eval.mapping import map_dataset
from evidenceflow.eval.v2 import evaluate_rankings, save_artifact
from evidenceflow.ingestion import load_document
from evidenceflow.storage.chroma import _restore_chunk


def read_chunks_readonly(database: Path, collection: str):
    """用 SQLite mode=ro 导出已有文本，避免 PersistentClient 创建/迁移正式 collection。"""
    with sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True) as connection:
        exists = connection.execute("SELECT id FROM collections WHERE name=?", (collection,)).fetchall()
        if len(exists) != 1:
            raise ValueError("expected exactly one existing collection")
        records = connection.execute(
            "SELECT e.id,e.embedding_id FROM embeddings e JOIN segments s ON s.id=e.segment_id "
            "WHERE s.collection=? ORDER BY e.embedding_id", (exists[0][0],)).fetchall()
        chunks = []
        for record_id, chunk_id in records:
            metadata = {key: next((v for v in values if v is not None), None)
                        for key, *values in connection.execute(
                            "SELECT key,string_value,int_value,float_value,bool_value FROM embedding_metadata WHERE id=?", (record_id,))}
            chunks.append(_restore_chunk(chunk_id, metadata.pop("chroma:document"), metadata))
        return chunks


def run(output_dir: Path) -> dict:
    """不改数据真值，用新 schema 包装旧题并明确映射失败的统计口径。"""
    # 输出目录必须全新；所有输入均读取，历史产物不会被覆盖。
    if output_dir.exists():
        raise FileExistsError(output_dir)
    settings = Settings()
    source_root = (ROOT / settings.source_docs_path).resolve()
    legacy_path = ROOT / "evals/research_retrieval/research_questions_v1.json"
    legacy = json.loads(legacy_path.read_text(encoding="utf-8"))
    documents = [load_document(path, source_root) for path in sorted(source_root.glob("*.pdf"))]
    versions = {doc.document_id: doc.content_hash for doc in documents}
    manifest = json.loads((ROOT / "evals/research_ingestion/academic_corpus_manifest_v1.json").read_text(encoding="utf-8"))
    expected = {paper["document_id"]: paper["sha256"] for paper in manifest["papers"]}
    if versions != expected or legacy["corpus_hash"] != manifest["corpus_hash"]:
        raise ValueError("current sources differ from frozen corpus")
    chunks = read_chunks_readonly(ROOT / settings.vector_db_path / "chroma.sqlite3", settings.collection_name)
    if {chunk.document_id for chunk in chunks} != set(versions):
        raise ValueError("index source inventory differs from corpus")
    dataset = adapt_research_v1(legacy, versions)
    index_payload = [chunk.model_dump(mode="json") for chunk in chunks]
    # 指纹只覆盖本次映射真正消费的 Chunk 文本与 metadata，不加载向量或模型。
    index_hash = hashlib.sha256(json.dumps(index_payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    mapping = map_dataset(dataset, documents, chunks, index_version=f"{settings.collection_name}:{index_hash}")
    baseline = json.loads((ROOT / "evals/research_retrieval/research_dense_baseline_v1.json").read_text(encoding="utf-8"))
    if baseline["dataset_hash"] != legacy["dataset_hash"]:
        raise ValueError("historical rankings target a different dataset")
    rankings = {q["question_id"]: [c["chunk_id"] for c in q["candidates"]] for q in baseline["question_results"]}
    if any(cid not in {c.chunk_id for c in chunks} for ids in rankings.values() for cid in ids):
        raise ValueError("historical candidate ID absent from current index")
    replay = evaluate_rankings(dataset, mapping, rankings, depths=(1, 3, 5, 10, 20),
                               configuration={"mode": "historical-ranking-replay", "not_a_new_retrieval_baseline": True})
    summary = {"phase": 3, "dataset_version": dataset.dataset_id, "corpus_version": dataset.corpus_version,
               "index_version": mapping.index_version, "question_count": len(dataset.questions),
               "group_count": len(mapping.groups), "status_counts": dict(Counter(g.status for g in mapping.groups)),
               "source_extraction_success_rate": sum(bool(d.segments) for d in documents) / len(documents),
               "unresolved": [g.model_dump() for g in mapping.groups if not g.chunk_sets],
               "historical_artifacts_overwritten": False,
               "gate": "PASS" if len(mapping.groups) == sum(len(q.required_evidence_groups) for q in dataset.questions) else "FAIL",
               "gate_scope": "all groups audited and mapping failures excluded; not all gold necessarily mappable"}
    save_artifact(dataset.model_dump(mode="json"), output_dir / "research_dataset_v2.json")
    save_artifact(mapping.model_dump(mode="json"), output_dir / "gold_mapping_v2.json")
    save_artifact(replay, output_dir / "historical_rankings_replay_v2.json")
    save_artifact(summary, output_dir / "gold_audit_summary_v2.json")
    return summary


def main():
    """显式选择新产物目录，默认不运行任何正式检索或生成。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir.resolve()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
