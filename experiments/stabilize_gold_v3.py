"""依据来源语义冻结 occurrence；不读取检索排名，生成 Phase 3.5 新产物。"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    sys.path.insert(0, str(path))

from evidenceflow.domain.evaluation_v2 import EvaluationDatasetV2
from evidenceflow.eval.stable_mapping import hyphen_rules, normalize_mapping, _positions, validate_source, map_stable_dataset
from evidenceflow.eval.v2 import save_artifact
from evidenceflow.ingestion import load_document
from experiments.audit_gold_v2 import read_chunks_readonly


# 基于 Phase 3.5 修改前原文 occurrence 审阅；选择过程不消费 Chunk/检索结果。
SOURCE_CHOICES = {
    ("ARR-006", "G1"): (1, "摘要中说明现有方法遗漏高阶关系的动机句"),
    ("ARR-010", "G1"): (1, "Figure 3 图注明确列出 SPD 模块"),
    ("ARR-012", "G1"): (2, "摘要解释 alignment 如何 refine missing proxies，排除标题"),
    ("ARR-013", "G1"): (2, "摘要解释 dual-generator 的形状先验与几何自相似性，排除标题"),
    ("ARR-024", "G1"): (1, "Experiment settings 的数据集总清单，排除 PCN 单项结果讨论"),
    ("ARR-025", "G1"): (1, "Table 1 的 ShapeNet-55 表注，排除同名小节标题"),
    ("ARR-026", "G1"): (1, "Table 3 表注明确 l1 Chamfer distance"),
    ("ARR-028", "G1"): (1, "Table 1 定量比较表及类别表头；Table 2 属另一指标表"),
    ("ARR-037", "G2"): (1, "正文首次定义 dynamic query embeddings，与 static query 对照"),
    ("ARR-038", "G2"): (2, "摘要定义 Patch Seeds 及局部表示，排除标题和关键词"),
    ("ARR-039", "G1"): (3, "明确说明 attention 总结上一 SPD splitting patterns 的句子"),
    ("ARR-040", "G1"): (3, "dual-generator 融合 learned shape priors 和 geometric self-similarities"),
    ("ARR-040", "G2"): (2, "摘要说明 hypergraphs 显式建模高阶关系，排除标题和模块枚举"),
}


def stabilize_dataset(old, documents):
    """仅用原页上下文为 Canonical Gold 增加稳定定位，原问题和 anchor 不变。"""
    raw = old.model_dump(mode="json")
    raw["schema_version"] = "2.1"
    raw["dataset_id"] = "research_questions_v2_gold_stable_v1"
    changes = []
    by_id = {d.document_id: d for d in documents}
    for q in raw["questions"]:
        for group in q["required_evidence_groups"]:
            for alternative in group["alternatives"]:
                doc = by_id[alternative["document_id"]]
                if doc.content_hash != alternative["content_version"]:
                    raise ValueError("source content hash mismatch")
                loc = alternative["locator"]
                old_loc = dict(loc)
                source = "\n".join(s.content for s in doc.segments if loc["page_start"] <= s.page_number <= loc["page_end"])
                rules = hyphen_rules(source, alternative["evidence_anchor"])
                normalized = normalize_mapping(source, rules)
                anchor = normalize_mapping(alternative["evidence_anchor"], rules)
                positions = _positions(normalized, anchor)
                selected, reason = SOURCE_CHOICES.get((q["question_id"], group["group_id"]), (1, "原页唯一 anchor；增加上下文以保留稳定来源定位"))
                if not positions or selected > len(positions) or (len(positions) != 1 and (q["question_id"], group["group_id"]) not in SOURCE_CHOICES):
                    raise ValueError(f"source selection requires review: {q['question_id']} {group['group_id']}")
                start = positions[selected - 1]
                end = start + len(anchor)
                loc.update(left_context=normalized[max(0, start - 64):start].strip() or None,
                           right_context=normalized[end:end + 64].strip() or None,
                           occurrence=selected, source_span_start=start, source_span_end=end)
                changes.append({"question_id": q["question_id"], "group_id": group["group_id"],
                                "document_id": doc.document_id, "file_name": doc.file_name,
                                "old_locator": old_loc, "new_locator": dict(loc), "anchor_unchanged": alternative["evidence_anchor"],
                                "reason": reason, "source_evidence": normalized[max(0, start - 180):end + 220],
                                "source_occurrence_count": len(positions), "normalization_rules": rules})
    return EvaluationDatasetV2.model_validate(raw), changes


def build():
    """先冻结全部 Source，再只读映射；无排名、Embedding 或 LLM 输入。"""
    base = ROOT / "evals/evaluation_v2/phase3_gold_audit_v1"
    old = EvaluationDatasetV2.model_validate_json((base / "research_dataset_v2.json").read_text(encoding="utf-8"))
    documents = [load_document(p, ROOT / "data/source_docs") for p in sorted((ROOT / "data/source_docs").glob("*.pdf"))]
    stable, changes = stabilize_dataset(old, documents)
    chunks = read_chunks_readonly(ROOT / "data/chroma/chroma.sqlite3", "ef_research_bge_m3_c512_o64_v1")
    index_version = json.loads((base / "gold_mapping_v2.json").read_text(encoding="utf-8"))["index_version"]
    index_hash = hashlib.sha256(json.dumps([c.model_dump(mode="json") for c in chunks], sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    if index_version.rsplit(":", 1)[-1] != index_hash:
        raise ValueError("current index differs from Phase 3; do not reuse its version")
    mapping, validations, trace = map_stable_dataset(stable, documents, chunks, index_version=index_version)
    source_groups = {(v["question_id"], v["group_id"]) for v in validations if v["source_valid"]}
    multi = [{"question_id": q.question_id, "source_valid": all((q.question_id, g.group_id) in source_groups for g in q.required_evidence_groups)}
             for q in stable.questions if q.question_type == "multi_document"]
    counts = Counter(g.status for g in mapping.groups) if mapping else Counter()
    total = sum(len(q.required_evidence_groups) for q in stable.questions)
    summary = {"phase": "3.5", "source_valid_groups": len(source_groups), "total_groups": total,
               "multi_document": multi, "mapping_counts": {s: counts[s] for s in ("FULLY_MAPPED", "FRAGMENTED_ACROSS_CHUNKS", "PARTIALLY_MAPPED", "UNMAPPED")},
               "gold_mappability_rate": (counts["FULLY_MAPPED"] + counts["FRAGMENTED_ACROSS_CHUNKS"]) / total,
               "fragmented_evidence_rate": counts["FRAGMENTED_ACROSS_CHUNKS"] / total,
               "partial_mapping_rate": counts["PARTIALLY_MAPPED"] / total,
               "unmapped_rate": counts["UNMAPPED"] / total,
               "source_gate": "PASS" if len(source_groups) == total == 44 and all(q["source_valid"] for q in multi) and len(multi) == 4 else "FAIL",
               "tests_gate": "PENDING", "gate": "PENDING", "next_phase": "BLOCKED"}
    artifacts = {"research_questions_v2_gold_stable.json": stable.model_dump(mode="json"),
                 "source_gold_validation_v3.json": {"groups": validations},
                 "gold_change_log_v3.json": {"changes": changes},
                 "corpus_alignment_trace_v3.json": {"groups": trace},
                 "phase35_summary.json": summary}
    if mapping:
        artifacts["gold_mapping_v3.json"] = mapping.model_dump(mode="json")
    return artifacts


def main():
    """显式产物目录；同名文件已存在时拒绝覆盖。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    artifacts = build()
    if any((args.output_dir / name).exists() for name in artifacts):
        raise FileExistsError("artifact already exists")
    for name, payload in artifacts.items():
        save_artifact(payload, args.output_dir / name)
    print(json.dumps(artifacts["phase35_summary.json"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
