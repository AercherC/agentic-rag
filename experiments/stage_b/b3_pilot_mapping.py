"""B3a-1 第 4 步：用新映射工具映射 canonical 40 题，并与当前 active 相关性投影逐题对照。

一次性脚本，只读：
1. 把 canonical 40 题**只按证据组**转成临时 `E2EDataset`（不写 claims：schema 允许 `required_claims=[]`，
   这些题不经过 `validate_dataset`，因此无需占位 claim），写 `artifacts/stage_b/B3a1_pilot_dataset.json`；
2. 调用 `scripts/e2e_dataset.py map` 映射到当前 active 索引，写 `artifacts/stage_b/B3a1_pilot_mapping.json`；
3. 用 `experiments/phase15b_active_relevance.project_active_relevance` 重算投影，确认 hash 与
   `A1_exact_baseline.json` 记录一致，再逐题逐组比较 chunk 集合与可用性；
4. 对不一致的组做诊断（文本基础 vs 规则），结果写 `artifacts/stage_b/B3a1_pilot_mapping_check.json`（键排序）。

不加载 embedding / reranker 模型，不发 LLM 请求；Chroma 只以 `create_if_missing=False` 打开并 `iter_all_chunks`。
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for extra in (ROOT, ROOT / "src"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from evidenceflow.config import Settings  # noqa: E402
from evidenceflow.domain.evaluation_v2 import EvaluationDatasetV2  # noqa: E402
from evidenceflow.eval.e2e.dataset import E2EDataset, with_fingerprint  # noqa: E402
from evidenceflow.eval.e2e.mapping import E2EMapping, is_usable  # noqa: E402
from evidenceflow.eval.stable_mapping import (  # noqa: E402
    _compact,
    _positions,
    map_stable_dataset,
    normalize_mapping,
    validate_source,
)
from evidenceflow.runtime import resolve_active_collection  # noqa: E402
from evidenceflow.storage.chroma import ChromaVectorStore  # noqa: E402
from experiments.phase15b_active_relevance import normalize_text as projection_normalize  # noqa: E402
from experiments.phase15b_active_relevance import project_active_relevance  # noqa: E402
from scripts.e2e_dataset import load_source_documents  # noqa: E402

GOLD_PATH = ROOT / "evals/evaluation_v2/phase35_gold_stabilization_v1/research_questions_v2_gold_stable.json"
A1_PATH = ROOT / "artifacts/stage_a/A1_exact_baseline.json"
OUT_DIR = ROOT / "artifacts/stage_b"
PILOT_DATASET = OUT_DIR / "B3a1_pilot_dataset.json"
PILOT_MAPPING = OUT_DIR / "B3a1_pilot_mapping.json"
CHECK_PATH = OUT_DIR / "B3a1_pilot_mapping_check.json"


def build_pilot_dataset(raw: dict) -> E2EDataset:
    """canonical 40 题 → 只含证据组的临时题集；题干、证据组原样保留。"""

    cases = []
    for q in raw["questions"]:
        cases.append({
            "case_id": q["question_id"], "family_id": q["family_id"], "query": q["query"],
            "query_language": q["query_language"], "question_type": q["question_type"],
            "tags": ["multi_doc"] if q["question_type"] == "multi_document" else [],
            "critical": False, "expected_status": "answered", "acceptable_abstention": False,
            "required_evidence_groups": q["required_evidence_groups"],
            "required_claims": [], "forbidden_claims": [], "unanswerable_kind": None,
            "provenance": {"drafted_by": "b3a1_pilot", "approved_by": None, "source": "canonical_40_groups_only"},
        })
    return with_fingerprint(E2EDataset.model_validate(
        {"dataset_id": "b3a1_pilot_canonical40_groups", "corpus_version": raw["corpus_version"], "cases": cases}))


def diagnose(alternative, document, chunks, new_sets, proj_ids) -> dict:
    """逐组诊断：区分“规则不同”（occurrence 定位 vs 全文档子串）与“文本基础不同”（chunk 与 Loader 原文无法整块对齐）。"""

    validation = validate_source(alternative, document)
    rules = validation.get("rules", [])
    full_source = "".join(_compact(normalize_mapping(s.content, rules)) for s in document.segments)
    anchor_proj = projection_normalize(alternative.evidence_anchor)
    by_id = {c.chunk_id: c for c in chunks}
    new_ids = sorted({c for bundle in new_sets for c in bundle})
    rows = []
    for cid in sorted(set(new_ids) | set(proj_ids)):
        chunk = by_id[cid]
        compact = _compact(normalize_mapping(chunk.content, rules))
        rows.append({
            "chunk_id": cid, "in_new": cid in new_ids, "in_projection": cid in proj_ids,
            "page_start": chunk.page_start, "page_end": chunk.page_end,
            # stable_mapping 只信任在 Loader 原文中唯一整块对齐的 chunk。
            "whole_chunk_alignments_in_source": len(_positions(full_source, compact)),
            # 投影规则：NFKC+casefold+空白折叠后，anchor 是否为 chunk 子串（不看页码、不看 occurrence）。
            "projection_substring_hit": anchor_proj in projection_normalize(chunk.content),
        })
    return {"source_valid": validation["source_valid"],
            "source_occurrence_count_on_locator_pages": validation.get("source_occurrence_count"),
            "anchor_occurrences_in_whole_document": len(_positions(
                normalize_mapping("\n".join(s.content for s in document.segments), rules),
                normalize_mapping(alternative.evidence_anchor, rules))),
            "hyphen_rules": rules, "chunks": rows}


def classify(new_usable: bool, new_ids: list[str], proj_ids: list[str], diag: list[dict]) -> str:
    """不一致类型（只描述现象，原因见 diagnostics）。"""

    if new_usable and not proj_ids:
        return "NEW_USABLE_PROJECTION_EMPTY"
    if not new_usable and proj_ids:
        return "NEW_UNUSABLE_PROJECTION_NONEMPTY"
    if set(new_ids) < set(proj_ids):
        return "NEW_SUBSET_OF_PROJECTION"
    return "OTHER"


def main() -> int:
    """执行对照并写出结果；不一致时退出码 2（Gate STOP 信号）。"""

    raw = json.loads(GOLD_PATH.read_text(encoding="utf-8"))
    a1 = json.loads(A1_PATH.read_text(encoding="utf-8"))
    pilot = build_pilot_dataset(raw)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PILOT_DATASET.write_text(json.dumps(pilot.model_dump(mode="json"), ensure_ascii=False, sort_keys=True, indent=2)
                             + "\n", encoding="utf-8", newline="\n")

    # 通过正式脚本生成映射（覆盖本脚本自己的上一次产物；脚本自身拒绝写 evals/）。
    proc = subprocess.run([sys.executable, str(ROOT / "scripts/e2e_dataset.py"), "map", "--dataset", str(PILOT_DATASET),
                           "--out", str(PILOT_MAPPING), "--overwrite"], cwd=ROOT, capture_output=True, text=True)
    if proc.returncode != 0:
        print(proc.stdout, proc.stderr, file=sys.stderr)
        return 1
    mapping = E2EMapping.model_validate_json(PILOT_MAPPING.read_text(encoding="utf-8"))

    # 当前 active 相关性投影（与 replay_runtime_retrieval 同一函数），并核对 hash。
    settings = Settings()
    collection = resolve_active_collection(settings)
    chunks = list(ChromaVectorStore(settings.vector_db_path, collection, create_if_missing=False).iter_all_chunks())
    projection = project_active_relevance(chunks, raw)
    projection_hash_ok = projection["active_relevance_projection_hash"] == a1["active_relevance_projection_hash"]

    # 交叉核对：新工具的组级结果应与直接调用 map_stable_dataset 完全一致（证明只是复用）。
    documents = load_source_documents(Path(settings.source_docs_path))
    legacy, _, _ = map_stable_dataset(EvaluationDatasetV2.model_validate(raw), documents, chunks, index_version="pilot")
    legacy_rows = {(g.question_id, g.group_id): (g.status, g.chunk_sets) for g in legacy.groups}
    tool_matches_map_stable_dataset = all(legacy_rows[(g.case_id, g.group_id)] == (g.status, g.chunk_sets)
                                          for g in mapping.groups) and len(legacy_rows) == len(mapping.groups)

    docs_by_id = {d.document_id: d for d in documents}
    alts = {(q["question_id"], g["group_id"]): g for q in raw["questions"] for g in q["required_evidence_groups"]}
    gold = EvaluationDatasetV2.model_validate(raw)
    gold_alts = {(q.question_id, g.group_id): g.alternatives for q in gold.questions for g in q.required_evidence_groups}
    per_question: dict[str, dict] = {}
    for g in mapping.groups:
        proj_group = next(x for x in projection["groups_by_question"][g.case_id] if x["group_id"] == g.group_id)
        new_ids = sorted({c for bundle in g.chunk_sets for c in bundle})
        proj_ids = proj_group["matched_chunk_ids"]
        new_usable = is_usable(g.status)
        same = new_usable == proj_group["indexable"] and new_ids == proj_ids
        row = {"group_id": g.group_id, "new_status": g.status, "new_usable": new_usable, "new_chunk_sets": g.chunk_sets,
               "projection_indexable": proj_group["indexable"], "projection_chunk_ids": proj_ids, "match": same}
        if not same:
            alternative = gold_alts[(g.case_id, g.group_id)][0]
            diag = diagnose(alternative, docs_by_id[alternative.document_id], chunks, g.chunk_sets, proj_ids)
            row["diff_type"] = classify(new_usable, new_ids, proj_ids, diag["chunks"])
            row["diagnostics"] = diag
        entry = per_question.setdefault(g.case_id, {"groups": []})
        entry["groups"].append(row)
    assert set(alts) == {(q, r["group_id"]) for q, e in per_question.items() for r in e["groups"]}

    a1_eligible = {qid: v["eligible"] for qid, v in a1["standard_ir"]["per_query"].items()}
    for qid, entry in per_question.items():
        entry["projection_eligible"] = bool(projection["qrels"][qid])
        entry["a1_eligible"] = a1_eligible[qid]
        entry["new_all_groups_usable"] = all(r["new_usable"] for r in entry["groups"])
        entry["new_any_group_usable"] = any(r["new_usable"] for r in entry["groups"])
        entry["match"] = all(r["match"] for r in entry["groups"])

    eligible = [q for q, e in per_question.items() if e["projection_eligible"]]
    diff_groups = [(q, r) for q, e in per_question.items() for r in e["groups"] if not r["match"]]
    counts: dict[str, int] = {}
    for _, r in diff_groups:
        counts[r["diff_type"]] = counts.get(r["diff_type"], 0) + 1
    summary = {
        "collection_name": mapping.collection_name,
        "index_signature": mapping.index_signature,
        # 映射文件现在自带实测索引身份，写进对照摘要便于 B3b 直接审计。
        "mapping_chunk_count": mapping.chunk_count,
        "mapping_chunk_id_hash": mapping.chunk_id_hash,
        "pilot_dataset_fingerprint": pilot.fingerprint,
        "projection_hash": projection["active_relevance_projection_hash"],
        "projection_hash_matches_a1": projection_hash_ok,
        "projection_eligible_matches_a1": all(per_question[q]["projection_eligible"] == a1_eligible[q] for q in a1_eligible),
        "projection_eligible_count": len(eligible),
        "projection_ineligible": sorted(q for q in per_question if q not in eligible),
        "new_tool_matches_map_stable_dataset": tool_matches_map_stable_dataset,
        "new_unusable_groups": sorted(f"{q}/{r['group_id']}:{r['new_status']}" for q, e in per_question.items()
                                      for r in e["groups"] if not r["new_usable"]),
        "eligible_questions_exact_match": sum(per_question[q]["match"] for q in eligible),
        "mismatched_questions": sorted({q for q, _ in diff_groups}),
        "mismatched_group_count": len(diff_groups),
        "total_group_count": len(mapping.groups),
        "diff_type_counts": counts,
        "gate_pass": not diff_groups and projection_hash_ok,
    }
    CHECK_PATH.write_text(json.dumps({"summary": summary, "per_question": per_question}, ensure_ascii=False,
                                     sort_keys=True, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["gate_pass"] else 2


if __name__ == "__main__":
    sys.exit(main())
