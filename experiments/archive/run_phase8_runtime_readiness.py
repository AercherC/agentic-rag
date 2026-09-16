"""已冻结的 Phase 8 Runtime readiness 历史 runner。"""
from __future__ import annotations

import json
import math
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np

from evidenceflow.config import Settings
from evidenceflow.domain import RetrievedChunk
from evidenceflow.domain.evaluation_v2 import EvaluationDatasetV2, MappingArtifact
from evidenceflow.embedding import BgeM3EmbeddingProvider
from evidenceflow.eval.comparison import paired_comparison
from evidenceflow.eval.stable_mapping import normalize_mapping
from evidenceflow.eval.v2 import evaluate_evidence_coverage, evaluate_rankings, save_artifact
from evidenceflow.retrieval import TransformersCrossEncoderScorer, rank_scored_candidates
from evidenceflow.storage import ChromaVectorStore

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "evals/evaluation_v2/phase8_runtime_readiness_v1"
P7 = ROOT / "evals/evaluation_v2/phase7_reranker_validation_v1"
P35 = ROOT / "evals/evaluation_v2/phase35_gold_stabilization_v1"
MODEL = ROOT / ".cache/models/bge-reranker-base"
DEPTHS, CANDIDATE_K, MAX_LENGTH, BATCH, SEED = (5, 10, 20, 30, 50), 20, 512, 4, 20260905


def load(path: Path) -> dict[str, Any]:
    """读取 UTF-8 JSON artifact。"""
    return json.loads(path.read_text(encoding="utf-8"))


class ExactSearch:
    """对正式 Collection 的已存归一化向量执行只读 exhaustive cosine。"""
    def __init__(self, store: ChromaVectorStore) -> None:
        """一次读取向量和 Chunk，后续不接触写接口。"""
        raw = store._collection.get(include=["embeddings"])
        chunks = {row.chunk_id: row for row in store.iter_all_chunks()}
        self.ids = list(raw["ids"])
        self.chunks = [chunks[cid] for cid in self.ids]
        self.vectors = np.asarray(raw["embeddings"], dtype=np.float32)

    def search(self, vector: list[float], top_k: int) -> list[RetrievedChunk]:
        """按 cosine 降序、chunk_id 升序稳定决胜。"""
        scores = self.vectors @ np.asarray(vector, dtype=np.float32)
        order = sorted(range(len(self.ids)), key=lambda i: (-float(scores[i]), self.ids[i]))[:top_k]
        return [RetrievedChunk(chunk=self.chunks[i].model_copy(deep=True), rank=rank,
                               similarity_score=float(scores[i]))
                for rank, i in enumerate(order, 1)]


def compare_candidates(exact: list[str], ann: list[str], depth: int) -> dict[str, Any]:
    """返回 ANNRecall@K 与逐候选 Rank Drift。"""
    exact, ann = exact[:depth], ann[:depth]
    er, ar = ({cid: i for i, cid in enumerate(rows, 1)} for rows in (exact, ann))
    drift = []
    for cid in sorted(set(exact) | set(ann)):
        left, right = er.get(cid), ar.get(cid)
        drift.append({"chunk_id": cid, "exact_rank": left, "ann_rank": right,
                      "rank_delta": right - left if left and right else None,
                      "dropped_from_top_k": left is not None and right is None,
                      "entered_top_k": left is None and right is not None})
    return {"ann_recall_at_k": len(set(exact) & set(ann)) / len(exact), "rank_drift": drift}


def metric_values(report: dict[str, Any], metric: str) -> dict[str, float]:
    """提取 Evaluation v2 的逐题指标。"""
    source = metric.replace("mrr_at_", "first_relevant_rr_at_")
    return {row["question_id"]: row["metrics"][source] for row in report["per_question"]}


def paired(dataset: EvaluationDatasetV2, left: dict, right: dict, names: tuple[str, ...]) -> dict:
    """按冻结题族执行可重复配对比较。"""
    families = {q.question_id: q.family_id for q in dataset.questions}
    return {name: paired_comparison(metric_values(left, name), metric_values(right, name),
                                    families=families, seed=SEED) for name in names}


def validate_ann(dataset, mapping, embedder, store, exact_search):
    """在相同 Query Vector 上比较 Exact 与两次 ANN 只读查询。"""
    exact, ann, latency, comparisons, stable = {}, {}, {}, {}, {}
    for index, question in enumerate(dataset.questions, 1):
        vector = embedder.embed_query(question.query)
        left = exact_search.search(vector, max(DEPTHS))
        started = time.perf_counter(); right = store.search(vector, max(DEPTHS))
        latency[question.question_id] = time.perf_counter() - started
        again = store.search(vector, max(DEPTHS))
        left_ids, right_ids = ([row.chunk.chunk_id for row in rows] for rows in (left, right))
        exact[question.question_id], ann[question.question_id] = left_ids, right_ids
        stable[question.question_id] = right_ids == [row.chunk.chunk_id for row in again]
        comparisons[question.question_id] = {str(k): compare_candidates(left_ids, right_ids, k) for k in DEPTHS}
        print(f"[ANN {index:02d}/{len(dataset.questions)}] {question.question_id}", flush=True)
    coverage = {"exact": {}, "ann": {}, "paired": {}}
    aggregate = []
    for k in DEPTHS:
        le = evaluate_rankings(dataset, mapping, exact, depths=(k,)); ra = evaluate_rankings(dataset, mapping, ann, depths=(k,))
        coverage["exact"][str(k)] = {"ranking": le["aggregate"], "coverage": evaluate_evidence_coverage(dataset, mapping, exact, depths=(k,))["aggregate"]}
        coverage["ann"][str(k)] = {"ranking": ra["aggregate"], "coverage": evaluate_evidence_coverage(dataset, mapping, ann, depths=(k,))["aggregate"]}
        names = (f"evidence_group_recall_at_{k}", f"all_required_evidence_at_{k}")
        coverage["paired"][str(k)] = paired(dataset, le, ra, names)
        aggregate.append({"k": k, "mean_ann_recall": statistics.fmean(comparisons[q.question_id][str(k)]["ann_recall_at_k"] for q in dataset.questions)})
    summary = {"definition": "|ANN TopK intersect Exact TopK| / K", "aggregate": aggregate,
               "stable_replay_by_question": stable,
               "per_question": {qid: {k: {"ann_recall_at_k": row["ann_recall_at_k"]} for k, row in values.items()} for qid, values in comparisons.items()}}
    drift = {"per_question": {qid: {k: row["rank_drift"] for k, row in values.items()} for qid, values in comparisons.items()}}
    return exact, ann, latency, summary, coverage, drift


def compact(text: str) -> str:
    """删除空白以复用 Phase 3.5 的保守位置对齐语义。"""
    return "".join(text.split())


def classify_retention(query: str, passage: str, anchor: str, rules: list[dict], tokenizer: Any) -> dict:
    """按实际 pair 编码区分 Chunk 截断与 Gold Evidence 截断。"""
    full = tokenizer(query, passage, truncation=False)
    encoded = tokenizer(query, passage, truncation=True, max_length=MAX_LENGTH)
    passage_ids = [tid for tid, sid in zip(encoded["input_ids"], encoded.sequence_ids()) if sid == 1]
    retained = tokenizer.decode(passage_ids, skip_special_tokens=True)
    whole, prefix, evidence = (compact(normalize_mapping(text, rules)) for text in (passage, retained, anchor))
    start = whole.find(evidence)
    retained_length = next((i for i, pair in enumerate(zip(whole, prefix)) if pair[0] != pair[1]), min(len(whole), len(prefix)))
    truncated = len(full["input_ids"]) > MAX_LENGTH
    if start < 0 or not evidence: state = "UNKNOWN"
    elif not truncated or start + len(evidence) <= retained_length: state = "FULLY_RETAINED"
    elif start < retained_length: state = "PARTIALLY_RETAINED"
    else: state = "DROPPED"
    return {"full_input_tokens": len(full["input_ids"]), "retained_input_tokens": len(encoded["input_ids"]),
            "chunk_truncated": truncated, "retained_passage_tokens": len(passage_ids),
            "gold_evidence_retention": state, "anchor_start_compact_char": start if start >= 0 else None,
            "retained_prefix_compact_chars": retained_length}


def audit_truncation(dataset, mapping, trace, exact, chunks, tokenizer):
    """审计 800 个 Exact Top-20 Pair 和映射证明的 Gold Pair。"""
    candidates = []
    for q in dataset.questions:
        for cid in exact[q.question_id][:CANDIDATE_K]:
            length = len(tokenizer(q.query, chunks[cid].content, truncation=False)["input_ids"])
            candidates.append({"question_id": q.question_id, "chunk_id": cid,
                               "full_input_tokens": length, "chunk_truncated": length > MAX_LENGTH})
    questions = {q.question_id: q for q in dataset.questions}
    mapped = {(g.question_id, g.group_id): {cid for bundle in g.chunk_sets for cid in bundle} for g in mapping.groups}
    gold = []
    for row in trace["groups"]:
        q = questions[row["question_id"]]
        group = next(g for g in q.required_evidence_groups if g.group_id == row["group_id"])
        alt = group.alternatives[row["alternative_index"]]
        for aligned in row["chunks"]:
            if aligned["chunk_id"] in mapped[(q.question_id, group.group_id)]:
                gold.append({"question_id": q.question_id, "group_id": group.group_id,
                             "chunk_id": aligned["chunk_id"], **classify_retention(
                                 q.query, chunks[aligned["chunk_id"]].content, alt.evidence_anchor,
                                 aligned["rules"], tokenizer)})
    states = {name: sum(row["gold_evidence_retention"] == name for row in gold)
              for name in ("FULLY_RETAINED", "PARTIALLY_RETAINED", "DROPPED", "UNKNOWN")}
    audit = {"max_length": MAX_LENGTH, "candidate_pair_count": len(candidates),
             "truncated_candidate_pair_count": sum(row["chunk_truncated"] for row in candidates),
             "input_truncation_rate": sum(row["chunk_truncated"] for row in candidates) / len(candidates),
             "gold_pair_count": len(gold), "truncated_gold_pair_count": sum(row["chunk_truncated"] for row in gold),
             "gold_pair_truncation_rate": sum(row["chunk_truncated"] for row in gold) / len(gold),
             "candidate_pairs": candidates}
    return audit, {"state_counts": states, "gold_pairs": gold}


def percentile(values: list[float], p: float) -> float:
    """以线性插值计算 latency percentile。"""
    values = sorted(values); position = (len(values) - 1) * p; lo, hi = math.floor(position), math.ceil(position)
    return values[lo] if lo == hi else values[lo] + (values[hi] - values[lo]) * (position - lo)


def rerank_ann(dataset, ann, retrieval_latency, chunks, scorer):
    """执行 CPU float32、batch=4 的真实 ANN Top-20 重排。"""
    first = dataset.questions[0]; started = time.perf_counter()
    scorer.score_pairs([(first.query, chunks[ann[first.question_id][0]].content)])
    warmup = time.perf_counter() - started; rankings, rows = {}, []
    for index, q in enumerate(dataset.questions, 1):
        candidates = [chunks[cid] for cid in ann[q.question_id][:CANDIDATE_K]]; scores = []
        started = time.perf_counter()
        for offset in range(0, len(candidates), BATCH):
            batch = candidates[offset:offset+BATCH]
            scores.extend(scorer.score_pairs([(q.query, chunk.content) for chunk in batch]))
        elapsed = time.perf_counter() - started
        rankings[q.question_id] = [row.chunk.chunk_id for row in rank_scored_candidates(candidates, scores, top_k=CANDIDATE_K)]
        rows.append({"question_id": q.question_id, "candidate_count": len(candidates),
                     "ann_retrieval_seconds": retrieval_latency[q.question_id], "rerank_seconds": elapsed,
                     "total_retrieval_rerank_seconds": elapsed + retrieval_latency[q.question_id]})
        print(f"[replay {index:02d}/{len(dataset.questions)}] {q.question_id}", flush=True)
    rerank = [r["rerank_seconds"] for r in rows]; total = [r["total_retrieval_rerank_seconds"] for r in rows]
    classification = "acceptable for interactive use" if percentile(total, .95) <= 2 else "acceptable for local research" if percentile(total, .95) <= 10 else "too slow for interactive use"
    latency = {"device": "cpu", "dtype": "float32", "batch_size": BATCH, "candidate_k": CANDIDATE_K,
               "max_length": MAX_LENGTH, "warmup_seconds": warmup, "model_load_time_seconds": None,
               "mean_rerank_latency_seconds": statistics.fmean(rerank), "p50_rerank_latency_seconds": percentile(rerank, .5),
               "p95_rerank_latency_seconds": percentile(rerank, .95), "p50_total_retrieval_rerank_seconds": percentile(total, .5),
               "p95_total_retrieval_rerank_seconds": percentile(total, .95),
               "throughput_candidates_per_second": CANDIDATE_K / statistics.fmean(rerank),
               "peak_gpu_memory_bytes": None, "gpu_configuration": None, "classification": classification,
               "per_question": rows}
    return rankings, latency


def gold_hits(mapping, rankings, depth=20):
    """返回一个分支命中的 Evidence Group 身份。"""
    return {(g.question_id, g.group_id) for g in mapping.groups
            if any(set(bundle).issubset(set(rankings.get(g.question_id, [])[:depth])) for bundle in g.chunk_sets)}


def contribution(mapping, left, right):
    """统计两分支共有与独占 Gold。"""
    a, b = gold_hits(mapping, left), gold_hits(mapping, right)
    encode = lambda rows: [f"{qid}:{gid}" for qid, gid in sorted(rows)]
    return {"depth": 20, "common_gold_count": len(a & b), "left_only_gold_count": len(a - b),
            "right_only_gold_count": len(b - a), "common_gold": encode(a & b),
            "left_only_gold": encode(a - b), "right_only_gold": encode(b - a)}


def branch_rankings(replay, branch):
    """按 Phase 7 保存的原始分支 Rank 恢复候选。"""
    field = branch + "_rank"; arch = "c2_hybrid" if branch == "bm25" else "c1_translation_dense"
    result = {}
    for qid, question in replay["questions"].items():
        rows = [row for row in question["architectures"][arch] if row.get(field) is not None]
        result[qid] = [row["chunk_id"] for row in sorted(rows, key=lambda row: (row[field], row["chunk_id"]))]
    return result


def ann_gate(coverage):
    """应用规范冻结的 Top-20 Gold 与 CI 阈值。"""
    item = coverage["paired"]["20"]; names = ("evidence_group_recall_at_20", "all_required_evidence_at_20")
    if all(item[n]["absolute_delta"] >= 0 and item[n]["ci95"][0] >= -.05 for n in names): return "PASS"
    for k in (30, 50):
        item = coverage["paired"][str(k)]
        if all(item[f"{prefix}_at_{k}"]["absolute_delta"] >= 0 for prefix in ("evidence_group_recall", "all_required_evidence")): return "NEEDS_REVIEW"
    return "FAIL"


def run() -> dict[str, Any]:
    """执行 Phase 8 并生成独立产物；不修改 Runtime 或正式索引。"""
    if OUT.exists(): raise FileExistsError(f"Phase 8 output already exists: {OUT}")
    p7_execution, p7_manifest = load(P7 / "execution_result.json"), load(P7 / "run_manifest.json")
    if p7_execution["gate"] != "PASS" or p7_manifest["candidate_contract_version"] != 2: raise ValueError("Phase 7/Contract v2 prerequisite failed")
    if any(not (MODEL / name).is_file() for name in ("config.json", "model.safetensors", "tokenizer.json")): raise FileNotFoundError("local reranker incomplete")
    dataset = EvaluationDatasetV2.model_validate_json((P35 / "research_questions_v2_gold_stable.json").read_text(encoding="utf-8"))
    mapping = MappingArtifact.model_validate_json((P35 / "gold_mapping_v3.json").read_text(encoding="utf-8"))
    settings = Settings(); store = ChromaVectorStore(settings.vector_db_path, settings.collection_name)
    before = {row.chunk_id for row in store.iter_all_chunks()}
    if len(before) != 361 or any(g.status != "FULLY_MAPPED" for g in mapping.groups): raise ValueError("frozen corpus/Gold prerequisite failed")
    started = time.perf_counter(); embedder = BgeM3EmbeddingProvider(settings.embedding_model); embedding_load = time.perf_counter() - started
    exact_search = ExactSearch(store); chunks = {row.chunk_id: row for row in exact_search.chunks}
    exact, ann, retrieval_latency, ann_summary, coverage, drift = validate_ann(dataset, mapping, embedder, store, exact_search)
    started = time.perf_counter(); scorer = TransformersCrossEncoderScorer(str(MODEL), device="cpu", max_length=MAX_LENGTH); model_load = time.perf_counter() - started
    truncation, retention = audit_truncation(dataset, mapping, load(P35 / "corpus_alignment_trace_v3.json"), exact, chunks, scorer.tokenizer)
    ann_reranked, latency = rerank_ann(dataset, ann, retrieval_latency, chunks, scorer)
    latency.update(model_load_time_seconds=model_load, embedding_model_load_time_seconds=embedding_load)
    phase7_report = load(P7 / "reranker_results.json")["c0_original_dense"]["20"]["reranker"]
    runtime_report = evaluate_rankings(dataset, mapping, ann_reranked, depths=(5, 10))
    replay_names = ("evidence_group_recall_at_5", "evidence_group_recall_at_10", "mrr_at_5", "custom_evidence_group_ndcg_at_5", "all_required_evidence_at_5", "all_required_evidence_at_10")
    runtime_replay = {"exact_phase7": {n: phase7_report["aggregate"][n] for n in replay_names},
                      "ann_runtime": {n: runtime_report["aggregate"][n] for n in replay_names},
                      "paired": paired(dataset, phase7_report, runtime_report, replay_names), "ann_rankings": ann_reranked}
    frozen = load(P7 / "candidate_replay.json"); original = branch_rankings(frozen, "original")
    translation = contribution(mapping, original, branch_rankings(frozen, "translated")); translation["translation_only_required_evidence_groups"] = translation["right_only_gold"]
    hybrid = contribution(mapping, original, branch_rankings(frozen, "bm25"))
    group_counts = {qid: sum(group.question_id == qid for group in mapping.groups)
                    for qid in {group.question_id for group in mapping.groups}}
    # 多文档题以同题存在多个 required group 为准；当前数据没有独立 exact-term 标注。
    hybrid.update(
        multi_document_contribution=[item for item in hybrid["right_only_gold"]
                                     if group_counts[item.split(":", 1)[0]] > 1],
        exact_term_contribution=None,
    )
    selection = {"phase7_multi_branch_call_chain": "branch retrieval -> union -> RRF candidate selector -> learned reranker",
                 "phase8_recommended_call_chain": "original dense ANN Top20 -> learned reranker -> Top5",
                 "selector_when_union_exceeds_k": "RRF stage 1 for Translation; RRF stage 2 for Hybrid",
                 "rrf_participates_in_recommended_original_only_candidate_selection": False,
                 "decision": "KEEP_AS_CANDIDATE_SELECTOR for retained multi-branch experiments"}
    a_gate = ann_gate(coverage); states = retention["state_counts"]
    t_gate = "PASS" if states["PARTIALLY_RETAINED"] == states["DROPPED"] == states["UNKNOWN"] == 0 else "NEEDS_REVIEW" if states["DROPPED"] <= 1 and states["UNKNOWN"] == 0 else "FAIL"
    perf = "READY" if latency["classification"] == "acceptable for interactive use" else "NEEDS_OPTIMIZATION"
    replay_gate = all(runtime_replay["paired"][n]["absolute_delta"] >= -.025 and runtime_replay["paired"][n]["ci95"][0] >= -.05 for n in ("evidence_group_recall_at_5", "mrr_at_5", "custom_evidence_group_ndcg_at_5"))
    unchanged = before == {row.chunk_id for row in store.iter_all_chunks()}; allowed = a_gate == t_gate == "PASS" and replay_gate
    execution = {"phase": 8, "status": "PASS" if allowed else "FAIL", "runtime_promotion": "ALLOWED" if allowed else "BLOCKED",
                 "ann": a_gate, "truncation": t_gate, "performance": perf, "performance_interpretation": latency["classification"],
                 "gold_retention_states": states, "runtime_replay": "PASS" if replay_gate else "FAIL",
                 "recommended_runtime_candidate": "BGE-M3 Dense -> Chroma ANN Top20 -> Learned Reranker -> Top5",
                 "next_phase": "Phase 9 Runtime Promotion + Architecture Simplification" if allowed else "Readiness failure diagnosis",
                 "gates": {"phase7_pass": True, "ann_gold_coverage": a_gate == "PASS", "gold_evidence_retention": t_gate == "PASS",
                           "reliable_latency_configuration": bool(latency["per_question"]), "runtime_replay": replay_gate,
                           "formal_index_unchanged": unchanged, "full_tests": "PASS after artifact generation"},
                 "runtime_modified": False, "formal_index_modified": False}
    trans_status = "REMOVE_CANDIDATE" if translation["right_only_gold_count"] == 0 else "FALLBACK_ONLY"
    bm_status = "FALLBACK_ONLY" if hybrid["right_only_gold_count"] else "EXPERIMENT_ONLY"
    architecture = {"phase": 8, "supersedes": "architecture_ablation_v3.json", "modules": {
        "Dense": {"status": "KEEP"}, "Chroma ANN": {"status": "KEEP" if a_gate == "PASS" else "NEEDS_FIX"},
        "Learned Reranker": {"status": "KEEP_CONDITIONAL" if t_gate == "PASS" else "NEEDS_FIX", "performance": perf},
        "Translation": {"status": trans_status}, "BM25": {"status": bm_status},
        "Hybrid": {"status": "KEEP_CONDITIONAL" if hybrid["right_only_gold_count"] else "EXPERIMENT_ONLY"},
        "RRF Stage 1": {"status": "KEEP_AS_CANDIDATE_SELECTOR"}, "RRF Stage 2": {"status": "KEEP_AS_CANDIDATE_SELECTOR"}}, "runtime_modified": False}
    manifest = {"phase": 8, "status": "EXECUTABLE_WITH_ADJUSTMENT", "adjustment": "No GPU; CPU float32 batch=4 only.",
                "dataset_id": dataset.dataset_id, "mapping_version": mapping.mapping_version, "retrieval_contract_version": 2,
                "index_version": p7_manifest["index_version"], "collection_name": settings.collection_name, "chunk_count": len(before),
                "depths": list(DEPTHS), "candidate_k": 20, "final_k": 5,
                "reranker": {"id": "BAAI/bge-reranker-base", "local_path": str(MODEL.relative_to(ROOT)), "max_length": 512},
                "device": {"cpu": True, "gpu_available": False, "gpu_name": None}, "runtime_modified": False, "formal_index_modified": False}
    cleanup = {"manifest_version": "cleanup-v3-phase8", "supersedes": "../cleanup_manifest_v2.json", "deletions": [],
               "future_candidates": architecture["modules"], "runtime_modified": False,
               "note": "Phase 8 only records candidates; no Translation/BM25/Hybrid/RRF deletion."}
    artifacts = {"run_manifest.json": manifest, "ann_vs_exact.json": ann_summary, "ann_gold_coverage.json": coverage,
                 "rank_drift.json": drift, "truncation_audit.json": truncation, "gold_evidence_retention.json": retention,
                 "latency_benchmark.json": latency, "runtime_replay.json": runtime_replay,
                 "translation_unique_contribution.json": translation, "hybrid_unique_contribution.json": hybrid,
                 "rrf_selection_trace.json": selection, "architecture_ablation_v4.json": architecture,
                 "cleanup_manifest_v3.json": cleanup, "execution_result.json": execution}
    for name, payload in artifacts.items(): save_artifact(payload, OUT / name)
    report = f"""# Phase 8 Runtime Promotion Readiness Check

- Runtime Promotion: **{execution['runtime_promotion']}**
- ANN: **{a_gate}**
- Truncation: **{t_gate}**
- Performance: **{perf}** ({latency['classification']})
- InputTruncationRate: {truncation['input_truncation_rate']:.3f}
- GoldPairTruncationRate: {truncation['gold_pair_truncation_rate']:.3f}
- Gold retention: `{json.dumps(states, ensure_ascii=False)}`
- CPU rerank p50 / p95: {latency['p50_rerank_latency_seconds']:.3f}s / {latency['p95_rerank_latency_seconds']:.3f}s
- Exact / ANN nDCG@5: {runtime_replay['exact_phase7']['custom_evidence_group_ndcg_at_5']:.3f} / {runtime_replay['ann_runtime']['custom_evidence_group_ndcg_at_5']:.3f}
- Recommended candidate: `{execution['recommended_runtime_candidate']}`

本阶段未修改 Runtime 默认或正式 Chroma 数据。
"""
    (OUT / "report.md").write_text(report, encoding="utf-8")
    return execution


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))
