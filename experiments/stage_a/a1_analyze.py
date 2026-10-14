"""A1：汇总确定性检查、交叉校验与 Exact-vs-HNSW 对比，并冻结新基线 B0'。

只读取 artifacts/stage_a/ 与冻结的 evals/ 产物；不写入 evals/。
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for extra in (ROOT, ROOT / "src"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from experiments.run_phase15b1_dense_candidate_depth import compare_replay  # noqa: E402

OUT = ROOT / "artifacts/stage_a"
FROZEN_B0 = ROOT / "evals/evaluation_v3/phase15b_hybrid_candidate_v2/production_reranked_baseline.json"
E15C1 = ROOT / "evals/evaluation_v3/phase15c1_exact_dense_architecture_v1"
KEYS = ("P@5", "Recall@5", "MRR@5", "MAP@5", "nDCG@5")
# Gate：R@5 与 nDCG@5 相对 HNSW B0 的下降不得超过该值（绝对值）。
MAX_DROP = 0.01


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def determinism() -> dict:
    """10 个独立进程的 Top20 ID 与分数必须逐元素完全一致。"""

    runs = [read(OUT / f"A1_cand_run{i}.json") for i in range(1, 11)]
    base = runs[0]
    mismatches = []
    for i, run in enumerate(runs[1:], start=2):
        for qid in base["candidate_run"]:
            if run["candidate_run"][qid] != base["candidate_run"][qid]:
                mismatches.append({"run": i, "qid": qid, "kind": "ids"})
            elif run["candidate_scores"][qid] != base["candidate_scores"][qid]:
                mismatches.append({"run": i, "qid": qid, "kind": "scores"})
    return {
        "run_count": len(runs),
        "question_count": len(base["candidate_run"]),
        "identical": not mismatches,
        "mismatches": mismatches,
        "devices": sorted({r["device"] for r in runs}),
        "backends": sorted({r["backend"] for r in runs}),
    }


def crosscheck_15c1() -> dict:
    """与 Phase 15C.1 冻结的 exact20.json 逐题比较 ID 与分数（不作为 Gate）。"""

    frozen = read(E15C1 / "candidates/exact20.json")
    by_q: dict[str, list[dict]] = {}
    for row in frozen["rows"]:
        by_q.setdefault(row["question_id"], []).append(row)
    run = read(OUT / "A1_cand_run1.json")
    id_diff, score_diff = [], []
    max_abs = 0.0
    for qid, rows in by_q.items():
        rows.sort(key=lambda r: r["candidate_rank"])
        f_ids = [r["chunk_id"] for r in rows]
        f_scores = [r["similarity_score"] for r in rows]
        if f_ids != run["candidate_run"][qid]:
            id_diff.append(qid)
            continue
        if f_scores != run["candidate_scores"][qid]:
            score_diff.append(qid)
            max_abs = max(max_abs, max(abs(a - b) for a, b in zip(f_scores, run["candidate_scores"][qid])))
    return {
        "compared_questions": len(by_q),
        "id_mismatch_questions": id_diff,
        "score_mismatch_questions": score_diff,
        "max_abs_score_diff": max_abs,
        "identical": not id_diff and not score_diff,
    }


def exact_vs_hnsw() -> dict:
    """完整回放两次一致性，以及与冻结 HNSW B0 的逐题 / 指标对比。"""

    r1 = read(OUT / "A1_exact_replay_1.json")
    r2 = read(OUT / "A1_exact_replay_2.json")
    b0 = read(FROZEN_B0)
    qids = sorted(b0["final_run"])
    replay_diff = [q for q in qids if r1["final_run"][q] != r2["final_run"][q]]
    score_diff = [
        q
        for q in qids
        if [t["reranker_score"] for t in r1["per_query"][q]["top5"]]
        != [t["reranker_score"] for t in r2["per_query"][q]["top5"]]
    ]
    # compare_replay 要求 pipeline 字符串相同；这里仅在比较时覆盖它，其余字段（身份、
    # projection hash、逐题 Top5、指标）按原样比较。
    probe = dict(r1)
    probe["pipeline"] = b0["pipeline"]
    cmp_result = compare_replay(b0, probe, b0["active_relevance_projection_hash"], qids)
    diffs = {
        q: {"hnsw_b0_top5": b0["final_run"][q], "exact_top5": r1["final_run"][q]}
        for q in cmp_result["run_mismatches"]
    }
    delta = {k: r1["summary_at_5"][k] - b0["summary_at_5"][k] for k in KEYS}
    gate_ok = all(delta[k] >= -MAX_DROP for k in ("Recall@5", "nDCG@5"))
    return {
        "two_replays_top5_identical": not replay_diff,
        "two_replays_mismatch_questions": replay_diff,
        "two_replays_reranker_scores_identical": not score_diff,
        "compare_replay": cmp_result,
        "differing_questions": diffs,
        "summary_hnsw_b0": {k: b0["summary_at_5"][k] for k in KEYS},
        "summary_exact": {k: r1["summary_at_5"][k] for k in KEYS},
        "delta_exact_minus_hnsw": delta,
        "gate_max_drop": MAX_DROP,
        "gate_recall_ndcg_ok": gate_ok,
    }


def e15c1_metrics_check() -> dict:
    """E20 的 @5 指标应与本次 Exact 回放及 B0 相同（交叉校验，不作为 Gate）。"""

    e20 = read(E15C1 / "final_ir_metrics.json")["summary_at_5"]["20"]
    r1 = read(OUT / "A1_exact_replay_1.json")["summary_at_5"]
    return {"max_abs_diff_vs_15c1_E20": max(abs(e20[k] - r1[k]) for k in KEYS), "e20": {k: e20[k] for k in KEYS}}


def main() -> int:
    d = determinism()
    write(OUT / "A1_determinism.json", d)
    x = crosscheck_15c1()
    write(OUT / "A1_crosscheck_15c1.json", {**x, "e20_metrics": e15c1_metrics_check()})
    v = exact_vs_hnsw()
    write(OUT / "A1_exact_vs_hnsw.json", v)
    print(json.dumps({"determinism": d["identical"], "crosscheck_15c1": x["identical"], "replays_identical": v["two_replays_top5_identical"], "delta": v["delta_exact_minus_hnsw"], "gate_ok": v["gate_recall_ndcg_ok"], "differing": list(v["differing_questions"])}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
