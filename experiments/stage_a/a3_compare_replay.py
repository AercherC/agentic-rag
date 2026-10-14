"""A3 Gate：经 lifecycle active 指针的线上回放，与冻结的 B0'（A1_exact_baseline.json）逐题比较。

只读取 artifacts/stage_a/ 的回放产物；比较内容：40 题 Top5 chunk ID 逐题一致、Dense Top20 候选 ID 与
分数逐元素一致、5 项指标与 B0' 一致、身份字段一致。结果写入 A3_replay_vs_b0prime.json。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for extra in (ROOT, ROOT / "src"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from experiments.run_phase15b1_dense_candidate_depth import compare_replay  # noqa: E402

OUT = ROOT / "artifacts/stage_a"
BASELINE = OUT / "A1_exact_baseline.json"
SUMMARY_KEYS = ("P@5", "Recall@5", "MRR@5", "MAP@5", "nDCG@5")


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def baseline_sha_ok() -> bool:
    """基线文件的实际 sha256 必须与 A1 冻结时记录的一致，防止对比到被改过的基线。"""

    recorded = (OUT / "A1_exact_baseline.sha256").read_text(encoding="utf-8").split()[0]
    return hashlib.sha256(BASELINE.read_bytes()).hexdigest() == recorded


def compare(baseline: dict, replay: dict) -> dict:
    """逐题比较 Top5 与 Top20 候选（ID 与分数），并复用 compare_replay 的身份校验。"""

    question_ids = sorted(baseline["final_run"])
    base_run, new_run = baseline["final_run"], replay["final_run"]
    top5_mismatches = [q for q in question_ids if list(base_run[q]) != list(new_run.get(q, []))]
    cand_id_mismatches = [
        q for q in question_ids if list(baseline["candidate_run"][q]) != list(replay["candidate_run"].get(q, []))
    ]
    cand_score_mismatches = [
        q
        for q in question_ids
        if baseline["candidate_scores"][q] != replay["candidate_scores"].get(q)
    ]
    # 精排分数同样逐题比较：Top5 ID 一致但分数漂移也应暴露。
    rerank_score_mismatches = [
        q
        for q in question_ids
        if [r["reranker_score"] for r in baseline["per_query"][q]["top5"]]
        != [r["reranker_score"] for r in replay["per_query"][q]["top5"]]
    ]
    summary_diff = {
        key: replay["summary_at_5"][key] - baseline["summary_at_5"][key] for key in SUMMARY_KEYS
    }
    summary_equal = all(math.isclose(v, 0.0, abs_tol=1e-12) for v in summary_diff.values())
    base_compare = compare_replay(
        baseline, replay, baseline.get("active_relevance_projection_hash"), question_ids
    )
    return {
        "compared_question_count": len(question_ids),
        "top5_mismatches": top5_mismatches,
        "top20_candidate_id_mismatches": cand_id_mismatches,
        "top20_candidate_score_mismatches": cand_score_mismatches,
        "reranker_score_mismatches": rerank_score_mismatches,
        "summary_baseline": {k: baseline["summary_at_5"][k] for k in SUMMARY_KEYS},
        "summary_replay": {k: replay["summary_at_5"][k] for k in SUMMARY_KEYS},
        "summary_diff": summary_diff,
        "summary_equal": summary_equal,
        "compare_replay": base_compare,
        "replay_collection_name": replay["collection_name"],
        "replay_device": replay["device"],
        "replay_backend": replay["backend"],
        "baseline_sha256_ok": baseline_sha_ok(),
        "parity": (
            baseline_sha_ok()
            and not top5_mismatches
            and not cand_id_mismatches
            and not cand_score_mismatches
            and summary_equal
            and base_compare["parity"]
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", type=Path, default=OUT / "A3_replay_runtime.json")
    parser.add_argument("--output", type=Path, default=OUT / "A3_replay_vs_b0prime.json")
    args = parser.parse_args()
    result = compare(read(BASELINE), read(args.replay))
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({k: result[k] for k in ("parity", "top5_mismatches", "top20_candidate_id_mismatches",
                                              "top20_candidate_score_mismatches", "reranker_score_mismatches",
                                              "summary_equal", "baseline_sha256_ok", "replay_collection_name", "replay_device")},
                     ensure_ascii=False))
    return 0 if result["parity"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
