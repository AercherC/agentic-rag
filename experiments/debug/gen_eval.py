"""临时评测：生成层（Generation-level）抽检。

项目目前所有评测都停留在检索层（Hit@K / MRR / nDCG），
没有任何证据说明"召回到之后，LLM 是否真的用对了、引用是否落在 Gold 页"。
本脚本补这一层，只读运行，不写 evals/。
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# 历史数据集仍保存在项目根的 evals 中。
ROOT = Path(__file__).resolve().parents[2]
os.environ["HF_HOME"] = str((ROOT / ".cache" / "huggingface").resolve())
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
sys.path.insert(0, str(ROOT / "src"))

from evidenceflow.config import Settings  # noqa: E402
from evidenceflow.runtime import build_rag_service  # noqa: E402

QUESTIONS = json.loads(
    (ROOT / "evals/research_retrieval/research_questions_v1.json").read_text(encoding="utf-8")
)["questions"]
BASELINE = json.loads(
    (ROOT / "evals/research_retrieval/research_dense_baseline_v1.json").read_text(encoding="utf-8")
)["question_results"]
RANK = {row["question_id"]: row["gold_group_rank"] for row in BASELINE}
FAIL = {row["question_id"]: row["primary_failure"] for row in BASELINE}
BY_ID = {q["question_id"]: q for q in QUESTIONS}


def pick() -> list[str]:
    """按检索层表现分层抽样，覆盖易/中/难/多文档/召回失败五档。"""

    hit1 = [q for q in QUESTIONS if RANK.get(q["question_id"]) == 1]
    mid = [q for q in QUESTIONS if isinstance(RANK.get(q["question_id"]), int) and 2 <= RANK[q["question_id"]] <= 5]
    multi = [q for q in QUESTIONS if q["question_type"] == "multi_document"]
    missed = [q for q in QUESTIONS if FAIL.get(q["question_id"]) == "candidate_retrieval_failure"]
    chosen = [q["question_id"] for q in (hit1[:5] + mid[:4] + multi[:4] + missed[:3])]
    # 去重并保持顺序
    return list(dict.fromkeys(chosen))


def overlap(a_start, a_end, b_start, b_end) -> bool:
    """判断两个页码区间是否相交；任一端为 None 视为不相交。"""

    if None in (a_start, a_end, b_start, b_end):
        return False
    return not (a_end < b_start or b_end < a_start)


def main() -> None:
    settings = Settings()
    service = build_rag_service(settings)
    ids = pick()
    print(f"抽样 {len(ids)} 题，开始生成层评测\n")

    rows = []
    for qid in ids:
        q = BY_ID[qid]
        gold = q["gold_sources"]
        try:
            ans = service.answer(q["query"])
            status = "ANSWERED" if ans.sources or ans.answer else "INSUFFICIENT_CONTEXT"
            cited = [
                {
                    "file_name": s.file_name,
                    "page_start": s.page_start,
                    "page_end": s.page_end,
                }
                for s in ans.sources
            ]
            err = None
        except Exception as exc:  # noqa: BLE001
            status, cited, err = "ERROR", [], f"{type(exc).__name__}: {exc}"

        # 引用是否命中 Gold（同一文档 且 页码区间相交）
        gold_hit = any(
            c["file_name"] == g["file_name"]
            and overlap(c["page_start"], c["page_end"], g["page_start"], g["page_end"])
            for c in cited
            for g in gold
        )
        doc_hit = any(c["file_name"] in {g["file_name"] for g in gold} for c in cited)

        rows.append(
            {
                "question_id": qid,
                "type": q["question_type"],
                "retrieval_rank": RANK.get(qid),
                "retrieval_failure": FAIL.get(qid),
                "status": status,
                "cited_count": len(cited),
                "cited": cited,
                "gold_files": [g["file_name"] for g in gold],
                "gold_pages": [[g["page_start"], g["page_end"]] for g in gold],
                "citation_gold_hit": gold_hit,
                "citation_doc_hit": doc_hit,
                "error": err,
            }
        )
        print(
            f"[{qid}] type={q['question_type']:<20} ret_rank={str(RANK.get(qid)):<5} "
            f"status={status:<20} cited={len(cited)} doc_hit={doc_hit} gold_hit={gold_hit}"
        )

    out = ROOT / "tmp" / "generation_eval_result.json"
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    n = len(rows)
    answered = [r for r in rows if r["status"] == "ANSWERED"]
    print("\n" + "=" * 72)
    print(f"样本 {n} 题")
    print(f"  生成成功（有引用）      : {len(answered)}/{n}")
    print(f"  引用命中 Gold 文档      : {sum(r['citation_doc_hit'] for r in rows)}/{n}")
    print(f"  引用命中 Gold 文档+页码 : {sum(r['citation_gold_hit'] for r in rows)}/{n}")
    print(f"  运行错误                : {sum(1 for r in rows if r['status'] == 'ERROR')}/{n}")

    by_rank = [r for r in rows if isinstance(r["retrieval_rank"], int) and r["retrieval_rank"] <= 5]
    if by_rank:
        ok = sum(r["citation_gold_hit"] for r in by_rank)
        print(f"\n  检索已命中 Top-5 的 {len(by_rank)} 题中，引用落到 Gold 页的有 {ok} 题")
        lost = [r["question_id"] for r in by_rank if not r["citation_gold_hit"]]
        if lost:
            print(f"  检索命中但引用没落到 Gold 页: {lost}")


if __name__ == "__main__":
    main()
