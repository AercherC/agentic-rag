# -*- coding: utf-8 -*-
"""B2：验证 RunTrace 挂钩不改变线上检索，且 trace 记录的 ID 与分数与冻结基线逐位一致。

做法（CPU、离线，不调用 LLM）：
1. 用 ``runtime.build_retriever(settings)`` 构建**一个**线上 retriever（与 ``replay_runtime_retrieval.py``
   相同的装配路径），读取同样的 40 道 canonical 题（``research_questions_v1.json`` 的 ``english_query``）。
2. 对每题：先在 ``trace_recorder=None`` 下 ``retrieve``；再把同一个 retriever 的公开属性
   ``trace_recorder`` 换成 ``RunTraceRecorder``，``start_run`` 后再 ``retrieve`` 一次，然后把属性恢复为 None。
   （``DenseRerankerRetriever.trace_recorder`` 本就是公开属性，未为实验新增任何接口。）
3. 断言：
   (a) 两次的 Top5 ``chunk_id``、``similarity_score``、``reranker_score``、``candidate_rank`` 逐位相等；
   (b) trace 中 ``retrieval.dense.completed`` 的 Top20 ``chunk_id`` / ``similarity_score`` 序列与
       ``artifacts/stage_a/A1_exact_baseline.json`` 的 ``candidate_run[q]`` / ``candidate_scores[q]`` 逐位相等；
   (c) ``rerank.completed`` 的按 rank 记录与该次 Top5 一致。
4. 结果写 ``artifacts/stage_b/B2_trace_parity.json``（键排序）。只读 A1 基线，不写它。

退出码：0 = 全部一致；1 = 存在不一致；2 = 设备不是 cpu。
文件名不以 ``test_`` 开头，不会被 pytest 收集。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
for extra in (ROOT, ROOT / "src"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from evidenceflow.config import Settings  # noqa: E402
from evidenceflow.model_device import resolve_model_device  # noqa: E402
from evidenceflow.runtime import build_retriever, resolve_active_collection  # noqa: E402
from evidenceflow.tracing import RunTraceRecorder  # noqa: E402
from experiments.run_production_reranked_baseline_replay import (  # noqa: E402
    QUESTIONS_V1_PATH,
    configure_offline_cache,
    read_json,
    sha256_file,
    write_json,
)

A1_BASELINE_PATH = ROOT / "artifacts" / "stage_a" / "A1_exact_baseline.json"
DEFAULT_OUTPUT = ROOT / "artifacts" / "stage_b" / "B2_trace_parity.json"


def _top5_rows(results) -> list[dict[str, Any]]:
    """把检索结果压成只含 ID、名次与分数的行，供逐位比较。"""

    return [
        {
            "rank": item.rank,
            "chunk_id": item.chunk.chunk_id,
            "candidate_rank": item.candidate_rank,
            "similarity_score": item.similarity_score,
            "reranker_score": item.reranker_score,
        }
        for item in results
    ]


def run(output: Path) -> dict[str, Any]:
    """执行 40 题的开/关 trace 对照并写出结果。"""

    configure_offline_cache()
    settings = Settings()
    # 数值 Gate 只认 CPU：GPU 浮点差异可能改变个别题排序。
    device = resolve_model_device(settings.model_device)
    if device != "cpu":
        raise SystemExit(2)

    baseline = read_json(A1_BASELINE_PATH)
    questions = read_json(QUESTIONS_V1_PATH)["questions"]
    if len(questions) != 40:
        raise RuntimeError("expected the frozen 40-question set")

    retriever = build_retriever(settings)
    recorder = RunTraceRecorder()
    per_question: dict[str, Any] = {}
    mismatches: list[dict[str, Any]] = []
    for question in questions:
        qid = question["question_id"]
        query = question["english_query"]
        # 关 trace：与线上默认路径完全相同。
        retriever.trace_recorder = None
        off = _top5_rows(retriever.retrieve(query))
        # 开 trace：同一个 retriever，只替换 recorder。
        retriever.trace_recorder = recorder
        recorder.start_run(query, run_id=qid)
        try:
            on = _top5_rows(retriever.retrieve(query))
        finally:
            retriever.trace_recorder = None
        events = {event.name: event.payload for event in recorder.events}
        dense = events["retrieval.dense.completed"]["candidates"]
        rerank = events["rerank.completed"]["results"]

        checks = {
            # (a) 开/关 trace 的 Top5 逐位相等（含全部分数与名次）。
            "a_top5_on_off_equal": on == off,
            # (b) trace 的 Dense Top20 与 A1 冻结基线逐位相等。
            "b_dense_ids_equal_a1": [c["chunk_id"] for c in dense] == baseline["candidate_run"][qid],
            "b_dense_scores_equal_a1": [c["similarity_score"] for c in dense]
            == baseline["candidate_scores"][qid],
            # (c) rerank.completed 的记录与本次 Top5 一致。
            "c_rerank_matches_top5": [
                (r["rank"], r["chunk_id"], r["candidate_rank"], r["reranker_score"]) for r in rerank
            ]
            == [(t["rank"], t["chunk_id"], t["candidate_rank"], t["reranker_score"]) for t in on],
        }
        consistent = all(checks.values())
        per_question[qid] = {"consistent": consistent, **checks}
        if not consistent:
            mismatches.append({"question_id": qid, **{k: v for k, v in checks.items() if not v}})

    payload = {
        "schema_version": "stage-b-b2-trace-parity-v1",
        "collection_name": resolve_active_collection(settings),
        "device": retriever.dense_retriever.embedding_provider.device,
        "query_count": len(questions),
        "query_source": "evals/research_retrieval/research_questions_v1.json:english_query",
        "a1_baseline_sha256": sha256_file(A1_BASELINE_PATH),
        "per_question": per_question,
        "mismatches": mismatches,
        "parity": not mismatches,
    }
    write_json(output, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    payload = run(args.output)
    print(
        json.dumps(
            {
                "parity": payload["parity"],
                "mismatch_count": len(payload["mismatches"]),
                "device": payload["device"],
                "collection_name": payload["collection_name"],
            },
            ensure_ascii=False,
        )
    )
    return 0 if payload["parity"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
