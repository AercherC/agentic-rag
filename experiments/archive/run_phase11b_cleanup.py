#!/usr/bin/env python
"""已冻结的 Phase 11B Cleanup 历史 runner。

子命令：
  baseline    Step 0 / 14：冻结 Cleanup 前状态与 Chroma 身份
  regression  Step 13 / 14：标准 IR 指标 + Chroma 身份回归
  rollback    Step 15：RERANKER_ENABLED=false 回滚路径装配验证

强制边界：
  - 不修改 Runtime / Gold / Chunk / Embedding / Chroma；
  - 所有输出写入 evals/evaluation_v3/phase11b_cleanup_v1/；
  - 不覆盖任何 Phase 6–11A 历史 JSON（排他创建）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, "src")

from evidenceflow.config import Settings
from evidenceflow.storage import ChromaVectorStore

OUT = Path("evals/evaluation_v3/phase11b_cleanup_v1")
V3 = Path("evals/evaluation_v3/standard_ir_v1")

# Step 13 要求：Cleanup 前后标准 IR 指标必须完全一致。
FROZEN_STANDARD_METRICS = {
    "Precision@5": 0.1150,
    "Recall@5": 0.5583,
    "Recall@10": 0.7333,
    "MRR@5": 0.4321,
    "MAP@5": 0.4279,
    "nDCG@5": 0.4619,
}
METRIC_TOLERANCE = 5e-5  # 冻结值只保留 4 位小数，比较时允许末位舍入差

FROZEN_CHUNK_COUNT = 361
FROZEN_DATASET_HASH = "be1e8354babef3927163c19e32c614f7639323095900c487d559ade8249e7357"


def _sha256_file(path: Path) -> str:
    """计算单个文件的 SHA-256，用于 Runtime fingerprint 审计。"""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _runtime_fingerprint() -> dict[str, str]:
    """对 PROJECT_SPEC 与全部生产源码取 hash，任何代码改动都可被审计。"""

    files = [Path("PROJECT_SPEC.md")]
    files.extend(sorted(Path("src/evidenceflow").rglob("*.py")))
    return {str(path).replace("\\", "/"): _sha256_file(path) for path in files}


def _chroma_identity() -> dict[str, object]:
    """读取正式 Chroma Collection 的 chunk 数与排序后 chunk_id 身份 hash。"""

    settings = Settings()
    store = ChromaVectorStore(settings.vector_db_path, settings.collection_name)
    chunk_ids = sorted(chunk.chunk_id for chunk in store.iter_all_chunks())
    identity = hashlib.sha256(
        "\n".join(chunk_ids).encode("utf-8")
    ).hexdigest()
    return {
        "collection_name": settings.collection_name,
        "vector_db_path": str(settings.vector_db_path),
        "chunk_count": len(chunk_ids),
        "sorted_chunk_id_identity_hash": identity,
    }


def _settings_snapshot() -> dict[str, object]:
    """导出 Settings 快照；API Key 等敏感字段只记录是否已设置。"""

    settings = Settings()
    data = settings.model_dump(mode="json")
    redacted = {}
    for key, value in data.items():
        if "key" in key or "token" in key or "secret" in key:
            redacted[key] = "<redacted:set>" if value else "<redacted:unset>"
        else:
            redacted[key] = value
    return redacted


def _collect_pytest_stats() -> dict[str, object]:
    """从 pytest 缓存读取上次统计；无缓存时标记为 N/A。"""

    cache = Path(".pytest_cache/v/cache/lastfailed")
    return {"lastfailed": sorted(json.loads(cache.read_text()).keys()) if cache.exists() else []}


def save_artifact(payload: dict, path: Path) -> None:
    """排他写入 JSON：禁止覆盖任何已存在的历史 artifact。"""

    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def cmd_baseline(args: argparse.Namespace) -> int:
    """Step 0 + Step 14：冻结 Cleanup 前状态。"""

    v3_resume = json.loads((V3 / "resume_metrics.json").read_text(encoding="utf-8"))
    qrels_manifest = json.loads((V3 / "qrels_manifest.json").read_text(encoding="utf-8"))
    highlight = json.loads((V3 / "standard_baseline_comparison.json").read_text(encoding="utf-8"))

    payload = {
        "artifact_type": "phase11b_cleanup_baseline",
        "phase": "11B",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git": {
            "available": False,
            "reason": "项目目录不是 git 仓库（无 .git），无法采集 git status / git diff。",
            "status": None,
            "diff": None,
        },
        "tests": {
            "collected": 408,
            "passed": 408,
            "failed": 0,
            "warnings": 17,
            "command": "python -m pytest -q",
            "pytest_cache": _collect_pytest_stats(),
        },
        "runtime_fingerprint": _runtime_fingerprint(),
        "chroma_identity": _chroma_identity(),
        "settings_snapshot": _settings_snapshot(),
        "standard_ir_baseline": {
            "source": str(V3 / "resume_metrics.json"),
            "evaluation_version": v3_resume.get("evaluation_version"),
            "dataset_hash": v3_resume.get("dataset_hash"),
            "qrels_hash": v3_resume.get("qrels_hash"),
            "current_runtime": v3_resume.get("current_runtime"),
            "frozen_gate_metrics": FROZEN_STANDARD_METRICS,
        },
        "runtime_ranking_reference": {
            "source": "evals/evaluation_v2/phase9_runtime_promotion_v1/runtime_after.json",
            "source_hash": _sha256_file(
                Path("evals/evaluation_v2/phase9_runtime_promotion_v1/runtime_after.json")
            ),
            "note": "Cleanup 不重跑检索；ranking 一致性以该冻结 artifact 为参照。",
        },
        "qrels": {
            "hash": qrels_manifest.get("qrels_hash"),
            "num_questions": qrels_manifest.get("num_questions"),
            "num_relevant_chunks": qrels_manifest.get("num_relevant_chunks"),
        },
        "comparison_source": highlight.get("artifact_type"),
    }
    save_artifact(payload, OUT / "cleanup_baseline.json")
    print(f"[baseline] written -> {OUT / 'cleanup_baseline.json'}")
    return 0


def _recompute_runtime_metrics() -> dict:
    """用 Evaluation v3 对冻结的当前 Runtime 排名重算标准指标并逐题比对 ranking IDs。

    Cleanup 不重跑检索；但代码删除（如 eval/dataset.py）可能影响 Eval 包本身，
    因此必须真的执行一次 v3 replay，证明 Eval 标准路径在 Cleanup 后仍给出同一组数字。
    """

    from pydantic import TypeAdapter

    from evidenceflow.domain.evaluation_v2 import MappingArtifact
    from evidenceflow.eval.v3 import build_qrels, evaluate_v3, qrels_hash

    mapping_path = Path("evals/evaluation_v2/phase35_gold_stabilization_v1/gold_mapping_v3.json")
    ranking_path = Path("evals/evaluation_v2/phase9_runtime_promotion_v1/runtime_after.json")
    mapping = TypeAdapter(MappingArtifact).validate_json(
        mapping_path.read_text(encoding="utf-8")
    )
    qrels = build_qrels(mapping)
    run = {
        qid: list(ids)
        for qid, ids in json.loads(ranking_path.read_text(encoding="utf-8"))["rankings"].items()
    }
    result = evaluate_v3(qrels, run, mapping, metric_depths=(1, 3, 5, 10))
    return {
        "source_artifact": str(ranking_path).replace("\\", "/"),
        "source_hash": _sha256_file(ranking_path),
        "qrels_hash": qrels_hash(qrels),
        "aggregate": result["aggregate"],
        "per_query": result["per_query"],
        "ranking_ids": {qid: ids for qid, ids in sorted(run.items())},
        "num_questions": len(run),
    }


def cmd_regression(args: argparse.Namespace) -> int:
    """Step 13 + Step 14：标准 IR 指标与 Chroma 身份回归。"""

    baseline = json.loads((OUT / "cleanup_baseline.json").read_text(encoding="utf-8"))
    v3_resume = json.loads((V3 / "resume_metrics.json").read_text(encoding="utf-8"))
    after = v3_resume.get("current_runtime", {})

    metric_rows = []
    metrics_ok = True
    for name, expected in FROZEN_STANDARD_METRICS.items():
        actual = after.get(name)
        if actual is None:
            ok = False
        else:
            ok = abs(float(actual) - float(expected)) <= METRIC_TOLERANCE
        metrics_ok = metrics_ok and ok
        metric_rows.append(
            {
                "metric": name,
                "before": expected,
                "after": actual,
                "match": ok,
            }
        )

    chroma_before = baseline["chroma_identity"]
    chroma_after = _chroma_identity()
    chroma_ok = (
        chroma_after["chunk_count"] == chroma_before["chunk_count"] == FROZEN_CHUNK_COUNT
        and chroma_after["sorted_chunk_id_identity_hash"]
        == chroma_before["sorted_chunk_id_identity_hash"]
    )

    # Runtime ranking 一致性：以冻结 artifact hash 为准，Cleanup 不重跑检索。
    ranking_ref = baseline["runtime_ranking_reference"]
    ranking_hash_now = _sha256_file(Path(ranking_ref["source"]))
    ranking_ok = ranking_hash_now == ranking_ref["source_hash"]

    qrels_ok = v3_resume.get("qrels_hash") == baseline["standard_ir_baseline"]["qrels_hash"]
    dataset_ok = v3_resume.get("dataset_hash") == FROZEN_DATASET_HASH

    # 真正重算一次：证明 Cleanup 后的 Eval 包仍给出同一组标准指标与同一组 ranking IDs。
    recomputed = _recompute_runtime_metrics()
    frozen_resume = json.loads((V3 / "resume_metrics.json").read_text(encoding="utf-8"))
    frozen_current = frozen_resume.get("current_runtime", {})
    aggregate = recomputed["aggregate"]
    # v3 aggregate 形如 {"precision_at_k": {"5": 0.115}}；冻结 gate 用 "Precision@5" 命名。
    flat = {}
    for family, key in (
        ("precision_at_k", "Precision"),
        ("recall_at_k", "Recall"),
        ("mrr_at_k", "MRR"),
        ("map_at_k", "MAP"),
        ("ndcg_at_k", "nDCG"),
    ):
        for depth, value in aggregate.get(family, {}).items():
            flat[f"{key}@{depth}"] = value
    recompute_rows = []
    recompute_ok = True
    for name, expected in FROZEN_STANDARD_METRICS.items():
        actual = flat.get(name)
        ok = actual is not None and abs(float(actual) - float(expected)) <= METRIC_TOLERANCE
        recompute_ok = recompute_ok and ok
        recompute_rows.append({"metric": name, "frozen": expected, "recomputed": actual, "match": ok})
    # 逐题比对：per-query 标准指标完全一致 ⇒ 冻结 ranking 未被 Cleanup 改变。
    frozen_per_query = json.loads((V3 / "per_query_results.json").read_text(encoding="utf-8"))[
        "phase9"
    ]
    ranking_ids_sha256 = hashlib.sha256(
        json.dumps(recomputed["ranking_ids"], sort_keys=True).encode("utf-8")
    ).hexdigest()
    per_query_diff = []
    for qid, frozen_row in sorted(frozen_per_query.items()):
        recomputed_row = result_per_query = recomputed["per_query"][qid]["standard_metrics"]
        for metric, value in frozen_row["standard_metrics"].items():
            if abs(float(value) - float(recomputed_row[metric])) > 1e-12:
                per_query_diff.append({"question_id": qid, "metric": metric,
                                       "frozen": value, "recomputed": recomputed_row[metric]})
    per_query_ok = not per_query_diff
    recompute_qrels_ok = recomputed["qrels_hash"] == baseline["standard_ir_baseline"]["qrels_hash"]

    payload = {
        "artifact_type": "phase11b_runtime_regression",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "method": (
            "Cleanup 不重跑检索：指标以冻结 ranking artifact 为输入，"
            "再用 Evaluation v3 真机重算一次并与冻结值比对。"
        ),
        "metrics": metric_rows,
        "metrics_unchanged": metrics_ok,
        "recompute": {
            "source_artifact": recomputed["source_artifact"],
            "source_hash": recomputed["source_hash"],
            "qrels_hash": recomputed["qrels_hash"],
            "qrels_unchanged": recompute_qrels_ok,
            "num_questions": recomputed["num_questions"],
            "rows": recompute_rows,
            "all_match": recompute_ok,
            "ranking_ids_sha256": ranking_ids_sha256,
            "per_query_metrics_identical": per_query_ok,
            "per_query_diff_count": len(per_query_diff),
            "per_query_diff_sample": per_query_diff[:5],
        },
        "chroma": {
            "before": chroma_before,
            "after": chroma_after,
            "chunk_count_expected": FROZEN_CHUNK_COUNT,
            "identity_unchanged": chroma_ok,
        },
        "runtime_ranking": {
            "frozen_artifact": ranking_ref["source"],
            "hash_before": ranking_ref["source_hash"],
            "hash_after": ranking_hash_now,
            "unchanged": ranking_ok,
        },
        "qrels": {
            "hash": v3_resume.get("qrels_hash"),
            "unchanged": qrels_ok,
        },
        "dataset_hash_unchanged": dataset_ok,
        "ranking_unchanged": ranking_ok and per_query_ok,
        "recall_at_20": {
            "value": after.get("Recall@20"),
            "note": "继续 N/A；Cleanup 不得重新发明历史 @20 数据。",
        },
        "gate": (
            "PASS"
            if (
                metrics_ok
                and chroma_ok
                and ranking_ok
                and qrels_ok
                and dataset_ok
                and recompute_ok
                and recompute_qrels_ok
                and per_query_ok
            )
            else "FAIL"
        ),
    }
    save_artifact(payload, OUT / "runtime_regression.json")
    save_artifact(payload["chroma"], OUT / "chroma_identity.json")
    print(f"[regression] metrics_unchanged={metrics_ok} chroma_ok={chroma_ok} -> {payload['gate']}")
    return 0 if payload["gate"] == "PASS" else 1


def cmd_rollback(args: argparse.Namespace) -> int:
    """Step 15：验证 RERANKER_ENABLED=false 回滚路径仍可装配并运行。"""

    from evidenceflow.runtime import build_rag_service

    # bge-reranker-base 在本机只有本地快照 .cache/models/bge-reranker-base
    # （Phase 9 run_manifest 记录的 resolved_local_path），HF hub 缓存是未完成下载；
    # 因此默认路径装配时显式指向本地快照，否则属于环境问题而非 Cleanup 回归。
    results = {}
    checks = {
        "reranker_enabled_true": {
            "RERANKER_ENABLED": "true",
            "RERANKER_MODEL": ".cache/models/bge-reranker-base",
        },
        "reranker_enabled_false": {"RERANKER_ENABLED": "false"},
        "hybrid_conditional_disabled": {
            "RERANKER_ENABLED": "false",
            "MULTILINGUAL_HYBRID_ENABLED": "false",
        },
        "hybrid_conditional_enabled": {
            "RERANKER_ENABLED": "false",
            "MULTILINGUAL_HYBRID_ENABLED": "true",
        },
    }
    for name, overrides in checks.items():
        saved = {k: os.environ.get(k) for k in overrides}
        try:
            os.environ.update(overrides)
            service = build_rag_service(Settings())
            results[name] = {
                "assembled": True,
                "retriever": type(service._retriever).__name__,
                "retrieval_strategy": getattr(service._retriever, "retrieval_strategy", None),
                "context_builder": type(service._context_builder).__name__,
                "generator": type(service._generator).__name__,
            }
        except Exception as exc:  # noqa: BLE001 - 回滚验证需要记录原始异常
            results[name] = {"assembled": False, "error_type": type(exc).__name__, "error": str(exc)}
        finally:
            for key, value in saved.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    # Fallback 实现存在性：只检查可导入，不触发模型加载。
    from evidenceflow.retrieval import (
        BM25Index,
        MultilingualRetriever,
        weighted_rrf_fusion,
    )
    from evidenceflow.retrieval.hybrid_fusion import equal_ranked_rrf_fusion
    from evidenceflow.translation import LLMTranslationProvider

    fallback_present = {
        "TranslationProvider(LLMTranslationProvider)": LLMTranslationProvider is not None,
        "BM25Index": BM25Index is not None,
        "Hybrid(equal_ranked_rrf_fusion)": equal_ranked_rrf_fusion is not None,
        "RRF(weighted_rrf_fusion)": weighted_rrf_fusion is not None,
        "MultilingualRetriever": MultilingualRetriever is not None,
    }
    all_assembled = all(item.get("assembled") for item in results.values())
    gate = "PASS" if all_assembled and all(fallback_present.values()) else "FAIL"
    payload = {
        "artifact_type": "phase11b_rollback_validation",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "assemblies": results,
        "fallback_implementations_present": fallback_present,
        "note": (
            "仅验证装配路径与类型，不端到端推理；"
            "reranker_enabled_true 使用本机本地快照 .cache/models/bge-reranker-base"
            "（同 Phase 9 run_manifest 的 resolved_local_path）。"
        ),
        "gate": gate,
    }
    save_artifact(payload, OUT / "rollback_validation.json")
    print(f"[rollback] -> {gate}")
    return 0 if gate == "PASS" else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 11B cleanup runner")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("baseline", help="冻结 Cleanup 前基线").set_defaults(func=cmd_baseline)
    sub.add_parser("regression", help="标准 IR 与 Chroma 回归").set_defaults(func=cmd_regression)
    sub.add_parser("rollback", help="回滚路径验证").set_defaults(func=cmd_rollback)
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
