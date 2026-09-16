"""已冻结的 Phase 12 Full Regression 历史脚本。

Phase 12 是 Evaluation Phase，不应改变任何 Runtime 行为。本脚本用**与 Phase 12 基线完全相同的
函数定义**（直接 import `run_phase12_rag_quality` 的 `_composition_fingerprint` / `_runtime_fingerprint`）
复核以下事实：

1. 标准 IR 指标与逐题排名与冻结 artifact 完全一致（复用 11C 的复算函数）；
2. Chroma chunk 数（361）与排序 chunk_id 身份 hash 不变；
3. Runtime composition fingerprint（Phase 12 口径）与 Phase 12 基线一致；
4. PROJECT_SPEC.md + src/evidenceflow/** 源码指纹与 Phase 12 基线一致（证明未动 Runtime 源码）；
5. Rollback / 装配路径与 Hybrid 开关注入仍可合法装配。

注意：11C 与 Phase 12 的 composition fingerprint **定义范围不同**（11C 取 5 个装配模块，
Phase 12 取 8 个模块 + 更宽的 Settings 子集），两者不可跨口径比较。本脚本统一使用 Phase 12 口径。
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

BASE = Path("evals/evaluation_v3/phase12_rag_quality_v1")


def _load(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _save(payload: Any, path: Path) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 12 全量回归")
    parser.add_argument("--tests-passed", type=int, required=True)
    parser.add_argument("--tests-failed", type=int, required=True)
    parser.add_argument("--tests-warnings", type=int, required=True)
    args = parser.parse_args()

    p11c = _load("p11c", "experiments/run_phase11c_technical_debt_closure.py")
    p12 = _load("p12", "experiments/run_phase12_rag_quality.py")

    baseline = json.loads((BASE / "phase12_baseline.json").read_text(encoding="utf-8"))

    # ---- 1. 标准 IR 指标 + 逐题差异（冻结 ranking 输入，不重跑检索） ----
    recomputed = p11c._recompute_runtime_metrics()
    flat = p11c._flatten(recomputed["aggregate"])
    metric_rows = []
    metrics_ok = True
    for name, expected in p11c.EXACT_STANDARD_METRICS.items():
        actual = flat.get(name)
        ok = actual is not None and abs(actual - expected) <= p11c.EXACT_TOLERANCE
        metrics_ok = metrics_ok and ok
        metric_rows.append({"metric": name, "expected": expected, "recomputed": actual, "match": ok})

    frozen_per_query = json.loads(
        (p11c.V3 / "per_query_results.json").read_text(encoding="utf-8")
    )["phase9"]
    per_query_diff = []
    for qid, frozen_row in sorted(frozen_per_query.items()):
        now_row = recomputed["per_query"][qid]["standard_metrics"]
        for metric, value in frozen_row["standard_metrics"].items():
            if abs(float(value) - float(now_row[metric])) > 1e-12:
                per_query_diff.append({"question_id": qid, "metric": metric})
    per_query_ok = not per_query_diff

    ranking_hash_now = p11c._sha256_file(p11c.RANKING_SOURCE)
    ranking_ok = ranking_hash_now == p11c.json.loads(
        (p11c.OUT / "baseline.json").read_text(encoding="utf-8")
    )["phase9_runtime_after_ranking"]["hash"]

    # ---- 2. Chroma 身份 ----
    chroma_now = p11c._chroma_identity()
    chroma_base = baseline["chroma_identity"]
    chroma_ok = (
        chroma_now["chunk_count"] == chroma_base["chunk_count"] == baseline["frozen_chunk_count"] == 361
        and chroma_now["sorted_chunk_id_identity_hash"] == chroma_base["sorted_chunk_id_identity_hash"]
    )

    # ---- 3. Runtime composition fingerprint（统一 Phase 12 口径） ----
    # 已知环境敏感性：settings_subset 含 `reranker_model`，其字面值受 RERANKER_MODEL 环境变量影响。
    #   - Phase 12 基线在**未设置**该变量的环境下冻结（取 Settings 默认值 BAAI/bge-reranker-base）；
    #   - 本地离线装配**必须**显式设置 RERANKER_MODEL=.cache/models/bge-reranker-base
    #     （该模型 HF hub 缓存为未完成下载，真实权重在 .cache/models 下）。
    # 二者指向同一模型，差异仅为部署路径字面量，不反映 Runtime 组成变更。
    # 因此采集 fingerprint 前临时恢复「基线采集环境」，避免把部署路径差异误判为组成漂移；
    # rollback 装配验证仍使用真实本地路径（见下）。
    env_reranker_before = os.environ.get("RERANKER_MODEL")
    saved_rrm = os.environ.pop("RERANKER_MODEL", None)
    try:
        comp_now = p12._composition_fingerprint()
    finally:
        if saved_rrm is not None:
            os.environ["RERANKER_MODEL"] = saved_rrm
    env_normalization = {
        "field": "reranker_model",
        "baseline_value": (baseline["runtime_composition_fingerprint"]["settings_subset"].get("reranker_model")),
        "value_before_alignment": env_reranker_before if env_reranker_before is not None else "<unset>",
        "action": "采集 composition fingerprint 前临时移除 RERANKER_MODEL，使 Settings 回落到与 Phase 12 基线相同的默认值。",
        "why": "reranker_model 属部署路径型配置；本地离线环境必须指向 .cache/models/bge-reranker-base，"
               "该差异不反映 Runtime 组成变更。装配等价性由 rollback 检查独立证明。",
        "recorded_as": "KNOWN_ENVIRONMENT_SENSITIVITY（Phase 12 为评测阶段，不改代码，仅记录）",
    }
    comp_base = baseline["runtime_composition_fingerprint"]
    comp_ok = comp_now["fingerprint"] == comp_base["fingerprint"]
    comp_diff_settings = {
        k: {"baseline": comp_base["settings_subset"].get(k), "now": v}
        for k, v in comp_now["settings_subset"].items()
        if comp_base["settings_subset"].get(k) != v
    }
    comp_diff_modules = sorted(
        m for m, h in comp_now["assembly_module_hashes"].items()
        if comp_base["assembly_module_hashes"].get(m) != h
    )

    # ---- 4. 生产源码 / 文档指纹（证明 Phase 12 未改 Runtime 源码） ----
    src_fp_now = p12._sha256_json(p12._runtime_fingerprint())
    src_fp_ok = src_fp_now == baseline["runtime_source_fingerprint"]

    # ---- 5. Rollback / 装配 ----
    from evidenceflow.runtime import build_rag_service
    from evidenceflow.config import Settings

    checks = {
        "reranker_enabled_true": {"RERANKER_ENABLED": "true", "RERANKER_MODEL": ".cache/models/bge-reranker-base"},
        "reranker_enabled_false": {"RERANKER_ENABLED": "false"},
        "hybrid_disabled": {"RERANKER_ENABLED": "false", "MULTILINGUAL_HYBRID_ENABLED": "false"},
        "hybrid_enabled": {"RERANKER_ENABLED": "false", "MULTILINGUAL_HYBRID_ENABLED": "true"},
    }
    assemblies: dict[str, Any] = {}
    for name, overrides in checks.items():
        saved = {k: os.environ.get(k) for k in overrides}
        try:
            os.environ.update(overrides)
            service = build_rag_service(Settings())
            assemblies[name] = {
                "assembled": True,
                "retriever": type(service._retriever).__name__,
                "context_builder": type(service._context_builder).__name__,
                "generator": type(service._generator).__name__,
            }
        except Exception as exc:  # noqa: BLE001
            assemblies[name] = {"assembled": False, "error": f"{type(exc).__name__}: {exc}"}
        finally:
            for key, value in saved.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    on = assemblies.get("reranker_enabled_true", {})
    off = assemblies.get("reranker_enabled_false", {})
    rollback_ok = (
        all(a.get("assembled") for a in assemblies.values())
        and on.get("retriever") == "DenseRerankerRetriever"
        and off.get("retriever") == "MultilingualRetriever"
    )

    tests_ok = args.tests_failed == 0
    gate = "PASS" if all([metrics_ok, per_query_ok, ranking_ok, chroma_ok, comp_ok, src_fp_ok, rollback_ok, tests_ok]) else "FAIL"

    payload = {
        "artifact_type": "phase12_regression",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "method": "以冻结 ranking artifact 复算标准 IR 指标；不重跑检索、不重建 Chroma；fingerprint 统一使用 Phase 12 口径。",
        "standard_metrics": metric_rows,
        "metrics_unchanged": metrics_ok,
        "per_query": {"num_questions": recomputed["num_questions"], "diff_count": len(per_query_diff), "identical": per_query_ok},
        "ranking": {"hash_unchanged": ranking_ok, "hash": ranking_hash_now},
        "chroma": {"before": chroma_base, "after": chroma_now, "identity_unchanged": chroma_ok},
        "runtime_composition": {
            "scope_note": "Phase 12 口径（8 个装配模块 + 更宽 Settings 子集），与 11C 的 5 模块口径不可跨比。",
            "baseline": comp_base["fingerprint"],
            "now": comp_now["fingerprint"],
            "unchanged": comp_ok,
            "diff_settings": comp_diff_settings,
            "diff_modules": comp_diff_modules,
            "env_normalization": env_normalization,
        },
        "runtime_source_fingerprint": {
            "scope": "PROJECT_SPEC.md + src/evidenceflow/**/*.py",
            "baseline": baseline["runtime_source_fingerprint"],
            "now": src_fp_now,
            "unchanged": src_fp_ok,
        },
        "rollback": {
            "assemblies": assemblies,
            "expectation": {"true": "DenseRerankerRetriever", "false": "MultilingualRetriever"},
            "pass": rollback_ok,
        },
        "recall_at_20": {"value": flat.get("Recall@20"), "note": "继续 N/A；冻结排名仅 Top-10。"},
        "tests": {"passed": args.tests_passed, "failed": args.tests_failed, "warnings": args.tests_warnings},
        "gate": gate,
    }
    _save(payload, BASE / "runtime_regression.json")
    _save(payload["chroma"], BASE / "chroma_identity.json")
    _save({
        "artifact_type": "phase12_rollback_validation",
        "created_at": payload["created_at"],
        "assemblies": assemblies,
        "expectation": payload["rollback"]["expectation"],
        "gate": "PASS" if rollback_ok else "FAIL",
    }, BASE / "rollback_validation.json")

    print(
        f"[regression] metrics={metrics_ok} per_query={per_query_ok} ranking={ranking_ok} "
        f"chroma={chroma_ok} composition={comp_ok} src_fp={src_fp_ok} rollback={rollback_ok} tests={tests_ok} -> {gate}"
    )
    if comp_diff_settings or comp_diff_modules:
        print("  composition diff settings:", comp_diff_settings)
        print("  composition diff modules :", comp_diff_modules)
    return 0 if gate == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
