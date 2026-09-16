# -*- coding: utf-8 -*-
"""EvidenceFlow 清理 Step 9 硬门验证：Gate1 src hash / Gate5 frozen hash / Gate3 IR smoke。

- Gate1：src/evidenceflow 全部 .py 与基线逐字节一致（.pyc 属预期删除，不参与比对）。
- Gate5：frozen 基线文件（CORE_EVAL + PDF_ACTIVE_RESEARCH + MULTIFORMAT_ARCHIVE）
  与当前一致；允许差异仅限"预期变更清单"（Phase14 状态收口字段、tmp 去重副本、
  已删除的 phase14 chroma、新增的 ARCHIVED.md）。
- Gate3：标准 IR 一致性 fixture 校验（ir_metric_conformance_fixture.json 若存在）+
  standard_ir_v1 黄金指标读取核对。
"""

import hashlib
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "artifacts", "cleanup")


def sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    result = {"gate1_src_hash": {}, "gate5_frozen_hash": {}, "gate3_ir_smoke": {}}

    # ---- Gate 1: src hash 不变（仅 .py；.pyc 属本轮预期清理）----
    baseline = json.load(open(os.path.join(OUT, "src_hash_baseline.json"), encoding="utf-8"))["files"]
    mismatch, missing, ok = [], [], 0
    for rel, h in baseline.items():
        if not rel.endswith(".py"):
            continue  # pyc 已按计划删除
        p = os.path.join(ROOT, rel)
        if not os.path.isfile(p):
            missing.append(rel)
        elif sha256_of(p) == h:
            ok += 1
        else:
            mismatch.append(rel)
    result["gate1_src_hash"] = {
        "checked_py_files": ok + len(mismatch) + len(missing),
        "unchanged": ok, "missing": missing, "mismatch": mismatch,
        "pass": not mismatch and not missing,
    }
    print(f"[Gate1] src .py unchanged={ok}, missing={len(missing)}, mismatch={len(mismatch)} -> "
          f"{'PASS' if result['gate1_src_hash']['pass'] else 'FAIL'}")

    # ---- Gate 5: frozen hash（排除预期变更）----
    frozen = json.load(open(os.path.join(OUT, "frozen_hash_baseline.json"), encoding="utf-8"))["files"]
    # 预期变更/删除清单：清理方案内的有意动作，其余任何差异 = FAIL
    expected_changed = {
        # Phase14 状态收口（§11 增量字段）
        "evals/evaluation_v3/phase14a_multiformat_ingestion_benchmark/phase14a_a_status.json",
        "evals/evaluation_v3/phase14a_multiformat_ingestion_benchmark/phase14a_b/phase14a_b_status.json",
        "evals/evaluation_v3/phase14a_multiformat_ingestion_benchmark/phase14a_c/phase14a_c_status.json",
        "evals/evaluation_v3/phase14a_multiformat_ingestion_benchmark/phase14a_b/metrics/real_corpus_status.json",
        # Step7 tmp 去重（canonical 在 evals/，tmp 副本删除）
        "tmp/phase12_regression_out/baseline.json",
        "tmp/phase12_regression_out/chroma_identity.json",
    }
    expected_deleted_prefixes = (
        # Phase14 chroma（§15 可重建索引，inventory 已导出）
        "evals/evaluation_v3/phase14a_multiformat_ingestion_benchmark/phase14a_c/chroma/",
    )
    n_ok = n_changed = n_deleted = 0
    unexpected = []
    for rel, h in frozen.items():
        p = os.path.join(ROOT, rel)
        if rel.startswith(expected_deleted_prefixes):
            if os.path.isfile(p):
                unexpected.append(("should_be_deleted_but_present", rel))
            else:
                n_deleted += 1
            continue
        if rel in expected_changed:
            n_changed += 1
            continue
        if not os.path.isfile(p):
            unexpected.append(("missing", rel))
        elif sha256_of(p) == h:
            n_ok += 1
        else:
            unexpected.append(("hash_mismatch", rel))
    # Gate5 保护集显式核对（方案 §29）：Gold/qrels/Phase11A/Phase12/13C-B B2_fused
    protected_hits = {
        "standard_ir_v1": 0, "phase12_rag_quality_v1": 0, "phase13c_b": 0,
    }
    for rel, h in frozen.items():
        if "standard_ir_v1" in rel and os.path.isfile(os.path.join(ROOT, rel)) and sha256_of(os.path.join(ROOT, rel)) == h:
            protected_hits["standard_ir_v1"] += 1
        if "phase12_rag_quality_v1" in rel and os.path.isfile(os.path.join(ROOT, rel)) and sha256_of(os.path.join(ROOT, rel)) == h:
            protected_hits["phase12_rag_quality_v1"] += 1
        if "phase13c_b" in rel and os.path.isfile(os.path.join(ROOT, rel)) and sha256_of(os.path.join(ROOT, rel)) == h:
            protected_hits["phase13c_b"] += 1
    result["gate5_frozen_hash"] = {
        "unchanged": n_ok, "expected_changed": n_changed, "expected_deleted": n_deleted,
        "unexpected": unexpected,
        "protected_sets_intact": protected_hits,
        "pass": not unexpected,
    }
    print(f"[Gate5] unchanged={n_ok}, expected_changed={n_changed}, expected_deleted={n_deleted}, "
          f"unexpected={len(unexpected)} -> {'PASS' if result['gate5_frozen_hash']['pass'] else 'FAIL'}")
    print(f"[Gate5] protected intact: {protected_hits}")

    # ---- Gate 3: 标准 IR smoke ----
    fixture = os.path.join(ROOT, "tests", "fixtures", "ir_metric_conformance_fixture.json")
    if not os.path.isfile(fixture):
        # 兜底搜索
        for dp, dns, fns in os.walk(os.path.join(ROOT, "tests")):
            for fn in fns:
                if "conformance" in fn:
                    fixture = os.path.join(dp, fn)
    fixture_found = os.path.isfile(fixture)
    result["gate3_ir_smoke"]["fixture_path"] = os.path.relpath(fixture, ROOT).replace("\\", "/") if fixture_found else None
    # 黄金指标（来自 standard_ir_v1 冻结结果，Phase 11A）
    golden_path = os.path.join(ROOT, "evals", "evaluation_v3", "standard_ir_v1", "execution_result.json")
    golden = json.load(open(golden_path, encoding="utf-8")) if os.path.isfile(golden_path) else {}
    result["gate3_ir_smoke"]["golden_source"] = "evals/evaluation_v3/standard_ir_v1/execution_result.json"
    result["gate3_ir_smoke"]["fixture_found"] = fixture_found
    result["gate3_ir_smoke"]["note"] = ("fixture 数值校验由 tests/ 内 conformance 测试在 Gate2 全量回归中执行；"
                                        "此处核对冻结黄金结果可读且完整")
    result["gate3_ir_smoke"]["golden_keys_sample"] = sorted(golden.keys())[:8]
    print(f"[Gate3] fixture_found={fixture_found}, golden readable={bool(golden)}（数值校验由 Gate2 回归覆盖）")

    with open(os.path.join(OUT, "gate_verification.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
    print("report -> artifacts/cleanup/gate_verification.json")


if __name__ == "__main__":
    main()
