# -*- coding: utf-8 -*-
"""EvidenceFlow cleanup Step 9 hard-gate verification: Gate1 src hash / Gate5 frozen hash / Gate3 IR smoke.

- Gate1: every .py under src/evidenceflow is byte-identical to the baseline (.pyc files are expected
  deletions and do not participate in the comparison).
- Gate5: the frozen baseline files (CORE_EVAL + PDF_ACTIVE_RESEARCH + MULTIFORMAT_ARCHIVE) match the
  current state; allowed differences are limited to the "expected change list" (Phase14 status
  closing fields, tmp deduplicated copies, the deleted phase14 chroma, the added ARCHIVED.md).
- Gate3: standard IR conformance fixture check (ir_metric_conformance_fixture.json if present) plus
  reading and cross-checking the standard_ir_v1 golden metrics.
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

    # ---- Gate 1: src hash unchanged (.py only; .pyc files are this round's expected cleanup) ----
    baseline = json.load(open(os.path.join(OUT, "src_hash_baseline.json"), encoding="utf-8"))["files"]
    mismatch, missing, ok = [], [], 0
    for rel, h in baseline.items():
        if not rel.endswith(".py"):
            continue  # pyc files were deleted as planned
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

    # ---- Gate 5: frozen hash (excluding expected changes) ----
    frozen = json.load(open(os.path.join(OUT, "frozen_hash_baseline.json"), encoding="utf-8"))["files"]
    # Expected change/deletion list: the deliberate actions in the cleanup plan; any other
    # difference = FAIL
    expected_changed = {
        # Phase14 status closing (§11 incremental fields)
        "evals/evaluation_v3/phase14a_multiformat_ingestion_benchmark/phase14a_a_status.json",
        "evals/evaluation_v3/phase14a_multiformat_ingestion_benchmark/phase14a_b/phase14a_b_status.json",
        "evals/evaluation_v3/phase14a_multiformat_ingestion_benchmark/phase14a_c/phase14a_c_status.json",
        "evals/evaluation_v3/phase14a_multiformat_ingestion_benchmark/phase14a_b/metrics/real_corpus_status.json",
        # Step7 tmp dedup (canonical lives in evals/, the tmp copies are deleted)
        "tmp/phase12_regression_out/baseline.json",
        "tmp/phase12_regression_out/chroma_identity.json",
    }
    expected_deleted_prefixes = (
        # Phase14 chroma (§15 rebuildable index, inventory already exported)
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
    # Explicit check of the Gate5 protected set (plan §29): Gold/qrels/Phase11A/Phase12/13C-B
    # B2_fused
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

    # ---- Gate 3: standard IR smoke ----
    fixture = os.path.join(ROOT, "tests", "fixtures", "ir_metric_conformance_fixture.json")
    if not os.path.isfile(fixture):
        # Fallback search
        for dp, dns, fns in os.walk(os.path.join(ROOT, "tests")):
            for fn in fns:
                if "conformance" in fn:
                    fixture = os.path.join(dp, fn)
    fixture_found = os.path.isfile(fixture)
    result["gate3_ir_smoke"]["fixture_path"] = os.path.relpath(fixture, ROOT).replace("\\", "/") if fixture_found else None
    # Golden metrics (from the standard_ir_v1 frozen results, Phase 11A)
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
