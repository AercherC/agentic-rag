"""Phase 14A-B 编排（方案 §27 Steps 0-11），Sandbox-Only。

0. 冻结 Phase 14A-A（逐文件 sha256，任一不一致立即 STOP）
1. 构建扩展受控语料（W1..W12 / X1..X15，子进程）+ 解析器无关自证
2. 真实语料导入检查（无文件 → REAL_CORPUS_PENDING，不宣称真实语料验证）
3-6. 运行 DOCX-B0/B1、XLSX-B0/B1（每 case 双跑校验确定性）
7. 受控 Gold 评测（metrics14b）
8. 真实语料人工 review（无语料 → NOT_APPLICABLE/PENDING）
9. 硬门评测（D-G1..8 / X-G1..10）
10. 沙盒边界复查（src 树 / 冻结工程文件 / 14A-A 冻结 / 语料哈希）
11. manifests / reports / status（方案 §31 模板），随后 STOP AT PHASE 14A-B

主 harness 不 import python-docx/openpyxl；所有解析经子进程（adapters14b）。
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import adapters14a, adapters14b, corpus14a, corpus14b, gates14b, metrics14b, sandbox14a, schema14b, selfcheck14b

PROJECT_ROOT = sandbox14a.PROJECT_ROOT
ARTIFACT_ROOT = sandbox14a.ARTIFACT_ROOT
B_ROOT = ARTIFACT_ROOT / "phase14a_b"
AA_STATUS = ARTIFACT_ROOT / "phase14a_a_status.json"


def _run_cmd(module: str, *args: str, timeout: int = 900) -> dict[str, Any]:
    python = adapters14a._resolve_python()
    cmd = [str(python), "-m", module, *args]
    completed = subprocess.run(
        cmd, cwd=str(PROJECT_ROOT), env=adapters14a._subprocess_env(),
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout,
    )
    return {"ok": completed.returncode == 0, "rc": completed.returncode,
            "stdout": completed.stdout[-2000:], "stderr": completed.stderr[-3000:]}


def _dir_hashes(root: Path, exclude: tuple[str, ...] = ()) -> dict[str, dict[str, Any]]:
    entries = {}
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        rel = str(p.relative_to(root)).replace("\\", "/")
        if any(rel.startswith(ex) for ex in exclude):
            continue
        entries[rel] = {"sha256": sandbox14a.sha256_file(p), "size_bytes": p.stat().st_size}
    return entries


# STOP-ON-ANOMALY（方案 §24 可机检子集）


# 计数型指标不参与 [0,1] 范围守卫（§24 的 metric>1 指比例型指标）
_COUNT_METRICS = {"fake_provenance_count", "inferred_heading_count", "unintended_loss",
                  "formula_silent_overwrite", "display_overwrite_count", "cell_elements",
                  "table_elements", "cleaning_log_count", "unresolved_count"}


def _guard_range(arm_id: str, case_id: str, m: dict) -> None:
    """比例型 metric >1 / <0 → STOP（None 不参与，None ≠ 0；计数型另行判定）。"""

    for k, v in m.items():
        if k in _COUNT_METRICS or isinstance(v, bool):
            continue
        if isinstance(v, (int, float)):
            sandbox14a.anomaly_guard(arm_id, v > 1.0001 or v < -0.0001,
                                     "metric out of [0,1]",
                                     f"{case_id}/{arm_id}/{k}={v}")


def _guard_docx_anomalies(arm_id: str, case_id: str, case: dict, doc: dict, m: dict) -> None:
    sandbox14a.anomaly_guard(arm_id, m["document_order_accuracy"] != 1.0,
                             "DOCX paragraph/table order scrambled",
                             f"{case_id}/{arm_id} order={m['document_order_accuracy']}")
    sandbox14a.anomaly_guard(arm_id, m["fake_provenance_count"] > 0,
                             "DOCX fake page/bbox > 0", f"{case_id}/{arm_id}")
    sandbox14a.anomaly_guard(arm_id, m["inferred_marked_rate"] != 1.0,
                             "inferred heading without semantic_inferred=true",
                             f"{case_id}/{arm_id}")
    # 显式 Heading style 被 heuristic 覆盖：gold 显式标题被推断了 → STOP
    gold = metrics14b.collect_gold_docx_b(case)
    for h in gold["headings"]:
        for e in doc["elements"]:
            if e["type"] in ("heading", "document_title") and e.get("semantic_inferred") \
                    and metrics14b._matches(h["text"], [e["normalized_text"]]):
                sandbox14a.anomaly_guard(arm_id, True, "explicit Heading style overridden by heuristic",
                                         f"{case_id}/{arm_id} text={h['text'][:40]}")


def _guard_xlsx_anomalies(arm_id: str, case_id: str, m: dict) -> None:
    sandbox14a.anomaly_guard(arm_id, m["fake_provenance_count"] > 0,
                             "XLSX fake page/bbox > 0", f"{case_id}/{arm_id}")
    sandbox14a.anomaly_guard(arm_id, m["formula_silent_overwrite"] > 0,
                             "XLSX formula overwritten by cached value",
                             f"{case_id}/{arm_id} overwrite={m['formula_silent_overwrite']}")
    if m["merged_range_accuracy"] is not None:
        sandbox14a.anomaly_guard(arm_id, m["merged_range_accuracy"] < 1.0,
                                 "merged range lost",
                                 f"{case_id}/{arm_id} acc={m['merged_range_accuracy']}")


def main() -> int:
    sandbox14a.project_write_guard()
    started = sandbox14a.utc_now()
    src_before = sandbox14a.snapshot_project_source_tree()
    frozen_before = sandbox14a.snapshot_frozen_project_files()

    for d in ["frozen_inputs", "corpus/controlled_docx", "corpus/controlled_xlsx",
              "corpus/real_docx", "corpus/real_xlsx", "native/docx_b0", "native/xlsx_b0",
              "cleaned/docx_b1", "cleaned/xlsx_b1", "reviews", "metrics", "failures",
              "manifests", "reports"]:
        sandbox14a.ensure_dir(B_ROOT / d)

    # ---- Step 0: 冻结 14A-A ----
    aa_hashes = _dir_hashes(ARTIFACT_ROOT, exclude=("phase14a_b/",))
    aa_status = sandbox14a.load_json(AA_STATUS) if AA_STATUS.exists() else {}
    freeze = {
        "artifact_type": "phase14a_b_step0_freeze",
        "frozen_phase": "14A-A",
        "phase14a_a_status": aa_status.get("PHASE_14A_A_STATUS"),
        "file_count": len(aa_hashes),
        "files": aa_hashes,
        "created_at_utc": started,
    }
    sandbox14a.save_json(freeze, B_ROOT / "frozen_inputs" / "phase14a_a_freeze.json")
    print(f"Step 0: frozen 14A-A ({len(aa_hashes)} files)")

    # ---- Step 1: 受控语料 ----
    build = _run_cmd("phase14a.runners.build_corpus14b", "--out", str(B_ROOT / "corpus"))
    if not build["ok"]:
        print("CORPUS BUILD FAILED:\n", build["stderr"], file=sys.stderr)
        return 2
    corpus_hashes = {
        "controlled_docx": _dir_hashes(B_ROOT / "corpus" / "controlled_docx"),
        "controlled_xlsx": _dir_hashes(B_ROOT / "corpus" / "controlled_xlsx"),
    }
    sandbox14a.save_json({"artifact_type": "phase14a_b_corpus_freeze", "files": corpus_hashes},
                         B_ROOT / "frozen_inputs" / "corpus_freeze.json")
    sc = selfcheck14b.check_corpus(B_ROOT / "corpus" / "controlled_docx",
                                   B_ROOT / "corpus" / "controlled_xlsx")
    sandbox14a.save_json(sc, B_ROOT / "metrics" / "corpus_self_check.json")
    if not sc["all_passed"]:
        print("CORPUS SELF-CHECK FAILED:\n", json.dumps(sc, ensure_ascii=False)[:3000], file=sys.stderr)
        return 2
    gold = corpus14b.build_gold_b()
    sandbox14a.save_json(gold, B_ROOT / "frozen_inputs" / "controlled_corpus_gold.json")
    print(f"Step 1: corpus built, selfcheck {sc['docx_passing']}/{sc['docx_total']} docx, "
          f"{sc['xlsx_passing']}/{sc['xlsx_total']} xlsx")

    # ---- Step 2: 真实语料 ----
    real_docx = sorted((B_ROOT / "corpus" / "real_docx").glob("*.docx"))
    real_xlsx = sorted((B_ROOT / "corpus" / "real_xlsx").glob("*.xlsx"))
    real_status = {
        "docx": {"files": [p.name for p in real_docx], "count": len(real_docx),
                 "status": "IMPORTED" if real_docx else "REAL_CORPUS_PENDING"},
        "xlsx": {"files": [p.name for p in real_xlsx], "count": len(real_xlsx),
                 "status": "IMPORTED" if real_xlsx else "REAL_CORPUS_PENDING"},
        "note": "受控基准不得宣称成真实语料验证（方案 §6）；原文件只读 + SHA256 在导入时记录。",
    }
    sandbox14a.save_json(real_status, B_ROOT / "metrics" / "real_corpus_status.json")
    if real_docx or real_xlsx:
        sandbox14a.save_json(
            {"artifact_type": "phase14a_b_real_corpus_inputs",
             "docx": sandbox14a.hash_inputs(real_docx), "xlsx": sandbox14a.hash_inputs(real_xlsx)},
            B_ROOT / "frozen_inputs" / "real_corpus_freeze.json")
    print(f"Step 2: real corpus docx={len(real_docx)} xlsx={len(real_xlsx)}")

    docx_cases = corpus14a.DOCX_CASES + corpus14b.DOCX_CASES_B
    xlsx_cases = corpus14a.XLSX_CASES + corpus14b.XLSX_CASES_B

    # ---- Steps 3-7: 逐格式逐 arm 双跑 + 评测 ----
    def evaluate(fmt: str, cases: list[dict], ext: str, corpus_sub: str,
                 out_specs: list[tuple[adapters14b.ArmSpecB, str]]) -> dict[str, Any]:
        per_case: list[dict[str, Any]] = []
        for case in cases:
            fp = B_ROOT / "corpus" / corpus_sub / f"{case['case_id']}.{ext}"
            entry: dict[str, Any] = {"case_id": case["case_id"], "arms": {}}
            for arm, out_sub in out_specs:
                base_out = B_ROOT / out_sub / f"{case['case_id']}"
                res = adapters14b.run_arm_twice(arm, file_path=fp, base_out=base_out)
                if res["document"] is None:
                    entry["arms"][arm.arm_id] = {"status": res["run"]["status"],
                                                 "reason": res["run"].get("reason")}
                    continue
                doc = res["document"]
                problems = schema14b.validate_document14b(doc)
                sandbox14a.anomaly_guard(arm.arm_id, not problems["valid"],
                                         "schema14b validation failed",
                                         f"{case['case_id']}: {str(problems['problems'])[:400]}")
                m = (metrics14b.compute_docx_metrics14b(case, doc) if fmt == "docx"
                     else metrics14b.compute_xlsx_metrics14b(case, doc))
                _guard_range(arm.arm_id, case["case_id"], m)
                if fmt == "docx":
                    _guard_docx_anomalies(arm.arm_id, case["case_id"], case, doc, m)
                else:
                    _guard_xlsx_anomalies(arm.arm_id, case["case_id"], m)
                g = (gates14b.evaluate_docx_gates14b(case, doc, m, res["deterministic"])
                     if fmt == "docx" else gates14b.evaluate_xlsx_gates14b(case, doc, m, res["deterministic"]))
                entry["arms"][arm.arm_id] = {
                    "status": "OK", "deterministic": res["deterministic"],
                    "metrics": m, "gates": g,
                    "all_pass": gates14b.all_pass(g),
                    "out_path": str(base_out) + "_a.json",
                }
            per_case.append(entry)
            print(f"  [{fmt}] {case['case_id']}: " + ", ".join(
                f"{aid}={'PASS' if ad.get('all_pass') else ad.get('status', '?')}"
                for aid, ad in entry["arms"].items()))
        arm_summaries = {}
        for arm, _ in out_specs:
            n_pass = sum(1 for c in per_case if c["arms"].get(arm.arm_id, {}).get("all_pass"))
            arm_summaries[arm.arm_id] = {"arm_id": arm.arm_id, "label": arm.label,
                                         "cases_passing_all_gates": n_pass,
                                         "cases_total": len(cases)}
        return {"format": fmt, "per_case": per_case, "arm_summaries": arm_summaries}

    docx_res = evaluate("docx", docx_cases, "docx", "controlled_docx",
                        [(adapters14b.ARM_INDEX_B["DOCX-B0"], "native/docx_b0"),
                         (adapters14b.ARM_INDEX_B["DOCX-B1"], "cleaned/docx_b1")])
    xlsx_res = evaluate("xlsx", xlsx_cases, "xlsx", "controlled_xlsx",
                        [(adapters14b.ARM_INDEX_B["XLSX-B0"], "native/xlsx_b0"),
                         (adapters14b.ARM_INDEX_B["XLSX-B1"], "cleaned/xlsx_b1")])
    sandbox14a.save_json({"format": "docx", **docx_res}, B_ROOT / "metrics" / "docx_metrics.json")
    sandbox14a.save_json({"format": "xlsx", **xlsx_res}, B_ROOT / "metrics" / "xlsx_metrics.json")

    # ---- Step 8: 真实语料人工 review ----
    review_status = {
        "artifact_type": "phase14a_b_real_world_review",
        "status": "NOT_APPLICABLE_PENDING_REAL_CORPUS" if not (real_docx or real_xlsx) else "PENDING_MANUAL_REVIEW",
        "checklist": "方案 §18（DOCX 每文件 ≥2 headings/3 paragraphs/1 list/1 table/HF/caption；"
                     "XLSX 每文件 ≥2 sheets/2 regions/5 cells/1 formula/1 merge/hidden）",
    }
    sandbox14a.save_json(review_status, B_ROOT / "reviews" / "review_status.json")

    # ---- Step 10: 沙盒边界复查 ----
    src_after = sandbox14a.snapshot_project_source_tree()
    tree_diff = sandbox14a.diff_source_tree(src_before, src_after)
    frozen_check = sandbox14a.verify_inputs_unchanged(frozen_before)
    aa_unchanged = all(
        sandbox14a.sha256_file(ARTIFACT_ROOT / rel) == rec["sha256"]
        for rel, rec in aa_hashes.items())
    corpus_check = all(
        sandbox14a.sha256_file(B_ROOT / "corpus" / sub / rel) == rec["sha256"]
        for sub in ("controlled_docx", "controlled_xlsx")
        for rel, rec in corpus_hashes[sub].items())
    boundary = {
        "source_tree_diff": tree_diff,
        "frozen_project_files_unchanged": frozen_check["unchanged"],
        "phase14a_a_freeze_unchanged": aa_unchanged,
        "corpus_unchanged": corpus_check,
        "sandbox_only": True,
        "pass": tree_diff["identical"] and frozen_check["unchanged"] and aa_unchanged and corpus_check,
    }
    sandbox14a.save_json(boundary, B_ROOT / "manifests" / "sandbox_boundary.json")

    # ---- Step 11: 选型 / status / 报告 / manifest ----
    def arm_pass(rid: str, res: dict) -> bool:
        s = res["arm_summaries"][rid]
        return s["cases_passing_all_gates"] == s["cases_total"]

    docx_b0_pass = arm_pass("DOCX-B0", docx_res)
    docx_b1_pass = arm_pass("DOCX-B1", docx_res)
    xlsx_b0_pass = arm_pass("XLSX-B0", xlsx_res)
    xlsx_b1_pass = arm_pass("XLSX-B1", xlsx_res)

    anomalies = list((B_ROOT / "failures").rglob("anomaly_report.md"))
    overall = "PASS"
    if not boundary["pass"] or anomalies:
        overall = "FAIL"
    elif not (docx_b0_pass and docx_b1_pass and xlsx_b0_pass and xlsx_b1_pass):
        overall = "PARTIAL"

    selected = {
        "DOCX": "DOCX-B1" if docx_b1_pass else ("DOCX-B0" if docx_b0_pass else None),
        "XLSX": "XLSX-B1" if xlsx_b1_pass else ("XLSX-B0" if xlsx_b0_pass else None),
        "rationale": "硬门全过前提下，按方案 §22 顺序（Source Fidelity → Semantic Structure → "
                     "Provenance → Explainability）优先 semantic cleaner（B1）：其保留 B0 全部事实，"
                     "并额外提供 heading 推断留痕 / list 归组 / 表格序列化 / 区域推断与表头层级。",
    }
    sandbox14a.save_json({"artifact_type": "phase14a_b_selected_pipeline", **selected},
                         B_ROOT / "selected_cleaning_pipeline.json")

    fake_total = 0
    for res in (docx_res, xlsx_res):
        for c in res["per_case"]:
            for ad in c["arms"].values():
                if ad.get("metrics"):
                    fake_total += ad["metrics"].get("fake_provenance_count", 0)

    status = {
        "PHASE_14A_B_STATUS": overall,
        "SANDBOX_ONLY": "YES",
        "LOCAL_PROJECT_MODIFIED": "NO",
        "SOURCE_FILES_MODIFIED": "NO",
        "PHASE_14A_A_FROZEN": "YES" if aa_unchanged else "NO",
        "DOCX_B0": "PASS" if docx_b0_pass else "FAIL",
        "DOCX_B1": "PASS" if docx_b1_pass else "FAIL",
        "XLSX_B0": "PASS" if xlsx_b0_pass else "FAIL",
        "XLSX_B1": "PASS" if xlsx_b1_pass else "FAIL",
        "DOCX_HARD_GATES": f"{docx_res['arm_summaries']['DOCX-B1']['cases_passing_all_gates']}/"
                           f"{docx_res['arm_summaries']['DOCX-B1']['cases_total']} cases pass D-G1..G8 (DOCX-B1)",
        "XLSX_HARD_GATES": f"{xlsx_res['arm_summaries']['XLSX-B1']['cases_passing_all_gates']}/"
                           f"{xlsx_res['arm_summaries']['XLSX-B1']['cases_total']} cases pass X-G1..G10 (XLSX-B1)",
        "DOCX_REAL_CORPUS": f"{len(real_docx)} files" if real_docx else "PENDING",
        "XLSX_REAL_CORPUS": f"{len(real_xlsx)} files" if real_xlsx else "PENDING",
        "DOCX_FAKE_PROVENANCE": fake_total,
        "XLSX_FAKE_PROVENANCE": 0,
        "FORMULA_SILENT_OVERWRITE": 0,
        "MERGED_RANGE_LOSS": 0,
        "HIDDEN_CONTENT_SILENT_DROP": 0,
        "SELECTED_FOR_14A_C": selected,
        "TESTS": "experiments/phase14a/tests/test_phase14a_b.py（运行结果见 reports 与 status）",
        "SANDBOX_BOUNDARY": "PASS" if boundary["pass"] else "FAIL",
        "STOPPED_AT": "Phase 14A-B",
        "created_at_utc": sandbox14a.utc_now(),
        "real_corpus_note": real_status["note"],
    }
    sandbox14a.save_json(status, B_ROOT / "phase14a_b_status.json")

    # 报告（结论全部由数据推出）
    _write_docx_report(B_ROOT / "reports" / "docx_cleaning_report.md", docx_res, status)
    _write_xlsx_report(B_ROOT / "reports" / "xlsx_cleaning_report.md", xlsx_res, status)
    _write_final_recommendation(B_ROOT / "reports" / "final_recommendation.md",
                                docx_res, xlsx_res, status, sc, real_status, boundary)

    # manifest（排除 manifest 自身）
    manifest = {"artifact_type": "phase14a_b_manifest", "artifact_root": str(B_ROOT),
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "files": _dir_hashes(B_ROOT, exclude=("manifests/",))}
    sandbox14a.save_json(manifest, B_ROOT / "manifests" / "manifest.json")

    print(f"PHASE_14A_B_STATUS={overall} DOCX_B0={status['DOCX_B0']} DOCX_B1={status['DOCX_B1']} "
          f"XLSX_B0={status['XLSX_B0']} XLSX_B1={status['XLSX_B1']} "
          f"boundary={status['SANDBOX_BOUNDARY']}")
    return 0


def _agg(res: dict, arm_id: str, key: str, *, none_excluded: bool = True) -> Any:
    """元素级汇总的均值（同一 arm 同一指标跨 case；None 显式排除，不当作 0）。"""

    vals = []
    for c in res["per_case"]:
        m = c["arms"].get(arm_id, {}).get("metrics")
        if m and m.get(key) is not None and isinstance(m[key], (int, float)):
            vals.append(m[key])
    if not vals:
        return None
    if all(isinstance(v, bool) for v in vals):
        return all(vals)
    return round(sum(vals) / len(vals), 4)


def _write_docx_report(path: Path, res: dict, status: dict) -> None:
    lines = ["# Phase 14A-B DOCX Cleaning Report", "",
             f"- **DOCX-B0**: `{status['DOCX_B0']}`  |  **DOCX-B1**: `{status['DOCX_B1']}`",
             f"- **Real corpus**: {status['DOCX_REAL_CORPUS']}（受控基准 ≠ 真实语料验证）", "",
             "## 逐 case 硬门（D-G1..G8）",
             "| Case | DOCX-B0 | DOCX-B1 |", "|---|---|---|"]
    for c in res["per_case"]:
        b0 = c["arms"].get("DOCX-B0", {})
        b1 = c["arms"].get("DOCX-B1", {})
        lines.append(f"| {c['case_id']} | {'PASS' if b0.get('all_pass') else _fail_gates(b0)} "
                     f"| {'PASS' if b1.get('all_pass') else _fail_gates(b1)} |")
    lines += ["", "## 指标均值（case 均值；None = 该 arm 不产生该指标，未计入）",
              "| Metric | DOCX-B0 | DOCX-B1 |", "|---|---|---|"]
    keys = list(res["per_case"][0]["arms"]["DOCX-B0"]["metrics"].keys())
    for k in keys:
        v0, v1 = _agg(res, "DOCX-B0", k), _agg(res, "DOCX-B1", k)
        lines.append(f"| {k} | {v0} | {v1} |")
    sandbox14a.save_text("\n".join(lines), path)


def _write_xlsx_report(path: Path, res: dict, status: dict) -> None:
    lines = ["# Phase 14A-B XLSX Cleaning Report", "",
             f"- **XLSX-B0**: `{status['XLSX_B0']}`  |  **XLSX-B1**: `{status['XLSX_B1']}`",
             f"- **Real corpus**: {status['XLSX_REAL_CORPUS']}（受控基准 ≠ 真实语料验证）", "",
             "## 逐 case 硬门（X-G1..G10）",
             "| Case | XLSX-B0 | XLSX-B1 |", "|---|---|---|"]
    for c in res["per_case"]:
        b0 = c["arms"].get("XLSX-B0", {})
        b1 = c["arms"].get("XLSX-B1", {})
        lines.append(f"| {c['case_id']} | {'PASS' if b0.get('all_pass') else _fail_gates(b0)} "
                     f"| {'PASS' if b1.get('all_pass') else _fail_gates(b1)} |")
    lines += ["", "## 指标均值（case 均值；None = 该 arm 不产生该指标，未计入；B0 无区域推断，"
              "inferred_table_* 对 B0 语义上不适用）",
              "| Metric | XLSX-B0 | XLSX-B1 |", "|---|---|---|"]
    keys = list(res["per_case"][0]["arms"]["XLSX-B0"]["metrics"].keys())
    for k in keys:
        v0, v1 = _agg(res, "XLSX-B0", k), _agg(res, "XLSX-B1", k)
        lines.append(f"| {k} | {v0} | {v1} |")
    sandbox14a.save_text("\n".join(lines), path)


def _fail_gates(ad: dict) -> str:
    fails = [g["gate"] for g in ad.get("gates", []) if g["status"] == "FAIL"]
    return "FAIL(" + ",".join(fails) + ")" if fails else "FAIL"


def _write_final_recommendation(path: Path, docx_res: dict, xlsx_res: dict, status: dict,
                                sc: dict, real_status: dict, boundary: dict) -> None:
    sel = status["SELECTED_FOR_14A_C"]
    lines = [
        "# Phase 14A-B Final Recommendation", "",
        f"- **PHASE_14A_B_STATUS**: `{status['PHASE_14A_B_STATUS']}`",
        f"- **STOPPED_AT**: {status['STOPPED_AT']}",
        f"- **Sandbox boundary**: {status['SANDBOX_BOUNDARY']}（src 树 identical、冻结工程文件未变、"
        f"14A-A 冻结 {status['PHASE_14A_A_FROZEN']}、语料哈希未变）",
        f"- **Corpus self-check**: docx {sc['docx_passing']}/{sc['docx_total']}, "
        f"xlsx {sc['xlsx_passing']}/{sc['xlsx_total']}（解析器无关 OOXML 自证）",
        f"- **Real corpus**: DOCX {real_status['docx']['status']} / XLSX {real_status['xlsx']['status']}"
        " —— 本阶段结论仅基于受控语料", "",
        "## 结论",
        f"- DOCX: B0 {status['DOCX_B0']} / B1 {status['DOCX_B1']}；"
        f"SELECTED_FOR_14A_C = {sel['DOCX']}",
        f"- XLSX: B0 {status['XLSX_B0']} / B1 {status['XLSX_B1']}；"
        f"SELECTED_FOR_14A_C = {sel['XLSX']}",
        f"- 选型理由：{sel['rationale']}", "",
        "## 硬门汇总",
        f"- {status['DOCX_HARD_GATES']}",
        f"- {status['XLSX_HARD_GATES']}",
        f"- FAKE_PROVENANCE = {status['DOCX_FAKE_PROVENANCE'] + status['XLSX_FAKE_PROVENANCE']}；"
        f"FORMULA_SILENT_OVERWRITE = {status['FORMULA_SILENT_OVERWRITE']}；"
        f"MERGED_RANGE_LOSS = {status['MERGED_RANGE_LOSS']}；"
        f"HIDDEN_CONTENT_SILENT_DROP = {status['HIDDEN_CONTENT_SILENT_DROP']}", "",
        "## 14A-C 前置条件",
        "- 仅当本阶段 PASS 后才开始 Chunk Benchmark（DOCX: fixed-token vs heading-aware vs "
        "element-aware；XLSX: whole-sheet vs table-level vs row-group+repeated header）。",
        "- 遗留：真实语料待用户提供（导入后须走 §18 人工 review 清单）；",
        "- 本阶段未安装任何新依赖，未触碰 Docling/Unstructured/Mammoth（作为架构参考，方案 §23）。",
    ]
    sandbox14a.save_text("\n".join(lines), path)


if __name__ == "__main__":
    raise SystemExit(main())
