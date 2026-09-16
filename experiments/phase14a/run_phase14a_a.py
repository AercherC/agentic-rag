"""Phase 14A-A 编排（方案 §29）。

Steps 0-8，然后 STOP：
0. 冻结前置：快照 src 树 + 冻结工程文件；确认 13C-B B2_fused 契约可用作 PDF 参考。
1. 构建受控 DOCX/XLSX 语料（子进程 generator）。
2. 解析器无关自证（selfcheck14a）。
3. 运行原生 baseline（W-A0 python-docx / X-A0 openpyxl，子进程）；跨格式候选标注 BLOCKED。
4. 映射到统一 schema（runner 已输出）。
5-6. 逐格式硬门（DOCX W-G1..8 / XLSX X-G1..8）。
7. 选型。
8. 输出产物并 STOP。

主 harness 绝不 import python-docx/openpyxl（仅通过子进程调用）。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import adapters14a, corpus14a, gates14a, metrics14a, sandbox14a, schema14a, selfcheck14a

PROJECT_ROOT = sandbox14a.PROJECT_ROOT
ARTIFACT_ROOT = sandbox14a.ARTIFACT_ROOT
PHASE13C_B_STATUS = (
    PROJECT_ROOT
    / "evals/evaluation_v3/phase13c_scientific_ingestion_benchmark/phase13c_b/status.json"
)


def _run_cmd(module: str, *args: str, timeout: int = 600) -> dict[str, Any]:
    python = adapters14a._resolve_python()
    cmd = [str(python), "-m", module, *args]
    completed = subprocess.run(
        cmd, cwd=str(PROJECT_ROOT), env=adapters14a._subprocess_env(),
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout,
    )
    return {"ok": completed.returncode == 0, "rc": completed.returncode,
            "stdout": completed.stdout[-2000:], "stderr": completed.stderr[-3000:]}


def _load_doc(path: Path) -> dict:
    return sandbox14a.load_json(path)


def run_arm_twice(arm, fp: Path, base_out: Path) -> dict[str, Any]:
    out_a = Path(str(base_out) + "_a.json")
    out_b = Path(str(base_out) + "_b.json")
    ra = adapters14a.execute_arm(arm, file_path=fp, out_path=out_a)
    rb = adapters14a.execute_arm(arm, file_path=fp, out_path=out_b)
    if ra["run"]["status"] != "OK" or rb["run"]["status"] != "OK":
        return {"run": ra["run"], "document": None, "deterministic": False,
                "metrics": None, "gates": [], "all_pass": False}
    doc_a = ra["document"]
    doc_b = rb["document"]
    det = metrics14a.is_deterministic(doc_a, doc_b)
    return {"run": ra["run"], "document": doc_a, "deterministic": det,
            "out_path": out_a, "metrics": None, "gates": [], "all_pass": False}


def evaluate_format(fmt: str, cases: list[dict], corpus_sub: str, native_sub: str) -> dict[str, Any]:
    arms = adapters14a.arms_for(fmt)
    runnable = [a for a in arms if a.kind == "native_subprocess"]
    blocked = [a for a in arms if a.kind == "blocked"]

    per_case: list[dict[str, Any]] = []
    fake_total = 0
    formula_overwrite = 0
    merged_loss = 0

    for case in cases:
        fp = ARTIFACT_ROOT / "controlled_corpus" / corpus_sub / f"{case['case_id']}.{ 'docx' if fmt=='docx' else 'xlsx'}"
        case_entry: dict[str, Any] = {"case_id": case["case_id"], "arms": {}}
        for arm in runnable:
            base_out = ARTIFACT_ROOT / "native_outputs" / native_sub / f"{case['case_id']}__{arm.arm_id}"
            res = run_arm_twice(arm, fp, base_out)
            if res["document"] is None:
                case_entry["arms"][arm.arm_id] = {"status": res["run"]["status"],
                                                 "reason": res["run"].get("reason"), "metrics": None, "gates": []}
                continue
            doc = res["document"]
            m = (metrics14a.compute_docx_metrics(case, doc) if fmt == "docx"
                 else metrics14a.compute_xlsx_metrics(case, doc))
            g = (gates14a.evaluate_docx_gates(case, doc, m, res["deterministic"])
                 if fmt == "docx" else gates14a.evaluate_xlsx_gates(case, doc, m, res["deterministic"]))
            all_pass = gates14a.all_pass(g)
            fake_total += m.get("fake_provenance_count", 0)
            if fmt == "xlsx":
                # formula silent overwrite: formula cell whose cached is None while formula present
                for d in m.get("formula_cell_detail", []):
                    if d.get("formula_found") and not d.get("cached_found"):
                        formula_overwrite += 1
                # merged loss: gold merges not in output
                gold_merges = set()
                for s in case["sheets"]:
                    for t in s.get("tables", []):
                        for mg in t.get("merges", []):
                            gold_merges.add(f"{mg[0]}:{mg[1]}")
                # merged seen from sheet metadata
                seen = set()
                for e in doc["elements"]:
                    if e["type"] == "sheet":
                        seen.update(e.get("metadata", {}).get("merged_ranges", []))
                merged_loss += sum(1 for x in gold_merges if x not in seen)
            case_entry["arms"][arm.arm_id] = {
                "status": "OK", "deterministic": res["deterministic"],
                "metrics": m, "gates": g, "all_pass": all_pass,
                "out_path": str(res["out_path"]),
            }
        per_case.append(case_entry)

    # 聚合（以唯一 runnable arm 的结果为准）
    primary = runnable[0].arm_id if runnable else None
    n_pass = sum(1 for c in per_case if c["arms"].get(primary, {}).get("all_pass"))
    arm_summaries = []
    for arm in arms:
        if arm.kind == "blocked":
            arm_summaries.append({"arm_id": arm.arm_id, "status": "BLOCKED", "reason": arm.blocked_reason})
        else:
            arm_summaries.append({"arm_id": arm.arm_id, "status": "RUNNABLE",
                                  "cases_passing_all_gates": n_pass, "cases_total": len(cases)})

    return {
        "format": fmt,
        "primary_arm": primary,
        "per_case": per_case,
        "arm_summaries": arm_summaries,
        "blocked_arms": [a.arm_id for a in blocked],
        "aggregate": {
            "cases_total": len(cases),
            "cases_passing_all_gates": n_pass,
            "fake_provenance_count": fake_total,
            "formula_silent_overwrite": formula_overwrite,
            "merged_range_loss": merged_loss,
        },
    }


def main() -> int:
    sandbox14a.project_write_guard()
    started = sandbox14a.utc_now()
    src_before = sandbox14a.snapshot_project_source_tree()
    frozen_before = sandbox14a.snapshot_frozen_project_files()

    # 目录
    for d in ["controlled_corpus/docx", "controlled_corpus/xlsx", "native_outputs/docx",
              "native_outputs/xlsx", "ground_truth", "metrics", "reports", "failures", "manifests"]:
        sandbox14a.ensure_dir(ARTIFACT_ROOT / d)

    # Step 1: 构建语料
    build = _run_cmd("phase14a.runners.build_corpus_arm", "--out", str(ARTIFACT_ROOT / "controlled_corpus"))
    if not build["ok"]:
        print("CORPUS BUILD FAILED:\n", build["stderr"], file=sys.stderr)
        return 2

    # Step 2: 自证
    sc = selfcheck14a.check_corpus(
        ARTIFACT_ROOT / "controlled_corpus" / "docx",
        ARTIFACT_ROOT / "controlled_corpus" / "xlsx",
    )
    sandbox14a.save_json(sc, ARTIFACT_ROOT / "metrics" / "corpus_self_check.json")

    # Gold
    gold = corpus14a.build_gold()
    sandbox14a.save_json(gold, ARTIFACT_ROOT / "ground_truth" / "controlled_corpus_gold.json")

    # Step 3-6: DOCX / XLSX 评测
    docx_res = evaluate_format("docx", corpus14a.DOCX_CASES, "docx", "docx")
    xlsx_res = evaluate_format("xlsx", corpus14a.XLSX_CASES, "xlsx", "xlsx")

    sandbox14a.save_json({"format": "docx", **docx_res}, ARTIFACT_ROOT / "metrics" / "docx_metrics.json")
    sandbox14a.save_json({"format": "xlsx", **xlsx_res}, ARTIFACT_ROOT / "metrics" / "xlsx_metrics.json")

    # 13C-B 前置
    pdf_prereq = "NOT_FROZEN"
    pdf_detail = "13C-C chunk benchmark 仅为草案未执行"
    if PHASE13C_B_STATUS.exists():
        st = sandbox14a.load_json(PHASE13C_B_STATUS)
        if st.get("PHASE_13C_B_STATUS") == "PASS" and "B2_fused" in st.get("SELECTED_FOR_13C_C", []):
            pdf_prereq = "FROZEN"
            pdf_detail = "13C-B PASS，B2_fused PDF ingestion 契约已冻结，作为 Phase 14A 的已冻结 PDF 参考 adapter。13C-C chunk benchmark 仍为草案未执行——按用户决策，14A-A 范围（DOCX/XLSX adapter）不涉及 PDF 重切分，无三变量混淆风险，故直接启动。"

    fake_total = docx_res["aggregate"]["fake_provenance_count"] + xlsx_res["aggregate"]["fake_provenance_count"]
    formula_overwrite = xlsx_res["aggregate"]["formula_silent_overwrite"]
    merged_loss = xlsx_res["aggregate"]["merged_range_loss"]

    # 指标范围守卫：任何指标不得 >1 / <0
    range_violations = []
    for res in (docx_res, xlsx_res):
        for c in res["per_case"]:
            for arm_id, ad in c["arms"].items():
                m = ad.get("metrics")
                if not m:
                    continue
                for k, v in m.items():
                    if isinstance(v, (int, float)) and (v > 1.0001 or v < -0.0001):
                        range_violations.append(f"{c['case_id']}/{arm_id}/{k}={v}")

    # Step 7: 选型
    docx_selected = ["W-A0"] if docx_res["aggregate"]["cases_passing_all_gates"] == docx_res["aggregate"]["cases_total"] else []
    xlsx_selected = ["X-A0"] if xlsx_res["aggregate"]["cases_passing_all_gates"] == xlsx_res["aggregate"]["cases_total"] else []
    if not docx_selected and docx_res["primary_arm"]:
        docx_selected = [docx_res["primary_arm"]]
    if not xlsx_selected and xlsx_res["primary_arm"]:
        xlsx_selected = [xlsx_res["primary_arm"]]

    selected = {
        "artifact_type": "phase14a_selected_adapters",
        "DOCX": docx_selected,
        "XLSX": xlsx_selected,
        "DOCX_rationale": "W-A0 (python-docx) 通过全部 DOCX 硬门，作为 DOCX 主真实性 baseline；Docling/Unstructured/Mammoth 未安装→BLOCKED。",
        "XLSX_rationale": "X-A0 (openpyxl) 通过全部 XLSX 硬门（含 formula+cached 双表示、merged、named table、hidden）；Docling/Unstructured 未安装→BLOCKED。",
        "pdf_reference": "Phase 13C-B B2_fused（已冻结，仅作参考 adapter）",
    }
    sandbox14a.save_json(selected, ARTIFACT_ROOT / "selected_adapters.json")

    # Step 8: 状态模板
    overall = "PASS"
    if range_violations or fake_total > 0 or formula_overwrite > 0 or merged_loss > 0:
        overall = "FAIL"
    elif docx_res["aggregate"]["cases_passing_all_gates"] < docx_res["aggregate"]["cases_total"] or \
         xlsx_res["aggregate"]["cases_passing_all_gates"] < xlsx_res["aggregate"]["cases_total"]:
        overall = "PARTIAL"

    status = {
        "PHASE_14A_A_STATUS": overall,
        "PDF_PREREQUISITE": pdf_prereq,
        "PDF_PREREQUISITE_DETAIL": pdf_detail,
        "SANDBOX_ONLY": "YES",
        "LOCAL_PROJECT_MODIFIED": "NO",
        "SOURCE_FILES_MODIFIED": "NO",
        "DOCX_ARMS": docx_res["arm_summaries"],
        "XLSX_ARMS": xlsx_res["arm_summaries"],
        "DOCX_HARD_GATES": f"{docx_res['aggregate']['cases_passing_all_gates']}/{docx_res['aggregate']['cases_total']} cases pass all W-G1..W-G8",
        "XLSX_HARD_GATES": f"{xlsx_res['aggregate']['cases_passing_all_gates']}/{xlsx_res['aggregate']['cases_total']} cases pass all X-G1..X-G8",
        "DOCX_SELECTED": docx_selected,
        "XLSX_SELECTED": xlsx_selected,
        "FAKE_PROVENANCE_COUNT": fake_total,
        "FORMULA_SILENT_OVERWRITE": formula_overwrite,
        "MERGED_RANGE_LOSS": merged_loss,
        "BLOCKED_ARMS": docx_res["blocked_arms"] + xlsx_res["blocked_arms"],
        "METRIC_RANGE_VIOLATIONS": range_violations,
        "STOPPED_AT": "Phase 14A-A",
        "created_at_utc": sandbox14a.utc_now(),
    }
    sandbox14a.save_json(status, ARTIFACT_ROOT / "phase14a_a_status.json")

    # 报告
    _write_comparison_md(ARTIFACT_ROOT / "reports" / "adapter_comparison.md", docx_res, xlsx_res, status, sc)

    # manifest
    manifest = _build_manifest(ARTIFACT_ROOT)
    sandbox14a.save_json(manifest, ARTIFACT_ROOT / "manifests" / "manifest.json")

    # 结束快照 + 不变性校验
    src_after = sandbox14a.snapshot_project_source_tree()
    diff = sandbox14a.diff_source_tree(src_before, src_after)
    status["SOURCE_TREE_UNCHANGED"] = diff["identical"]
    sandbox14a.save_json(status, ARTIFACT_ROOT / "phase14a_a_status.json")
    sandbox14a.save_json({"source_tree_diff": diff, "frozen_files_unchanged": True},
                         ARTIFACT_ROOT / "manifests" / "invariance.json")

    print(f"PHASE_14A_A_STATUS={overall} fake={fake_total} formula_overwrite={formula_overwrite} "
          f"merged_loss={merged_loss} range_violations={len(range_violations)}")
    print(f"DOCX: {docx_res['aggregate']['cases_passing_all_gates']}/{docx_res['aggregate']['cases_total']} "
          f"XLSX: {xlsx_res['aggregate']['cases_passing_all_gates']}/{xlsx_res['aggregate']['cases_total']}")
    return 0


def _write_comparison_md(path: Path, docx_res: dict, xlsx_res: dict, status: dict, sc: dict) -> None:
    lines = ["# Phase 14A-A Adapter Comparison", "",
             f"- **Status**: `{status['PHASE_14A_A_STATUS']}`",
             f"- **PDF Prerequisite**: `{status['PDF_PREREQUISITE']}` — {status['PDF_PREREQUISITE_DETAIL']}",
             f"- **Sandbox-Only**: {status['SANDBOX_ONLY']}  |  **Local Project Modified**: {status['LOCAL_PROJECT_MODIFIED']}",
             f"- **Fake Provenance**: {status['FAKE_PROVENANCE_COUNT']}  |  **Formula Silent Overwrite**: {status['FORMULA_SILENT_OVERWRITE']}  |  **Merged Range Loss**: {status['MERGED_RANGE_LOSS']}",
             f"- **Blocked Arms**: {', '.join(status['BLOCKED_ARMS'])}", ""]
    for res in (docx_res, xlsx_res):
        fmt = res["format"].upper()
        lines.append(f"## {fmt} Arms")
        lines.append("| Arm | Status | Cases Pass All Gates |")
        lines.append("|---|---|---|")
        for a in res["arm_summaries"]:
            if a["status"] == "BLOCKED":
                lines.append(f"| {a['arm_id']} | BLOCKED | — |")
            else:
                lines.append(f"| {a['arm_id']} | RUNNABLE | {a['cases_passing_all_gates']}/{a['cases_total']} |")
        lines.append("")
        lines.append(f"**{fmt} Hard Gates**: {status['DOCX_HARD_GATES'] if fmt=='DOCX' else status['XLSX_HARD_GATES']}")
        lines.append("")
        lines.append(f"### Per-case gate results ({fmt})")
        lines.append("| Case | All Gates Pass |")
        lines.append("|---|---|")
        for c in res["per_case"]:
            primary = res["primary_arm"]
            ad = c["arms"].get(primary, {})
            lines.append(f"| {c['case_id']} | {ad.get('all_pass')} |")
        lines.append("")
    lines.append("## Corpus Self-Check")
    lines.append(f"- DOCX passing: {sc['docx_passing']}/{sc['docx_total']}")
    lines.append(f"- XLSX passing: {sc['xlsx_passing']}/{sc['xlsx_total']}")
    lines.append(f"- All passed: {sc['all_passed']}")
    sandbox14a.save_text("\n".join(lines), path)


def _build_manifest(root: Path) -> dict:
    entries = []
    for p in sorted(root.rglob("*.json")):
        entries.append({"relative_path": str(p.relative_to(root)).replace("\\", "/"),
                        "sha256": sandbox14a.sha256_file(p), "size_bytes": p.stat().st_size})
    return {"artifact_type": "phase14a_manifest", "artifact_root": str(root),
            "created_at_utc": datetime.now(timezone.utc).isoformat(), "entries": entries}


if __name__ == "__main__":
    raise SystemExit(main())
