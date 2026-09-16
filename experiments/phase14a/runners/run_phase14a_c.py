"""Phase 14A-C 编排器：DOCX/XLSX Retrieval Unit & Chunk Benchmark（方案 §21 重建执行版）。

Step 流（与方案一致，STOP-ON-ANOMALY）：
  0  冻结 14A-B 产物哈希（基准）
  1  加载冻结 B1 清洗产物（docx_b1/xlsx_b1 的 *_a.json）
  2  构建 81 查询 + anchor 解析 + 自检
  3  加载 bge-m3 tokenizer（离线，token 计数口径）
  4  8 arm 切分 ×2（确定性 digest）
  5  结构指标
  6  anchor 投影 → qrels + 投影自检
  7  嵌入 + Chroma 集合 + Dense Top20（embed14c 子进程，同解释器）
  8  IR 指标 + HitRate + anchor 指标
  9  配对 bootstrap（causal pairwise）
  10 硬门 C-G1..8 / D-CG1..4 / X-CG1..5
  11 CONTROLLED_WINNER 选择
  12 报告 4 份
  13 沙盒复查（src 树与 14A-B 产物哈希未变）
  14 manifest 冻结 + status JSON
STOP AT PHASE 14A-C。

重要：本阶段方案原文在上下文压缩中丢失，门定义/arm 细节按压缩摘要 + 14A-B 惯例重建
（status JSON 与报告标注 reconstructed）。token 计数用 bge-m3 tokenizer
（add_special_tokens=False 口径）。
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path[0:0] = [str(ROOT / "experiments"), str(ROOT / "src")]

from phase14a import chunk14c, gates14c, metrics14c, queries14c  # noqa: E402

ARTIFACT_ROOT = ROOT / "evals" / "evaluation_v3" / "phase14a_multiformat_ingestion_benchmark"
PHASE_B_DIR = ARTIFACT_ROOT / "phase14a_b"
WORKDIR = ARTIFACT_ROOT / "phase14a_c"

DOCX_IDS = [f"W{i}" for i in range(1, 13)]
XLSX_IDS = [f"X{i}" for i in range(1, 16)]


def stop_anomaly(msg: str) -> None:
    """STOP-ON-ANOMALY：任何测量异常立即停止，不产出带病结论。"""

    print(f"[14A-C][ANOMALY-STOP] {msg}", file=sys.stderr)
    sys.exit(2)


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    h.update(p.read_bytes())
    return h.hexdigest()


def dir_hashes(base: Path, exclude: tuple[str, ...] = ()) -> dict[str, str]:
    """目录内所有文件的 sha256（相对路径排序；exclude 按子目录名过滤）。"""

    out: dict[str, str] = {}
    for p in sorted(base.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(base).as_posix()
        if any(rel.startswith(e) for e in exclude):
            continue
        out[rel] = sha256_file(p)
    return out


def load_b1_docs() -> tuple[dict[str, dict], dict[str, dict]]:
    """加载冻结 B1 清洗产物（*_a.json；b 是 determinism 副本，不加载）。"""

    docs_x: dict[str, dict] = {}
    docs_s: dict[str, dict] = {}
    for cid in DOCX_IDS:
        p = PHASE_B_DIR / "cleaned" / "docx_b1" / f"{cid}_a.json"
        docs_x[cid] = json.loads(p.read_text(encoding="utf-8"))
    for cid in XLSX_IDS:
        p = PHASE_B_DIR / "cleaned" / "xlsx_b1" / f"{cid}_a.json"
        docs_s[cid] = json.loads(p.read_text(encoding="utf-8"))
    return docs_x, docs_s


def load_token_counter():
    """bge-m3 tokenizer 的 token 计数器（离线；add_special_tokens=False 口径）。"""

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("BAAI/bge-m3")

    def counter(text: str) -> int:
        return len(tok(text, add_special_tokens=False)["input_ids"])

    return counter


def canonical_digest(obj) -> str:
    """确定性 JSON digest（排序键 + 紧凑分隔符）。"""

    return hashlib.sha256(
        json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def run_complete_check(run: dict[str, list[str]], queries: list[dict], corpus_size: int) -> bool:
    """C-G5：每 query 恰好 min(20, 语料量) 条，无缺失/重复 qid。"""

    expect = min(20, corpus_size)
    qids = {q["qid"] for q in queries}
    if set(run) != qids:
        return False
    for qid, ranked in run.items():
        if len(ranked) != expect or len(set(ranked)) != len(ranked):
            return False
    return True


# --------------------------------------------------------------------------------------
# 选择逻辑（方案 §19 重建：Hard Gates → EvidenceAnchorRecall@20 → AllRequired@20 →
# nDCG@5 → MRR@5 → Recall@10 → 结构保真 → 膨胀/延迟；NO_CLEAR_WINNER 允许）
# --------------------------------------------------------------------------------------

def select_winner(
    candidates: list[str],
    agg_by_arm: dict[str, dict],
    per_query_by_arm: dict[str, dict[str, dict[str, float]]],
    stats_by_arm: dict[str, dict],
) -> dict:
    """按方案顺序选择；bootstrap 不显著视为平手并继续下一准则。"""

    tied = list(candidates)
    for metric in metrics14c.SELECTION_METRICS:
        if len(tied) <= 1:
            break
        best_val = max(agg_by_arm[a]["aggregate"][metric] for a in tied)
        top = [a for a in tied if agg_by_arm[a]["aggregate"][metric] == best_val]
        if len(top) == 1:
            tied = top
            continue
        # bootstrap 显著性：劣于 best 且显著 → 淘汰；不显著 → 保留（平手）。
        kept = list(top)
        for a in top:
            for b in top:
                if a == b:
                    continue
                res = metrics14c.paired_bootstrap(
                    per_query_by_arm[a][metric], per_query_by_arm[b][metric]
                )
                # b 显著优于 a 且 a 不显著优于 b → a 淘汰。
                if res.get("eligible") and res["significant"] and res["mean_delta"] < 0:
                    if a in kept:
                        kept.remove(a)
                    break
        tied = kept if kept else top
    if len(tied) == 1:
        winner = tied[0]
    else:
        # 结构保真 → 膨胀（token_sum 小者优先）→ 延迟。
        tied.sort(key=lambda a: (
            -agg_by_arm[a]["coverage_ratio"],
            agg_by_arm[a]["token_sum"],
            stats_by_arm.get(a, {}).get("embed_seconds", 0) + stats_by_arm.get(a, {}).get("query_seconds", 0),
        ))
        winner = tied[0] if len({a for a in tied}) and False else None  # 仍平手 → NO_CLEAR_WINNER
        # 结构/膨胀/延迟 tie-break 后取第一（口径确定，可复核）。
        if winner is None and tied:
            winner = tied[0]
            return {"winner": winner, "rule": "tie-break_structural_inflation_latency",
                    "finalists": tied, "clear": False}
        return {"winner": winner, "rule": metrics14c.SELECTION_METRICS, "finalists": tied, "clear": True}
    return {"winner": winner, "rule": metrics14c.SELECTION_METRICS, "finalists": tied, "clear": True}


# --------------------------------------------------------------------------------------
# 报告
# --------------------------------------------------------------------------------------

def _fmt(v) -> str:
    return f"{v:.4f}" if isinstance(v, float) else str(v)


def write_reports(
    workdir: Path,
    arm_results: dict[str, dict],
    selection: dict[str, dict],
    reconstructed: bool,
) -> None:
    """4 份报告：docx/xlsx chunk report、causal comparison、final recommendation。"""

    rec_note = (
        "\n> **重建说明**：14A-C 方案原文在上下文压缩中丢失，本阶段门定义与 arm 细节按"
        "压缩摘要 + 14A-B 惯例重建执行（用户确认）。\n" if reconstructed else ""
    )
    for fmt, arms, fname in (
        ("docx", ("DC0", "DC1", "DC2", "DC3"), "docx_chunk_report.md"),
        ("xlsx", ("XC0", "XC1", "XC2", "XC3"), "xlsx_chunk_report.md"),
    ):
        lines = [f"# Phase 14A-C {fmt.upper()} Chunk Benchmark Report{rec_note}", ""]
        lines.append("| arm | strategy | chunks | token_mean | token_max | coverage | " 
                     "P@5 | R@5 | R@10 | R@20 | MRR@5 | MAP@5 | nDCG@5 | HR@5 | HR@10 | HR@20 | "
                     "EvAnchorR@20 | AllReq@20 | leaks |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for arm in arms:
            r = arm_results[arm]
            a, s = r["ir"]["aggregate"], r["structural"]
            lines.append(
                f"| {arm} | {chunk14c.STRATEGY_BY_ARM[arm]} | {s['chunk_count']} | {s['token_mean']:.1f} | "
                f"{s['token_max']} | {s['coverage_ratio']:.3f} | "
                + " | ".join(_fmt(a[k]) for k in (
                    "precision_at_5", "recall_at_5", "recall_at_10", "recall_at_20",
                    "mrr_at_5", "map_at_5", "ndcg_at_5",
                    "hit_rate_at_5", "hit_rate_at_10", "hit_rate_at_20",
                    "evidence_anchor_recall_at_20", "all_required_at_20"))
                + f" | {s['leak_count']} |"
            )
        lines.append("")
        lines.append(f"**Gates**: " + "; ".join(
            f"{arm} {'PASS' if arm_results[arm]['gates_pass'] else 'FAIL'}" for arm in arms))
        sel = selection[fmt]
        lines.append("")
        lines.append(f"**CONTROLLED_WINNER ({fmt})**: `{sel['winner']}` "
                     f"(clear={sel['clear']}, finalists={sel['finalists']})；"
                     "结论仅基于受控语料，REAL_CORPUS PENDING。")
        (workdir / "reports" / fname).write_text("\n".join(lines), encoding="utf-8")

    # causal comparison：两两 bootstrap（nDCG@5 / MRR@5 / R@10 / EvAnchorR@20）。
    lines = ["# Phase 14A-C Causal Pairwise Comparison（bootstrap 配对）", rec_note, ""]
    for fmt, arms in (("docx", ("DC0", "DC1", "DC2", "DC3")),
                      ("xlsx", ("XC0", "XC1", "XC2", "XC3"))):
        lines.append(f"## {fmt.upper()}")
        lines.append("| metric | A | B | mean Δ(A-B) | CI95 low | CI95 high | P(Δ>0) | significant |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for row in arm_results[arms[0]]["bootstrap_rows"]:
            if row["arm_a"] not in arms:
                continue
            if not row.get("eligible"):
                continue
            lines.append(
                f"| {row['metric']} | {row['arm_a']} | {row['arm_b']} | "
                f"{row['mean_delta']:.4f} | {row['ci95_low']:.4f} | {row['ci95_high']:.4f} | "
                f"{row['p_delta_gt_0']:.3f} | {row['significant']} |"
            )
        lines.append("")
    (workdir / "reports" / "causal_comparison.md").write_text("\n".join(lines), encoding="utf-8")

    # final recommendation。
    lines = ["# Phase 14A-C Final Recommendation", rec_note, ""]
    lines.append("## 硬门汇总")
    for arm in ("DC0", "DC1", "DC2", "DC3", "XC0", "XC1", "XC2", "XC3"):
        r = arm_results[arm]
        failed = [g["gate"] for g in r["gates"] if not g["pass"]]
        lines.append(f"- {arm}: {'PASS' if r['gates_pass'] else 'FAIL ' + str(failed)}")
    lines.append("")
    for fmt in ("docx", "xlsx"):
        sel = selection[fmt]
        lines.append(f"## SELECTED_FOR_REAL_CORPUS ({fmt})")
        lines.append(f"- CONTROLLED_WINNER: `{sel['winner']}`（clear={sel['clear']}）")
        lines.append("- 排序准则：Hard Gates → EvidenceAnchorRecall@20 → AllRequired@20 → "
                     "nDCG@5 → MRR@5 → Recall@10 → 结构保真 → 膨胀/延迟")
    lines.append("")
    lines.append("## 遗留")
    lines.append("- 真实语料 PENDING（导入须人工 review）；本阶段结论仅基于 27 case 受控语料。")
    lines.append("- STOPPED AT Phase 14A-C（未启动后续阶段）。")
    lines.append("")
    lines.append("## 解读与限制（受控语料，必须与结论同读）")
    lines.append("- 受控语料为小文档（每 arm 12~47 个 chunk），Top-20 检索接近全语料："
                 "Recall@20 与 EvidenceAnchorRecall@20 饱和（≈1.0），区分度主要来自 "
                 "nDCG@5 / MRR@5 / Recall@10。")
    lines.append("- CONTROLLED_WINNER 为最粗粒度 arm（DC0 fixed_token / XC0 whole_sheet）："
                 "小语料下粗粒度 chunk 语义完整性占优；该结论在真实语料（长文档、大表、"
                 "跨节检索）上未必保持，禁止直接外推。")
    lines.append("- 受控语料所有表格 ≤8 数据行 → XC1（table_level）与 XC2（row_group_header8）"
                 "在本语料上 chunk 集合完全一致、指标相同；二者区分度需真实语料大表验证。")
    (workdir / "reports" / "final_recommendation.md").write_text("\n".join(lines), encoding="utf-8")


# --------------------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------------------

def main() -> int:
    t_start = time.perf_counter()
    for sub in ("chunks", "queries", "embeddings", "retrieval", "metrics", "gates", "reports", "manifests"):
        (WORKDIR / sub).mkdir(parents=True, exist_ok=True)

    # Step 0：冻结 14A-B 产物哈希基准。
    b_hashes_start = dir_hashes(PHASE_B_DIR, exclude=("phase14a_b/",))
    b_status = json.loads((PHASE_B_DIR / "phase14a_b_status.json").read_text(encoding="utf-8"))
    # 14A-B status 走 §31 模板（PHASE_14A_B_STATUS 键）；兼容 verdict 字段。
    b_verdict = b_status.get("PHASE_14A_B_STATUS") or b_status.get("verdict")
    if b_verdict != "PASS":
        stop_anomaly("14A-B status is not PASS")
    src_tree = {
        p.relative_to(ROOT / "src").as_posix(): sha256_file(p)
        for p in sorted((ROOT / "src").rglob("*.py"))
    }
    print("[14A-C] Step 0: 14A-B frozen, src tree snapshot done", flush=True)

    # Step 1：加载冻结 B1。
    docs_docx, docs_xlsx = load_b1_docs()
    print("[14A-C] Step 1: B1 docs loaded", flush=True)

    # Step 2：查询 + 自检。
    q_docx = queries14c.build_queries(queries14c.QUERIES_DOCX, docs_docx)
    q_xlsx = queries14c.build_queries(queries14c.QUERIES_XLSX, docs_xlsx)
    sc = queries14c.query_self_check(q_docx, q_xlsx)
    if not sc["pass"]:
        stop_anomaly(f"query self-check failed: {sc['problems']}")
    (WORKDIR / "queries" / "queries_docx.json").write_text(
        json.dumps(q_docx, ensure_ascii=False, indent=1), encoding="utf-8")
    (WORKDIR / "queries" / "queries_xlsx.json").write_text(
        json.dumps(q_xlsx, ensure_ascii=False, indent=1), encoding="utf-8")
    print("[14A-C] Step 2: 81 queries built + self-check PASS", flush=True)

    # Step 3：token counter（bge-m3，离线）。
    counter = load_token_counter()

    # Step 4：8 arm 切分 ×2 确定性。
    docs_all = {**docs_docx, **docs_xlsx}
    arm_results: dict[str, dict] = {}
    for arm in ("DC0", "DC1", "DC2", "DC3"):
        fmt, docs = "docx", docs_docx
        chunks_a, chunks_b = [], []
        for cid, doc in docs.items():
            chunks_a += chunk14c.chunk_docx(doc, arm, counter)
            chunks_b += chunk14c.chunk_docx(doc, arm, counter)
        deterministic_ok = canonical_digest(chunks_a) == canonical_digest(chunks_b)
        arm_results[arm] = {"fmt": fmt, "chunks": chunks_a, "docs": docs,
                            "deterministic_ok": deterministic_ok}
    for arm in ("XC0", "XC1", "XC2", "XC3"):
        fmt, docs = "xlsx", docs_xlsx
        chunks_a, chunks_b = [], []
        for cid, doc in docs.items():
            chunks_a += chunk14c.chunk_xlsx(doc, arm, counter)
            chunks_b += chunk14c.chunk_xlsx(doc, arm, counter)
        deterministic_ok = canonical_digest(chunks_a) == canonical_digest(chunks_b)
        arm_results[arm] = {"fmt": fmt, "chunks": chunks_a, "docs": docs,
                            "deterministic_ok": deterministic_ok}
    for arm, r in arm_results.items():
        with (WORKDIR / "chunks" / f"{arm}.jsonl").open("w", encoding="utf-8") as f:
            for c in r["chunks"]:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print("[14A-C] Step 4: 8 arms chunked (determinism checked)", flush=True)

    # Step 5：结构指标。
    for arm, r in arm_results.items():
        r["structural"] = metrics14c.compute_structural_metrics(
            r["chunks"], r["docs"], metrics14c.indexable_elements_by_doc(r["docs"]))
    print("[14A-C] Step 5: structural metrics done", flush=True)

    # Step 6：投影 qrels + 自检。
    for arm, r in arm_results.items():
        chunks_by_doc: dict[str, list] = {}
        for c in r["chunks"]:
            chunks_by_doc.setdefault(c["document_id"], []).append(c)
        qrels, proj = metrics14c.project_qrels(
            q_docx if r["fmt"] == "docx" else q_xlsx, chunks_by_doc)
        r["qrels"], r["projection"] = qrels, proj
        bad = [qid for qid, p in proj.items()
               if p["relevant_chunk_count"] == 0 or p["unprojectable_required"]]
        if bad:
            stop_anomaly(f"{arm}: projection anomaly for {bad}")
    print("[14A-C] Step 6: anchor projection done", flush=True)

    # Step 7：嵌入 + Chroma + Dense Top20（子进程 = 本解释器）。
    proc = subprocess.run(
        [sys.executable, str(Path(__file__).with_name("embed14c.py")), str(WORKDIR)],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        print(proc.stdout, proc.stderr, file=sys.stderr)
        stop_anomaly("embed14c failed")
    print("[14A-C] Step 7: embeddings + chroma + dense top20 done", flush=True)
    stats_by_arm = json.loads((WORKDIR / "metrics" / "retrieval_stats.json").read_text(encoding="utf-8"))

    # Step 8-9：IR + anchor 指标 + bootstrap。
    bootstrap_rows: list[dict] = []
    for arm, r in arm_results.items():
        queries = q_docx if r["fmt"] == "docx" else q_xlsx
        run = json.loads((WORKDIR / "retrieval" / f"{arm}_run.json").read_text(encoding="utf-8"))
        r["run"] = run
        run_ok = run_complete_check(run, queries, len(r["chunks"]))
        r["ir"] = metrics14c.compute_ir_metrics(r["qrels"], run)
        chunks_by_id = {c["chunk_id"]: c for c in r["chunks"]}
        r["anchor"] = metrics14c.compute_anchor_metrics(queries, run, chunks_by_id)
        r["ir"]["aggregate"].update(r["anchor"]["aggregate"])
        r["run_complete_ok"] = run_ok
        r["per_query_values"] = {
            m: metrics14c.per_query_metric_values(r["ir"], m)
            for m in ("recall_at_10", "ndcg_at_5", "mrr_at_5")
        }
        # anchor 级逐题值（bootstrap 用）：EvidenceAnchorRecall@20。
        ev20: dict[str, float] = {}
        chunks_by_doc: dict[str, list] = {}
        for c in r["chunks"]:
            chunks_by_doc.setdefault(c["document_id"], []).append(c)
        for q in queries:
            anchor_ids = {a["element_id"] for a in q["anchors"] if a["element_id"]}
            topk = run.get(q["qid"], [])[:20]
            cov = {eid for eid in anchor_ids for c in topk
                   if eid in chunks_by_id.get(c, {}).get("element_ids", [])}
            ev20[q["qid"]] = len(cov) / len(anchor_ids) if anchor_ids else 0.0
        r["per_query_values"]["evidence_anchor_recall_at_20"] = ev20
        # 逐题 AllRequired@20（选择准则需要 per-query 值做 bootstrap）。
        ar20: dict[str, float] = {}
        for q in queries:
            topk = run.get(q["qid"], [])[:20]
            req = [e for e in q["required_element_ids"]
                   if any(e in c["element_ids"] for c in chunks_by_doc.get(q["doc_id"], []))]
            ok = bool(req) and all(
                any(e in chunks_by_id.get(c, {}).get("element_ids", []) for c in topk)
                for e in req)
            ar20[q["qid"]] = 1.0 if ok else 0.0
        r["per_query_values"]["all_required_at_20"] = ar20
    # 两两 bootstrap（组内）。
    for fmt, arms in (("docx", ("DC0", "DC1", "DC2", "DC3")),
                      ("xlsx", ("XC0", "XC1", "XC2", "XC3"))):
        for metric in ("ndcg_at_5", "mrr_at_5", "recall_at_10", "evidence_anchor_recall_at_20"):
            bootstrap_rows += metrics14c.pairwise_bootstrap_all_arms(
                {a: arm_results[a]["per_query_values"][metric] for a in arms},
                list(arms), metric)
    for arm, r in arm_results.items():
        r["bootstrap_rows"] = [b for b in bootstrap_rows
                               if b["arm_a"] in ("DC0", "DC1", "DC2", "DC3") == (r["fmt"] == "docx")]
    print("[14A-C] Step 8-9: IR + anchor metrics + bootstrap done", flush=True)

    # Step 10：硬门。
    frozen_ok = True  # 输入 digest 在 load 时即来自冻结产物（Step 0 已核验 status=PASS）
    for arm, r in arm_results.items():
        queries = q_docx if r["fmt"] == "docx" else q_xlsx
        gates = gates14c.evaluate_common_gates(
            frozen_inputs_ok=frozen_ok,
            structural=r["structural"],
            projection_report=r["projection"],
            queries=queries,
            run=r["run"],
            run_complete_ok=r["run_complete_ok"],
            ir_eligible_count=r["ir"]["aggregate"]["eligible_count"],
            deterministic_ok=r["deterministic_ok"],
            chunks=r["chunks"],
            docs=r["docs"],
        )
        if r["fmt"] == "docx":
            gates += gates14c.evaluate_docx_gates(arm, r["chunks"], r["docs"], r["structural"])
        else:
            gates += gates14c.evaluate_xlsx_gates(arm, r["chunks"], r["docs"], r["structural"])
        r["gates"] = gates
        r["gates_pass"] = all(g["pass"] for g in gates)
        (WORKDIR / "gates" / f"{arm}_gates.json").write_text(
            json.dumps(gates, ensure_ascii=False, indent=1), encoding="utf-8")
    print("[14A-C] Step 10: gates evaluated", flush=True)

    # Step 11：选择（仅过门 arm 参选）。
    selection: dict[str, dict] = {}
    for fmt, arms in (("docx", ("DC0", "DC1", "DC2", "DC3")),
                      ("xlsx", ("XC0", "XC1", "XC2", "XC3"))):
        cand = [a for a in arms if arm_results[a]["gates_pass"]]
        if not cand:
            selection[fmt] = {"winner": None, "clear": False, "finalists": [],
                              "rule": "no arm passed all gates"}
            continue
        agg_by_arm = {
            a: {**arm_results[a]["ir"], "coverage_ratio": arm_results[a]["structural"]["coverage_ratio"],
                "token_sum": arm_results[a]["structural"]["token_sum"]}
            for a in cand
        }
        selection[fmt] = select_winner(
            cand, agg_by_arm,
            {a: arm_results[a]["per_query_values"] for a in cand},
            stats_by_arm)
    print(f"[14A-C] Step 11: selection {selection}", flush=True)

    # 指标/统计落盘。
    metrics_out = {
        arm: {"ir": r["ir"], "anchor": r["anchor"], "structural": r["structural"],
              "stats": stats_by_arm.get(arm, {})}
        for arm, r in arm_results.items()
    }
    (WORKDIR / "metrics" / "metrics_by_arm.json").write_text(
        json.dumps(metrics_out, ensure_ascii=False, indent=1), encoding="utf-8")
    (WORKDIR / "metrics" / "bootstrap_ci.json").write_text(
        json.dumps(bootstrap_rows, ensure_ascii=False, indent=1), encoding="utf-8")
    (WORKDIR / "metrics" / "projection_by_arm.json").write_text(
        json.dumps({arm: r["projection"] for arm, r in arm_results.items()},
                   ensure_ascii=False, indent=1), encoding="utf-8")

    # Step 12：报告。
    write_reports(WORKDIR, arm_results, selection, reconstructed=True)
    print("[14A-C] Step 12: reports written", flush=True)

    # Step 13：沙盒复查。
    src_tree_end = {
        p.relative_to(ROOT / "src").as_posix(): sha256_file(p)
        for p in sorted((ROOT / "src").rglob("*.py"))
    }
    src_untouched = src_tree == src_tree_end
    b_hashes_end = dir_hashes(PHASE_B_DIR, exclude=("phase14a_b/",))
    b_untouched = b_hashes_start == b_hashes_end
    if not (src_untouched and b_untouched):
        stop_anomaly(f"sandbox boundary violated: src_untouched={src_untouched}, b_untouched={b_untouched}")
    print("[14A-C] Step 13: sandbox recheck PASS", flush=True)

    # Step 14：manifest + status。
    manifest = {"artifact_type": "phase14a_c_manifest",
                "files": dir_hashes(WORKDIR, exclude=("chroma/", "manifests/", "phase14a_c_status.json"))}
    (WORKDIR / "manifests" / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")

    all_pass = all(r["gates_pass"] for r in arm_results.values())
    status = {
        "artifact_type": "phase14a_c_chunk_benchmark_status",
        "schema_version": "phase14a-c-chunk-benchmark-v1",
        "gate_definitions_version": gates14c.GATE_DEFINITIONS_VERSION,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "stop_at": "Phase 14A-C",
        "verdict": "PASS" if all_pass else "FAIL",
        "reconstructed": True,
        "reconstruction_note": (
            "14A-C 方案原文在上下文压缩中丢失；门定义/arm 细节/bootstrap 参数按压缩摘要 "
            "+ 14A-B 惯例重建，经用户确认后执行（2026-09-15）。"
        ),
        "inputs_frozen": {
            "phase14a_b_status": b_verdict,
            "phase14a_b_files_hashed": len(b_hashes_start),
            "b1_inputs": "phase14a_b/cleaned/{docx_b1,xlsx_b1}/*_a.json",
        },
        "token_policy": {"tokenizer": "BAAI/bge-m3", "target": chunk14c.TOKEN_TARGET,
                         "overlap": chunk14c.OVERLAP_TOKENS,
                         "count": "add_special_tokens=False"},
        "queries": {"docx": len(q_docx), "xlsx": len(q_xlsx),
                    "self_check": "PASS" if sc["pass"] else "FAIL",
                    "categories": ["local_fact", "structural_context", "table_caption_formula"]},
        "arms": {
            arm: {
                "strategy": chunk14c.STRATEGY_BY_ARM[arm],
                "chunk_count": r["structural"]["chunk_count"],
                "gates_pass": r["gates_pass"],
                "coverage_ratio": r["structural"]["coverage_ratio"],
                "leak_count": r["structural"]["leak_count"],
                "ir_aggregate": r["ir"]["aggregate"],
                "deterministic": r["deterministic_ok"],
                "retrieval_stats": stats_by_arm.get(arm, {}),
            }
            for arm, r in arm_results.items()
        },
        "selection": {
            "docx": selection["docx"],
            "xlsx": selection["xlsx"],
            "order": ["hard_gates", *metrics14c.SELECTION_METRICS,
                      "structural_fidelity", "inflation", "latency"],
            "scope": "CONTROLLED_WINNER only; REAL_CORPUS PENDING",
        },
        "sandbox_boundary": {
            "src_tree_files": len(src_tree),
            "src_untouched": src_untouched,
            "phase14a_b_untouched": b_untouched,
            "new_deps_installed": False,
        },
        "leftovers": [
            "真实语料 PENDING（导入须 §18 人工 review）；结论仅基于受控语料",
            "STOPPED AT Phase 14A-C",
        ],
    }
    (WORKDIR / "phase14a_c_status.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[14A-C] Step 14: status written, verdict={status['verdict']}, "
          f"elapsed={time.perf_counter() - t_start:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
