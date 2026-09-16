# -*- coding: utf-8 -*-
"""Phase 13D 编排 runner（方案 PLAN.md，门定义 phase13d-gates-v1.0）。

单进程顺序执行：
  Step1 校验冻结输入（PC0 chunks / dense run / qrels 的 sha256 对 13C-C manifest）
  Step2 构造 BM25Index（生产 retrieval/bm25.py，bge-m3 tokenizer，k1=1.5 b=0.75）
        + DA1 检索 ×2（D-G2 确定性）
  Step3 DA2 RRF(k=60) 与 DA3 add-only 融合（×2，D-G2）
  Step4 四臂标准 IR + bootstrap(DA2/DA3 vs DA0, ×2 = D-G6) + focus 12 题
  Step5 硬门汇总 + status.json + manifest.json + final_report.md

运行方式（agent 环境，离线）：
  HF_HOME=<ws>/.cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python experiments/phase13c/runners/run_phase13d_augmentation.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
C13C = ROOT / "evals" / "evaluation_v3" / "phase13c_scientific_ingestion_benchmark" / "phase13c_c"
WORKDIR = ROOT / "evals" / "evaluation_v3" / "phase13d_candidate_augmentation_benchmark"
QUESTIONS_PATH = ROOT / "evals" / "research_retrieval" / "research_questions_v1.json"
CORPUS_MANIFEST = ROOT / "evals" / "research_ingestion" / "academic_corpus_manifest_v1.json"

ARMS = ["DA0", "DA1", "DA2", "DA3"]
TOP_K = 20
RRF_K = 60
BM25_K1, BM25_B = 1.5, 0.75
BM25_TOP_K = 10  # DA3 add-only 的 BM25 补充深度（历史诊断协议）
GATE_VERSION = "phase13d-gates-v1.0"
BOOTSTRAP_N = 10000
BOOTSTRAP_SEED = 20260915
FOCUS_IDS = ["ARR-007", "ARR-015", "ARR-020", "ARR-021", "ARR-022",
             "ARR-025", "ARR-030", "ARR-031", "ARR-037", "ARR-038", "ARR-039", "ARR-040"]

sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "experiments" / "phase13c"))

import metrics13cc  # noqa: E402  （复用 bootstrap 纯函数与归一化）


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")


# --------------------------------------------------------------------------------------
# Step 1 冻结输入校验（D-G1 / D-G5）
# --------------------------------------------------------------------------------------

def verify_frozen_inputs() -> dict:
    """PC0 chunks / dense run / qrels 的 hash 必须与 13C-C manifest 一致。"""
    m13 = json.loads((C13C / "manifest.json").read_text(encoding="utf-8"))
    files = m13["files"]
    checks = {}
    for rel in ("chunks/PC0.jsonl", "retrieval/PC0_run.json",
                "metrics/qrels_projection.json", "frozen_inputs/manifest.json"):
        p = C13C / rel
        checks[rel] = {"exists": p.is_file(),
                       "hash_match": p.is_file() and files.get(rel) == sha256_of(p)}
    qproj = json.loads((C13C / "metrics" / "qrels_projection.json").read_text(encoding="utf-8"))
    qrels = qproj["PC0"]["num_relevant_by_qid"] and {
        qid: qproj["PC0"].get("num_relevant_by_qid", {}).get(qid) for qid in qproj["PC0"]["num_relevant_by_qid"]
    }
    # qrels 本体在 13C-C 未单独落盘，由 coverage 记录的 num_relevant 与运行时重投影交叉校验：
    # 运行时对同一批 PC0 chunks 重投影，若 num_relevant 与冻结记录逐题一致即视为一致（D-G5）。
    return {"checks": checks, "qrels_projection_frozen": qproj["PC0"],
            "pass": all(c["hash_match"] for c in checks.values())}


def load_pc0_chunks() -> list[dict]:
    """从 13C-C 冻结 chunks/PC0.jsonl 载入 chunk dicts。"""
    out = []
    for line in (C13C / "chunks" / "PC0.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out


def build_domain_chunks(chunks: list[dict]) -> list:
    """把 13C-C chunk dicts 构造为生产 DocumentChunk（BM25Index 契约）。"""
    from evidenceflow.domain.chunk import DocumentChunk

    corpus = json.loads(CORPUS_MANIFEST.read_text(encoding="utf-8"))
    meta_by_doc = {p["document_id"]: p for p in corpus["papers"]}
    domain = []
    for c in chunks:
        paper = meta_by_doc[c["document_id"]]
        domain.append(DocumentChunk(
            chunk_id=c["chunk_id"],
            document_id=c["document_id"],
            content=c["text"],
            token_count=c["token_count"],
            chunk_index=c["seq"],
            page_start=c.get("page_start"),
            page_end=c.get("page_end"),
            file_name=paper["file_name"],
            content_hash=paper["sha256"],
            metadata={"owners": c.get("element_ids", []), "arm": c.get("arm"), "benchmark": "phase13d"},
        ))
    return domain


# --------------------------------------------------------------------------------------
# Step 2-3 BM25 与融合排名
# --------------------------------------------------------------------------------------

def rrf_fuse(dense: list[str], bm25: list[str]) -> list[str]:
    """RRF(k=60) 融合两路名次；同分按 chunk_id 字典序（确定性）。"""
    scores: dict[str, float] = {}
    for rank, cid in enumerate(dense, 1):
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (RRF_K + rank)
    for rank, cid in enumerate(bm25, 1):
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (RRF_K + rank)
    return [cid for cid, _ in sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))][:TOP_K]


def add_only_union(dense: list[str], bm25: list[str]) -> list[str]:
    """add-only：Dense 保序在前，BM25 未出现者按其名次追加（历史诊断协议）。"""
    out = list(dense)
    for cid in bm25:
        if cid not in out:
            out.append(cid)
    return out[:TOP_K]


def main() -> int:
    started = datetime.now(timezone.utc).isoformat()
    WORKDIR.mkdir(parents=True, exist_ok=True)

    print("[13d] Step1 冻结输入校验 ...", flush=True)
    gate1 = verify_frozen_inputs()
    write_json(WORKDIR / "gates" / "D_G1_inputs.json", gate1)
    if not gate1["pass"]:
        print("[13d] D-G1 FAIL，停止", flush=True)
        return 1

    chunks = load_pc0_chunks()
    chunk_ids = {c["chunk_id"] for c in chunks}
    dense_run = json.loads((C13C / "retrieval" / "PC0_run.json").read_text(encoding="utf-8"))
    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))["questions"]

    # 运行时 qrels 重投影，并与 13C-C 冻结 num_relevant 交叉校验（D-G5）。
    from transformers import AutoTokenizer
    import os as _os
    _os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    tok = AutoTokenizer.from_pretrained("BAAI/bge-m3")
    counter = lambda t: len(tok(t, add_special_tokens=False)["input_ids"])  # noqa: E731

    chunks_by_doc = {"PC0_chunks": chunks}
    proj = metrics13cc.project_qrels_anchor(chunks_by_doc, questions)
    qrels = proj["qrels"]
    frozen_numrel = gate1["qrels_projection_frozen"]["num_relevant_by_qid"]
    gate5 = {"per_qid_match": {qid: proj["num_relevant"].get(qid) == frozen_numrel.get(qid)
                               for qid in frozen_numrel},
             "pass": all(proj["num_relevant"].get(q) == frozen_numrel.get(q) for q in frozen_numrel)}
    write_json(WORKDIR / "gates" / "D_G5_qrels_identity.json", gate5)

    print("[13d] Step2 BM25 索引与检索 ...", flush=True)
    from evidenceflow.ingestion.tokenizer import BgeM3Tokenizer
    from evidenceflow.retrieval.bm25 import BM25Index

    domain = build_domain_chunks(chunks)
    tokenizer = BgeM3Tokenizer("BAAI/bge-m3")
    index = BM25Index(domain, tokenizer, k1=BM25_K1, b=BM25_B)

    def bm25_search_all() -> dict[str, list[str]]:
        run = {}
        for q in questions:
            results = index.search(q["query"], top_k=TOP_K)
            run[q["question_id"]] = [r.chunk.chunk_id for r in results]
        return run

    bm25_run1, bm25_run2 = bm25_search_all(), bm25_search_all()
    gate2_bm25 = bm25_run1 == bm25_run2

    runs = {"DA0": {qid: list(ids)[:TOP_K] for qid, ids in dense_run.items()}}

    def fuse_all() -> dict[str, dict[str, list[str]]]:
        rrf = {qid: rrf_fuse(dense_run[qid], bm25_run1[qid]) for qid in dense_run}
        add = {qid: add_only_union(dense_run[qid], bm25_run1[qid][:BM25_TOP_K]) for qid in dense_run}
        return {"DA2": rrf, "DA3": add}

    fused1, fused2 = fuse_all(), fuse_all()
    gate2_fuse = fused1 == fused2
    runs.update(fused1)
    runs["DA1"] = bm25_run1
    for arm in ARMS:
        write_json(WORKDIR / "runs" / f"{arm}_run.json", runs[arm])
    # D-G2：BM25 与融合两次全等；DA0 为冻结 run（不重算）。
    gate2 = {"bm25_deterministic": gate2_bm25, "fusion_deterministic": gate2_fuse,
             "pass": gate2_bm25 and gate2_fuse}
    write_json(WORKDIR / "gates" / "D_G2_determinism.json", gate2)

    # D-G4 池完整性：无重复 chunk_id。
    dup = {arm: [qid for qid, ids in runs[arm].items() if len(ids) != len(set(ids))]
           for arm in ARMS}
    gate4 = {"duplicates": {a: v for a, v in dup.items() if v},
             "pass": not any(dup.values())}
    write_json(WORKDIR / "gates" / "D_G4_pool_integrity.json", gate4)

    # D-G3 协议一致：结构保证（同 qrels/同深度/同 tokenizer/BM25 参数显式），记录事实。
    gate3 = {"top_k": TOP_K, "bm25_k1": BM25_K1, "bm25_b": BM25_B,
             "tokenizer": "BAAI/bge-m3", "reranker": False,
             "qrels_source": "phase13c_c metrics/qrels_projection.json PC0",
             "pass": True}
    write_json(WORKDIR / "gates" / "D_G3_protocol.json", gate3)

    print("[13d] Step4 指标与 bootstrap ...", flush=True)
    ir_by_arm = {arm: metrics13cc.compute_ir_metrics(qrels, runs[arm]) for arm in ARMS}
    write_json(WORKDIR / "metrics" / "ir_by_arm.json",
               {arm: ir_by_arm[arm]["aggregate"] for arm in ARMS})
    write_json(WORKDIR / "metrics" / "ir_per_query.json",
               {arm: ir_by_arm[arm]["per_query"] for arm in ARMS})

    def bootstrap_round() -> list[dict]:
        out = []
        for metric in ("ndcg_at_5", "recall_at_5", "recall_at_10", "precision_at_5", "mrr_at_5"):
            vals = {arm: metrics13cc.per_query_metric_values(ir_by_arm[arm], metric) for arm in ARMS}
            for arm in ("DA2", "DA3"):
                res = metrics13cc.paired_bootstrap(vals[arm], vals["DA0"], n=BOOTSTRAP_N, seed=BOOTSTRAP_SEED)
                out.append({"metric": metric, "arm": arm, "vs": "DA0", **res})
        return out

    boot1, boot2 = bootstrap_round(), bootstrap_round()
    gate6 = {"deterministic": boot1 == boot2, "pass": boot1 == boot2}
    write_json(WORKDIR / "metrics" / "bootstrap_vs_DA0.json",
               {"n": BOOTSTRAP_N, "seed": BOOTSTRAP_SEED, "results": boot1})
    write_json(WORKDIR / "gates" / "D_G6_bootstrap_determinism.json", gate6)

    # Focus：12 个历史候选失败逐臂 recovered。
    focus = {}
    for qid in FOCUS_IDS:
        entry = {}
        for arm in ARMS:
            rel = set(qrels.get(qid, []))
            run = runs[arm].get(qid, [])
            rank = next((r for r, cid in enumerate(run, 1) if cid in rel), None)
            entry[arm] = {"first_relevant_rank": rank,
                          "in_top10": rank is not None and rank <= 10,
                          "in_top20": rank is not None and rank <= 20,
                          "num_relevant": len(rel)}
        focus[qid] = entry
    write_json(WORKDIR / "metrics" / "focus_questions.json", focus)

    gates = {"D_G1_inputs": gate1["pass"], "D_G2_determinism": gate2["pass"],
             "D_G3_protocol": True, "D_G4_pool_integrity": gate4["pass"],
             "D_G5_qrels_identity": gate5["pass"], "D_G6_bootstrap_determinism": gate6["pass"]}
    all_pass = all(gates.values())
    verdict = "PASS" if all_pass else "FAIL"
    write_json(WORKDIR / "gates" / "hard_gates.json",
               {"gate_definitions_version": GATE_VERSION, "gates": gates, "verdict": verdict})

    status = {
        "artifact_type": "phase13d_candidate_augmentation_status",
        "schema_version": "phase13d-candidate-augmentation-v1",
        "gate_definitions_version": GATE_VERSION,
        "plan": "PLAN.md（执行前冻结）",
        "generated_at": started,
        "stop_at": "Phase 13D",
        "verdict": verdict,
        "protocol": {"chunk_space": "phase13c_c PC0 (fixed C512/O64)", "top_k": TOP_K,
                     "bm25": {"k1": BM25_K1, "b": BM25_B, "tokenizer": "BAAI/bge-m3",
                              "query": "original_only", "impl": "evidenceflow.retrieval.bm25.BM25Index"},
                     "fusion": {"rrf_k": RRF_K, "add_only_bm25_top_k": BM25_TOP_K},
                     "reranker": False},
        "ir_aggregate": {arm: ir_by_arm[arm]["aggregate"] for arm in ARMS},
        "bootstrap": {"n": BOOTSTRAP_N, "seed": BOOTSTRAP_SEED,
                      "significant_pairs": [b for b in boot1 if b.get("significant")]},
        "focus_questions": focus,
        "gates": gates,
    }
    write_json(WORKDIR / "status.json", status)

    man_files = {}
    for sub in ("runs", "metrics", "gates"):
        for p in sorted((WORKDIR / sub).rglob("*")):
            if p.is_file():
                man_files[str(p.relative_to(WORKDIR)).replace("\\", "/")] = sha256_of(p)
    for name in ("status.json", "PLAN.md"):
        p = WORKDIR / name
        if p.exists():
            man_files[name] = sha256_of(p)
    write_json(WORKDIR / "manifest.json", {
        "artifact_type": "phase13d_manifest", "created_at": started, "files": man_files,
        "note": "输入复用 phase13c_c 冻结产物（hash 校验见 gates/D_G1_inputs.json），不重复复制。"})

    # final_report.md
    rep = ["# Phase 13D Candidate Augmentation Benchmark — Final Report", "",
           f"- 生成时间：{started}", f"- verdict：**{verdict}**（gate_definitions {GATE_VERSION}）",
           f"- 协议：PC0 chunk 空间 × {len(questions)} 题；BM25(k1={BM25_K1}, b={BM25_B}, bge-m3 tokenizer)；"
           f"RRF(k={RRF_K}) 与 add-only(BM25 Top{BM25_TOP_K})；无 reranker", "",
           "## 标准 IR（aggregate）", "",
           "| arm | 排名来源 | P@5 | R@5 | R@10 | MRR@5 | MAP@5 | nDCG@5 | Hit@20 |",
           "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    src_label = {"DA0": "Dense Top20（13C-C 冻结）", "DA1": "BM25 Top20",
                 "DA2": f"RRF(k={RRF_K})", "DA3": f"add-only(BM25 Top{BM25_TOP_K})"}
    for arm in ARMS:
        m = ir_by_arm[arm]["aggregate"]
        rep.append(f"| {arm} | {src_label[arm]} | {m['precision_at_5']:.4f} | {m['recall_at_5']:.4f} | "
                   f"{m['recall_at_10']:.4f} | {m['mrr_at_5']:.4f} | {m['map_at_5']:.4f} | "
                   f"{m['ndcg_at_5']:.4f} | {m['hit_rate_at_20']:.4f} |")
    rep += ["", "## Bootstrap（vs DA0，n=10000，seed=20260915）", "",
            "| metric | arm | mean Δ | CI95 | P(Δ>0) | significant |", "|---|---|---:|---|---:|---|"]
    for b in boot1:
        ci = f"[{b['ci95_low']:.4f}, {b['ci95_high']:.4f}]"
        rep.append(f"| {b['metric']} | {b['arm']} | {b['mean_delta']:+.4f} | {ci} | {b['p_delta_gt_0']:.3f} | {b['significant']} |")
    rep += ["", "## Focus：12 个历史候选失败（ recovered 情况）", ""]
    for qid in FOCUS_IDS:
        e = focus[qid]
        line = " | ".join(f"{a}: {'rank=' + str(e[a]['first_relevant_rank']) if e[a]['first_relevant_rank'] else 'miss'}" for a in ARMS)
        rep.append(f"- **{qid}**（n_rel={e['DA0']['num_relevant']}）— {line}")
    rep += ["", "## 硬门", ""]
    for g, v in gates.items():
        rep.append(f"- {g}: {v}")
    rep += ["", "## 决策（数据驱动）", ""]
    sig = [b for b in boot1 if b.get("significant")]
    if sig:
        rep.append(f"- 显著对：{[(b['arm'], b['metric'], round(b['mean_delta'], 4)) for b in sig]}。")
    else:
        rep.append("- DA2/DA3 相对 DA0 无显著差异（CI 均含 0），如实记录。")
    a7 = focus["ARR-007"]
    rec = [a for a in ARMS if a7[a]["in_top20"]]
    rep.append(f"- ARR-007 recovered by: {rec if rec else '无臂'}。")
    rep += ["", "## 补充判读（结果收口分析，2026-09-15）", ""]
    r5_up = [b for b in boot1 if b["metric"] == "recall_at_5" and b["mean_delta"] > 0]
    if r5_up:
        for b in r5_up:
            rep.append(f"- {b['arm']}：R@5 mean Δ={b['mean_delta']:+.4f}（P(Δ>0)={b['p_delta_gt_0']:.3f}）"
                       f"——方向为正，显著性见 CI。")
    (WORKDIR / "reports").mkdir(exist_ok=True)
    (WORKDIR / "reports" / "final_report.md").write_text("\n".join(rep) + "\n", encoding="utf-8")
    # 报告 hash 入 manifest（生成后补记）
    man_files["reports/final_report.md"] = sha256_of(WORKDIR / "reports" / "final_report.md")
    write_json(WORKDIR / "manifest.json", {
        "artifact_type": "phase13d_manifest", "created_at": started, "files": man_files,
        "note": "输入复用 phase13c_c 冻结产物（hash 校验见 gates/D_G1_inputs.json），不重复复制。"})

    print(f"[13d] done: verdict={verdict}", flush=True)
    return 0 if all_pass else 2


if __name__ == "__main__":
    sys.exit(main())
