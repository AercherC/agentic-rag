# -*- coding: utf-8 -*-
"""Phase 13E 编排 runner（方案 PLAN.md，门定义 phase13e-gates-v1.0）。

三新臂 PH1/PH2/PH3（standalone + section/fixed body），参照臂 PC0/PC1/PC2 直接
复用 13C-C 冻结 chunks/runs/逐题 IR（零重算）。步骤：
  Step1 E-G1 输入校验（13C-C manifest hash）
  Step2 参照臂重投影一致性（E-G6：num_relevant 逐题对齐 13C-C）
  Step3 新臂切分 ×2（E-G3）+ E-G4/E-G5 + E-G2 覆盖
  Step4 新臂嵌入 + Chroma Top20（断点续跑：npz 缓存复用）
  Step5 IR + 双基线 bootstrap（vs PC0 与 vs PC1，×2 = E-G7）+ focus
  Step6 硬门汇总 + status.json + manifest.json + final_report.md
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
C13C = ROOT / "evals" / "evaluation_v3" / "phase13c_scientific_ingestion_benchmark" / "phase13c_c"
WORKDIR = ROOT / "evals" / "evaluation_v3" / "phase13e_hybrid_chunk_benchmark"
QUESTIONS_PATH = ROOT / "evals" / "research_retrieval" / "research_questions_v1.json"

NEW_ARMS = ["PH1", "PH2", "PH3"]
REF_ARMS = ["PC0", "PC1", "PC2"]
MODEL_ID = "BAAI/bge-m3"
TOP_K = 20
GATE_VERSION = "phase13e-gates-v1.0"
BOOTSTRAP_N = 10000
BOOTSTRAP_SEED = 20260915
FOCUS_QIDS = ["ARR-007", "ARR-022", "ARR-037", "ARR-040"]

sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "experiments" / "phase13c"))

import chunk13cc  # noqa: E402
import metrics13cc  # noqa: E402
from phase13c.runners._common import chroma_metadatas_for_records  # noqa: E402


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")


def load_docs() -> dict[str, dict]:
    """加载 13C-C 冻结 B2_fused 文档（重映射到生产 document_id，同 13C-C 契约）。"""
    remap = json.loads((C13C / "gates" / "C_G1_inputs.json").read_text(encoding="utf-8"))
    docs = {}
    for name, entry in remap["checks"]["document_id_remap"].items():
        d = json.loads((C13C / "frozen_inputs" / "b2_fused" / f"{name}.json").read_text(encoding="utf-8"))
        d["document_id"] = entry["production_document_id"]
        docs[name] = d
    return docs


def embed_and_retrieve(chunks_by_arm: dict[str, list[dict]], questions: list[dict]):
    """与 13C-C 完全同协议：CLS 池化 + L2 归一化 + cosine Top20；npz 断点续跑。"""
    import os
    import chromadb
    import numpy as np
    import torch
    from transformers import AutoModel, AutoTokenizer

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModel.from_pretrained(MODEL_ID)
    model.eval()

    def embed(texts: list[str]) -> "np.ndarray":
        out = []
        with torch.no_grad():
            for i in range(0, len(texts), 16):
                batch = texts[i:i + 16]
                enc = tok(batch, padding=True, truncation=True, max_length=512, return_tensors="pt")
                cls = torch.nn.functional.normalize(model(**enc).last_hidden_state[:, 0], dim=-1)
                out.append(cls.cpu().numpy().astype("float32"))
        return np.concatenate(out, axis=0) if out else np.zeros((0, 1024), dtype="float32")

    queries = [{"qid": q["question_id"], "text": q["english_query"]} for q in questions]
    (WORKDIR / "embeddings").mkdir(parents=True, exist_ok=True)
    (WORKDIR / "retrieval").mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(WORKDIR / "chroma"))
    stats = {}
    for arm in NEW_ARMS:
        chunks = chunks_by_arm[arm]
        npz = WORKDIR / "embeddings" / f"{arm}.npz"
        if npz.exists():
            data = np.load(npz, allow_pickle=False)
            vecs = data["embeddings"]
            if [str(x) for x in data["chunk_ids"]] != [c["chunk_id"] for c in chunks]:
                raise RuntimeError(f"{arm}: 缓存向量与当前切分不一致，须删除 npz 重算")
            embed_seconds = 0.0
        else:
            import time
            t0 = time.perf_counter()
            vecs = embed([c["text"] for c in chunks])
            embed_seconds = time.perf_counter() - t0
            np.savez_compressed(npz, chunk_ids=np.array([c["chunk_id"] for c in chunks]), embeddings=vecs)
        col_name = f"ef_13e_{arm.lower()}"
        try:
            client.delete_collection(col_name)
        except Exception:
            pass
        col = client.create_collection(name=col_name, metadata={"hnsw:space": "cosine", "arm": arm})
        col.add(ids=[c["chunk_id"] for c in chunks], embeddings=vecs.tolist(),
                documents=[c["text"] for c in chunks],
                metadatas=chroma_metadatas_for_records(chunks))
        q_vecs = embed([q["text"] for q in queries])
        run = {}
        for q, qv in zip(queries, q_vecs):
            res = col.query(query_embeddings=[qv.tolist()], n_results=min(TOP_K, len(chunks)))
            run[q["qid"]] = list(res["ids"][0])
        write_json(WORKDIR / "retrieval" / f"{arm}_run.json", run)
        stats[arm] = {"collection": col_name, "chunk_count": len(chunks),
                      "embed_seconds": round(embed_seconds, 1), "dim": int(vecs.shape[1])}
        print(f"[13e] {arm}: {len(chunks)} chunks done", flush=True)
    write_json(WORKDIR / "metrics" / "retrieval_stats.json", stats)
    return stats


def main() -> int:
    started = datetime.now(timezone.utc).isoformat()
    WORKDIR.mkdir(parents=True, exist_ok=True)

    # ---- Step1 E-G1：13C-C manifest hash 校验 ----
    m13 = json.loads((C13C / "manifest.json").read_text(encoding="utf-8"))["files"]
    ref_files = (["chunks/PC0.jsonl", "chunks/PC1.jsonl", "chunks/PC2.jsonl",
                  "retrieval/PC0_run.json", "retrieval/PC1_run.json", "retrieval/PC2_run.json",
                  "metrics/ir_per_query.json", "metrics/qrels_projection.json",
                  "frozen_inputs/manifest.json"])
    checks = {rel: m13.get(rel) == sha256_of(C13C / rel) for rel in ref_files}
    gate1 = {"checks": checks, "pass": all(checks.values())}
    write_json(WORKDIR / "gates" / "E_G1_inputs.json", gate1)
    if not gate1["pass"]:
        print("[13e] E-G1 FAIL，停止", flush=True)
        return 1

    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))["questions"]
    docs = load_docs()

    # ---- Step2 参照臂重投影一致性（E-G6）----
    ref_chunks = {arm: [json.loads(l) for l in (C13C / "chunks" / f"{arm}.jsonl")
                        .read_text(encoding="utf-8").splitlines() if l.strip()]
                  for arm in REF_ARMS}
    ref_runs = {arm: json.loads((C13C / "retrieval" / f"{arm}_run.json").read_text(encoding="utf-8"))
                for arm in REF_ARMS}
    frozen_numrel = json.loads((C13C / "metrics" / "qrels_projection.json")
                               .read_text(encoding="utf-8"))
    e6 = {}
    ref_qrels = {}
    for arm in REF_ARMS:
        proj = metrics13cc.project_qrels_anchor({arm: ref_chunks[arm]}, questions)
        ref_qrels[arm] = proj["qrels"]
        e6[arm] = all(proj["num_relevant"].get(q) == frozen_numrel[arm]["num_relevant_by_qid"].get(q)
                      for q in frozen_numrel[arm]["num_relevant_by_qid"])
    gate6 = {"ref_qrels_reprojection": e6,
             "pass": all(e6.values()),
             "protocol": "同 13C-C：bge-m3/Top20/anchor 严格制（结构保证，逐条记录）"}
    write_json(WORKDIR / "gates" / "E_G6_protocol.json", gate6)

    # ---- Step3 新臂切分 + 结构门 ----
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    counter = lambda t: len(tok(t, add_special_tokens=False)["input_ids"])  # noqa: E731

    chunks_by_arm, gate3, gate4, gate5, arm_stats = {}, {}, {}, {}, {}
    new_qrels, new_cov = {}, {}
    for arm in NEW_ARMS:
        run1 = {name: chunk13cc.chunk_pdf(d, arm, counter) for name, d in docs.items()}
        run2 = {name: chunk13cc.chunk_pdf(d, arm, counter) for name, d in docs.items()}
        gate3[arm] = {"pass": all([c["chunk_id"] for c in run1[n]] == [c["chunk_id"] for c in run2[n]]
                                  and [c["text"] for c in run1[n]] == [c["text"] for c in run2[n]]
                                  for n in docs)}
        chunks_by_arm[arm] = [c for n in sorted(run1) for c in run1[n]]
        gate4[arm] = metrics13cc.check_provenance(run1, docs)
        gate5[arm] = metrics13cc.check_token_budget(chunks_by_arm[arm], chunk13cc.TOKEN_TARGET)
        proj = metrics13cc.project_qrels_anchor(run1, questions)
        new_qrels[arm] = proj["qrels"]
        new_cov[arm] = {"coverage_rate": proj["coverage_rate"],
                        "uncovered": proj["uncovered"],
                        "num_relevant_by_qid": proj["num_relevant"]}
        out = WORKDIR / "chunks" / f"{arm}.jsonl"
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            for c in chunks_by_arm[arm]:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
        tokens = [c["token_count"] for c in chunks_by_arm[arm]]
        arm_stats[arm] = {"chunk_count": len(tokens),
                          "token_mean": round(sum(tokens) / len(tokens), 1),
                          "token_max": max(tokens)}
        print(f"[13e] {arm}: chunked {len(tokens)} chunks", flush=True)
    gate2 = {arm: {"coverage_rate": new_cov[arm]["coverage_rate"],
                   "pass": not new_cov[arm]["uncovered"]} for arm in NEW_ARMS}
    write_json(WORKDIR / "gates" / "E_G2_coverage.json", gate2)
    write_json(WORKDIR / "gates" / "E_G3_G4_G5_structure.json",
               {"gate3": gate3, "gate4": gate4, "gate5": gate5, "arm_stats": arm_stats})
    write_json(WORKDIR / "metrics" / "qrels_projection.json",
               {arm: {"coverage_rate": new_cov[arm]["coverage_rate"],
                      "uncovered": new_cov[arm]["uncovered"],
                      "num_relevant_by_qid": new_cov[arm]["num_relevant_by_qid"]} for arm in NEW_ARMS})

    # ---- Step4 嵌入 + 检索 ----
    stats = embed_and_retrieve(chunks_by_arm, questions)
    new_runs = {arm: json.loads((WORKDIR / "retrieval" / f"{arm}_run.json").read_text(encoding="utf-8"))
                for arm in NEW_ARMS}

    # ---- Step5 IR + 双基线 bootstrap ----
    ir_by_arm = {arm: metrics13cc.compute_ir_metrics(new_qrels[arm], new_runs[arm]) for arm in NEW_ARMS}
    # 参照臂逐题 IR 从 13C-C 冻结文件包装（per_query_metric_values 兼容结构）。
    frozen_pq = json.loads((C13C / "metrics" / "ir_per_query.json").read_text(encoding="utf-8"))
    ir_ref = {arm: {"per_query": frozen_pq[arm]} for arm in REF_ARMS}
    write_json(WORKDIR / "metrics" / "ir_by_arm.json",
               {arm: ir_by_arm[arm]["aggregate"] for arm in NEW_ARMS})
    write_json(WORKDIR / "metrics" / "ir_per_query.json",
               {arm: ir_by_arm[arm]["per_query"] for arm in NEW_ARMS})

    def bootstrap_round() -> list[dict]:
        out = []
        for metric in ("ndcg_at_5", "recall_at_5", "recall_at_10", "mrr_at_5"):
            vals_new = {arm: metrics13cc.per_query_metric_values(ir_by_arm[arm], metric) for arm in NEW_ARMS}
            vals_ref = {arm: metrics13cc.per_query_metric_values(ir_ref[arm], metric) for arm in REF_ARMS}
            for arm in NEW_ARMS:
                for base in ("PC0", "PC1"):
                    res = metrics13cc.paired_bootstrap(vals_new[arm], vals_ref[base],
                                                       n=BOOTSTRAP_N, seed=BOOTSTRAP_SEED)
                    out.append({"metric": metric, "arm": arm, "vs": base, **res})
        return out

    boot1, boot2 = bootstrap_round(), bootstrap_round()
    gate7 = {"deterministic": boot1 == boot2, "pass": boot1 == boot2}
    write_json(WORKDIR / "metrics" / "bootstrap.json",
               {"n": BOOTSTRAP_N, "seed": BOOTSTRAP_SEED, "results": boot1})
    write_json(WORKDIR / "gates" / "E_G7_bootstrap_determinism.json", gate7)

    # focus
    focus = {}
    all_runs = {**new_runs, **ref_runs}
    all_qrels = {**new_qrels, **ref_qrels}
    for qid in FOCUS_QIDS:
        entry = {}
        for arm in NEW_ARMS + REF_ARMS:
            rel = set(all_qrels[arm].get(qid, []))
            rank = next((r for r, cid in enumerate(all_runs[arm].get(qid, []), 1) if cid in rel), None)
            entry[arm] = {"first_relevant_rank": rank, "in_top20": rank is not None and rank <= 20,
                          "num_relevant": len(rel)}
        focus[qid] = entry
    write_json(WORKDIR / "metrics" / "focus_questions.json", focus)

    gates = {"E_G1_inputs": gate1["pass"],
             "E_G2_coverage": {a: gate2[a]["pass"] for a in NEW_ARMS},
             "E_G3_determinism": {a: gate3[a]["pass"] for a in NEW_ARMS},
             "E_G4_provenance": {a: gate4[a]["pass"] for a in NEW_ARMS},
             "E_G5_token_budget": {a: gate5[a]["pass"] for a in NEW_ARMS},
             "E_G6_protocol": gate6["pass"],
             "E_G7_bootstrap_determinism": gate7["pass"]}
    all_pass = (gates["E_G1_inputs"] and gates["E_G6_protocol"] and gates["E_G7_bootstrap_determinism"]
                and all(gates["E_G2_coverage"].values()) and all(gates["E_G3_determinism"].values())
                and all(gates["E_G4_provenance"].values()) and all(gates["E_G5_token_budget"].values()))
    verdict = "PASS" if all_pass else "FAIL"
    write_json(WORKDIR / "gates" / "hard_gates.json",
               {"gate_definitions_version": GATE_VERSION, "gates": gates, "verdict": verdict})

    # status + manifest + report
    status = {"artifact_type": "phase13e_hybrid_chunk_status",
              "schema_version": "phase13e-hybrid-chunk-v1",
              "gate_definitions_version": GATE_VERSION,
              "plan": "PLAN.md（执行前冻结）", "generated_at": started, "stop_at": "Phase 13E",
              "verdict": verdict,
              "arm_stats": {**arm_stats,
                            **{arm: {"source": "phase13c_c frozen", "chunk_count": len(ref_chunks[arm])}
                               for arm in REF_ARMS}},
              "ir_aggregate_new": {arm: ir_by_arm[arm]["aggregate"] for arm in NEW_ARMS},
              "ir_aggregate_ref_source": "phase13c_c metrics/ir_by_arm.json",
              "bootstrap": {"n": BOOTSTRAP_N, "seed": BOOTSTRAP_SEED,
                            "significant_pairs": [b for b in boot1 if b.get("significant")]},
              "focus_questions": focus, "gates": gates}
    write_json(WORKDIR / "status.json", status)

    man_files = {}
    for sub in ("chunks", "embeddings", "retrieval", "metrics", "gates"):
        for p in sorted((WORKDIR / sub).rglob("*")):
            if p.is_file():
                man_files[str(p.relative_to(WORKDIR)).replace("\\", "/")] = sha256_of(p)
    for name in ("status.json", "PLAN.md"):
        p = WORKDIR / name
        if p.exists():
            man_files[name] = sha256_of(p)
    write_json(WORKDIR / "manifest.json", {
        "artifact_type": "phase13e_manifest", "created_at": started, "files": man_files,
        "note": "参照臂复用 13C-C 冻结产物；chroma 为临时不入 manifest。"})

    # final_report.md
    ref_agg = json.loads((C13C / "metrics" / "ir_by_arm.json").read_text(encoding="utf-8"))
    rep = ["# Phase 13E Hybrid Chunk Benchmark — Final Report", "",
           f"- 生成时间：{started}", f"- verdict：**{verdict}**（gate_definitions {GATE_VERSION}）",
           "- 协议：同 13C-C（BGE-M3 Dense Top20、无 reranker、anchor 严格制）；standalone=caption/table/formula 独立 chunk", "",
           "## Arm 结构统计", "",
           "| arm | 策略 | chunks | token_mean | token_max | 来源 |", "|---|---|---:|---:|---:|---|"]
    label = {"PH1": "standalone+section O0", "PH2": "standalone+section O64",
             "PH3": "standalone+fixed O64", "PC0": "fixed C512/O64",
             "PC1": "section O0", "PC2": "element-aware"}
    for arm in NEW_ARMS:
        s = arm_stats[arm]
        rep.append(f"| {arm} | {label[arm]} | {s['chunk_count']} | {s['token_mean']} | {s['token_max']} | 本轮 |")
    for arm in REF_ARMS:
        rep.append(f"| {arm} | {label[arm]} | {len(ref_chunks[arm])} | — | — | 13C-C 冻结 |")
    rep += ["", "## 标准 IR（aggregate）", "",
            "| arm | P@5 | R@5 | R@10 | MRR@5 | MAP@5 | nDCG@5 | Hit@20 |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for arm in NEW_ARMS + REF_ARMS:
        m = ir_by_arm[arm]["aggregate"] if arm in NEW_ARMS else ref_agg[arm]
        rep.append(f"| {arm} | {m['precision_at_5']:.4f} | {m['recall_at_5']:.4f} | {m['recall_at_10']:.4f} | "
                   f"{m['mrr_at_5']:.4f} | {m['map_at_5']:.4f} | {m['ndcg_at_5']:.4f} | {m['hit_rate_at_20']:.4f} |")
    rep += ["", "## Bootstrap（n=10000，seed=20260915）", "",
            "| metric | arm | vs | mean Δ | CI95 | P(Δ>0) | significant |", "|---|---|---|---:|---|---:|---|"]
    for b in boot1:
        ci = f"[{b['ci95_low']:+.4f}, {b['ci95_high']:+.4f}]"
        rep.append(f"| {b['metric']} | {b['arm']} | {b['vs']} | {b['mean_delta']:+.4f} | {ci} | {b['p_delta_gt_0']:.3f} | {b['significant']} |")
    rep += ["", "## Focus", ""]
    for qid in FOCUS_QIDS:
        e = focus[qid]
        line = " | ".join(f"{a}: {'rank=' + str(e[a]['first_relevant_rank']) if e[a]['first_relevant_rank'] else 'miss'}"
                          for a in NEW_ARMS + REF_ARMS)
        rep.append(f"- **{qid}** — {line}")
    rep += ["", "## 硬门", ""]
    for g, v in gates.items():
        rep.append(f"- {g}: {v}")
    (WORKDIR / "reports").mkdir(exist_ok=True)
    (WORKDIR / "reports" / "final_report.md").write_text("\n".join(rep) + "\n", encoding="utf-8")
    man_files["reports/final_report.md"] = sha256_of(WORKDIR / "reports" / "final_report.md")
    write_json(WORKDIR / "manifest.json", {
        "artifact_type": "phase13e_manifest", "created_at": started, "files": man_files,
        "note": "参照臂复用 13C-C 冻结产物；chroma 为临时不入 manifest。"})

    print(f"[13e] done: verdict={verdict}", flush=True)
    return 0 if all_pass else 2


if __name__ == "__main__":
    sys.exit(main())
