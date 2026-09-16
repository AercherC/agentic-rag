# -*- coding: utf-8 -*-
"""Phase 13C-C 编排 runner（方案 PLAN.md §2-§9，门定义 phase13c-c-gates-v1.0）。

单进程顺序执行：
  Step1 冻结输入（B2_fused ×10 从 AppData temp 复制 + sha256 manifest；40 题校验）
  Step2 四臂切分 ×2 次运行（C-G3 确定性）+ C-G4 provenance + C-G5 token 预算
  Step3 BGE-M3 离线嵌入 + 每臂独立 Chroma（cosine）Dense Top20 检索
  Step4 anchor 严格制 qrels 逐臂投影（C-G2 覆盖）
  Step5 标准 IR 指标 + paired bootstrap（n=10000，运行两次全等 = C-G7）
  Step6 focus 题诊断（ARR-007/037/040）+ 硬门汇总 + status.json + final_report.md

运行方式（agent 环境，离线）：
  HF_HOME=<ws>/.cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  PYTHONPATH="experiments;src" python runners/run_phase13c_c.py
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]  # runners/ -> phase13c -> experiments -> 仓库根
WORKDIR = ROOT / "evals" / "evaluation_v3" / "phase13c_scientific_ingestion_benchmark" / "phase13c_c"
TEMP_B2 = Path(r"C:\Users\chchxh\AppData\Local\Temp\evidenceflow_phase13c\cleaned\B2_fused")
QUESTIONS_PATH = ROOT / "evals" / "research_retrieval" / "research_questions_v1.json"
PER_QUERY_PATH = ROOT / "evals" / "evaluation_v3" / "standard_ir_v1" / "per_query_results.json"

REAL_DOCS = [
    "adapointr", "hyper_pcn", "pcn_point_completion_network", "pfnet_point_fractal_network",
    "pointattn", "pointr_geometry_aware_transformers", "proxyformer", "seedformer",
    "snowflakenet", "svdformer",
]
ARMS = ["PC0", "PC1", "PC2", "PC3"]
MODEL_ID = "BAAI/bge-m3"
TOP_K = 20
FROZEN_DATASET_HASH = "be1e8354babef3927163c19e32c614f7639323095900c487d559ade8249e7357"
FROZEN_CORPUS_HASH = "1e48b401aa48fc59d94bba627deaa3005a7627860e4fe5815761d45c6ba68675"
GATE_VERSION = "phase13c-c-gates-v1.0"
BOOTSTRAP_N = 10000
BOOTSTRAP_SEED = 20260915
FOCUS_QIDS = ["ARR-007", "ARR-037", "ARR-040"]

sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "experiments" / "phase13c"))

import metrics13cc  # noqa: E402
from phase13c.runners._common import chroma_metadatas_for_records  # noqa: E402
import chunk13cc  # noqa: E402


def sha256_of(path: Path) -> str:
    """流式 sha256。"""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")


# --------------------------------------------------------------------------------------
# Step 1 冻结输入
# --------------------------------------------------------------------------------------

def freeze_inputs() -> dict[str, Any]:
    """复制 B2_fused ×10 到 frozen_inputs 并生成 sha256 manifest（C-G1 基础）。"""
    out_dir = WORKDIR / "frozen_inputs" / "b2_fused"
    out_dir.mkdir(parents=True, exist_ok=True)
    files = {}
    for name in REAL_DOCS:
        src = TEMP_B2 / f"{name}.json"
        if not src.is_file():
            raise FileNotFoundError(f"B2_fused 源文件缺失（temp 易失）：{src}")
        dst = out_dir / f"{name}.json"
        if not dst.exists() or sha256_of(dst) != sha256_of(src):
            shutil.copyfile(src, dst)
        files[f"b2_fused/{name}.json"] = sha256_of(dst)
    questions_sha = sha256_of(QUESTIONS_PATH)
    manifest = {
        "artifact_type": "phase13c_c_frozen_inputs",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "files": files,
        "questions_file_sha256": questions_sha,
    }
    write_json(WORKDIR / "frozen_inputs" / "manifest.json", manifest)
    return manifest


def load_inputs():
    """加载冻结文档与 40 题数据集，重映射 document_id 并做 C-G1 一致性校验。

    B2_fused ScientificDocument 内部 document_id 为语料名（13C 沙盒命名空间），
    而 40 题 gold 使用生产 sha256 document_id。此处用语料 manifest 的
    file_name → document_id 契约把文档重映射到生产 ID 空间（原值记录在
    remap 表中随 manifest 冻结），使 chunk/qrels/检索集合共用一个 ID 空间。
    """
    corpus = json.loads(
        (ROOT / "evals" / "research_ingestion" / "academic_corpus_manifest_v1.json")
        .read_text(encoding="utf-8"))
    id_by_stem = {}
    for p in corpus.get("papers", []):
        stem = p.get("file_name", "").removesuffix(".pdf")
        id_by_stem[stem] = p.get("document_id")
    remap = {}
    docs: dict[str, dict] = {}
    for name in REAL_DOCS:
        p = WORKDIR / "frozen_inputs" / "b2_fused" / f"{name}.json"
        d = json.loads(p.read_text(encoding="utf-8"))
        prod_id = id_by_stem.get(name)
        if not prod_id:
            raise KeyError(f"语料 manifest 缺少 {name}.pdf 的 document_id 映射")
        remap[name] = {"b2_fused_document_id": d.get("document_id"), "production_document_id": prod_id}
        d["document_id"] = prod_id  # 重映射到生产 ID 空间（原值保留在 remap 表）
        docs[name] = d
    qdata = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    checks = {
        "dataset_hash_matches_frozen": qdata.get("dataset_hash") == FROZEN_DATASET_HASH,
        "corpus_hash_matches_frozen": qdata.get("corpus_hash") == FROZEN_CORPUS_HASH,
        "question_count": len(qdata["questions"]),
        "english_query_nonempty": all(
            (q.get("english_query") or "").strip() for q in qdata["questions"]
        ),
        "doc_ids_consistent": all(
            d.get("document_id") for d in docs.values()
        ),
        "document_id_remap": remap,
    }
    # 每篇文档 document_id 需与 40 题 gold 中的 document_id 对得上（语料对齐前提）。
    gold_doc_ids = {s["document_id"] for q in qdata["questions"] for s in q["gold_sources"]}
    checks["gold_doc_ids_covered"] = gold_doc_ids.issubset({d["document_id"] for d in docs.values()})
    # pass 只聚合布尔检查项；document_id_remap 是冻结的映射证据，不参与布尔判定。
    bool_items = {k: v for k, v in checks.items() if isinstance(v, bool)}
    checks["pass"] = all(bool_items.values())
    return docs, qdata, checks


def load_token_counter():
    """bge-m3 tokenizer 计数器（离线；add_special_tokens=False 口径，同 14A-C）。"""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_ID)

    def counter(text: str) -> int:
        return len(tok(text, add_special_tokens=False)["input_ids"])

    return counter


# --------------------------------------------------------------------------------------
# Step 2 四臂切分 + 结构门
# --------------------------------------------------------------------------------------

def chunk_all_arms(docs: dict[str, dict], counter) -> tuple[dict[str, dict[str, list]], dict[str, Any]]:
    """切分两次（确定性 C-G3），写 chunks/PC*.jsonl，并做 C-G4/C-G5 检查。"""
    chunks_by_arm: dict[str, dict[str, list]] = {}
    gate3, gate4, gate5, arm_stats = {}, {}, {}, {}
    for arm in ARMS:
        run1 = {name: chunk13cc.chunk_pdf(doc, arm, counter) for name, doc in docs.items()}
        run2 = {name: chunk13cc.chunk_pdf(doc, arm, counter) for name, doc in docs.items()}
        # C-G3：两次切分 chunk_id 序列与文本全等。
        deterministic = all(
            [c["chunk_id"] for c in run1[n]] == [c["chunk_id"] for c in run2[n]]
            and [c["text"] for c in run1[n]] == [c["text"] for c in run2[n]]
            for n in docs
        )
        gate3[arm] = {"deterministic": deterministic, "pass": deterministic}
        chunks_by_arm[arm] = run1
        # 结构检查（C-G4 provenance / C-G5 token 预算）。
        gate4[arm] = metrics13cc.check_provenance(run1, docs)
        flat = [c for chunks in run1.values() for c in chunks]
        gate5[arm] = metrics13cc.check_token_budget(flat, chunk13cc.TOKEN_TARGET)
        tokens = [c["token_count"] for c in flat]
        arm_stats[arm] = {
            "chunk_count": len(flat),
            "token_mean": round(sum(tokens) / len(tokens), 1) if tokens else 0.0,
            "token_max": max(tokens) if tokens else 0,
            "doc_chunk_counts": {n: len(run1[n]) for n in sorted(run1)},
        }
        # 落盘 chunks JSONL（评测证据）。
        out = WORKDIR / "chunks" / f"{arm}.jsonl"
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            for name in sorted(run1):
                for c in run1[name]:
                    f.write(json.dumps(c, ensure_ascii=False) + "\n")
    return chunks_by_arm, {"gate3": gate3, "gate4": gate4, "gate5": gate5, "arm_stats": arm_stats}


# --------------------------------------------------------------------------------------
# Step 3 嵌入 + 检索（镜像 embed14c 协议：CLS 池化 + L2 归一化 + cosine Top20）
# --------------------------------------------------------------------------------------

def embed_and_retrieve(chunks_by_arm: dict[str, dict[str, list]], questions: list[dict]) -> dict[str, Any]:
    import chromadb
    import numpy as np
    import torch
    from transformers import AutoModel, AutoTokenizer

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModel.from_pretrained(MODEL_ID)
    model.eval()

    def embed(texts: list[str]) -> "np.ndarray":
        """bge-m3 dense embedding（CLS 池化 + L2 归一化），批量 16，truncation 512。"""
        out = []
        with torch.no_grad():
            for i in range(0, len(texts), 16):
                batch = texts[i:i + 16]
                enc = tok(batch, padding=True, truncation=True, max_length=512, return_tensors="pt")
                hidden = model(**enc).last_hidden_state
                cls = torch.nn.functional.normalize(hidden[:, 0], dim=-1)
                out.append(cls.cpu().numpy().astype("float32"))
        return np.concatenate(out, axis=0) if out else np.zeros((0, 1024), dtype="float32")

    queries = [{"qid": q["question_id"], "text": q["english_query"]} for q in questions]
    (WORKDIR / "embeddings").mkdir(parents=True, exist_ok=True)
    (WORKDIR / "retrieval").mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(WORKDIR / "chroma"))
    stats = {}
    for arm in ARMS:
        chunks = [c for name in sorted(chunks_by_arm[arm]) for c in chunks_by_arm[arm][name]]
        npz_path = WORKDIR / "embeddings" / f"{arm}.npz"
        if npz_path.exists():
            # 断点续跑：向量已落盘则直接载入（避免重复 30 分钟 CPU 嵌入）。
            data = np.load(npz_path, allow_pickle=False)
            vecs = data["embeddings"]
            cached_ids = [str(x) for x in data["chunk_ids"]]
            if cached_ids != [c["chunk_id"] for c in chunks]:
                raise RuntimeError(f"{arm}: 已缓存向量与当前切分 chunk_id 不一致，须删除 npz 重算")
            embed_seconds = 0.0
        else:
            t0 = time.perf_counter()
            vecs = embed([c["text"] for c in chunks])
            embed_seconds = time.perf_counter() - t0
        col_name = f"ef_13cc_{arm.lower()}"
        try:
            client.delete_collection(col_name)
        except Exception:
            pass
        col = client.create_collection(name=col_name, metadata={"hnsw:space": "cosine", "arm": arm})
        col.add(ids=[c["chunk_id"] for c in chunks], embeddings=vecs.tolist(),
                documents=[c["text"] for c in chunks],
                metadatas=chroma_metadatas_for_records(chunks))
        t1 = time.perf_counter()
        q_vecs = embed([q["text"] for q in queries])
        run = {}
        for q, qv in zip(queries, q_vecs):
            res = col.query(query_embeddings=[qv.tolist()], n_results=min(TOP_K, len(chunks)))
            run[q["qid"]] = list(res["ids"][0])
        query_seconds = time.perf_counter() - t1
        np.savez_compressed(
            WORKDIR / "embeddings" / f"{arm}.npz",
            chunk_ids=np.array([c["chunk_id"] for c in chunks]), embeddings=vecs)
        write_json(WORKDIR / "retrieval" / f"{arm}_run.json", run)
        stats[arm] = {
            "collection": col_name, "chunk_count": len(chunks), "query_count": len(queries),
            "embed_seconds": round(embed_seconds, 1), "query_seconds": round(query_seconds, 1),
            "dim": int(vecs.shape[1]),
        }
        print(f"[13cc] {arm}: {len(chunks)} chunks embedded+retrieved", flush=True)
    write_json(WORKDIR / "metrics" / "retrieval_stats.json", stats)
    return stats


# --------------------------------------------------------------------------------------
# Step 4-6 qrels / 指标 / bootstrap / focus / 报告
# --------------------------------------------------------------------------------------

def first_relevant_rank(run: dict[str, list[str]], relevant: list[str]) -> int | None:
    """首个 relevant chunk 的 1-based 名次（无则 None）。"""
    for rank, cid in enumerate(run, 1):
        if cid in relevant:
            return rank
    return None


def main() -> int:
    started = datetime.now(timezone.utc).isoformat()
    WORKDIR.mkdir(parents=True, exist_ok=True)

    print("[13cc] Step1 冻结输入 ...", flush=True)
    manifest = freeze_inputs()
    docs, qdata, g1_checks = load_inputs()
    questions = qdata["questions"]
    gate1 = {"checks": g1_checks, "inputs_manifest": manifest, "pass": g1_checks["pass"]}
    if not gate1["pass"]:
        write_json(WORKDIR / "gates" / "C_G1_inputs.json", gate1)
        print("[13cc] C-G1 FAIL，停止", flush=True)
        return 1
    write_json(WORKDIR / "gates" / "C_G1_inputs.json", gate1)

    print("[13cc] Step2 四臂切分 + 结构门 ...", flush=True)
    counter = load_token_counter()
    chunks_by_arm, struct = chunk_all_arms(docs, counter)
    write_json(WORKDIR / "gates" / "C_G3_G4_G5_structure.json", struct)

    print("[13cc] Step3 嵌入 + 检索 ...", flush=True)
    stats = embed_and_retrieve(chunks_by_arm, questions)

    print("[13cc] Step4 qrels 投影 + Step5 指标 ...", flush=True)
    runs = {arm: json.loads((WORKDIR / "retrieval" / f"{arm}_run.json").read_text(encoding="utf-8"))
            for arm in ARMS}
    qrels_by_arm, coverage_by_arm, ir_by_arm, gate2 = {}, {}, {}, {}
    for arm in ARMS:
        flat_chunks = {name: chunks_by_arm[arm][name] for name in chunks_by_arm[arm]}
        proj = metrics13cc.project_qrels_anchor(flat_chunks, questions)
        qrels_by_arm[arm] = proj["qrels"]
        coverage_by_arm[arm] = {
            "coverage_rate": proj["coverage_rate"],
            "gold_source_count": proj["gold_source_count"],
            "uncovered_count": len(proj["uncovered"]),
            "uncovered": proj["uncovered"],
            "num_relevant_by_qid": proj["num_relevant"],
        }
        # C-G2：coverage = 1.0 才 PASS。
        gate2[arm] = {
            "coverage_rate": proj["coverage_rate"],
            "pass": len(proj["uncovered"]) == 0,
        }
        ir_by_arm[arm] = metrics13cc.compute_ir_metrics(proj["qrels"], runs[arm])
    write_json(WORKDIR / "metrics" / "qrels_projection.json",
               {arm: coverage_by_arm[arm] for arm in ARMS})
    write_json(WORKDIR / "gates" / "C_G2_coverage.json", gate2)
    write_json(WORKDIR / "metrics" / "ir_by_arm.json",
               {arm: ir_by_arm[arm]["aggregate"] for arm in ARMS})
    write_json(WORKDIR / "metrics" / "ir_per_query.json",
               {arm: ir_by_arm[arm]["per_query"] for arm in ARMS})

    # bootstrap：PC1/2/3 vs PC0（n=10000，seed 固定），运行两次全等 = C-G7。
    def bootstrap_round() -> list[dict]:
        out = []
        for metric in ("ndcg_at_5", "recall_at_5", "recall_at_10", "mrr_at_5"):
            vals = {arm: metrics13cc.per_query_metric_values(ir_by_arm[arm], metric) for arm in ARMS}
            for arm in ARMS[1:]:
                res = metrics13cc.paired_bootstrap(vals[arm], vals["PC0"], n=BOOTSTRAP_N, seed=BOOTSTRAP_SEED)
                out.append({"metric": metric, "arm": arm, "vs": "PC0", **res})
        return out

    boot1, boot2 = bootstrap_round(), bootstrap_round()
    gate7 = {"deterministic": boot1 == boot2, "pass": boot1 == boot2}
    write_json(WORKDIR / "metrics" / "bootstrap_vs_PC0.json", {"n": BOOTSTRAP_N, "seed": BOOTSTRAP_SEED, "results": boot1})
    write_json(WORKDIR / "gates" / "C_G7_bootstrap_determinism.json", gate7)

    # Step6 focus 题诊断（含生产冻结排名参照）。
    per_query_prod = json.loads(PER_QUERY_PATH.read_text(encoding="utf-8")) if PER_QUERY_PATH.exists() else {}
    prod = per_query_prod.get("phase10", {})
    focus = {}
    for qid in FOCUS_QIDS:
        entry: dict[str, Any] = {"question_id": qid}
        for arm in ARMS:
            rel = qrels_by_arm[arm].get(qid, [])
            rank = first_relevant_rank(runs[arm].get(qid, []), rel)
            entry[arm] = {
                "num_relevant": len(rel), "first_relevant_rank": rank,
                "in_top10": rank is not None and rank <= 10,
                "in_top20": rank is not None and rank <= 20,
            }
        prow = prod.get(qid)
        if prow:
            sm = prow.get("standard_metrics", {})
            entry["production_phase10_reference"] = {
                "recall_at_10": sm.get("recall_at_10"), "recall_at_20": sm.get("recall_at_20"),
                "ndcg_at_5": sm.get("ndcg_at_5"),
            }
        focus[qid] = entry
    write_json(WORKDIR / "metrics" / "focus_questions.json", focus)

    # 硬门汇总与 verdict。
    gates = {
        "C_G1_inputs": gate1["pass"],
        "C_G2_coverage": {a: gate2[a]["pass"] for a in ARMS},
        "C_G3_determinism": {a: struct["gate3"][a]["pass"] for a in ARMS},
        "C_G4_provenance": {a: struct["gate4"][a]["pass"] for a in ARMS},
        "C_G5_token_budget": {a: struct["gate5"][a]["pass"] for a in ARMS},
        "C_G6_protocol_consistency": True,  # 同进程同模型同投影代码路径（结构保证），见 status 说明
        "C_G7_bootstrap_determinism": gate7["pass"],
    }
    all_pass = (
        gates["C_G1_inputs"] and all(gates["C_G2_coverage"].values())
        and all(gates["C_G3_determinism"].values()) and all(gates["C_G4_provenance"].values())
        and all(gates["C_G5_token_budget"].values()) and gates["C_G7_bootstrap_determinism"]
    )
    verdict = "PASS" if all_pass else "FAIL"
    write_json(WORKDIR / "gates" / "hard_gates.json",
               {"gate_definitions_version": GATE_VERSION, "gates": gates, "verdict": verdict})

    # status.json + final_report.md。
    status = {
        "artifact_type": "phase13c_c_chunk_benchmark_status",
        "schema_version": "phase13c-c-chunk-benchmark-v1",
        "gate_definitions_version": GATE_VERSION,
        "plan": "PLAN.md（执行前冻结）",
        "generated_at": started,
        "stop_at": "Phase 13C-C",
        "verdict": verdict,
        "inputs_frozen": {"b2_fused_docs": len(REAL_DOCS), "questions": len(questions),
                          "dataset_hash": FROZEN_DATASET_HASH},
        "retrieval_protocol": {"embedding": MODEL_ID, "top_k": TOP_K, "reranker": False,
                               "qrels": "anchor_strict"},
        "arm_stats": struct["arm_stats"],
        "ir_aggregate": {arm: ir_by_arm[arm]["aggregate"] for arm in ARMS},
        "bootstrap": {"n": BOOTSTRAP_N, "seed": BOOTSTRAP_SEED,
                      "significant_pairs": [b for b in boot1 if b.get("significant")]},
        "focus_questions": focus,
        "gates": gates,
    }
    write_json(WORKDIR / "status.json", status)

    # manifest（冻结不可变产物；chroma 为临时不入 manifest）。
    man_files = {}
    for sub in ("frozen_inputs", "chunks", "embeddings", "retrieval", "metrics", "gates"):
        d = WORKDIR / sub
        if d.is_dir():
            for p in sorted(d.rglob("*")):
                if p.is_file():
                    man_files[str(p.relative_to(WORKDIR)).replace("\\", "/")] = sha256_of(p)
    for name in ("status.json", "PLAN.md"):
        p = WORKDIR / name
        if p.exists():
            man_files[name] = sha256_of(p)
    write_json(WORKDIR / "manifest.json", {
        "artifact_type": "phase13c_c_manifest", "created_at": started, "files": man_files})

    # final_report.md。
    rep = ["# Phase 13C-C PDF Chunk Benchmark — Final Report", "",
           f"- 生成时间：{started}", f"- verdict：**{verdict}**（gate_definitions {GATE_VERSION}）",
           f"- 协议：B2_fused × {len(REAL_DOCS)} 文档、{len(questions)} 题、BGE-M3 Dense Top20、无 reranker、anchor 严格制 qrels", "",
           "## Arm 结构统计", "",
           "| arm | 策略 | chunks | token_mean | token_max |", "|---|---|---:|---:|---:|"]
    for arm in ARMS:
        s = struct["arm_stats"][arm]
        rep.append(f"| {arm} | {chunk13cc.ARM_SPECS[arm][0]} | {s['chunk_count']} | {s['token_mean']} | {s['token_max']} |")
    rep += ["", "## 标准 IR（aggregate，40 题）", "", "| arm | P@5 | R@5 | R@10 | MRR@5 | MAP@5 | nDCG@5 | Hit@20 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for arm in ARMS:
        m = ir_by_arm[arm]["aggregate"]
        rep.append(f"| {arm} | {m['precision_at_5']:.4f} | {m['recall_at_5']:.4f} | {m['recall_at_10']:.4f} | "
                   f"{m['mrr_at_5']:.4f} | {m['map_at_5']:.4f} | {m['ndcg_at_5']:.4f} | {m['hit_rate_at_20']:.4f} |")
    rep += ["", "## Bootstrap（arm vs PC0，n=10000，seed=20260915）", "",
            "| metric | arm | mean Δ | CI95 | P(Δ>0) | significant |", "|---|---|---:|---|---:|---|"]
    for b in boot1:
        ci = f"[{b['ci95_low']:.4f}, {b['ci95_high']:.4f}]"
        rep.append(f"| {b['metric']} | {b['arm']} | {b['mean_delta']:+.4f} | {ci} | {b['p_delta_gt_0']:.3f} | {b['significant']} |")
    rep += ["", "## Focus Questions", ""]
    for qid in FOCUS_QIDS:
        e = focus[qid]
        line = " | ".join(f"{a}: rank={e[a]['first_relevant_rank']} top20={e[a]['in_top20']}" for a in ARMS)
        prod_note = ""
        if "production_phase10_reference" in e:
            pr = e["production_phase10_reference"]
            prod_note = f"（生产 phase10：R@10={pr['recall_at_10']}，R@20={pr['recall_at_20']}）"
        rep.append(f"- **{qid}** — {line} {prod_note}")
    rep += ["", "## 硬门", ""]
    for g, v in gates.items():
        rep.append(f"- {g}: {v}")
    rep += ["", "## 决策（数据驱动）", ""]
    boot_ndcg = [b for b in boot1 if b["metric"] == "ndcg_at_5"]
    sig = [b["arm"] for b in boot_ndcg if b.get("significant")]
    if sig:
        rep.append(f"- 结构策略相对 PC0 的 nDCG@5 提升显著：{', '.join(sig)}（详见 bootstrap 表）。")
    else:
        rep.append("- 13C-B 结构收益未转化为 chunk 级 Dense 检索显著提升（nDCG@5 CI 均含 0），如实记录。")
    t20 = {qid: any(focus[qid][a]["in_top20"] for a in ARMS) for qid in ("ARR-037", "ARR-040")}
    for qid, hit in t20.items():
        if hit:
            rep.append(f"- {qid}：至少一个 PC 臂进入 Top20。")
        else:
            rep.append(f"- {qid}：四臂仍全部 Top20 miss → 支持进入 Sparse/BM25 candidate augmentation，不再回改 parser。")
    (WORKDIR / "reports").mkdir(exist_ok=True)
    (WORKDIR / "reports" / "final_report.md").write_text("\n".join(rep) + "\n", encoding="utf-8")

    print(f"[13cc] done: verdict={verdict}", flush=True)
    return 0 if all_pass else 2


if __name__ == "__main__":
    sys.exit(main())
