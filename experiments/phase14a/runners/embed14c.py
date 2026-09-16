"""Phase 14A-C 嵌入与检索 runner（方案 §13-§14 重建执行版）。

在 agent 环境 python 下运行（transformers/chromadb/torch 就绪），离线加载
BAAI/bge-m3（HF_HOME + HF_HUB_OFFLINE=1 + TRANSFORMERS_OFFLINE=1）。

职责：
- 读入 8 个 arm 的 chunks JSONL 与查询 JSON；
- 每 arm 独立 Chroma persistent 集合（ef_14ac_docx_dc0..3 / ef_14ac_xlsx_xc0..3，cosine）；
- Dense Top20 检索（无 reranker——14A-C 只评 Dense 检索单元质量）；
- 输出 run JSON（qid → ranked chunk_ids）与延迟/嵌入统计。

用法：python runners/embed14c.py <workdir> <python_of_this_process 无需>
输入：<workdir>/chunks/<ARM>.jsonl、<workdir>/queries/queries_{docx|xlsx}.json
输出：<workdir>/embeddings/<ARM>.npz、<workdir>/retrieval/<ARM>_run.json、
      <workdir>/metrics/retrieval_stats.json
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

WORKDIR = Path(sys.argv[1]) if len(sys.argv) > 1 else None
MODEL_ID = "BAAI/bge-m3"
TOP_K = 20

# arm → (格式, 集合名)。
ARM_SPECS = {
    "DC0": ("docx", "ef_14ac_docx_dc0"),
    "DC1": ("docx", "ef_14ac_docx_dc1"),
    "DC2": ("docx", "ef_14ac_docx_dc2"),
    "DC3": ("docx", "ef_14ac_docx_dc3"),
    "XC0": ("xlsx", "ef_14ac_xlsx_xc0"),
    "XC1": ("xlsx", "ef_14ac_xlsx_xc1"),
    "XC2": ("xlsx", "ef_14ac_xlsx_xc2"),
    "XC3": ("xlsx", "ef_14ac_xlsx_xc3"),
}


def main() -> int:
    import chromadb
    import torch
    from transformers import AutoModel, AutoTokenizer

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    workdir = WORKDIR
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModel.from_pretrained(MODEL_ID)
    model.eval()

    def embed(texts: list[str]) -> np.ndarray:
        """bge-m3 dense embedding（CLS 池化 + L2 归一化），批量 16。"""

        out = []
        with torch.no_grad():
            for i in range(0, len(texts), 16):
                batch = texts[i : i + 16]
                enc = tok(batch, padding=True, truncation=True, max_length=512,
                          return_tensors="pt")
                hidden = model(**enc).last_hidden_state
                # CLS 池化（bge-m3 dense 用法）+ L2 归一化。
                cls = hidden[:, 0]
                cls = torch.nn.functional.normalize(cls, dim=-1)
                out.append(cls.cpu().numpy().astype(np.float32))
        return np.concatenate(out, axis=0) if out else np.zeros((0, 1024), dtype=np.float32)

    client = chromadb.PersistentClient(path=str(workdir / "chroma"))
    stats: dict[str, dict] = {}
    for arm, (fmt, col_name) in ARM_SPECS.items():
        chunk_path = workdir / "chunks" / f"{arm}.jsonl"
        queries_path = workdir / "queries" / f"queries_{fmt}.json"
        if not chunk_path.exists() or not queries_path.exists():
            continue
        chunks = [json.loads(l) for l in chunk_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        queries = json.loads(queries_path.read_text(encoding="utf-8"))
        qtexts = [q["text"] for q in queries]

        # 嵌入（计时）。
        t0 = time.perf_counter()
        chunk_vecs = embed([c["text"] for c in chunks])
        embed_seconds = time.perf_counter() - t0

        # 独立集合（重建保证可复现）。
        try:
            client.delete_collection(col_name)
        except Exception:
            pass
        col = client.create_collection(
            name=col_name, metadata={"hnsw:space": "cosine", "arm": arm}
        )
        col.add(
            ids=[c["chunk_id"] for c in chunks],
            embeddings=chunk_vecs.tolist(),
            documents=[c["text"] for c in chunks],
            # Phase14 记录不是 EvidenceFlow Domain Chunk，仍保存可重建的标量 provenance。
            metadatas=[
                {
                    "schema_version": c.get("schema_version", "phase14a-c-chunk-v1"),
                    "document_id": c["document_id"],
                    "arm": c["arm"],
                    "strategy": c["strategy"],
                    "seq": c["seq"],
                    "token_count": c["token_count"],
                    "element_ids_json": json.dumps(
                        c.get("element_ids", []), ensure_ascii=False, separators=(",", ":")
                    ),
                }
                for c in chunks
            ],
        )

        # 检索（计时）。
        t1 = time.perf_counter()
        q_vecs = embed(qtexts)
        run: dict[str, list[str]] = {}
        for q, qv in zip(queries, q_vecs):
            res = col.query(query_embeddings=[qv.tolist()], n_results=min(TOP_K, len(chunks)))
            run[q["qid"]] = list(res["ids"][0])
        query_seconds = time.perf_counter() - t1

        np.savez_compressed(
            workdir / "embeddings" / f"{arm}.npz",
            chunk_ids=np.array([c["chunk_id"] for c in chunks]),
            embeddings=chunk_vecs,
        )
        (workdir / "retrieval" / f"{arm}_run.json").write_text(
            json.dumps(run, ensure_ascii=False, indent=1), encoding="utf-8"
        )
        stats[arm] = {
            "collection": col_name,
            "chunk_count": len(chunks),
            "query_count": len(queries),
            "embed_seconds": round(embed_seconds, 3),
            "query_seconds": round(query_seconds, 3),
            "dim": int(chunk_vecs.shape[1]),
        }
        print(f"[embed14c] {arm}: {len(chunks)} chunks, {len(queries)} queries done", flush=True)

    out = workdir / "metrics" / "retrieval_stats.json"
    out.write_text(json.dumps(stats, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
