# EvidenceFlow

EvidenceFlow is a scientific-document RAG system focused on evidence-grounded
retrieval and answering over research PDFs.

当前 active corpus 为 10 篇点云补全论文；知识库与评测集均与旧领域数据隔离。

## Capability Layers（按能力层组织的正式架构）

### 1. Scientific PDF Ingestion

```text
Documents -> Clean -> Structured ScientificDocument -> Hybrid Chunk (BGE-M3, ≤512 tokens)
```

- PDF 解析 / cleaning 融合基准（13C-B）：SELECTED = B2_fused，契约已冻结。
- 产物：[evals/evaluation_v3/phase13c_scientific_ingestion_benchmark](evals/evaluation_v3/phase13c_scientific_ingestion_benchmark/)。

### 2. Indexing & Retrieval

```text
BGE-M3 Embedding -> Chroma
Query -> BGE-M3 Dense -> Exact Dense Top20（进程内 ExactDenseIndex，numpy.exact）
```

- 正式 collection 由 lifecycle active 指针（`artifacts/phase15a/index_state.json`）决定，当前为
  `ef_research_bge_m3_hybrid_c512_v2`（817 chunks / 10 papers）。Chroma 是唯一持久化来源；Exact 检索在
  多进程间逐题可复现。ACTIVE 索引不允许直接 ingest，重建走 `scripts/build_staging_index.py`。
- Corpus signature：`bb184726b7f98756bd519d70e1a2a1f1224b5108e689680b2c328023ba1f4332`。

### 3. Learned Reranking

```text
Exact Dense Top20 -> BGE Reranker -> Top5
```

- `runtime.py` 是 ingest / ask 的唯一生产依赖装配入口（Composition Root）；
  CLI 只触发装配、不含业务逻辑；问答编排只位于 `RAGService`。

### 4. Evidence Context

```text
Top5 -> Context Builder
```

### 5. Answer & Citation

```text
Context Builder -> LLM (rag-generator-v3, JSON 输出) -> Citation Validation -> RAGAnswer(status)
```

- 结构化生成：模型只返回 `{"status": "answered" | "insufficient_context", "answer": "..."}`，
  经 Pydantic 校验，格式错误最多修复重试一次，仍失败报 `Generation format error`。
- `RAGAnswer.status` 区分“已回答”与“证据不足拒答”；`ask` 输出中显示 `Status:` 行。`answered`
  必须带合法引用，`insufficient_context` 可以没有引用。
- 默认 LLM 为 DeepSeek（`deepseek-flash`，`LLM_THINKING=disabled`）；换用其他服务时需同时覆盖
  `LLM_BASE_URL`、`LLM_MODEL`，并把 `LLM_THINKING` 设为 `auto`。
- LLM endpoint、model 和 API key 仅通过 `.env` 或环境变量配置；项目不保存真实密钥。

### 6. Evaluation

冻结 40 题 Research Retrieval Dataset，Gold 使用 `document_id + page_range + evidence_anchor`，
不依赖 Chunk ID。当前正式基线 B0'（Exact Dense Top20 → Reranker → Top5，CPU 回放，40 题 / eligible 38）
标准 IR 指标：

| Metric | Value |
|---|---:|
| Precision@5 | 0.2684 |
| Recall@5 | 0.5103 |
| MRR@5 | 0.5175 |
| MAP@5 | 0.3631 |
| nDCG@5 | 0.4903 |

来源：[`artifacts/stage_a/A1_exact_baseline.json`](artifacts/stage_a/A1_exact_baseline.json)
（sha256 `422fd629b22f26caf9452e3598bf805d01df9b9cf24e6d42b5a6890e40a1411f`；与此前 HNSW 基线逐题 Top5 一致）。
指标 canonical 实现：`src/evidenceflow/eval/ir_metrics.py`（TREC / BEIR 语义）。

## Retired Retrieval Paths

Phase 15B v2 已将 Hybrid 判定为 `DO_NOT_PROMOTE`。Translation、Multilingual、BM25
Production branch、RRF 以及以 Chroma HNSW 作为线上候选检索均已退休；历史源码可从 git tag
`stage-a-start` 恢复，冻结结果继续由 `evals/` 保存。Runtime rollback 使用已验证索引、
冻结 artifact 与版本历史，不再依赖配置开关维护第二套生产算法。

## Usage

使用已有 Conda 环境，并复用本地模型缓存：

```powershell
conda activate agent-gpu
$env:HF_HOME = (Resolve-Path .cache/huggingface)
$env:TRANSFORMERS_OFFLINE = '1'
python -m evidenceflow ask "PoinTr 如何表示用于 Transformer 处理的点云？"
```

`python -m evidenceflow ingest` 只用于全新环境（没有 lifecycle 状态文件、目标不是 ACTIVE）；对当前 ACTIVE
索引执行会被拒绝（退出码 2）。

## Design Docs

- 当前规范：[PROJECT_SPEC.md](PROJECT_SPEC.md)（v0.27）；v0.26 完整历史见
  [docs/history/PROJECT_SPEC_v0.26_full.md](docs/history/PROJECT_SPEC_v0.26_full.md)。
- Agentic RAG 架构审查与阶段计划：[docs/agentic_rag/](docs/agentic_rag/)
  （[01_ARCHITECTURE_REVIEW.md](docs/agentic_rag/01_ARCHITECTURE_REVIEW.md)、
  [02_STAGE_A_EXECUTION_PLAN.md](docs/agentic_rag/02_STAGE_A_EXECUTION_PLAN.md)）；Stage A 总结见
  [artifacts/stage_a/STAGE_A_SUMMARY.md](artifacts/stage_a/STAGE_A_SUMMARY.md)。

## Experimental Work

Controlled experiments evaluated DOCX/XLSX ingestion (Phase 14A-A/B/C all PASS
on controlled corpus), but these adapters are **not part of the current runtime**
(`RUNTIME_PROMOTED = FALSE`)。Office 分支已归档：
[docs/history/phase14_multiformat.md](docs/history/phase14_multiformat.md)。

## Phase History

Phase 11B Cleanup、Phase 11C Technical Debt Closure 等历史阶段记录已移至
[docs/evaluation/history.md](docs/evaluation/history.md)。

## 已知限制

学术 PDF 跨行断词普遍存在；表格被线性化，行列结构未保留；References /
Supplementary 暂无结构化 section metadata；多文档题只能稳定召回部分来源。当前系统是 text-centric
Academic RAG，尚无证据支持立即引入多模态解析。

# RAG
