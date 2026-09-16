# EvidenceFlow

EvidenceFlow is a scientific-document RAG system focused on evidence-grounded
retrieval and answering over research PDFs.

当前 active corpus 为 10 篇点云补全论文；知识库与评测集均与旧领域数据隔离。

## Capability Layers（按能力层组织的正式架构）

### 1. Scientific PDF Ingestion

```text
Documents -> Clean -> Structured ScientificDocument -> Chunk (BGE-M3 C512/O64)
```

- PDF 解析 / cleaning 融合基准（13C-B）：SELECTED = B2_fused，契约已冻结。
- 产物：[evals/evaluation_v3/phase13c_scientific_ingestion_benchmark](evals/evaluation_v3/phase13c_scientific_ingestion_benchmark/)。

### 2. Indexing & Retrieval

```text
BGE-M3 Embedding -> Chroma
Query -> BGE-M3 Dense -> Chroma ANN Top20
```

- 正式 collection：`ef_research_bge_m3_hybrid_c512_v2`（817 chunks / 10 papers）。
- Corpus signature：`bb184726b7f98756bd519d70e1a2a1f1224b5108e689680b2c328023ba1f4332`。

### 3. Learned Reranking

```text
ANN Top20 -> BGE Reranker -> Top5
```

- `runtime.py` 是 ingest / ask 的唯一生产依赖装配入口（Composition Root）；
  CLI 只触发装配、不含业务逻辑；问答编排只位于 `RAGService`。

### 4. Evidence Context

```text
Top5 -> Context Builder
```

### 5. Answer & Citation

```text
Context Builder -> LLM -> Citation Validation -> RAGAnswer
```

- LLM endpoint、model 和 API key 仅通过 `.env` 或环境变量配置；项目不保存真实密钥。

### 6. Evaluation

冻结 40 题 Research Retrieval Dataset，Gold 使用 `document_id + page_range + evidence_anchor`，
不依赖 Chunk ID。当前正式 Runtime（Dense Top20 → Reranker → Top5）标准 IR 指标：

| Metric | Value |
|---|---:|
| Precision@5 | 0.2684 |
| Recall@5 | 0.5103 |
| MRR@5 | 0.5175 |
| MAP@5 | 0.3631 |
| nDCG@5 | 0.4903 |

来源：[Phase 15B v2](evals/evaluation_v3/phase15b_hybrid_candidate_v2/)。
指标 canonical 实现：`src/evidenceflow/eval/ir_metrics.py`（TREC / BEIR 语义）。

## Retired Retrieval Paths

Phase 15B v2 已将 Hybrid 判定为 `DO_NOT_PROMOTE`。Translation、Multilingual、BM25
Production branch 与 RRF 已从正式 Runtime、Settings 和 Domain Model 退休；历史源码仅存放于
`experiments/archive/`，冻结结果继续由 `evals/` 保存。Runtime rollback 使用已验证索引、
冻结 artifact 与版本历史，不再依赖配置开关维护第二套生产算法。

## Usage

使用已有 Conda 环境，并复用本地模型缓存：

```powershell
conda activate agent
$env:HF_HOME = (Resolve-Path .cache/huggingface)
$env:TRANSFORMERS_OFFLINE = '1'
python -m evidenceflow ingest data/source_docs
python -m evidenceflow ask "PoinTr 如何表示用于 Transformer 处理的点云？"
```

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
