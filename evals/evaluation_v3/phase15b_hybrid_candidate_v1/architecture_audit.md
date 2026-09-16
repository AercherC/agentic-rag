# Phase 15B Architecture Audit

审查日期：2026-09-16

本审查只覆盖 Phase 15B 离线 candidate benchmark 所需的复用边界；没有修改
Runtime、Production Chroma、Embedding、Reranker、Gold 或 qrels。

## Frozen identity observed

| 项目 | 当前值 | 结果 |
|---|---|---|
| active collection | `ef_research_bge_m3_hybrid_c512_v2` | PASS |
| corpus signature | `bb184726b7f98756bd519d70e1a2a1f1224b5108e689680b2c328023ba1f4332` | PASS |
| index signature | `b4e9432c7170615ade2c53803a887fa86d0f793949934a57db421f0da57739e5` | PASS |
| document count | 10 | PASS |
| chunk count | 817 | PASS |
| active lifecycle | `ACTIVE` | PASS |
| Chroma distance | cosine | PASS |

证据：`artifacts/phase15a/baseline_snapshot.json`、
`artifacts/phase15a/active_manifest.json` 与 `scripts/verify_active_index.py` 的只读 smoke
结果。当前 smoke 结果为 collection readable、metadata coverage 100%、
`iter_all_chunks()` 恢复 817 条、sample query 成功。

## Q1 — BM25 是否可独立复用

`BM25_REUSE = YES`

`src/evidenceflow/retrieval/bm25.py` 的 `BM25Index` 直接接收
`Sequence[DocumentChunk]` 与 `TokenizerProtocol`，使用冻结参数 `k1=1.5`、`b=0.75`，
`search(query, top_k=...)` 独立返回 `BM25RetrievedChunk`。它不接收
`TranslationProvider`，不调用 RRF，也不依赖 `evidenceflow.eval`；每个结果保留原始
`chunk_id`，并用 `chunk_id` 做同分稳定排序。现有 `tests/test_lexical.py` 已覆盖确定性
BM25 行为，因此 Phase 15B 复用此 primitive，不新建第二套索引。

## Q2 — Dense Top40 是否可用

`DENSE40_SUPPORTED = YES`

`src/evidenceflow/retrieval/retriever.py` 的 `DenseRetriever.retrieve()` 接受调用级
`top_k`，该参数只传给当前 `VectorStore.search()`。`src/evidenceflow/storage/chroma.py`
只将请求深度限制为 `min(top_k, record_count)`，不会改变 query embedding、collection、
distance 或 stored vectors。因此 benchmark 可请求 `top_k=20` 与 `top_k=40`，不需要修改
production default（当前 `DenseRerankerRetriever` 仍固定使用 20）。

## Q3 — Learned Reranker 是否可接收最多 40 个候选

`RERANK40_SUPPORTED = YES`

`src/evidenceflow/retrieval/reranker.py` 的 `LearnedReranker` 当前默认
`max_candidates=100`，其 `TransformersCrossEncoderScorer` 使用现有
`BAAI/bge-reranker-base`、`max_length=512`、CPU、`float32`。`rerank()` 只对唯一候选
评分并排序，不改变候选 identity；现有 batch 与 stable tie-break 逻辑可直接复用，
不需要训练、改 checkpoint 或修改 production wrapper。

## Q4 — Candidate Pool 是否可评价

`CANDIDATE_EVAL_SUPPORTED = YES`

标准 v3 qrels 与 Gold Mapping 的 question identity 已确认是 40 个 `ARR-001` 至
`ARR-040`，其冻结 qrels hash 为
`9337c26327097085dd3dc400a73e6c0baa79f375944d2be401edd39d014c9bb9`，mapping hash 为
`42d067404ae01e204bdfcfce87f008d11a237d24215e9b05fb89642a6c004203`。Phase 15B 使用
当前 active chunk 文本对冻结 evidence anchor 做只读投影，用于 Candidate HitRate、
EvidenceAnchorRecall 和 AllRequired；这不会重写 canonical qrels 或 Gold。

当前投影确认 ARR-003 与 ARR-033 的 anchor 在 frozen production representation 中
不可表示，保持 `UNINDEXABLE`。它们不进入 candidate-miss 统计，也不从 canonical
输入中删除。其余 42 个 evidence groups 可在当前 representation 中评价。

## Reuse / change boundary

### FILES_TO_REUSE

- `src/evidenceflow/retrieval/bm25.py`
- `src/evidenceflow/retrieval/retriever.py`
- `src/evidenceflow/retrieval/reranker.py`
- `src/evidenceflow/storage/chroma.py`
- `src/evidenceflow/embedding/bge_m3.py`
- `src/evidenceflow/ingestion/tokenizer.py`
- `src/evidenceflow/eval/ir_metrics.py`
- `src/evidenceflow/eval/comparison.py`
- `artifacts/phase15a/baseline_snapshot.json`
- `artifacts/phase15a/active_manifest.json`
- `evals/evaluation_v2/phase35_gold_stabilization_v1/gold_mapping_v3.json`
- `evals/evaluation_v2/phase35_gold_stabilization_v1/research_questions_v2_gold_stable.json`
- `evals/evaluation_v3/standard_ir_v1/qrels.json`
- `evals/evaluation_v3/standard_ir_v1/qrels_manifest.json`

### FILES_TO_ADD

- `experiments/run_phase15b_hybrid_candidate_benchmark.py`
- `tests/test_phase15b_hybrid_candidate.py`
- `evals/evaluation_v3/phase15b_hybrid_candidate_v1/` 下的 benchmark artifacts

### FILES_TO_MODIFY

- 无生产模块修改。

### FILES_NOT_TO_TOUCH

- `src/evidenceflow/runtime.py`
- `src/evidenceflow/config.py`
- `src/evidenceflow/retrieval/multilingual.py`
- `src/evidenceflow/retrieval/hybrid_fusion.py`
- `src/evidenceflow/retrieval/fusion.py`
- `src/evidenceflow/domain/`
- `data/chroma/`
- `evals/evaluation_v2/`
- `evals/evaluation_v3/standard_ir_v1/`
- `artifacts/phase15a/`
- Gold、qrels、source PDF 与现有 Phase 6–14 artifacts

## Audit decision

四个问题均满足 Phase 15B Step 1 条件，允许进入 Step 2。后续 runner 只在 offline
experiment path 中执行 Dense20、Dense40、BM25-20 与 `chunk_id` set union；不调用
`equal_ranked_rrf_fusion` 或 `weighted_rrf_fusion`，也不改变 production retrieval path。
