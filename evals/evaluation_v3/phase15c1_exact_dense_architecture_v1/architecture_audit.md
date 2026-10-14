# EvidenceFlow Phase 15C.1 Architecture Audit

## Scope

只读检查 `src/evidenceflow/`、`tests/`、`experiments/`、`scripts/` 与 `evals/`。
当前 Production 保持 `DenseRetriever -> ChromaVectorStore.search -> LearnedReranker`。

## Consumer classification

| Area | Classification | Finding |
|---|---|---|
| `src/evidenceflow/runtime.py` | PRODUCTION | Composition Root 仍只装配 HNSW Dense Top20 -> Reranker Top5。 |
| `src/evidenceflow/retrieval/retriever.py` | PRODUCTION / CANONICAL_REUSABLE | `DenseRetriever` 只依赖最小 `search()` 合同，可消费 Chroma 或 ExactDenseIndex。 |
| `src/evidenceflow/storage/chroma.py` | PRODUCTION / CANONICAL_REUSABLE | Chroma 继续负责持久化、metadata、lifecycle 和既有 HNSW `search()`；新增只读 embedding iterator。 |
| `src/evidenceflow/retrieval/exact_dense.py` | CANONICAL_REUSABLE | 唯一 exact candidate primitive；生产、评测和 HNSW audit 共用。 |
| `experiments/run_phase15b1_dense_candidate_depth.py` | EXPERIMENT_ONLY | 复用评测指标与 artifact 适配器，不再实现 exact search。 |
| `experiments/run_phase15c_dense30_promotion.py` | HISTORICAL | Phase 15C STOP 证据与回滚脚本，未进入 Production import graph。 |
| `experiments/archive/run_phase7_reranker_validation.py` / `run_phase8_runtime_readiness.py` | HISTORICAL | 旧 exhaustive cosine helper 只作历史证据；当前 canonical primitive 已覆盖其 executable 职责。 |
| `evals/evaluation_v3/*` | HISTORICAL_EVIDENCE | Gold、qrels、mapping、projection 与 frozen reports 只读保护。 |

## Answers

1. Production DenseRetriever 过去直接绑定完整 VectorStore；本阶段收窄为只读 `search()` 依赖，生产调用语义不变。
2. Chroma 原先没有稳定的 embedding iterator；本阶段增加 `iter_all_chunks_with_embeddings()`，按 `chunk_id ASC` 返回完整 Domain Chunk 与普通 float 列表。
3. Phase 7/8 有历史重复 exhaustive cosine；它们不在当前 Runtime，Phase 15C.1 不删除历史证据，ExactDenseIndex 成为唯一新的 executable primitive。
4. Phase 7/8 helper 不再被当前 Production/Evaluation 调用；历史 artifact 保留以满足可追溯性。
5. HNSW-specific Settings/code 仍由 Chroma 的持久化 Collection 与 Production fallback 使用；本阶段不删除 Chroma query capability，也不增加 backend 开关。

## Boundary

FAISS/NumPy ExactDenseIndex 是进程内派生索引，不写 `.faiss`、`.index` 或第二份持久化向量真源；本阶段不修改 `runtime.py` 的 Production path。
