# Phase 15B.5 Architecture Audit

## STATUS

PASS。审计范围为 `src/`、`tests/`、`experiments/`、`scripts/`、Settings、README 与当前规范。

## Classification

| 分类 | 内容 | 决策 |
|---|---|---|
| PRODUCTION | Loader/Cleaner/Hybrid Chunk、BGE-M3 Embedding、Chroma、DenseRetriever、LearnedReranker、ContextBuilder、Generator、Citation、RAGService | KEEP |
| ROLLBACK_REQUIRED | Active/Previous index identity、manifest/signatures、staging validation、activation/rollback | KEEP |
| CURRENT_EXPERIMENT | 生产 40Q replay、active relevance projection | KEEP |
| HISTORICAL_EXPERIMENT | Phase 6–12 runner、Phase 15B Hybrid benchmark、旧 research retrieval runner | ARCHIVE |
| COMPATIBILITY_ONLY | `eval.lexical`、`eval.hybrid` | DELETE |
| ONE_SHOT_MAINTENANCE | Phase 11B/11C cleanup/audit/finalize 脚本 | ARCHIVE |
| DUPLICATED | Phase-specific 历史测试重复保护旧 Runtime | DELETE WITH FEATURE |
| ZERO_CONSUMER | `top_k`、`reranker_enabled`、全部 `multilingual_*` Settings | DELETE |
| FROZEN_ARTIFACT | Phase15A、Phase15B v1/v2、Gold/qrels/mapping、Phase13F | KEEP, HASH VERIFIED |

## Production dependency graph

```text
build_rag_service
  -> BgeM3EmbeddingProvider
  -> ChromaVectorStore
  -> DenseRetriever(top_k=20)
  -> LearnedReranker(BAAI/bge-reranker-base)
  -> DenseRerankerRetriever(final_top_k=5)
  -> ContextBuilder
  -> Generator
  -> RAGService
```

Production 不再构造 Translation、BM25、MultilingualRetriever 或 RRF。

## Settings audit

- Before: 33 fields.
- After: 21 fields.
- Removed: `top_k`, `reranker_enabled`, `multilingual_candidate_top_k`,
  `multilingual_candidate_pool_k`, `multilingual_final_top_k`, `multilingual_rrf_k`,
  `multilingual_original_weight`, `multilingual_translation_weight`,
  `multilingual_hybrid_enabled`, `multilingual_hybrid_bm25_k1`,
  `multilingual_hybrid_bm25_b`, `multilingual_hybrid_rrf_k`.

## Evaluation and lifecycle review

`eval.ir_metrics` remains the canonical standard IR implementation. Candidate metrics, failure attribution
and bootstrap remain experiment-local because no second long-term consumer justifies a new abstraction.

Phase15A manifest/signature、source diff、staging validation、activation/rollback 均有真实 Gate 与测试
消费者；`BuildJob` 仍是同步状态记录，没有 queue、worker、heartbeat 或分布式锁。本轮未发现可在
不削弱索引安全合同的前提下删除的防御层。
