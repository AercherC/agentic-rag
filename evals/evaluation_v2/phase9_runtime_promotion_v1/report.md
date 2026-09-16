# Phase 9 Runtime Promotion + Architecture Simplification

- Runtime Promotion: **PASS**
- New default: `BGE-M3 Dense -> Chroma ANN Top20 -> Learned Reranker -> Top5`
- Rollback: **PASS** via `RERANKER_ENABLED=false`
- Translation / BM25 / Hybrid default: **disabled**
- RRF default final ranking: **disabled**
- Phase 8 quality reproduced: **True**
- nDCG@5 / MRR@5: 0.4634 / 0.4321
- CPU total p50 / p95: 4.889s / 5.027s
- Functional Runtime: **READY**
- Interactive Performance: **NEEDS_OPTIMIZATION**
- Formal Chroma unchanged: **True** (361 chunks)

Phase 9 在此停止；未执行模型替换、低精度、GPU、Context 或数据层优化。
